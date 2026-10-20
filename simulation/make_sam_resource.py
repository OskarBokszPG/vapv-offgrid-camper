# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: make_sam_resource.py
Opis: budowa plików zasobu słonecznego SAM z szeregów PVGIS. Model
      generacji nie jest zasilany plikiem EPW z PVGIS, tylko plikami CSV
      zasobu słonecznego budowanymi tutaj. Powód jest jeden i jest
      rozstrzygający: PVGIS zapisuje w nagłówku EPW ``TimeZone = 1``,
      podczas gdy wiersze danych są w UTC (dowód:
      ``weather_qc.epw_timebase_evidence``). Czytnik pogodowy SSC
      wyznacza położenie Słońca na podstawie pola strefy z nagłówka,
      więc podanie mu pliku PVGIS bez korekty daje błąd godziny w
      geometrii słonecznej.

Autor: Oskar Boksz
Uczelnia: Politechnika Gdańska
Wydział: Wydział Elektrotechniki i Automatyki
Kierunek: Elektrotechnika (studia II stopnia)
Poziom / Tytuł: mgr inż.

Temat pracy: Projekt i analiza efektywności energetycznej autonomicznego
             systemu zasilania off-grid typu VAPV dla pojazdu do celów
             turystycznych
Promotor: dr inż. Piotr Kołodziejek

Wsparcie AI / Narzędzia: Kod źródłowy napisany z wykorzystaniem narzędzia
                         Claude Code (Anthropic) na podstawie założeń,
                         modeli i decyzji projektowych autora. Autor
                         zweryfikował działanie kodu testami jednostkowymi
                         i kontrolami bilansu opisanymi w pracy.

Data utworzenia: 2026-10-19
Ostatnia modyfikacja: 2026-10-20
Wersja: 1.0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

import config
import weather_qc as qc
from config import LOCATIONS, Location

# Katalog plików zasobu słonecznego. Pliki są wytwarzane z danych surowych
# i mogą zostać odtworzone w każdej chwili, więc nie trafiają do raw/.
RESOURCE_DIR = config.WEATHER_DIR / "sam_resource"

# Kolumny danych wymagane przez czytnik SSC.
DATA_COLUMNS = ["Year", "Month", "Day", "Hour", "Minute",
                "GHI", "DNI", "DHI", "Tdry", "Wspd"]

# Próg wysokości Słońca, poniżej którego nie wyznaczamy DNI z rzutu składowej
# bezpośredniej. Przy sin(h) dążącym do zera iloraz rośnie bez ograniczenia
# i drobny błąd Gb daje wartości niefizyczne.
DNI_ELEVATION_GATE_DEG = 1.0
# Górne ograniczenie DNI. Stała słoneczna wynosi ok. 1361 W/m2; wartości
# powyżej tego progu na powierzchni Ziemi są niemożliwe.
DNI_MAX_W_M2 = 1400.0

# Kryteria bramki weryfikacyjnej położenia Słońca.
GATE_MAE_LIMIT_DEG = 0.5
GATE_SHIFT_LIMIT_DEG = 0.5


# ===========================================================================
# 1. Budowa plików zasobu
# ===========================================================================

def verify_elevation(loc: Location) -> tuple[bool, float | None]:
    """
    Sprawdza zgodność wysokości n.p.m. z config.py z nagłówkiem pliku PVGIS.

    Wysokość wpisana do tabeli lokalizacji jest wartością deklarowaną; tutaj
    porównywana jest z wartością zwróconą przez PVGIS dla tej komórki siatki.
    Niezgodność nie przerywa pracy, ale jest zgłaszana — wysokość wpływa na
    masę optyczną atmosfery, więc nie powinna się rozjeżdżać między plikami.
    """
    series_path = config.raw_series_path(loc, "PVGIS-SARAH3")
    if not series_path.exists():
        return True, None

    with open(series_path, "r", encoding="utf-8", errors="replace") as handle:
        for _ in range(10):
            line = handle.readline()
            if line.lower().startswith("elevation"):
                try:
                    reported = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    return True, None
                return abs(reported - loc.elevation_m) < 1.0, reported
    return True, None


def derive_dni(beam_horizontal: pd.Series, sun_elevation: pd.Series) -> np.ndarray:
    """
    Wyznacza DNI z poziomej składowej bezpośredniej i wysokości Słońca.

    PVGIS z parametrem ``components=1`` i kątem nachylenia 0 zwraca składowe
    w płaszczyźnie poziomej: ``Gb(i)`` bezpośrednią, ``Gd(i)`` rozproszoną.
    Składowa bezpośrednia na powierzchni poziomej jest rzutem natężenia
    promieniowania bezpośredniego:

        Gb(i) = DNI · cos(theta_z) = DNI · sin(h)

    stąd DNI = Gb(i) / sin(h). Dla h <= 1 stopnia iloraz jest niestabilny
    numerycznie i przyjmowane jest DNI = 0 — przy tak niskim Słońcu składowa
    bezpośrednia i tak jest pomijalna, a wartość niefizyczna psułaby model
    kąta padania. Wynik jest dodatkowo ograniczany z góry do 1400 W/m2.

    Konsekwencja metodyczna, do zapisania w 6.1: DNI w tych plikach nie jest
    wielkością niezależną — powstaje z GHI i geometrii. Test zamknięcia
    bilansu promieniowania (GHI = DNI·cos(theta_z) + DHI) jest dla nich
    tożsamościowy i nie niesie informacji. Niezależne są tylko GHI i DHI.
    """
    sin_elevation = np.sin(np.radians(np.asarray(sun_elevation, dtype=float)))
    beam = np.asarray(beam_horizontal, dtype=float)

    dni = np.zeros_like(beam, dtype=float)
    usable = np.asarray(sun_elevation, dtype=float) > DNI_ELEVATION_GATE_DEG
    dni[usable] = beam[usable] / sin_elevation[usable]
    return np.clip(np.nan_to_num(dni, nan=0.0, posinf=0.0, neginf=0.0),
                   0.0, DNI_MAX_W_M2)


def load_series_with_components(loc: Location) -> pd.DataFrame:
    """
    Wczytuje szereg SARAH3 wraz ze składowymi potrzebnymi do pliku zasobu.

    ``weather_qc.parse_seriescalc`` zwraca już DNI, ale ustawia wartości
    nieoznaczone jako NaN; plik zasobu wymaga liczb. Dlatego składowa
    bezpośrednia wyznaczana jest tu ponownie z GHI i DHI, a DNI liczone
    funkcją ``derive_dni``.
    """
    series_path = config.raw_series_path(loc, "PVGIS-SARAH3")
    if not series_path.exists():
        raise FileNotFoundError(
            f"Brak szeregu SARAH3 dla {loc.key}: {series_path}. "
            f"Uruchom fetch_weather.py."
        )

    frame, _ = qc.parse_seriescalc(series_path, loc, "SARAH3")

    # Składowa bezpośrednia na płaszczyźnie poziomej = GHI - DHI. Dla danych
    # PVGIS przy kącie 0 jest to dokładnie kolumna Gb(i), odtworzona tu
    # z wielkości zapisanych w formacie uporządkowanym.
    beam_horizontal = (frame["GHI"] - frame["DHI"]).clip(lower=0.0)

    result = pd.DataFrame(index=frame.index)
    result["GHI"] = frame["GHI"].clip(lower=0.0).to_numpy()
    result["DHI"] = frame["DHI"].clip(lower=0.0).to_numpy()
    result["DNI"] = derive_dni(beam_horizontal, frame["H_sun_pvgis"])
    result["Tdry"] = frame["T2m"].to_numpy()
    result["Wspd"] = frame["WS10m"].clip(lower=0.0).to_numpy()
    return result


def resource_path(loc: Location, year: int) -> Path:
    """Ścieżka pliku zasobu słonecznego dla lokalizacji i roku."""
    return RESOURCE_DIR / f"{loc.key}_sarah3_{year}.csv"


def write_resource_file(
    loc: Location, year: int, data: pd.DataFrame, drop_leap_day: bool
) -> tuple[Path, int]:
    """
    Zapisuje jeden rok w formacie CSV zasobu słonecznego SAM.

    Struktura pliku: wiersz nazw pól nagłówka, wiersz wartości nagłówka,
    wiersz nazw kolumn danych, dane. ``Time Zone = 0`` oznacza, że znaczniki
    czasu są w UTC — tak jak dane PVGIS.

    Kolumna ``Minute`` zawiera rzeczywiste przesunięcie minutowe znacznika
    PVGIS (11 min dla Łeby i Suwałk, 10 min dla Leska), a nie zero. Dzięki temu
    SSC wyznacza położenie Słońca dla chwili, której wartość promieniowania
    faktycznie dotyczy, a nie dla pełnej godziny.
    """
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    target = resource_path(loc, year)

    year_data = data.loc[data.index.year == year].copy()
    if drop_leap_day:
        # 29 lutego usuwany, aby każdy rok miał 8760 wierszy i lata były
        # porównywalne. Decyzja zapisana jawnie — pomija ok. 0,3 % energii
        # w latach przestępnych.
        mask = ~((year_data.index.month == 2) & (year_data.index.day == 29))
        year_data = year_data.loc[mask]

    stamps = year_data.index
    block = pd.DataFrame({
        "Year": stamps.year,
        "Month": stamps.month,
        "Day": stamps.day,
        "Hour": stamps.hour,
        "Minute": stamps.minute,
        "GHI": np.round(year_data["GHI"].to_numpy(), 2),
        "DNI": np.round(year_data["DNI"].to_numpy(), 2),
        "DHI": np.round(year_data["DHI"].to_numpy(), 2),
        "Tdry": np.round(year_data["Tdry"].to_numpy(), 2),
        "Wspd": np.round(year_data["Wspd"].to_numpy(), 2),
    })[DATA_COLUMNS]

    header_names = ["Source", "Location ID", "City", "State", "Country",
                    "Latitude", "Longitude", "Time Zone", "Elevation"]
    header_values = ["PVGIS-SARAH3", loc.wmo, loc.key, "", "Poland",
                     f"{loc.lat}", f"{loc.lon}", "0", f"{loc.elevation_m:.0f}"]

    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(",".join(header_names) + "\n")
        handle.write(",".join(str(v) for v in header_values) + "\n")
        handle.write(",".join(DATA_COLUMNS) + "\n")
        block.to_csv(handle, index=False, header=False, lineterminator="\n")

    return target, len(block)


def build_all(drop_leap_day: bool = True) -> pd.DataFrame:
    """Buduje pliki zasobu dla wszystkich lokalizacji i lat wielolecia."""
    rows: list[dict[str, object]] = []

    for loc in LOCATIONS.values():
        matches, reported = verify_elevation(loc)
        if not matches:
            print(f"  UWAGA {loc.key}: wysokość w config.py "
                  f"({loc.elevation_m} m) różni się od zgłoszonej przez PVGIS "
                  f"({reported} m)")

        data = load_series_with_components(loc)
        minutes = sorted(set(data.index.minute.tolist()))

        for year in range(config.SERIES_START_YEAR, config.SERIES_END_YEAR + 1):
            target, count = write_resource_file(loc, year, data, drop_leap_day)
            rows.append({
                "location": loc.key,
                "year": year,
                "file": target.name,
                "rows": count,
                "minute_offset": minutes[0] if len(minutes) == 1 else str(minutes),
                "elevation_m": loc.elevation_m,
                "time_zone": 0,
                "ghi_sum_kwh_m2": round(
                    float(data.loc[data.index.year == year, "GHI"].sum()) / 1000.0, 2
                ),
            })

    return pd.DataFrame(rows)


# ===========================================================================
# 2. Bramka weryfikacyjna położenia Słońca
# ===========================================================================

def _run_model_for_resource(resource_file: Path):
    """Wykonuje Pvsamv1 na zadanym pliku zasobu i zwraca obiekt modelu."""
    import PySAM.Pvsamv1 as Pvsamv1
    import run_pv

    model = Pvsamv1.default("FlatPlatePVNone")
    configuration = run_pv.zbuduj_konfiguracje(resource_file, None)
    run_pv._zastosuj(model, configuration)
    model.execute(0)
    return model


def _expected_timestamps(resource_file: Path) -> pd.DatetimeIndex:
    """
    Odtwarza znaczniki czasu UTC z pliku zasobu słonecznego.

    Czyta kolumny Year/Month/Day/Hour/Minute i buduje z nich indeks ze strefą
    UTC. Odczyt z pliku, a nie z pamięci, jest celowy: bramka ma sprawdzać to,
    co SSC faktycznie dostało.
    """
    block = pd.read_csv(resource_file, skiprows=2)
    stamps = pd.to_datetime(block[["Year", "Month", "Day", "Hour", "Minute"]])
    return pd.DatetimeIndex(stamps).tz_localize("UTC")


def _epw_timestamps(epw_file: Path, loc: Location) -> pd.DatetimeIndex:
    """Znaczniki czasu UTC pliku EPW, wyznaczone jak w weather_qc.parse_epw."""
    frame, _ = qc.parse_epw(epw_file, loc)
    return frame.index


def verify_sun_position(
    loc: Location,
    resource_file: Path,
    timestamps: pd.DatetimeIndex,
    label: str,
) -> dict[str, object]:
    """
    Porównuje położenie Słońca wyznaczone przez SSC z obliczeniami pvlib.

    Porównanie prowadzone jest wyłącznie dla godzin dziennych (wysokość Słońca
    wg pvlib powyżej 5 stopni). Nocą kąt zenitalny jest słabo określony,
    a różnice w modelu refrakcji dawałyby pozorne rozbieżności.

    Oprócz błędu bezwzględnego sprawdzane jest, czy przesunięcie osi czasu
    o +/-1 h nie dawałoby lepszego dopasowania. Jeśli dawałoby, to znaczy, że
    SSC interpretuje znaczniki inaczej niż zakładamy — właśnie ten przypadek
    bramka ma wychwycić.
    """
    model = _run_model_for_resource(resource_file)
    outputs = model.export()["Outputs"]

    ssc_zenith = np.asarray(outputs["sol_zen"], dtype=float)
    ssc_altitude = np.asarray(outputs["sol_alt"], dtype=float)

    if len(ssc_zenith) != len(timestamps):
        raise ValueError(
            f"{label}: SSC zwrócił {len(ssc_zenith)} wartości położenia Słońca, "
            f"a plik ma {len(timestamps)} znaczników."
        )

    reference = pvlib.solarposition.get_solarposition(
        timestamps, loc.lat, loc.lon, altitude=loc.elevation_m
    )
    pvlib_zenith = reference["zenith"].to_numpy()
    pvlib_altitude = reference["elevation"].to_numpy()

    daylight = pvlib_altitude > 5.0
    zenith_error = ssc_zenith[daylight] - pvlib_zenith[daylight]
    altitude_error = ssc_altitude[daylight] - pvlib_altitude[daylight]

    # Sprawdzenie, czy przesunięcie o całą godzinę nie pasuje lepiej.
    shift_scores: dict[int, float] = {}
    for shift in (-1, 0, 1):
        shifted = pvlib.solarposition.get_solarposition(
            timestamps + pd.Timedelta(hours=shift), loc.lat, loc.lon,
            altitude=loc.elevation_m,
        )["zenith"].to_numpy()
        shift_scores[shift] = float(
            np.mean(np.abs(ssc_zenith[daylight] - shifted[daylight]))
        )
    best_shift = min(shift_scores, key=shift_scores.get)

    zenith_mae = float(np.mean(np.abs(zenith_error)))
    passed = (
        zenith_mae < GATE_MAE_LIMIT_DEG
        and best_shift == 0
        and abs(float(np.mean(zenith_error))) < GATE_SHIFT_LIMIT_DEG
    )

    return {
        "label": label,
        "location": loc.key,
        "resource_file": resource_file.name,
        "n_daylight_hours": int(daylight.sum()),
        "zenith_mae_deg": round(zenith_mae, 4),
        "zenith_mbe_deg": round(float(np.mean(zenith_error)), 4),
        "zenith_max_abs_deg": round(float(np.max(np.abs(zenith_error))), 4),
        "altitude_mae_deg": round(float(np.mean(np.abs(altitude_error))), 4),
        "mae_shift_minus1h": round(shift_scores[-1], 4),
        "mae_shift_0h": round(shift_scores[0], 4),
        "mae_shift_plus1h": round(shift_scores[1], 4),
        "best_shift_h": best_shift,
        "gate_mae_limit_deg": GATE_MAE_LIMIT_DEG,
        "passed": bool(passed),
    }


def run_gate() -> tuple[pd.DataFrame, bool]:
    """
    Uruchamia bramkę na pliku zasobu SAM oraz, dla porównania, na pliku EPW.

    Wynik dla EPW jest materiałem do pracy: pokazuje, ile wynosi błąd, którego
    unika się rezygnując z podawania SSC pliku PVGIS wprost.
    """
    rows: list[dict[str, object]] = []
    loc = next(iter(LOCATIONS.values()))

    # --- plik zasobu SAM, Time Zone = 0 -----------------------------------
    resource_file = resource_path(loc, 2020)
    if not resource_file.exists():
        raise FileNotFoundError(
            f"Brak pliku zasobu {resource_file}. Uruchom bez --gate-only."
        )
    rows.append(verify_sun_position(
        loc, resource_file, _expected_timestamps(resource_file),
        "CSV zasobu SAM (Time Zone = 0)",
    ))

    # --- plik EPW z PVGIS, nagłówek deklaruje Time Zone = 1 ---------------
    epw_file = config.working_tmy_path(loc)
    if epw_file.exists():
        rows.append(verify_sun_position(
            loc, epw_file, _epw_timestamps(epw_file, loc),
            "EPW z PVGIS (nagłówek Time Zone = 1)",
        ))

    report = pd.DataFrame(rows)
    gate_passed = bool(report.loc[report["label"].str.startswith("CSV"), "passed"].all())
    return report, gate_passed


# ===========================================================================
# 3. Uruchomienie
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Budowa plików zasobu słonecznego SAM z szeregów PVGIS."
    )
    parser.add_argument("--no-build", action="store_true", dest="no_build",
                        help="pomiń budowę plików, uruchom samą bramkę")
    parser.add_argument("--gate-only", action="store_true", dest="gate_only",
                        help="to samo co --no-build")
    parser.add_argument("--keep-leap-day", action="store_true", dest="keep_leap",
                        help="zachowaj 29 lutego (lata przestępne po 8784 h)")
    args = parser.parse_args()

    config.ensure_dirs()

    if not (args.no_build or args.gate_only):
        print("=" * 72)
        print("Budowa plików zasobu słonecznego SAM")
        print("=" * 72)
        summary = build_all(drop_leap_day=not args.keep_leap)
        summary.to_csv(config.OUT_DIR / "sam_resource_files.csv", index=False)
        print(f"  zbudowano {len(summary)} plików w {RESOURCE_DIR}")
        counts = sorted(summary["rows"].unique().tolist())
        print(f"  liczba wierszy na plik: {counts}")
        print(f"  przesunięcia minutowe : "
              f"{sorted(set(summary['minute_offset'].astype(str)))}")

    print("\n" + "=" * 72)
    print("Bramka weryfikacyjna — położenie Słońca w SSC vs pvlib")
    print("=" * 72)
    report, gate_passed = run_gate()
    report.to_csv(config.OUT_DIR / "qc_ssc_sun_position.csv", index=False)

    for _, record in report.iterrows():
        print(f"\n  {record['label']}")
        print(f"    godziny dzienne        : {record['n_daylight_hours']}")
        print(f"    MAE kąta zenitalnego   : {record['zenith_mae_deg']:.4f}°")
        print(f"    MBE kąta zenitalnego   : {record['zenith_mbe_deg']:+.4f}°")
        print(f"    maksimum |błędu|       : {record['zenith_max_abs_deg']:.4f}°")
        print(f"    MAE przy przes. -1/0/+1 h: "
              f"{record['mae_shift_minus1h']:.3f} / "
              f"{record['mae_shift_0h']:.3f} / "
              f"{record['mae_shift_plus1h']:.3f}°")
        print(f"    najlepsze przesunięcie : {record['best_shift_h']:+d} h")
        print(f"    wynik                  : "
              f"{'ZALICZONA' if record['passed'] else 'NIEZALICZONA'}")

    print(f"\nZapisano: {config.OUT_DIR / 'qc_ssc_sun_position.csv'}")

    if not gate_passed:
        print("\nBRAMKA NIEZALICZONA dla pliku zasobu SAM — przerwano.")
        return 1
    print("\nBramka zaliczona dla pliku zasobu SAM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
