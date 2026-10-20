# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: weather_qc.py
Opis: wczytanie, ujednolicenie osi czasu i kontrola jakości danych.

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
from config import LOCATIONS, Location

# ---------------------------------------------------------------------------
# Progi kontroli jakości
# ---------------------------------------------------------------------------
# Wartości przyjęte przez autora na podstawie zakresów fizycznie możliwych dla
# szerokości geograficznej Polski. Nie pochodzą z normy ani z publikacji —
# „No verified source available for this claim" dla konkretnych liczb.
NIGHT_IRRADIANCE_LIMIT = 5.0      # W/m2 — dopuszczalne zero instrumentalne nocą
NIGHT_ELEVATION_DEG = 0.0         # poniżej tej wysokości Słońca oczekujemy zera
KT_ELEVATION_GATE_DEG = 5.0       # kt liczony dopiero powyżej tej wysokości
KT_LIMIT = 1.0                    # wskaźnik przejrzystości nie może przekroczyć 1
CLOSURE_ELEVATION_GATE_DEG = 10.0  # test zamknięcia tylko wysoko nad horyzontem
CLOSURE_REL_TOLERANCE = 0.08      # dopuszczalne odchylenie względne zamknięcia
DHI_OVER_GHI_TOLERANCE = 0.05     # DHI może minimalnie przekroczyć GHI
DHI_ABSOLUTE_TOLERANCE = 5.0      # W/m2 — przy bardzo małych wartościach
T2M_MIN, T2M_MAX = -40.0, 40.0    # degC
WS10M_MIN, WS10M_MAX = 0.0, 40.0  # m/s

# Kolumny docelowego formatu uporządkowanego.
TIDY_COLUMNS = ["GHI", "DNI", "DHI", "T2m", "WS10m", "source", "location"]

# ---------------------------------------------------------------------------
# Baza czasu plików EPW z PVGIS — ustalenie doświadczalne
# ---------------------------------------------------------------------------
# PVGIS deklaruje w wierszu LOCATION pliku EPW strefę czasową 1 (UTC+1), ale
# wiersze danych zapisane są w UTC. Pole strefy opisuje czas urzędowy
# lokalizacji, nie bazę czasu danych.
#
# Dowód (funkcja ``epw_timebase_evidence``, tabela out/qc_epw_timebase.csv):
# profil dobowy GHI z pliku TMY skorelowano z profilem z szeregu SARAH3 tej
# samej lokalizacji, którego znaczniki są jawnie w UTC. Maksimum korelacji
# wypada dla przesunięcia +1 h względem interpretacji „czas lokalny"
# (r > 0,9996 dla wszystkich trzech lokalizacji), a środek ciężkości profilu
# zgadza się wtedy z SARAH3 z dokładnością do 2 min. Przy interpretacji
# „czas lokalny" maksimum dobowe wypadałoby ok. 70-80 min przed południem
# słonecznym, co jest fizycznie nieuzasadnione.
#
# KONSEKWENCJA DLA MODELU PV: czytnik pliku pogodowego SAM/SSC wyznacza
# położenie Słońca na podstawie pola strefy czasowej z nagłówka. Podanie mu
# pliku PVGIS bez korekty grozi błędem godziny w geometrii słonecznej.
# Zagadnienie otwarte — do sprawdzenia po uruchomieniu PySAM (rozdz. 6.1).
EPW_DECLARED_TZ_APPLIES = False


# ===========================================================================
# 1. Wczytanie plików EPW
# ===========================================================================

def read_epw_header(path: Path) -> dict[str, object]:
    """
    Odczytuje z nagłówka EPW pola potrzebne do ustalenia osi czasu.

    Zwracane pola:
      ``tz_field``        — strefa czasowa zadeklarowana w wierszu LOCATION,
      ``elevation_m``     — wysokość n.p.m. zapisana przez PVGIS,
      ``time_offset_h``   — wartość „Irradiance Time Offset (h)" z COMMENTS 2.

    Konwencja EPW: pole ``hour`` przyjmuje wartości 1-24 i oznacza godzinę
    **kończącą** interwał (hour = 1 to interwał 00:00-01:00). Dodatkowo PVGIS
    zapisuje w komentarzu przesunięcie czasowe promieniowania względem końca
    interwału — wartość ujemna cofa moment reprezentatywny.
    """
    header: dict[str, object] = {
        "tz_field": None,
        "elevation_m": None,
        "time_offset_h": None,
        "source_comment": None,
    }
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for _ in range(8):
            line = handle.readline()
            if not line:
                break
            if line.startswith("LOCATION"):
                fields = line.strip().split(",")
                if len(fields) >= 10:
                    header["tz_field"] = float(fields[8])
                    header["elevation_m"] = float(fields[9])
            elif line.startswith("COMMENTS 1"):
                header["source_comment"] = line.strip()[11:]
            elif line.startswith("COMMENTS 2"):
                # Format: "COMMENTS 2,Irradiance Time Offset (h):-0.8145"
                if "Irradiance Time Offset" in line:
                    try:
                        header["time_offset_h"] = float(line.strip().split(":")[-1])
                    except ValueError:
                        pass
    return header


def parse_epw(path: Path, loc: Location) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Wczytuje plik EPW do formatu uporządkowanego z osią czasu w UTC.

    Oś czasu budowana jest w dwóch krokach, każdy udokumentowany, ponieważ
    pomyłka w którymkolwiek daje przesunięcie generacji o całe godziny:

      1. z pól rok/miesiąc/dzień/godzina odtwarzany jest **początek** interwału
         (pole ``hour`` ma wartości 1-24 i oznacza godzinę kończącą, więc
         początek interwału to ``hour-1`` godzin od północy),
      2. do początku interwału dodawane jest przesunięcie momentu
         reprezentatywnego: ``1 h + time_offset_h`` (dla -0,8145 h daje to
         ok. 11 min po początku godziny, co odpowiada znacznikom ``:11``
         w szeregach ``seriescalc`` z tej samej bazy).

    Zadeklarowana w nagłówku strefa czasowa **nie jest odejmowana** — wiersze
    danych PVGIS są już w UTC (patrz komentarz przy ``EPW_DECLARED_TZ_APPLIES``).
    Założenie to jest weryfikowane testem położenia maksimum dobowego GHI
    (funkcja ``time_axis_report``), a nie przyjmowane na wiarę.
    """
    header = read_epw_header(path)

    # Plik EPW ma dokładnie 8 wierszy nagłówka; dalej same dane, bez nazw kolumn.
    raw = pd.read_csv(path, skiprows=8, header=None, low_memory=False)

    year = raw[0].astype(int)
    month = raw[1].astype(int)
    day = raw[2].astype(int)
    hour = raw[3].astype(int)   # 1-24, godzina kończąca interwał

    interval_start_local = pd.to_datetime(
        {"year": year, "month": month, "day": day, "hour": hour - 1}
    )

    offset_h = header["time_offset_h"]
    if offset_h is None:
        # Brak komentarza z przesunięciem — przyjmujemy środek interwału
        # i zapisujemy ten fakt, bo zmienia interpretację testów geometrycznych.
        representative_offset_h = 0.5
        header["time_offset_source"] = "brak w nagłówku; przyjęto środek interwału"
    else:
        representative_offset_h = 1.0 + float(offset_h)
        header["time_offset_source"] = "COMMENTS 2 pliku EPW"
    header["representative_offset_min"] = representative_offset_h * 60.0

    tz_field = header["tz_field"] if header["tz_field"] is not None else 0.0
    applied_tz_h = float(tz_field) if EPW_DECLARED_TZ_APPLIES else 0.0
    header["declared_tz_applied"] = EPW_DECLARED_TZ_APPLIES

    instant_utc = (
        interval_start_local
        + pd.to_timedelta(representative_offset_h, unit="h")
        - pd.to_timedelta(applied_tz_h, unit="h")
    )

    frame = pd.DataFrame({
        # Kolumny EPW liczone od zera: 6 = temperatura powietrza, 13 = GHI,
        # 14 = DNI, 15 = DHI, 21 = prędkość wiatru.
        "GHI": raw[13].astype(float),
        "DNI": raw[14].astype(float),
        "DHI": raw[15].astype(float),
        "T2m": raw[6].astype(float),
        "WS10m": raw[21].astype(float),
        "source": "TMY",
        "location": loc.key,
    })
    frame.index = pd.DatetimeIndex(instant_utc, name="timestamp_utc").tz_localize("UTC")

    # EPW oznacza brak danych wartością 9999 (promieniowanie) i 99.9 (temperatura).
    for column in ("GHI", "DNI", "DHI"):
        frame.loc[frame[column] >= 9999.0, column] = np.nan

    header["rows"] = len(frame)
    header["file"] = path.name
    return frame, header


# ===========================================================================
# 2. Wczytanie szeregów seriescalc
# ===========================================================================

def parse_seriescalc(
    path: Path, loc: Location, source: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Wczytuje szereg godzinowy PVGIS ``seriescalc`` do formatu uporządkowanego.

    Struktura pliku: kilka wierszy metadanych, wiersz nazw kolumn zaczynający
    się od ``time``, dane, wiersz pusty, opis kolumn, stopka. Wiersz nazw
    kolumn odnajdywany jest przez przeszukanie, a nie przez stałą liczbę
    wierszy do pominięcia, ponieważ liczba wierszy metadanych zależy od bazy.

    OŚ CZASU — decyzja jawna. Znaczniki mają postać ``YYYYMMDD:HHMM``, gdzie
    ``MM`` nie jest zerem (dla SARAH3 wynosi 11). Minuta odpowiada momentowi
    obserwacji satelitarnej dla danego piksela: skan Meteosat przechodzi przez
    szerokość geograficzną punktu kilkanaście minut po nominalnym początku
    slotu. Wartość jest więc chwilowa dla ``HH:MM``, a reprezentuje godzinę
    rozpoczynającą się o ``HH:00``. Przyjęto:
      - do testów geometrycznych (położenie Słońca, kt, zamknięcie bilansu)
        używany jest **dokładny znacznik** ``HH:MM``, bo tylko on odpowiada
        rzeczywistemu momentowi pomiaru,
      - do agregacji energii znacznik zaokrąglany jest w dół do ``HH:00``.
    Ta sama konwencja wynika z komentarza „Irradiance Time Offset" w plikach
    EPW tej samej bazy, co stanowi niezależne potwierdzenie interpretacji.

    SKŁADOWE PROMIENIOWANIA. Przy ``angle=0`` i ``components=1`` PVGIS zwraca
    składowe w płaszczyźnie poziomej: ``Gb(i)`` bezpośrednią, ``Gd(i)``
    rozproszoną i ``Gr(i)`` odbitą. Nie ma kolumny GHI ani DNI — oba są
    wyprowadzane:
        GHI = Gb(i) + Gd(i) + Gr(i)          (dla płaszczyzny poziomej Gr = 0)
        DNI = Gb(i) / sin(H_sun)              dla H_sun > 0
    Konsekwencja metodyczna: test zamknięcia bilansu promieniowania traci dla
    tych plików sens, bo DNI powstało z przekształcenia GHI. Test zachowuje
    wartość wyłącznie dla plików EPW, gdzie trzy składowe są niezależne.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    header_index = None
    for index, line in enumerate(lines):
        if line.startswith("time,"):
            header_index = index
            break
    if header_index is None:
        raise ValueError(f"Nie znaleziono wiersza nagłówka 'time,' w {path.name}")

    # Dane kończą się pierwszym wierszem, który nie zaczyna się od cyfry.
    data_lines: list[str] = []
    for line in lines[header_index + 1:]:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            break
        data_lines.append(stripped)

    columns = [name.strip() for name in lines[header_index].strip().split(",")]
    records = [line.split(",") for line in data_lines]
    raw = pd.DataFrame(records, columns=columns)

    # Znacznik czasu: YYYYMMDD:HHMM. Traktowany jako UTC — założenie
    # weryfikowane testem położenia maksimum dobowego GHI.
    instant_utc = pd.to_datetime(raw["time"], format="%Y%m%d:%H%M", utc=True)
    minute_offsets = sorted(instant_utc.dt.minute.unique().tolist())

    beam = raw["Gb(i)"].astype(float)
    diffuse = raw["Gd(i)"].astype(float)
    reflected = raw["Gr(i)"].astype(float)
    sun_elevation = raw["H_sun"].astype(float)

    ghi = beam + diffuse + reflected

    # DNI z rzutu składowej bezpośredniej. Dla małych wysokości Słońca dzielenie
    # przez sinus jest niestabilne, dlatego poniżej 1 stopnia DNI nie jest
    # wyznaczane (pozostaje NaN) zamiast przyjmować wartości rozbieżne.
    sin_elevation = np.sin(np.radians(sun_elevation))
    dni = pd.Series(np.nan, index=raw.index, dtype=float)
    usable = sun_elevation > 1.0
    dni.loc[usable] = beam.loc[usable] / sin_elevation.loc[usable]

    frame = pd.DataFrame({
        "GHI": ghi.to_numpy(),
        "DNI": dni.to_numpy(),
        "DHI": diffuse.to_numpy(),
        "T2m": raw["T2m"].astype(float).to_numpy(),
        "WS10m": raw["WS10m"].astype(float).to_numpy(),
        "source": source,
        "location": loc.key,
    })
    frame.index = pd.DatetimeIndex(instant_utc.to_numpy(), name="timestamp_utc")

    # PVGIS oznacza w kolumnie Int rekordy o promieniowaniu odtworzonym
    # (interpolowanym) — własny znacznik jakości bazy, przenoszony do wyniku.
    if "Int" in raw.columns:
        frame["pvgis_reconstructed"] = raw["Int"].astype(float).to_numpy() > 0.5
    else:
        frame["pvgis_reconstructed"] = False

    # Wysokość Słońca podana przez PVGIS zachowywana do porównania z pvlib —
    # zgodność obu jest dodatkowym testem poprawności osi czasu.
    frame["H_sun_pvgis"] = sun_elevation.to_numpy()

    meta: dict[str, object] = {
        "file": path.name,
        "rows": len(frame),
        "minute_offsets": minute_offsets,
        "first": str(frame.index[0]),
        "last": str(frame.index[-1]),
        "source": source,
    }
    return frame, meta


# ===========================================================================
# 3. Wczytanie wszystkiego naraz
# ===========================================================================

def load_all(verbose: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """
    Wczytuje wszystkie dostępne zbiory do jednej ramki uporządkowanej.

    Zwraca krotkę (ramka, lista metadanych plików). Brakujące pliki są
    pomijane z komunikatem — kontrola jakości ma działać także wtedy, gdy
    część danych nie została jeszcze pobrana.
    """
    frames: list[pd.DataFrame] = []
    metadata: list[dict] = []

    for loc in LOCATIONS.values():
        # Plik TMY.
        tmy_path = config.raw_tmy_path(loc)
        if tmy_path.exists():
            frame, meta = parse_epw(tmy_path, loc)
            meta["location"] = loc.key
            frames.append(frame)
            metadata.append(meta)
            if verbose:
                print(f"  {loc.key:9s} TMY     {meta['rows']:6d} wierszy  "
                      f"tz={meta['tz_field']}  offset={meta['representative_offset_min']:.1f} min")
        elif verbose:
            print(f"  {loc.key:9s} TMY     BRAK PLIKU {tmy_path.name}")

        # Szeregi wieloletnie.
        for database in config.RADIATION_DATABASES:
            source = database.replace("PVGIS-", "")
            series_path = config.raw_series_path(loc, database)
            if not series_path.exists():
                if verbose:
                    print(f"  {loc.key:9s} {source:7s} BRAK PLIKU {series_path.name}")
                continue
            frame, meta = parse_seriescalc(series_path, loc, source)
            meta["location"] = loc.key
            frames.append(frame)
            metadata.append(meta)
            if verbose:
                print(f"  {loc.key:9s} {source:7s} {meta['rows']:6d} wierszy  "
                      f"minuty={meta['minute_offsets']}  "
                      f"{meta['first'][:10]}..{meta['last'][:10]}")

    if not frames:
        raise RuntimeError(
            "Nie wczytano żadnego pliku. Uruchom najpierw fetch_weather.py."
        )

    combined = pd.concat(frames)
    return combined, metadata


# ===========================================================================
# 4. Czas lokalny
# ===========================================================================

def tmy_source_years() -> pd.DataFrame:
    """
    Odczytuje, z jakiego roku kalendarzowego pochodzi każdy miesiąc pliku TMY.

    Typowy rok meteorologiczny nie jest rokiem rzeczywistym — składa się
    z miesięcy wybranych z różnych lat okresu bazowego. Tabela jest konieczna
    do oceny reprezentatywności (etap 5a): jeśli miesiące TMY pochodzą
    w znacznej części spoza okresu, z którym TMY jest porównywany, porównanie
    nie jest testem reprezentatywności tego samego okresu i trzeba to
    powiedzieć wprost.
    """
    rows: list[dict[str, object]] = []
    for loc in LOCATIONS.values():
        path = config.raw_tmy_path(loc)
        if not path.exists():
            continue
        raw = pd.read_csv(path, skiprows=8, header=None, usecols=[0, 1],
                          names=["year", "month"], low_memory=False)
        per_month = raw.groupby("month")["year"].agg(
            lambda series: int(series.mode().iloc[0])
        )
        for month, year in per_month.items():
            rows.append({
                "location": loc.key,
                "month": int(month),
                "source_year": int(year),
                "within_series_period": bool(
                    config.SERIES_START_YEAR <= year <= config.SERIES_END_YEAR
                ),
            })
    return pd.DataFrame(rows)


def to_local_time(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Przelicza indeks czasu z UTC na czas lokalny Europe/Warsaw.

    Konwersja uwzględnia czas letni. Służy do raportowania dobowego
    i do definiowania profili obciążenia, które użytkownik zadaje w czasie
    zegarowym, nie w UTC. Do łączenia z generacją PV wynik musi zostać
    przeliczony z powrotem na UTC — patrz ``battery_balance.load_profile_utc``.
    """
    local = frame.copy()
    local.index = local.index.tz_convert(config.LOCAL_TZ)
    local.index.name = "timestamp_local"
    return local


# ===========================================================================
# 5. Geometria słoneczna
# ===========================================================================

def add_sun_elevation(frame: pd.DataFrame, loc: Location) -> pd.DataFrame:
    """
    Dolicza wyłącznie wysokość Słońca — wariant tani obliczeniowo.

    Pełna funkcja ``add_solar_geometry`` wyznacza dodatkowo promieniowanie
    pozaziemskie uśrednione po godzinie, co wymaga jedenastu wywołań modelu
    położenia Słońca na rekord. Przy szeregach 10-letnich i analizach, którym
    wystarcza maska dnia, koszt ten jest zbędny.
    """
    result = frame.copy()
    position = pvlib.solarposition.get_solarposition(result.index, loc.lat, loc.lon)
    result["sun_elevation"] = position["apparent_elevation"].to_numpy()
    return result


def add_solar_geometry(frame: pd.DataFrame, loc: Location) -> pd.DataFrame:
    """
    Dolicza położenie Słońca i promieniowanie pozaziemskie dla jednej lokalizacji.

    Położenie Słońca liczone jest dla **dokładnych** znaczników czasu, nie dla
    pełnych godzin — przy znacznikach ``:11`` różnica w wysokości Słońca
    dochodzi do kilku stopni, co przy progach testów ma znaczenie.

    Do progów wysokości używana jest wysokość pozorna (z refrakcją), do
    promieniowania pozaziemskiego kąt zenitalny geometryczny — taka jest
    konwencja definicji stałej słonecznej rzutowanej na poziom.
    """
    result = frame.copy()
    position = pvlib.solarposition.get_solarposition(
        result.index, loc.lat, loc.lon
    )
    result["sun_elevation"] = position["apparent_elevation"].to_numpy()
    result["sun_zenith"] = position["zenith"].to_numpy()

    extra_dni = pvlib.irradiance.get_extra_radiation(result.index)
    cos_zenith = np.cos(np.radians(result["sun_zenith"].to_numpy()))
    result["etr_horizontal"] = np.clip(
        np.asarray(extra_dni, dtype=float) * cos_zenith, 0.0, None
    )

    # Promieniowanie pozaziemskie uśrednione po godzinie, do której odnosi się
    # rekord. Wariant konieczny, ponieważ zbiory mają różny charakter wartości:
    # SARAH3 podaje wartość chwilową (znacznik :11), ERA5 średnią godzinową
    # (znacznik :30). Porównanie średniej godzinowej z wartością chwilową
    # w środku godziny daje pozorne przekroczenia granicy pozaziemskiej przy
    # wschodzie i zachodzie Słońca, gdy wysokość Słońca szybko rośnie.
    hour_start = result.index.floor("h")
    accumulator = np.zeros(len(result), dtype=float)
    sub_samples = 10
    for step in range(sub_samples):
        sample_time = hour_start + pd.to_timedelta(
            (step + 0.5) * 60.0 / sub_samples, unit="m"
        )
        sample_position = pvlib.solarposition.get_solarposition(
            sample_time, loc.lat, loc.lon
        )
        sample_extra = np.asarray(
            pvlib.irradiance.get_extra_radiation(sample_time), dtype=float
        )
        sample_cos = np.cos(np.radians(sample_position["zenith"].to_numpy()))
        accumulator += np.clip(sample_extra * sample_cos, 0.0, None)
    result["etr_horizontal_hour_mean"] = accumulator / sub_samples

    # Wskaźnik przejrzystości atmosfery. Liczony wyłącznie powyżej progu
    # wysokości Słońca — przy małym mianowniku iloraz traci sens.
    kt = pd.Series(np.nan, index=result.index, dtype=float)
    gate = result["sun_elevation"] > KT_ELEVATION_GATE_DEG
    kt.loc[gate] = (
        result.loc[gate, "GHI"] / result.loc[gate, "etr_horizontal"]
    )
    result["kt"] = kt.to_numpy()
    return result


# ===========================================================================
# 6. Diagnostyka osi czasu
# ===========================================================================

def time_axis_report(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Sprawdza położenie maksimum dobowego GHI względem południa słonecznego.

    Test wykrywa najgroźniejszą klasę błędu — pomyłkę strefy czasowej.
    Dla każdej pary (lokalizacja, zbiór) wyznaczana jest godzina UTC
    o najwyższym średnim GHI i porównywana z przybliżonym południem
    słonecznym ``12:00 - lon/15``. Przesunięcie równe lub większe od 1 h
    oznacza błąd obsługi czasu.

    Dodatkowo porównywana jest wysokość Słońca podana przez PVGIS z wyliczoną
    przez pvlib dla tych samych znaczników; rozbieżność potwierdziłaby, że
    znaczniki nie są w UTC.
    """
    rows: list[dict[str, object]] = []
    for (location, source), group in frame.groupby(
        ["location", "source"], observed=True
    ):
        loc = LOCATIONS[str(location)]
        hourly_mean = group.groupby(group.index.hour)["GHI"].mean()
        peak_hour = int(hourly_mean.idxmax())
        expected = loc.solar_noon_utc_h

        # Różnica cykliczna w zakresie -12..+12 h.
        shift = (peak_hour - expected + 12.0) % 24.0 - 12.0

        row: dict[str, object] = {
            "location": location,
            "source": source,
            "peak_hour_utc": peak_hour,
            "solar_noon_utc_h": round(expected, 2),
            "shift_h": round(shift, 2),
            "verdict": "OK" if abs(shift) < 1.0 else "BŁĄD OSI CZASU",
        }

        # Zgodność wysokości Słońca z pvlib (tylko tam, gdzie PVGIS ją podaje).
        if "H_sun_pvgis" in group.columns and group["H_sun_pvgis"].notna().any():
            with_geometry = add_solar_geometry(group, loc)
            daylight = with_geometry["H_sun_pvgis"] > 5.0
            if daylight.any():
                difference = (
                    with_geometry.loc[daylight, "sun_elevation"]
                    - with_geometry.loc[daylight, "H_sun_pvgis"]
                )
                row["sun_elev_mae_deg"] = round(float(difference.abs().mean()), 3)
                row["sun_elev_max_abs_deg"] = round(float(difference.abs().max()), 3)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["location", "source"]).reset_index(drop=True)


def epw_timebase_evidence(max_lag_h: int = 3) -> pd.DataFrame:
    """
    Dowód doświadczalny bazy czasu plików EPW z PVGIS.

    Profil dobowy GHI z pliku TMY porównywany jest z profilem z szeregu SARAH3
    tej samej lokalizacji, którego znaczniki są jawnie w UTC. Dla każdego
    przesunięcia całogodzinnego liczony jest współczynnik korelacji profili
    oraz środek ciężkości profilu TMY. Przesunięcie o najwyższej korelacji
    wskazuje właściwą interpretację bazy czasu.

    Funkcja jest zarazem testem regresyjnym: jeśli PVGIS zmieni konwencję
    zapisu, kolumna ``best_lag_h`` przestanie wynosić zero i kontrola jakości
    zgłosi błąd osi czasu.
    """
    rows: list[dict[str, object]] = []

    for loc in LOCATIONS.values():
        tmy_path = config.raw_tmy_path(loc)
        series_path = config.raw_series_path(loc, "PVGIS-SARAH3")
        if not (tmy_path.exists() and series_path.exists()):
            continue

        tmy, _ = parse_epw(tmy_path, loc)
        series, _ = parse_seriescalc(series_path, loc, "SARAH3")

        reference = series.groupby(series.index.hour)["GHI"].mean()
        reference = reference.reindex(range(24), fill_value=0.0)
        reference_normalised = reference / reference.sum()
        reference_centroid = float(
            (reference_normalised.index.to_numpy() * reference_normalised.to_numpy()).sum()
        )

        for lag in range(-max_lag_h, max_lag_h + 1):
            shifted = tmy.copy()
            shifted.index = shifted.index + pd.to_timedelta(lag, unit="h")
            profile = shifted.groupby(shifted.index.hour)["GHI"].mean()
            profile = profile.reindex(range(24), fill_value=0.0)
            profile_normalised = profile / profile.sum()
            correlation = float(np.corrcoef(
                profile_normalised.to_numpy(), reference_normalised.to_numpy()
            )[0, 1])
            centroid = float(
                (profile_normalised.index.to_numpy() * profile_normalised.to_numpy()).sum()
            )
            rows.append({
                "location": loc.key,
                "lag_h": lag,
                "correlation_with_sarah3": round(correlation, 6),
                "tmy_centroid_h_utc": round(centroid, 3),
                "sarah3_centroid_h_utc": round(reference_centroid, 3),
                "centroid_difference_h": round(centroid - reference_centroid, 3),
                "solar_noon_h_utc": round(loc.solar_noon_utc_h, 2),
            })

    evidence = pd.DataFrame(rows)
    if evidence.empty:
        return evidence

    # Dla każdej lokalizacji wskazanie przesunięcia o najwyższej korelacji.
    best = (
        evidence.loc[evidence.groupby("location")["correlation_with_sarah3"].idxmax()]
        [["location", "lag_h"]]
        .rename(columns={"lag_h": "best_lag_h"})
    )
    return evidence.merge(best, on="location", how="left")


# ===========================================================================
# 7. Kontrola jakości
# ===========================================================================

def run_checks(frame: pd.DataFrame, loc: Location) -> pd.DataFrame:
    """
    Wykonuje testy jakości dla jednej pary (lokalizacja, zbiór).

    Zwraca ramkę z kolumnami znaczników ``flag_*`` typu logicznego. Rekordy
    nie są usuwane ani modyfikowane.
    """
    data = add_solar_geometry(frame, loc)

    night = data["sun_elevation"] < NIGHT_ELEVATION_DEG
    kt_gate = data["sun_elevation"] > KT_ELEVATION_GATE_DEG
    closure_gate = data["sun_elevation"] > CLOSURE_ELEVATION_GATE_DEG

    flags = pd.DataFrame(index=data.index)

    # --- wartości ujemne ---------------------------------------------------
    flags["flag_ghi_negative"] = data["GHI"] < 0.0
    flags["flag_dni_negative"] = data["DNI"] < 0.0
    flags["flag_dhi_negative"] = data["DHI"] < 0.0

    # --- promieniowanie nocą ----------------------------------------------
    flags["flag_night_irradiance"] = night & (data["GHI"] > NIGHT_IRRADIANCE_LIMIT)

    # --- granica pozaziemska ----------------------------------------------
    # Test podstawowy: wartość chwilowa w momencie reprezentatywnym rekordu.
    flags["flag_ghi_above_etr"] = data["GHI"] > data["etr_horizontal"] + 1.0
    # Test uściślony: granica uśredniona po godzinie, właściwa dla zbiorów
    # podających średnie godzinowe. Różnica obu liczników pokazuje, ile
    # przekroczeń wynika wyłącznie z niezgodności charakteru wartości.
    flags["flag_ghi_above_etr_hourly"] = (
        data["GHI"] > data["etr_horizontal_hour_mean"] + 1.0
    )

    # --- wskaźnik przejrzystości ------------------------------------------
    flags["flag_kt_above_1"] = kt_gate & (data["kt"] > KT_LIMIT)

    # --- relacja składowych -----------------------------------------------
    flags["flag_dhi_above_ghi"] = data["DHI"] > (
        data["GHI"] * (1.0 + DHI_OVER_GHI_TOLERANCE) + DHI_ABSOLUTE_TOLERANCE
    )

    # --- zamknięcie bilansu promieniowania --------------------------------
    # Dla szeregów seriescalc test jest tożsamościowy (DNI wyprowadzono z GHI),
    # dlatego jego wynik interpretować wolno tylko dla plików EPW.
    reconstructed = (
        data["DNI"] * np.cos(np.radians(data["sun_zenith"])) + data["DHI"]
    )
    relative_deviation = pd.Series(np.nan, index=data.index, dtype=float)
    meaningful = closure_gate & (data["GHI"] > 20.0)
    relative_deviation.loc[meaningful] = (
        (reconstructed.loc[meaningful] - data.loc[meaningful, "GHI"])
        / data.loc[meaningful, "GHI"]
    )
    flags["flag_closure"] = relative_deviation.abs() > CLOSURE_REL_TOLERANCE
    flags["closure_rel_dev"] = relative_deviation

    # --- zakresy meteorologiczne ------------------------------------------
    flags["flag_t2m_range"] = (data["T2m"] < T2M_MIN) | (data["T2m"] > T2M_MAX)
    flags["flag_ws_range"] = (data["WS10m"] < WS10M_MIN) | (data["WS10m"] > WS10M_MAX)

    # --- brak danych -------------------------------------------------------
    flags["flag_missing"] = data[["GHI", "DHI", "T2m", "WS10m"]].isna().any(axis=1)

    # --- znacznik własny PVGIS --------------------------------------------
    if "pvgis_reconstructed" in data.columns:
        flags["flag_pvgis_reconstructed"] = data["pvgis_reconstructed"].fillna(False)
    else:
        flags["flag_pvgis_reconstructed"] = False

    flags["kt"] = data["kt"]
    return flags


def expected_hours(frame: pd.DataFrame, source: str) -> int:
    """
    Zwraca oczekiwaną liczbę godzin dla zbioru.

    Dla TMY jest to 8760 (rok typowy jest zawsze nieprzestępny). Dla szeregów
    wieloletnich liczba wynika z kalendarza objętego zakresem danych, łącznie
    z dniami 29 lutego w latach przestępnych.
    """
    if source == "TMY":
        return 8760
    start = frame.index.min().floor("h")
    end = frame.index.max().floor("h")
    return int((end - start).total_seconds() // 3600) + 1


def summarise(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wykonuje kontrolę jakości dla wszystkich par (lokalizacja, zbiór).

    Zwraca krotkę (podsumowanie zbiorcze, rozkład wskaźnika kt).
    """
    summary_rows: list[dict[str, object]] = []
    kt_rows: list[dict[str, object]] = []

    for (location, source), group in frame.groupby(
        ["location", "source"], observed=True
    ):
        loc = LOCATIONS[str(location)]
        group = group.sort_index()
        flags = run_checks(group, loc)

        total = len(group)
        duplicates = int(group.index.duplicated().sum())
        expected = expected_hours(group, str(source))

        # Luki w osi czasu. Dla TMY badana jest ciągłość kalendarza
        # znormalizowanego (miesiąc, dzień, godzina), ponieważ typowy rok
        # składa się z miesięcy pochodzących z różnych lat kalendarzowych
        # i pole roku zmienia się na granicach miesięcy. Liczenie różnic
        # między znacznikami dałoby tu 11 pozornych nieciągłości.
        if str(source) == "TMY":
            calendar = pd.MultiIndex.from_arrays(
                [group.index.month, group.index.day, group.index.hour]
            )
            expected_calendar = pd.MultiIndex.from_arrays([
                pd.date_range("2019-01-01", periods=8760, freq="h").month,
                pd.date_range("2019-01-01", periods=8760, freq="h").day,
                pd.date_range("2019-01-01", periods=8760, freq="h").hour,
            ])
            gaps = int(len(expected_calendar.difference(calendar)))
        else:
            deltas = pd.Series(group.index).diff().dropna()
            gaps = int((deltas != pd.Timedelta(hours=1)).sum())

        row: dict[str, object] = {
            "location": location,
            "source": source,
            "rows": total,
            "expected_hours": expected,
            "missing_hours": expected - total + duplicates,
            "duplicates": duplicates,
            "time_gaps": gaps,
            "first_utc": str(group.index.min()),
            "last_utc": str(group.index.max()),
        }

        flag_columns = [c for c in flags.columns if c.startswith("flag_")]
        for column in flag_columns:
            row[column] = int(flags[column].sum())
        row["flags_total"] = int(
            flags[[c for c in flag_columns if c != "flag_pvgis_reconstructed"]]
            .any(axis=1).sum()
        )

        # Statystyki pomocnicze, nie testy.
        kt_valid = flags["kt"].dropna()
        row["kt_n"] = int(len(kt_valid))
        if len(kt_valid):
            row["kt_mean"] = round(float(kt_valid.mean()), 4)
            row["kt_p50"] = round(float(kt_valid.quantile(0.50)), 4)
            row["kt_p95"] = round(float(kt_valid.quantile(0.95)), 4)
            row["kt_p99"] = round(float(kt_valid.quantile(0.99)), 4)
            row["kt_max"] = round(float(kt_valid.max()), 4)
            for lower, upper in [
                (0.0, 0.2), (0.2, 0.4), (0.4, 0.6),
                (0.6, 0.8), (0.8, 1.0), (1.0, 99.0),
            ]:
                share = float(
                    ((kt_valid >= lower) & (kt_valid < upper)).mean()
                )
                kt_rows.append({
                    "location": location,
                    "source": source,
                    "kt_bin_low": lower,
                    "kt_bin_high": upper,
                    "share": round(share, 6),
                })

        closure_valid = flags["closure_rel_dev"].dropna()
        row["closure_n"] = int(len(closure_valid))
        if len(closure_valid):
            row["closure_mean_rel_dev"] = round(float(closure_valid.mean()), 5)
            row["closure_p95_abs_rel_dev"] = round(
                float(closure_valid.abs().quantile(0.95)), 5
            )
        row["closure_meaningful"] = (source == "TMY")

        summary_rows.append(row)

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["location", "source"])
        .reset_index(drop=True)
    )
    kt_distribution = pd.DataFrame(kt_rows)
    return summary, kt_distribution


def markdown_table(summary: pd.DataFrame) -> str:
    """Buduje zwięzłą tabelę Markdown z najważniejszymi wynikami kontroli."""
    columns = [
        ("location", "Lokalizacja"),
        ("source", "Zbiór"),
        ("rows", "Rekordy"),
        ("missing_hours", "Brak godz."),
        ("duplicates", "Dubl."),
        ("time_gaps", "Luki"),
        ("flag_night_irradiance", "Noc>0"),
        ("flag_ghi_above_etr", "GHI>ETR"),
        ("flag_ghi_above_etr_hourly", "GHI>ETR godz."),
        ("flag_kt_above_1", "kt>1"),
        ("flag_dhi_above_ghi", "DHI>GHI"),
        ("flag_t2m_range", "T zakres"),
        ("flag_ws_range", "WS zakres"),
        ("flag_pvgis_reconstructed", "Odtw. PVGIS"),
        ("kt_p99", "kt p99"),
    ]
    present = [(key, label) for key, label in columns if key in summary.columns]

    lines = ["| " + " | ".join(label for _, label in present) + " |"]
    lines.append("|" + "|".join("---" for _ in present) + "|")
    for _, record in summary.iterrows():
        cells = []
        for key, _ in present:
            value = record[key]
            cells.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ===========================================================================
# 8. Uruchomienie samodzielne
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kontrola jakości danych meteorologicznych."
    )
    parser.add_argument("--time-check", action="store_true", dest="time_check",
                        help="wykonaj tylko diagnostykę osi czasu")
    args = parser.parse_args()

    config.ensure_dirs()

    print("=" * 72)
    print("Wczytywanie danych wejściowych")
    print("=" * 72)
    frame, metadata = load_all()
    print(f"\nRazem rekordów: {len(frame)}")

    print("\n" + "=" * 72)
    print("Baza czasu plików EPW — dowód doświadczalny")
    print("=" * 72)
    evidence = epw_timebase_evidence()
    if not evidence.empty:
        evidence.to_csv(config.OUT_DIR / "qc_epw_timebase.csv", index=False)
        best_lags = evidence[["location", "best_lag_h"]].drop_duplicates()
        print(evidence[evidence["lag_h"].abs() <= 1].to_string(index=False))
        print(f"\nNajlepsze przesunięcie na lokalizację (0 = parser poprawny):")
        print(best_lags.to_string(index=False))
        if (best_lags["best_lag_h"] != 0).any():
            print("\nUWAGA: parser EPW wymaga korekty bazy czasu — patrz "
                  "EPW_DECLARED_TZ_APPLIES.")

    print("\n" + "=" * 72)
    print("Diagnostyka osi czasu")
    print("=" * 72)
    time_report = time_axis_report(frame)
    print(time_report.to_string(index=False))

    time_report.to_csv(config.OUT_DIR / "qc_time_axis.csv", index=False)

    if (time_report["verdict"] != "OK").any():
        print("\nPRZERWANO: wykryto przesunięcie osi czasu >= 1 h.")
        print("Dalsza kontrola jakości byłaby liczona na błędnej osi czasu.")
        return 1

    if args.time_check:
        return 0

    print("\n" + "=" * 72)
    print("Lata źródłowe miesięcy TMY")
    print("=" * 72)
    source_years = tmy_source_years()
    if not source_years.empty:
        source_years.to_csv(config.OUT_DIR / "tmy_source_years.csv", index=False)
        pivot = source_years.pivot(
            index="location", columns="month", values="source_year"
        )
        print(pivot.to_string())
        outside = source_years.groupby("location")["within_series_period"].apply(
            lambda flags: int((~flags).sum())
        )
        print(f"\nMiesiące spoza okresu {config.SERIES_START_YEAR}–"
              f"{config.SERIES_END_YEAR} (na 12):")
        print(outside.to_string())

    print("\n" + "=" * 72)
    print("Kontrola jakości")
    print("=" * 72)
    summary, kt_distribution = summarise(frame)

    summary.to_csv(config.OUT_DIR / "qc_summary.csv", index=False)
    kt_distribution.to_csv(config.OUT_DIR / "qc_kt_distribution.csv", index=False)

    table = markdown_table(summary)
    print()
    print(table)
    (config.OUT_DIR / "qc_summary.md").write_text(
        "# Kontrola jakości danych meteorologicznych\n\n" + table + "\n",
        encoding="utf-8",
    )

    print(f"\nZapisano: {config.OUT_DIR / 'qc_summary.csv'}")
    print(f"Zapisano: {config.OUT_DIR / 'qc_kt_distribution.csv'}")
    print(f"Zapisano: {config.OUT_DIR / 'qc_summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
