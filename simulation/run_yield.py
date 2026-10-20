# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: run_yield.py
Opis: pierwszy zweryfikowany uzysk energii z generatora PV. Model
      ``Pvsamv1`` uruchamiany jest dla trzech lokalizacji referencyjnych
      i dziesięciu kolejnych lat (2014-2023), na plikach zasobu
      słonecznego budowanych przez ``make_sam_resource`` — nie na
      plikach EPW z PVGIS. Powód jest udokumentowany bramką
      weryfikacyjną (``out/qc_ssc_sun_position.csv``): SSC stosuje
      strefę czasową z nagłówka EPW, przez co plik PVGIS podany wprost
      daje błąd godziny w geometrii słonecznej.

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

import numpy as np
import pandas as pd

import config
import make_sam_resource as resource
import run_pv
from config import LOCATIONS, Location

MONTH_NAMES_PL = [
    "sty", "lut", "mar", "kwi", "maj", "cze",
    "lip", "sie", "wrz", "paź", "lis", "gru",
]


def simulate_year(loc: Location, year: int) -> tuple[pd.Series, dict[str, object]]:
    """
    Wykonuje model generacji dla jednej pary (lokalizacja, rok).

    Zwraca krotkę (godzinowy szereg mocy DC w kW z indeksem UTC, metadane).
    """
    resource_file = resource.resource_path(loc, year)
    if not resource_file.exists():
        raise FileNotFoundError(
            f"Brak pliku zasobu {resource_file}. Uruchom make_sam_resource.py."
        )

    model = resource._run_model_for_resource(resource_file)
    series_kw, source = run_pv._odczytaj_szereg_dc(model)

    if series_kw.size != 8760:
        raise ValueError(
            f"{loc.key} {year}: szereg ma {series_kw.size} wartości zamiast 8760."
        )

    stamps = resource._expected_timestamps(resource_file)
    hourly = pd.Series(series_kw, index=stamps, name="dc_kw")

    outputs = model.export()["Outputs"]
    metadata: dict[str, object] = {
        "output_name": source,
        "resource_file": resource_file.name,
        "peak_w": round(float(series_kw.max()) * 1000.0, 1),
        "hours_generating": int((series_kw > 0.001).sum()),
    }
    # Nasłonecznienie w płaszczyźnie modułu — kontrola, czy transpozycja
    # zadziałała (przy pochyleniu 2 stopni powinno być nieco wyższe od GHI).
    #
    # UWAGA NA JEDNOSTKI. Wielkości ``annual_poa_*`` SSC podaje w kWh, czyli
    # jako energię padającą na całą powierzchnię pola, a ``subarray1_poa_*``
    # w kWh/m2. Przy polu 1,9924 m2 różnią się one właśnie tym współczynnikiem
    # (1108,9 kWh/m2 x 1,9924 m2 = 2209,5 kWh). Pomylenie ich prowadzi do
    # wniosku, że transpozycja zawyża nasłonecznienie o ponad połowę — wniosku
    # fałszywego, który przy pochyleniu 2 stopni jest z góry niemożliwy.
    if "subarray1_poa_nom" in outputs:
        poa_nominal = np.asarray(outputs["subarray1_poa_nom"], dtype=float)
        metadata["poa_nom_kwh_m2"] = round(float(poa_nominal.sum()) / 1000.0, 1)
    if "subarray1_poa_front" in outputs:
        poa_effective = np.asarray(outputs["subarray1_poa_front"], dtype=float)
        metadata["poa_eff_kwh_m2"] = round(float(poa_effective.sum()) / 1000.0, 1)
    if "gh" in outputs:
        ghi = np.asarray(outputs["gh"], dtype=float)
        metadata["ghi_kwh_m2"] = round(float(ghi.sum()) / 1000.0, 1)
    return hourly, metadata


def loss_breakdown(loc: Location, year: int) -> dict[str, object]:
    """
    Zestawia bilans strat od nasłonecznienia do energii stałoprądowej.

    Tabela jest potrzebna w rozdz. 6.3: bez niej nie da się uzasadnić, skąd
    bierze się stosunek energii oddanej do nasłonecznienia. Rozkład pozwala
    też wychwycić sytuację, w której któraś strata nie została przyjęta przez
    SSC — suma składników musi odtwarzać wynik końcowy.

    UWAGA NAZEWNICZA. Pozycja ``module_effect_pct`` nie jest samą stratą
    temperaturową. SSC nie rozdziela temperatury od zachowania modelu przy
    niskim natężeniu promieniowania: różnica między energią brutto a nominalną
    obejmuje oba zjawiska plus przesunięcie punktu mocy maksymalnej i jest
    raportowana przez SSC jako jedna pozycja ``annual_dc_module_loss_percent``.
    Rozkład na składnik czysto temperaturowy i resztę liczy osobny skrypt
    ``module_effect_check.py``.
    """
    resource_file = resource.resource_path(loc, year)
    model = resource._run_model_for_resource(resource_file)
    outputs = model.export()["Outputs"]

    area = run_pv.MODUL_KATALOG["powierzchnia"]
    stc_efficiency = (
        run_pv.MODUL_KATALOG["p_mp"] / (area * 1000.0)
    )

    poa_nominal_kwh = float(outputs["annual_poa_nom"])
    poa_effective_kwh = float(outputs["annual_poa_eff"])
    dc_nominal = float(outputs["annual_dc_nominal"])
    dc_gross = float(outputs["annual_subarray1_dc_gross"])
    wiring = float(outputs["annual_dc_wiring_loss"])
    diodes = float(outputs["annual_subarray1_dc_diodes_loss"])
    # Niedopasowanie i derating mocy znamionowej (nośnik LID) wprowadzono
    # w run_pv jako straty przyjęte. Bez nich suma składników nie odtwarzała
    # energii wyjściowej modelu — kontrola domknięcia poniżej to wychwytuje.
    mismatch = float(outputs["annual_subarray1_dc_mismatch_loss"])
    nameplate = float(outputs["annual_subarray1_dc_nameplate_loss"])
    dc_net = dc_gross - mismatch - wiring - diodes - nameplate
    # Energia stałoprądowa zgłoszona przez model. Różnica wobec sumy składników
    # jest kontrolą domknięcia: ujawnia stratę zadaną w konfiguracji, a pominiętą
    # w tym zestawieniu (tak wyszło przy wprowadzeniu niedopasowania i LID).
    dc_net_model = float(outputs["annual_dc_net"])
    residual = dc_net - dc_net_model

    return {
        "location": loc.key,
        "year": year,
        "module_area_m2": area,
        "stc_efficiency_pct": round(stc_efficiency * 100.0, 2),
        "poa_nominal_kwh": round(poa_nominal_kwh, 1),
        "soiling_loss_pct": round(
            float(outputs["annual_poa_soiling_loss_percent"]), 2
        ),
        "cover_iam_loss_pct": round(
            float(outputs["annual_poa_cover_loss_percent"]), 2
        ),
        "poa_effective_kwh": round(poa_effective_kwh, 1),
        "dc_nominal_kwh": round(dc_nominal, 1),
        "dc_gross_kwh": round(dc_gross, 1),
        # Dodatnia wartość oznacza pracę powyżej sprawności katalogowej.
        "module_effect_pct": round(
            (dc_gross - dc_nominal) / dc_nominal * 100.0, 2
        ),
        "ssc_module_loss_pct": round(
            float(outputs["annual_dc_module_loss_percent"]), 2
        ),
        "dc_mismatch_loss_kwh": round(mismatch, 2),
        "dc_wiring_loss_kwh": round(wiring, 2),
        "dc_diodes_loss_kwh": round(diodes, 2),
        "dc_nameplate_lid_loss_kwh": round(nameplate, 2),
        "dc_net_kwh": round(dc_net, 1),
        "dc_net_model_kwh": round(dc_net_model, 1),
        "closure_residual_kwh": round(residual, 3),
        # Stosunek energii stałoprądowej do energii, jaką dałoby pole
        # o sprawności katalogowej przy nasłonecznieniu nominalnym.
        "dc_performance_ratio": round(
            dc_net / (poa_nominal_kwh * stc_efficiency), 4
        ),
    }


def run_all(selected: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Wykonuje model dla wszystkich lokalizacji i lat wielolecia.

    Zwraca krotkę (zestawienie roczne, zestawienie miesięczne, szeregi godzinowe).
    """
    annual_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    hourly_store: dict[tuple[str, int], pd.Series] = {}

    locations = (
        [LOCATIONS[selected]] if selected else list(LOCATIONS.values())
    )

    for loc in locations:
        print(f"\n{loc.label}")
        for year in range(config.SERIES_START_YEAR, config.SERIES_END_YEAR + 1):
            hourly, metadata = simulate_year(loc, year)
            hourly_store[(loc.key, year)] = hourly

            annual_kwh = float(hourly.sum())
            annual_rows.append({
                "location": loc.key,
                "year": year,
                "dc_yield_kwh": round(annual_kwh, 2),
                "specific_yield_kwh_kwp": round(
                    annual_kwh / (run_pv.MODUL_KATALOG["p_mp"] / 1000.0), 1
                ),
                "peak_w": metadata["peak_w"],
                "hours_generating": metadata["hours_generating"],
                "resource_file": metadata["resource_file"],
            })

            by_month = hourly.groupby(hourly.index.month).sum()
            for month in range(1, 13):
                monthly_rows.append({
                    "location": loc.key,
                    "year": year,
                    "month": month,
                    "month_name": MONTH_NAMES_PL[month - 1],
                    "dc_yield_kwh": round(float(by_month.get(month, 0.0)), 3),
                })

            print(f"  {year}: {annual_kwh:7.1f} kWh   "
                  f"szczyt {metadata['peak_w']:5.0f} W   "
                  f"godzin {metadata['hours_generating']}")

    return (
        pd.DataFrame(annual_rows),
        pd.DataFrame(monthly_rows),
        hourly_store,
    )


def summarise_annual(annual: pd.DataFrame) -> pd.DataFrame:
    """
    Zestawia statystyki uzysku rocznego dla każdej lokalizacji.

    Celowo bez P50 i P90: dziesięć lat to za mało, żeby mówić o kwantylach
    rozkładu. Podawane są mediana, skrajne lata i rozstęp względny.
    """
    rows: list[dict[str, object]] = []
    for location, group in annual.groupby("location"):
        values = group["dc_yield_kwh"]
        worst = group.loc[values.idxmin()]
        best = group.loc[values.idxmax()]
        median = float(values.median())
        rows.append({
            "location": location,
            "n_years": int(len(group)),
            "median_kwh": round(median, 1),
            "mean_kwh": round(float(values.mean()), 1),
            "std_kwh": round(float(values.std()), 2),
            "cv_pct": round(float(values.std()) / float(values.mean()) * 100.0, 2),
            "worst_year": int(worst["year"]),
            "worst_kwh": round(float(worst["dc_yield_kwh"]), 1),
            "worst_vs_median_pct": round(
                (float(worst["dc_yield_kwh"]) - median) / median * 100.0, 2
            ),
            "best_year": int(best["year"]),
            "best_kwh": round(float(best["dc_yield_kwh"]), 1),
            "range_kwh": round(
                float(best["dc_yield_kwh"]) - float(worst["dc_yield_kwh"]), 1
            ),
            "specific_median_kwh_kwp": round(
                median / (run_pv.MODUL_KATALOG["p_mp"] / 1000.0), 1
            ),
        })
    return pd.DataFrame(rows)


def worst_window_yield(hourly_store: dict) -> pd.DataFrame:
    """
    Wylicza energię z generatora w oknach najniekorzystniejszych z etapu 5b.

    Okna (daty) wczytywane są z ``out/val_worst_windows.csv`` — wyznaczono je
    wcześniej na sumach dobowych GHI z SARAH3. Tutaj sumowana jest energia
    stałoprądowa z modelu w tych samych przedziałach dat, co daje bezpośredni
    argument do oceny autonomii zimowej (rozdz. 5 i 7).
    """
    windows_path = config.OUT_DIR / "val_worst_windows.csv"
    if not windows_path.exists():
        return pd.DataFrame()

    windows = pd.read_csv(windows_path)
    windows = windows.loc[windows["scope"] == "całe wielolecie"]

    rows: list[dict[str, object]] = []
    for _, window in windows.iterrows():
        location = str(window["location"])
        start = pd.Timestamp(window["window_start"], tz="UTC")
        end = pd.Timestamp(window["window_end"], tz="UTC") + pd.Timedelta(days=1)
        days = int(window["window_days"])

        # Okno może przechodzić przez granicę roku (Lesko: 20.12 - 02.01),
        # dlatego sklejane są szeregi wszystkich lat tej lokalizacji.
        parts = [
            series for (key, _year), series in hourly_store.items()
            if key == location
        ]
        if not parts:
            continue
        combined = pd.concat(parts).sort_index()
        window_data = combined.loc[(combined.index >= start) & (combined.index < end)]

        if window_data.empty:
            continue

        total_kwh = float(window_data.sum())
        rows.append({
            "location": location,
            "window_days": days,
            "window_start": str(window["window_start"]),
            "window_end": str(window["window_end"]),
            "ghi_sum_kwh_m2": float(window["ghi_sum_kwh_m2"]),
            "pv_energy_kwh": round(total_kwh, 3),
            "pv_energy_kwh_per_day": round(total_kwh / days, 4),
            "hours_covered": int(len(window_data)),
        })

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Uzysk energii z generatora PV dla wielolecia."
    )
    parser.add_argument("--location", type=str, default=None,
                        choices=sorted(LOCATIONS),
                        help="policz tylko jedną lokalizację")
    args = parser.parse_args()

    config.ensure_dirs()

    print("=" * 72)
    print("Uzysk energii z generatora PV — Pvsamv1, wielolecie 2014-2023")
    print("=" * 72)
    run_pv.wypisz_wersje()
    print(f"  moc znamionowa pola  : {run_pv.MODUL_KATALOG['p_mp']:.0f} Wp")
    print(f"  pochylenie / azymut  : {run_pv.POCHYLENIE_DEG}° / {run_pv.AZYMUT_DEG}°")
    print(f"  model cieplny        : standoff {run_pv.STANDOFF_MODEL_CIEPLNY} "
          f"(szczelina 1,5-2,5 cala)")
    print(f"  plik zasobu          : CSV SAM, Time Zone = 0 (UTC)")

    annual, monthly, hourly_store = run_all(args.location)

    annual.to_csv(config.OUT_DIR / "pv_yield_annual.csv", index=False)
    monthly.to_csv(config.OUT_DIR / "pv_yield_monthly.csv", index=False)

    summary = summarise_annual(annual)
    summary.to_csv(config.OUT_DIR / "pv_yield_summary.csv", index=False)

    print("\n" + "=" * 72)
    print("Uzysk roczny — mediana, rok najgorszy, rok najlepszy")
    print("(bez P50/P90 — przy n = 10 kwantyle rozkładu nie są wiarygodne)")
    print("=" * 72)
    print(summary[[
        "location", "n_years", "median_kwh", "worst_year", "worst_kwh",
        "worst_vs_median_pct", "best_year", "best_kwh", "range_kwh",
        "cv_pct", "specific_median_kwh_kwp",
    ]].to_string(index=False))

    # Rozkład miesięczny mediany — materiał do rysunku w rozdz. 6.3.
    median_monthly = (
        monthly.groupby(["location", "month", "month_name"])["dc_yield_kwh"]
        .median().reset_index()
        .rename(columns={"dc_yield_kwh": "median_dc_yield_kwh"})
    )
    median_monthly.to_csv(
        config.OUT_DIR / "pv_yield_monthly_median.csv", index=False
    )
    print("\nMediana uzysku miesięcznego [kWh]:")
    pivot = median_monthly.pivot(
        index="location", columns="month_name", values="median_dc_yield_kwh"
    )[MONTH_NAMES_PL]
    print(pivot.round(2).to_string())

    # --- bilans strat dla roku mediany ------------------------------------
    # Rok odniesienia: 2020, obecny w każdej lokalizacji i bez skrajnych
    # odchyleń. Bilans służy uzasadnieniu stosunku energii do nasłonecznienia.
    breakdown = pd.DataFrame([
        loss_breakdown(LOCATIONS[key], 2020)
        for key in (sorted(LOCATIONS) if args.location is None else [args.location])
    ])
    breakdown.to_csv(config.OUT_DIR / "pv_loss_breakdown_2020.csv", index=False)
    print("\n" + "=" * 72)
    print("Bilans strat DC, rok 2020 (uzasadnienie do rozdz. 6.3)")
    print("=" * 72)
    print(breakdown[[
        "location", "poa_nominal_kwh", "soiling_loss_pct", "cover_iam_loss_pct",
        "dc_nominal_kwh", "module_effect_pct", "dc_mismatch_loss_kwh",
        "dc_wiring_loss_kwh", "dc_diodes_loss_kwh", "dc_nameplate_lid_loss_kwh",
        "dc_net_kwh", "closure_residual_kwh", "dc_performance_ratio",
    ]].to_string(index=False))

    # --- okna najniekorzystniejsze ----------------------------------------
    windows = worst_window_yield(hourly_store)
    if not windows.empty:
        windows.to_csv(config.OUT_DIR / "pv_yield_worst_windows.csv", index=False)
        print("\n" + "=" * 72)
        print("Energia z generatora w oknach najniekorzystniejszych (etap 5b)")
        print("=" * 72)
        print(windows[[
            "location", "window_days", "window_start", "window_end",
            "ghi_sum_kwh_m2", "pv_energy_kwh", "pv_energy_kwh_per_day",
        ]].to_string(index=False))

    print(f"\nTabele zapisane w {config.OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
