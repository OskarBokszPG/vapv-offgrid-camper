# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: thermal_check.py
Opis: kontrola modelu temperatury ogniwa. Powód powstania. W bilansie
      strat (``pv_loss_breakdown_2020.csv``) wkład temperatury dla Hela
      jest DODATNI (ok. +0,8 %), czyli model zgłasza sprawność wyższą od
      katalogowej. Kierunek jest fizycznie możliwy — przy pochyleniu 2
      stopni, nad morzem i w klimacie umiarkowanym temperatura ogniwa
      przez większość roku jest niższa od 25 stopni C przyjętych w
      warunkach STC — ale ten sam objaw daje błąd polegający na podaniu
      modelowi cieplnemu prędkości wiatru z wysokości 10 m bez
      przeliczenia na wysokość modułu. Wiatr 10-metrowy jest szybszy od
      przyziemnego, więc zawyża chłodzenie i zaniża temperaturę ogniwa.

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
import math
import sys

import numpy as np
import pandas as pd
import pvlib

import config
import make_sam_resource as resource
import run_pv
from config import LOCATIONS, Location

# Wysokość odniesienia prędkości wiatru w pliku zasobu (PVGIS: WS10m).
WIND_REFERENCE_HEIGHT_M = 10.0
# Wysokość modułu nad dachem pojazdu przyjęta do profilu logarytmicznego.
MODULE_HEIGHT_M = 2.0
# Chropowatość terenu dla profilu logarytmicznego, teren otwarty.
ROUGHNESS_LENGTH_M = 0.03
# Współczynnik przeliczenia wiatru stosowany w modelu NOCT w SAM przy montażu
# niskim (jedna kondygnacja). Źródło: implementacja modelu NOCT w SSC.
NOCT_WIND_FACTOR_LOW_MOUNT = 0.51
# Temperatura odniesienia warunków STC.
T_STC_C = 25.0


def log_wind_factor(
    height_m: float = MODULE_HEIGHT_M,
    reference_m: float = WIND_REFERENCE_HEIGHT_M,
    roughness_m: float = ROUGHNESS_LENGTH_M,
) -> float:
    """Współczynnik przeliczenia prędkości wiatru z profilu logarytmicznego."""
    return math.log(height_m / roughness_m) / math.log(reference_m / roughness_m)


def weighted_temperature_effect(
    poa: np.ndarray, cell_temp: np.ndarray, gamma_per_k: float
) -> float:
    """
    Ważony nasłonecznieniem wkład temperatury do uzysku, w procentach.

    Dodatnia wartość oznacza pracę poniżej 25 stopni C, czyli sprawność wyższą
    od katalogowej.
    """
    mask = poa > 0.0
    if not mask.any():
        return float("nan")
    factor = 1.0 + gamma_per_k * (cell_temp[mask] - T_STC_C)
    return float((np.sum(poa[mask] * factor) / np.sum(poa[mask]) - 1.0) * 100.0)


def check_location(loc: Location, year: int) -> list[dict[str, object]]:
    """Zestawia warianty modelu cieplnego dla jednej lokalizacji i roku."""
    model = resource._run_model_for_resource(resource.resource_path(loc, year))
    outputs = model.export()["Outputs"]

    poa = np.asarray(outputs["subarray1_poa_eff"], dtype=float)
    cell_ssc = np.asarray(outputs["subarray1_celltemp"], dtype=float)
    t_air = np.asarray(outputs["tdry"], dtype=float)
    wind10 = np.asarray(outputs["wspd"], dtype=float)

    # Współczynnik temperaturowy mocy z danych katalogowych modułu, %/K -> 1/K.
    gamma = run_pv.MODUL_KATALOG["gamma_pmp"] / 100.0

    variants: dict[str, np.ndarray] = {
        "SSC NOCT (model w potoku)": cell_ssc,
        "Faiman, wiatr 10 m bez przeliczenia": pvlib.temperature.faiman(
            poa, t_air, wind10
        ),
        "Faiman, wiatr x 0,51 (NOCT, montaz niski)": pvlib.temperature.faiman(
            poa, t_air, wind10 * NOCT_WIND_FACTOR_LOW_MOUNT
        ),
        "Faiman, wiatr 2 m (profil log.)": pvlib.temperature.faiman(
            poa, t_air, wind10 * log_wind_factor()
        ),
    }

    rows: list[dict[str, object]] = []
    mask = poa > 0.0
    for name, cell in variants.items():
        cell = np.asarray(cell, dtype=float)
        rows.append({
            "location": loc.key,
            "year": year,
            "variant": name,
            "cell_temp_mean_c": round(float(cell[mask].mean()), 2),
            "cell_temp_max_c": round(float(cell.max()), 2),
            # Udział nasłonecznienia zbieranego poniżej 25 stopni C — to on
            # decyduje o znaku wkładu temperatury.
            "poa_share_below_stc_pct": round(
                float(poa[mask & (cell < T_STC_C)].sum() / poa[mask].sum() * 100.0), 1
            ),
            "temperature_effect_pct": round(
                weighted_temperature_effect(poa, cell, gamma), 2
            ),
            "mean_bias_vs_ssc_k": round(float((cell[mask] - cell_ssc[mask]).mean()), 2),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--location", choices=sorted(LOCATIONS))
    args = parser.parse_args()

    config.ensure_dirs()
    keys = [args.location] if args.location else sorted(LOCATIONS)

    print("=" * 78)
    print("Kontrola modelu temperatury ogniwa")
    print("=" * 78)
    print(f"  rok                     : {args.year}")
    print(f"  wsp. wiatru 10 m -> 2 m : {log_wind_factor():.3f} (profil log., "
          f"z0 = {ROUGHNESS_LENGTH_M} m)")
    print(f"  wsp. wiatru w NOCT      : {NOCT_WIND_FACTOR_LOW_MOUNT} "
          f"(montaz niski)")
    print(f"  gamma modulu            : {run_pv.MODUL_KATALOG['gamma_pmp']} %/K")

    rows: list[dict[str, object]] = []
    for key in keys:
        rows.extend(check_location(LOCATIONS[key], args.year))

    table = pd.DataFrame(rows)
    out_path = config.OUT_DIR / "thermal_model_check.csv"
    table.to_csv(out_path, index=False)

    print()
    print(table.to_string(index=False))
    print(f"\n  zapisano: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
