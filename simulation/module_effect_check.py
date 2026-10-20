# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: module_effect_check.py
Opis: rozkład „efektu modułu" na czynniki. Powód powstania. Po zmianie
      modułu z generycznego 300 Wp na AIKO 470 Wp wielkość raportowana w
      bilansie strat jako „wpływ temperatury" zmieniła się dla Hela z
      +0,85 % na −0,03 %. Samo zmniejszenie modułu współczynnika
      temperaturowego (|γ| z 0,37 na 0,26 %/K) nie tłumaczy zmiany znaku
      — przy samym skalowaniu dałoby ok. +0,6 %. Skrypt rozkłada różnicę
      na czynniki.

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
import copy
import sys

import numpy as np
import pandas as pd

import config
import make_sam_resource as resource
import run_pv
from config import LOCATIONS, Location

T_STC_C = 25.0

# Moduł generyczny używany w pierwszym przebiegu potoku. Zachowany wyłącznie
# jako punkt odniesienia tej analizy — do obliczeń projektowych nie wraca.
MODUL_GENERYCZNY = {
    "nazwa": "Generyczny moduł mono 60-ogniwowy, ok. 300 Wp (odniesienie)",
    "p_mp": 300.0, "v_mp": 32.3, "i_mp": 9.29, "v_oc": 39.6, "i_sc": 9.83,
    "n_ogniw": 60, "powierzchnia": 1.632, "t_noct": 45.0,
    "alpha_isc": 0.0005 * 9.83, "beta_voc": -0.0029 * 39.6,
    "gamma_pmp": -0.37, "technologia": 0,
    # Modul generyczny opisywal ogniwa typu p — spadek poczatkowy 1,5 %.
    "lid_pct": 1.5,
}


def _wariant(nazwa: str, modul: dict, loc: Location, year: int) -> dict:
    """Uruchamia model dla zadanych danych modułu i zwraca składniki efektu."""
    oryginal = run_pv.MODUL_KATALOG
    run_pv.MODUL_KATALOG = modul
    try:
        model = resource._run_model_for_resource(resource.resource_path(loc, year))
        out = model.export()["Outputs"]

        dc_nominal = float(out["annual_dc_nominal"])
        dc_gross = float(out["annual_subarray1_dc_gross"])
        efekt_modulu = (dc_gross - dc_nominal) / dc_nominal * 100.0

        poa = np.asarray(out["subarray1_poa_eff"], dtype=float)
        cell = np.asarray(out["subarray1_celltemp"], dtype=float)
        gamma = modul["gamma_pmp"] / 100.0
        maska = poa > 0.0
        # Składnik czysto temperaturowy: liniowy, ważony nasłonecznieniem.
        skl_temp = (np.sum(poa[maska] * (1.0 + gamma * (cell[maska] - T_STC_C)))
                    / np.sum(poa[maska]) - 1.0) * 100.0
        return {
            "wariant": nazwa,
            "p_mp_W": modul["p_mp"],
            "gamma_pct_K": modul["gamma_pmp"],
            "t_noct_C": modul["t_noct"],
            "n_ogniw": modul["n_ogniw"],
            "efekt_modulu_pct": round(efekt_modulu, 3),
            "ssc_module_loss_pct": round(float(out["annual_dc_module_loss_percent"]), 3),
            "skladnik_temperaturowy_pct": round(skl_temp, 3),
            "reszta_niskie_natezenie_pct": round(efekt_modulu - skl_temp, 3),
            "t_ogniwa_srednia_C": round(float(cell[maska].mean()), 2),
            "poa_ponizej_stc_pct": round(
                float(poa[maska & (cell < T_STC_C)].sum() / poa[maska].sum() * 100.0), 1),
            "dc_nominal_kWh": round(dc_nominal, 1),
            "dc_gross_kWh": round(dc_gross, 1),
        }
    finally:
        run_pv.MODUL_KATALOG = oryginal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--location", default="Leba", choices=sorted(LOCATIONS))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_dirs()
    loc = LOCATIONS[args.location]

    aiko = copy.deepcopy(run_pv.MODUL_KATALOG)

    aiko_noct45 = copy.deepcopy(aiko)
    aiko_noct45["t_noct"] = MODUL_GENERYCZNY["t_noct"]

    gener_gamma_aiko = copy.deepcopy(MODUL_GENERYCZNY)
    gener_gamma_aiko["gamma_pmp"] = aiko["gamma_pmp"]

    warianty = [
        ("0. generyczny 300 Wp, γ=-0,37, NOCT 45 (stan sprzed zmiany)", MODUL_GENERYCZNY),
        ("1. generyczny 300 Wp, γ=-0,26 AIKO, NOCT 45", gener_gamma_aiko),
        ("2. AIKO 470 Wp, γ=-0,26, NOCT 45 (jak w generycznym)", aiko_noct45),
        ("3. AIKO 470 Wp, γ=-0,26, NOCT 47,5 (stan obecny)", aiko),
    ]

    wiersze = [_wariant(n, m, loc, args.year) for n, m in warianty]
    tab = pd.DataFrame(wiersze)
    sciezka = config.OUT_DIR / "module_effect_decomposition.csv"
    tab.to_csv(sciezka, index=False)

    print("=" * 94)
    print("Rozkład efektu modułu — %s, rok %d" % (loc.label, args.year))
    print("=" * 94)
    kol = ["wariant", "efekt_modulu_pct", "ssc_module_loss_pct",
           "skladnik_temperaturowy_pct", "reszta_niskie_natezenie_pct",
           "t_ogniwa_srednia_C", "poa_ponizej_stc_pct"]
    print(tab[kol].to_string(index=False))

    e = tab["efekt_modulu_pct"].tolist()
    print()
    print("Przyrosty między wariantami [pkt proc.]:")
    print("  γ  (0 -> 1), sam współczynnik temperaturowy          : %+.3f" % (e[1] - e[0]))
    print("  moduł (1 -> 2), parametry CEC i geometria przy tym γ : %+.3f" % (e[2] - e[1]))
    print("  NOCT (2 -> 3), 45 -> 47,5 stC                        : %+.3f" % (e[3] - e[2]))
    print("  razem (0 -> 3)                                      : %+.3f" % (e[3] - e[0]))
    print()
    print("zapisano:", sciezka)
    return 0


if __name__ == "__main__":
    sys.exit(main())
