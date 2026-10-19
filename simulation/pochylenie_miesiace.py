# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: pochylenie_miesiace.py
Opis: miesięczny przyrost energii przy module pochylonym. Uzupełnia
      ``wariant_pochylenia.py`` o rozkład przyrostu energii
      stałoprądowej w roku. Rozstrzygające dla okna pracy zdalnej są
      miesiące przejściowe, więc przyrost roczny nie wystarcza do oceny
      wariantu.

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
Ostatnia modyfikacja: 2026-10-19
Wersja: 1.0.0
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config
import run_grid
import run_pv
import wariant_pochylenia as wp
from config import LOCATIONS

POCHYLENIE_DEG = 30.0


def miesieczne_sumy(moc_kw: np.ndarray, indeks: pd.DatetimeIndex) -> pd.Series:
    szereg = pd.Series(moc_kw, index=indeks)
    return szereg.groupby(szereg.index.month).sum()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_dirs()
    beta = float(sys.argv[1]) if len(sys.argv) > 1 else POCHYLENIE_DEG
    lata = range(config.SERIES_START_YEAR, config.SERIES_END_YEAR + 1)

    wiersze, pojemnosci = [], []
    for loc in LOCATIONS.values():
        for rok in lata:
            plaska, indeks = run_grid.generacja(loc, rok, run_pv.MODUL_KATALOG)
            pochylona = wp._na_postoju(
                wp._generacja(loc, rok, beta, wp.AZYMUT_POLUDNIE),
                plaska, loc, rok, wp.SCENARIUSZ)
            # Ocena wymagania W1a dla kilku pojemności: scenariusz oceny
            # wymagania, czyli wariant z falownikiem.
            for ah in run_grid.POJEMNOSCI_AH:
                for szereg, etykieta in ((plaska, "poziomo"),
                                         (pochylona, "pochylony")):
                    _, w = wp._bilans(loc, rok, szereg, wp.SCENARIUSZ_W1, ah)
                    pojemnosci.append({
                        "lokalizacja": loc.label, "rok": rok,
                        "wariant": etykieta, "pojemnosc_Ah": ah,
                        "LPSP": round(w["LPSP"], 5)})
            sp = miesieczne_sumy(plaska, indeks)
            sn = miesieczne_sumy(pochylona, indeks)
            for miesiac in sp.index:
                wiersze.append({
                    "lokalizacja": loc.label, "rok": rok,
                    "miesiac": int(miesiac),
                    "E_poziomo_kWh": round(float(sp[miesiac]), 3),
                    "E_pochylony_kWh": round(float(sn[miesiac]), 3),
                })
        print("  %s: gotowe" % loc.label, flush=True)

    tab = pd.DataFrame(wiersze)
    tab["przyrost_pct"] = (100.0 * (tab["E_pochylony_kWh"]
                                    - tab["E_poziomo_kWh"])
                           / tab["E_poziomo_kWh"]).round(2)
    tab.insert(0, "pochylenie_deg", beta)
    tab.to_csv(config.OUT_DIR / "pochylenie_miesiace.csv", index=False)

    poj = pd.DataFrame(pojemnosci)
    poj.to_csv(config.OUT_DIR / "pochylenie_pojemnosci.csv", index=False)

    przeglad = (tab.groupby("miesiac")["przyrost_pct"].median().round(1))
    print(przeglad.to_string())
    print()
    najgorszy = (poj.groupby(["wariant", "pojemnosc_Ah"])["LPSP"].max() * 100.0)
    print(najgorszy.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
