# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: kontrola_mppt.py
Opis: Nazwa pliku / Moduł: kontrola_mppt.py Opis: Kontrola ograniczenia
      prądu wyjściowego regulatora ładowania. Wyznacza największą
      godzinową moc stałoprądową generatora w całym szeregu, przelicza
      ją na stronę wyjściową regulatora przez jego sprawność i porównuje
      z iloczynem prądu znamionowego i napięcia podtrzymania. Zapisuje
      makra liczbowe do rozdziału 5.

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

import sys

import numpy as np
import pandas as pd

import battery_balance as bb
import config
import fmt
import run_grid
import run_pv
from config import LOCATIONS

GENERATED = config.SIM_DIR.parent / "tex" / "generated"
BS = chr(92)
NL = chr(10)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_dirs()
    par = bb.ParametryUkladu()
    lata = range(config.SERIES_START_YEAR, config.SERIES_END_YEAR + 1)

    wiersze = []
    for loc in LOCATIONS.values():
        for rok in lata:
            run_grid.generacja(loc, rok, run_pv.MODUL_KATALOG)
            # Szczyt przed ograniczeniem regulatora: szereg zwracany przez
            # generacje jest juz obciety progiem, wiec nie opisuje mocy
            # dostepnej po stronie generatora.
            klucz = (loc.key, rok, run_pv.MODUL_KATALOG["nazwa"],
                     round(run_grid.MNOZNIK_PROMIENIOWANIA, 4))
            wiersze.append({"lokalizacja": loc.label, "rok": rok,
                            "P_dc_max_kW": round(
                                run_grid._SZCZYT_DC[klucz], 4),
                            "E_obciete_kWh": round(
                                run_grid._OBCIECIE_MPPT[klucz], 3)})
        print("  %s: gotowe" % loc.label, flush=True)

    tab = pd.DataFrame(wiersze)
    tab.to_csv(config.OUT_DIR / "kontrola_mppt.csv", index=False)

    dc_max_kw = float(tab["P_dc_max_kW"].max())
    wy_max_kw = dc_max_kw * par.sprawnosc_mppt
    limit_kw = run_grid.MPPT_MOC_WYJSCIOWA_W / 1000.0
    zapas_pct = 100.0 * (limit_kw - wy_max_kw) / limit_kw

    makra = (
        ("MpptPradZnamionowy", fmt.pl_tex(run_grid.MPPT_PRAD_WYJSCIOWY_A, 0)),
        ("MpptNapiecie", fmt.pl_tex(run_grid.MPPT_NAPIECIE_PODTRZYMANIA_V, 1)),
        ("MpptLimitMocy", fmt.pl_tex(run_grid.MPPT_MOC_WYJSCIOWA_W, 0)),
        ("MpptSprawnosc", fmt.pl_tex(par.sprawnosc_mppt, 2)),
        ("MpptMocDcMax", fmt.pl_tex(1000.0 * dc_max_kw, 0)),
        ("MpptMocWyjsciowaMax", fmt.pl_tex(1000.0 * wy_max_kw, 0)),
        ("MpptZapasMocy", fmt.pl_tex(abs(zapas_pct), 1)),
        ("MpptEnergiaObcieta", fmt.pl_tex(
            float(tab["E_obciete_kWh"].max()), 2)),
        ("MpptGodzinObciecia", fmt.pl_tex(
            float((tab["E_obciete_kWh"] > 0.0).sum()), 0)),
        ("MpptLatObciecia", fmt.pl_tex(
            float((tab.groupby("rok")["E_obciete_kWh"].max() > 0.0).sum()),
            0)),
    )
    tresc = ["%% WYGENEROWANO AUTOMATYCZNIE: simulation/kontrola_mppt.py"
             " --- nie edytować ręcznie",
             "%% Dane: out/kontrola_mppt.csv"]
    for nazwa, wartosc in makra:
        tresc.append(BS + "newcommand{" + BS + nazwa + "}{" + wartosc + "}")
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "liczby-mppt.tex").write_text(NL.join(tresc) + NL,
                                               encoding="utf-8", newline="\n")

    print("P_dc,max      = %.4f kW" % dc_max_kw)
    print("po regulatorze= %.4f kW" % wy_max_kw)
    print("limit 30 A    = %.4f kW" % limit_kw)
    print("zapas         = %.1f %%" % zapas_pct)
    print("obcieta max   = %.3f kWh/rok w jednej lokalizacji"
          % tab["E_obciete_kWh"].max())
    return 0


if __name__ == "__main__":
    sys.exit(main())
