# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: przerwy.py
Opis: godziny niedoboru i najdłuższa przerwa w pokryciu obciążenia.

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

import pandas as pd

import battery_balance as bb
import config
import fmt
import grid_analiza
import run_grid
import run_pv
from config import LOCATIONS

GENERATED = config.SIM_DIR.parent / "tex" / "generated"
SCENARIUSZ = "A-AC"
LADOWARKA = grid_analiza.OAT_LADOWARKA
BS = chr(92)
NL = chr(10)
# Przyrostki nazw makr: nazwa makra moze zawierac wylacznie litery.
PRZYROSTEK = {"Łeba": "Leba", "Suwałki": "Suwalki", "Lesko": "Lesko"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_dirs()
    wym = pd.read_csv(config.OUT_DIR / "requirements_check.csv")
    ah = grid_analiza.pojemnosc_projektowa_ah(wym)
    magazyn = "LFP %d Ah" % ah
    par = bb.ParametryUkladu(pojemnosc_kwh=run_grid.pojemnosc_kwh(ah))
    lata = list(range(config.SERIES_START_YEAR, config.SERIES_END_YEAR + 1))

    wiersze, makra = [], {}
    for klucz, loc in LOCATIONS.items():
        soc = None
        najgorszy = None
        for rok in lata:
            _, w = run_grid.przebieg(
                loc, rok, run_pv.MODUL_KATALOG, par, SCENARIUSZ, LADOWARKA,
                config.ALTERNATOR_PRAD_A, soc_poczatkowy_wzgl=soc,
                rozbieg_godzin=(run_grid.ROZBIEG_GODZIN
                                if rok == lata[0] else 0))
            soc = w.get("SoC_koncowy_wzgledny")
            if najgorszy is None or w["LPSP"] > najgorszy["LPSP"]:
                najgorszy = dict(w, rok=rok)
        assert najgorszy is not None
        wiersze.append({
            "lokalizacja": loc.label, "magazyn": magazyn,
            "scenariusz": SCENARIUSZ, "rok": najgorszy["rok"],
            "LPSP": round(najgorszy["LPSP"], 5),
            "godziny_deficytu": int(najgorszy["godziny_deficytu"]),
            "najdluzsza_przerwa_h": int(najgorszy["najdluzsza_przerwa_h"]),
        })
        p = PRZYROSTEK[loc.label]
        makra["GodzinyDeficytu" + p] = int(najgorszy["godziny_deficytu"])
        makra["NajdluzszaPrzerwa" + p] = int(najgorszy["najdluzsza_przerwa_h"])
        print("  %s: gotowe" % loc.label, flush=True)

    tab = pd.DataFrame(wiersze)
    tab.to_csv(config.OUT_DIR / "przerwy.csv", index=False)

    makra["GodzinyDeficytuMax"] = int(tab["godziny_deficytu"].max())
    makra["NajdluzszaPrzerwaMax"] = int(tab["najdluzsza_przerwa_h"].max())
    makra["PrzerwaDobyMax"] = round(
        float(tab["najdluzsza_przerwa_h"].max()) / 24.0, 1)

    linie = ["%% WYGENEROWANO AUTOMATYCZNIE: simulation/przerwy.py"
             " --- nie edytować ręcznie",
             "%% Dane: out/przerwy.csv"]
    for nazwa in sorted(makra):
        wartosc = makra[nazwa]
        miejsca = 0 if float(wartosc) == int(wartosc) else 1
        linie.append(BS + "newcommand{" + BS + nazwa + "}{"
                     + fmt.num(wartosc, miejsca) + "}")
    (GENERATED / "liczby-przerwy.tex").write_text(NL.join(linie) + NL,
                                                  encoding="utf-8",
                                                  newline="\n")
    print(tab.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
