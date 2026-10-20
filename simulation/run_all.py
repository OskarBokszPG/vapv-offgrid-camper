# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: run_all.py
Opis: Uruchamia potok obliczeniowy w kolejności zależności: kontrola
      danych pogodowych, zasoby słoneczne, profile obciążeń, uzysk
      generatora, siatka przebiegów, ocena wymagań W1a-W4, analiza
      wrażliwości, scenariusze łączne, kontrola ograniczenia regulatora,
      godziny niedoboru oraz wariant z modułem podnoszonym. Wyniki
      zapisywane są jako pliki CSV w simulation/out/ wraz z manifestami
      przebiegów. Tryb --kontrola liczy jeden przebieg roczny na wskazanym
      pliku zasobu, bez całego katalogu danych.

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
import subprocess
import sys
import time
from pathlib import Path

KATALOG = Path(__file__).resolve().parent

# Kolejnosc wynika z zaleznosci: kazdy krok czyta wyniki poprzednich.
KROKI = (
    ("kontrola jakości danych pogodowych", "weather_qc.py"),
    ("zasoby słoneczne: sumy miesięczne i roczne", "ghi_stats.py"),
    ("profile obciążeń", "loads.py"),
    ("uzysk roczny generatora", "run_yield.py"),
    ("siatka przebiegów symulacyjnych", "run_grid.py"),
    ("ocena wymagań W1a-W4, analiza wrażliwości i scenariusze łączne",
     "grid_analiza.py"),
    ("raport siatki: łańcuch energii, bilans kontrolny, progi",
     "grid_raport.py"),
    ("kontrola ograniczenia prądu wyjściowego regulatora",
     "kontrola_mppt.py"),
    ("godziny niedoboru i najdłuższa przerwa", "przerwy.py"),
    ("wariant z modułem podnoszonym na postoju", "wariant_pochylenia.py"),
)


def przebieg_kontrolny(plik_zasobu: Path, lokalizacja: str, rok: int,
                       pojemnosc_ah: int) -> int:
    """Jeden bilans roczny na wskazanym pliku zasobu slonecznego.

    Sluzy sprawdzeniu, czy kopia kodu daje te same wskazniki co potok
    pracy: wynik porownuje sie z wierszem out/grid_results.csv o tej samej
    lokalizacji, roku, module, magazynie, ladowarce i scenariuszu.
    """
    import battery_balance as bb
    import config
    import run_grid
    import run_pv

    import make_sam_resource as resource

    if not plik_zasobu.exists():
        print("Brak pliku zasobu: %s" % plik_zasobu)
        return 2
    # Przebieg czyta plik zasobu dwa razy: raz silnik obliczeniowy, raz
    # model obciazen (temperatura powietrza). Katalog zasobow ustawiany
    # jest wiec na katalog wskazanego pliku.
    run_grid.ZASOB_WYMUSZONY = plik_zasobu
    resource.RESOURCE_DIR = plik_zasobu.parent
    loc = next((v for v in config.LOCATIONS.values()
                if v.label == lokalizacja or v.key == lokalizacja), None)
    if loc is None:
        print("Nieznana lokalizacja: %s" % lokalizacja)
        return 2
    par = bb.ParametryUkladu(
        pojemnosc_kwh=run_grid.pojemnosc_kwh(pojemnosc_ah))
    _, w = run_grid.przebieg(
        loc, rok, run_pv.MODUL_KATALOG, par, "A-AC", "Orion XS 50",
        config.ALTERNATOR_PRAD_A)
    print("lokalizacja      : %s" % loc.label)
    print("rok              : %d" % rok)
    print("magazyn          : LFP %d Ah" % pojemnosc_ah)
    print("plik zasobu      : %s" % plik_zasobu.name)
    print("LPSP             : %.5f" % w["LPSP"])
    print("SSR_PV           : %.4f" % w["SSR_PV"])
    print("E_obciazenie_kWh : %.2f" % w["E_obciazenie_kWh"])
    print("godziny deficytu : %d" % w["godziny_deficytu"])
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Potok obliczeniowy")
    parser.add_argument("--kontrola", metavar="PLIK_ZASOBU",
                        help="jeden przebieg roczny na wskazanym pliku")
    parser.add_argument("--lokalizacja", default="Łeba")
    parser.add_argument("--rok", type=int, default=2016)
    parser.add_argument("--pojemnosc", type=int, default=300,
                        help="pojemność magazynu w Ah (domyślnie projektowa)")
    args = parser.parse_args()

    if args.kontrola:
        return przebieg_kontrolny(Path(args.kontrola).resolve(),
                                  args.lokalizacja, args.rok, args.pojemnosc)

    poczatek = time.time()
    for opis, skrypt in KROKI:
        print("=" * 74)
        print(">>> %s  (%s)" % (opis, skrypt), flush=True)
        krok = time.time()
        wynik = subprocess.run([sys.executable, str(KATALOG / skrypt)],
                               cwd=KATALOG, check=False)
        if wynik.returncode != 0:
            print("PRZERWANO: %s zakończył się kodem %d"
                  % (skrypt, wynik.returncode))
            return wynik.returncode
        print("    czas: %.1f s" % (time.time() - krok))
    print("=" * 74)
    print("Gotowe. Łączny czas: %.1f s" % (time.time() - poczatek))
    print("Wyniki: simulation/out/*.csv oraz manifesty przebiegów")
    return 0


if __name__ == "__main__":
    sys.exit(main())
