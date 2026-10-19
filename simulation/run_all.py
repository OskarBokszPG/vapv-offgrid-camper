# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: run_all.py
Opis: Uruchamia potok obliczeniowy w kolejności zależności: kontrola
      danych pogodowych, zasoby słoneczne, profile obciążeń, uzysk
      generatora, siatka przebiegów, ocena wymagań i raport siatki.
      Wyniki zapisywane są jako pliki CSV w simulation/out/ wraz
      z manifestami przebiegów. Skład dokumentu nie wchodzi w zakres
      tego skryptu.

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

import subprocess
import sys
import time
from pathlib import Path

KATALOG = Path(__file__).resolve().parent

KROKI = (
    ("kontrola jakości danych pogodowych", "weather_qc.py"),
    ("zasoby słoneczne: sumy miesięczne i roczne", "ghi_stats.py"),
    ("profile obciążeń", "loads.py"),
    ("uzysk roczny generatora", "run_yield.py"),
    ("siatka przebiegów symulacyjnych", "run_grid.py"),
    ("analiza siatki i ocena wymagań", "grid_analiza.py"),
    ("raport siatki: łańcuch energii i przekroje", "grid_raport.py"),
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
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
