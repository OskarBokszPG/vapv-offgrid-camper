# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: main.py
Opis: spięcie całego potoku obliczeniowego. Skrypt łączy trzy etapy w
      jeden przebieg i wypisuje zbiorcze zestawienie wyników dla
      lokalizacji referencyjnych:

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

import argparse
import sys
from pathlib import Path

import pandas as pd

import config
import fetch_weather
import battery_balance
import run_pv


def zestawienie_wynikow(wiersze: list[dict[str, object]]) -> pd.DataFrame:
    """Buduje ramkę zbiorczą z wynikami dla wszystkich lokalizacji."""
    return pd.DataFrame(wiersze).set_index("Lokalizacja")


def wypisz_tabele(zestawienie: pd.DataFrame) -> None:
    """Wypisuje zestawienie zbiorcze w formie tabeli tekstowej."""
    print()
    print("=" * 72)
    print("ZESTAWIENIE ZBIORCZE")
    print("=" * 72)

    naglowek = (
        f"{'Lokalizacja':<12} {'Uzysk PV':>10} {'SSR':>8} {'LPSP':>8} "
        f"{'Deficyt':>9} {'Godz. def.':>11}"
    )
    print(naglowek)
    print(f"{'':<12} {'[kWh/rok]':>10} {'[-]':>8} {'[-]':>8} {'[kWh]':>9} {'[h]':>11}")
    print("-" * 72)

    for nazwa, wiersz in zestawienie.iterrows():
        print(
            f"{nazwa:<12} "
            f"{wiersz['Uzysk_kWh']:>10.1f} "
            f"{wiersz['SSR']:>8.3f} "
            f"{wiersz['LPSP']:>8.3f} "
            f"{wiersz['Deficyt_kWh']:>9.1f} "
            f"{int(wiersz['Godziny_deficytu']):>11d}"
        )
    print("-" * 72)


def przebieg_pelny(
    parametry: battery_balance.ParametryUkladu,
    force_download: bool,
) -> int:
    """Wykonuje pełny potok: pobranie danych, symulacja PV, bilans magazynu."""

    # --- Etap 1: dane meteorologiczne --------------------------------------
    # Pobranie dotyczy wyłącznie plików TMY; szeregi wieloletnie służą analizie
    # reprezentatywności (weather_validation) i nie są potrzebne do przelotu.
    fetch_weather.fetch_raw(force=force_download, tmy_only=True)
    pliki = fetch_weather.make_working_copies()
    if not pliki:
        print("\nPrzerwano: brak jakichkolwiek danych pogodowych.")
        return 1

    # --- Etapy 2 i 3: generacja i bilans, lokalizacja po lokalizacji -------
    print("\n" + "=" * 72)
    print("Symulacja generacji PV i bilans magazynu")
    print("=" * 72)

    wiersze: list[dict[str, object]] = []
    # Indeks ze strefą UTC — generacja PV jest wyznaczana w UTC, a profil
    # obciążenia zadany w czasie lokalnym wymaga przeliczenia (patrz
    # battery_balance.load_profile_utc). Indeks bez strefy uniemożliwiłby to
    # przeliczenie i cicho wymusiłby profil płaski.
    indeks = run_pv.indeks_godzinowy().tz_localize("UTC")

    for nazwa, epw in pliki.items():
        print(f"\n[{nazwa}]")
        try:
            moc_pv_kw, metadane = run_pv.symuluj_pv(epw)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            print(f"  BŁĄD symulacji dla {nazwa}: {e}")
            continue

        print(f"  odczytane wyjście SSC : {metadane['zrodlo_wyjscia']}")
        run_pv.wypisz_podsumowanie(moc_pv_kw, nazwa)

        przebiegi, wskazniki = battery_balance.bilans_godzinowy(
            moc_pv_kw, parametry, indeks=indeks
        )
        battery_balance.wypisz_wskazniki(wskazniki, nazwa)

        wiersze.append({
            "Lokalizacja": nazwa,
            "Uzysk_kWh": wskazniki["E_pv_kWh"],
            "SSR": wskazniki["SSR"],
            "LPSP": wskazniki["LPSP"],
            "Deficyt_kWh": wskazniki["E_deficyt_kWh"],
            "Godziny_deficytu": wskazniki["godziny_deficytu"],
        })

    if not wiersze:
        print("\nPrzerwano: żadna lokalizacja nie została policzona.")
        return 1

    wypisz_tabele(zestawienie_wynikow(wiersze))
    print("\nUWAGA: powyższe liczby służą weryfikacji potoku obliczeniowego.")
    print("Nie są to wyniki końcowe pracy — konfiguracja generatora, profil")
    print("obciążenia i dobór magazynu mają charakter roboczy.")
    return 0


def przebieg_bez_pysam(parametry: battery_balance.ParametryUkladu) -> int:
    """
    Sprawdza etapy 1 i 3 potoku bez udziału PySAM.

    Tryb pomocniczy, uruchamiany gdy pakiet nrel-pysam nie jest dostępny dla
    zainstalowanej wersji interpretera. Pozwala potwierdzić, że pobieranie
    danych oraz model bilansu działają, zanim skonfigurowane zostanie
    środowisko zdolne uruchomić SSC.
    """
    print("\n" + "!" * 72)
    print("TRYB ZASTĘPCZY — bez PySAM.")
    print("Generacja PV pochodzi z przebiegu SYNTETYCZNEGO, nie z modelu SAM.")
    print("Wyników liczbowych nie wolno cytować w pracy.")
    print("!" * 72)

    fetch_weather.fetch_raw(force=False, tmy_only=True)
    pliki = fetch_weather.make_working_copies()
    print(f"\nEtap 1 zweryfikowany: dostępnych {len(pliki)} plików EPW.")

    moc_pv_kw = battery_balance._syntetyczna_generacja()
    przebiegi, wskazniki = battery_balance.bilans_godzinowy(moc_pv_kw, parametry)
    battery_balance.wypisz_wskazniki(wskazniki, "przebieg syntetyczny")

    print("\nEtap 3 zweryfikowany: bilans zamyka się, wskaźniki policzone.")
    print("Etap 2 (PySAM/SSC) pozostaje niezweryfikowany.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Potok obliczeniowy: PVGIS -> PySAM Pvsamv1 -> bilans magazynu."
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="pobierz pliki EPW ponownie, nawet jeśli już istnieją",
    )
    parser.add_argument(
        "--pojemnosc", type=float, default=1.28,
        help="pojemność znamionowa magazynu w kWh (domyślnie 1.28 = 100 Ah / 12,8 V)",
    )
    parser.add_argument(
        "--obciazenie", type=float, default=1.0,
        help="dobowe zapotrzebowanie na energię w kWh (domyślnie 1.0)",
    )
    parser.add_argument(
        "--bez-pysam", action="store_true", dest="bez_pysam",
        help="sprawdź etapy 1 i 3 na przebiegu syntetycznym, z pominięciem PySAM",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Autonomiczny system zasilania VAPV — potok obliczeniowy")
    print("Weryfikacja drożności narzędzi")
    print("=" * 72)

    # Wersje narzędzi wypisywane są zawsze na starcie — numer wersji SSC jest
    # potrzebny do opisu środowiska obliczeniowego w rozdz. 6.1.
    run_pv.wypisz_wersje()

    parametry = battery_balance.ParametryUkladu(
        pojemnosc_kwh=args.pojemnosc,
        zapotrzebowanie_dobowe_kwh=args.obciazenie,
    )

    if args.bez_pysam:
        return przebieg_bez_pysam(parametry)

    if not run_pv.PYSAM_DOSTEPNY:
        print("BŁĄD: pakiet nrel-pysam nie jest dostępny w tym interpreterze.")
        print(f"       {run_pv._BLAD_IMPORTU}")
        print()
        print("       Pakiet nrel-pysam udostępniany jest w postaci kół binarnych")
        print("       wyłącznie dla Pythona 3.9-3.13. Zainstaluj interpreter z tego")
        print("       zakresu, utwórz środowisko wirtualne i powtórz instalację:")
        print()
        print("           py -3.13 -m venv .venv")
        print("           .venv\\Scripts\\activate")
        print("           pip install -r requirements.txt")
        print()
        print("       Aby sprawdzić pozostałe etapy potoku: python main.py --bez-pysam")
        return 2

    return przebieg_pelny(parametry, args.force_download)


if __name__ == "__main__":
    sys.exit(main())
