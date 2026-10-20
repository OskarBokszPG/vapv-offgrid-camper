# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: ghi_stats.py
Opis: sumy promieniowania na płaszczyznę poziomą dla rozdz. 4.5.
      Dlaczego osobny skrypt. Tabela i rysunek z zasobami słonecznymi
      powstawały dotąd z pliku ``out/ghi_monthly_stats.csv`` wyliczonego
      doraźnie, poza potokiem: żaden skrypt projektu go nie odtwarzał.
      Ten skrypt zamyka tę lukę — liczy statystyki z surowych szeregów
      PVGIS i zapisuje manifest przebiegu, dzięki czemu rysunek i tabela
      wiedzą, z czego powstały.

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

import config
import manifest
import weather_qc
from config import LOCATIONS

BAZA = "PVGIS-SARAH3"
PRZEBIEG = "zasoby_sloneczne"


def sumy_miesieczne() -> pd.DataFrame:
    """Sumy miesięczne nasłonecznienia dla każdej lokalizacji i roku, kWh/m²."""
    wiersze = []
    for loc in LOCATIONS.values():
        sciezka = config.raw_series_path(loc, BAZA)
        if not sciezka.exists():
            raise FileNotFoundError(
                f"Brak szeregu {sciezka.name}. Uruchom najpierw fetch_weather.py."
            )
        ramka, _ = weather_qc.parse_seriescalc(sciezka, loc, "SARAH3")
        # Do agregacji energii znacznik zaokrąglany jest w dół do pełnej
        # godziny — konwencja z weather_qc.parse_seriescalc.
        godziny = ramka.index.floor("h")
        grupa = (ramka["GHI"].groupby([godziny.year, godziny.month]).sum() / 1000.0)
        for (rok, miesiac), wartosc in grupa.items():
            wiersze.append({"lokalizacja": loc.label, "klucz": loc.key,
                            "rok": int(rok), "miesiac": int(miesiac),
                            "ghi_kwh_m2": round(float(wartosc), 3)})
    return pd.DataFrame(wiersze)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_dirs()

    miesieczne = sumy_miesieczne()
    okres = (int(miesieczne["rok"].min()), int(miesieczne["rok"].max()))

    staty = (miesieczne.groupby(["lokalizacja", "miesiac"])["ghi_kwh_m2"]
             .agg(["median", "min", "max"]).round(1).reset_index())
    sciezka_m = config.OUT_DIR / "ghi_monthly_stats.csv"
    staty.to_csv(sciezka_m, index=False)

    roczne = (miesieczne.groupby(["lokalizacja", "rok"])["ghi_kwh_m2"]
              .sum().round(2).reset_index()
              .rename(columns={"ghi_kwh_m2": "ghi_roczne_kwh_m2"}))
    sciezka_r = config.OUT_DIR / "ghi_annual_sums.csv"
    roczne.to_csv(sciezka_r, index=False)

    manifest.zapisz(
        PRZEBIEG,
        skrypt="simulation/ghi_stats.py",
        konfiguracja={
            "baza_promieniowania": BAZA,
            "okres_lat": list(okres),
            "lokalizacje": [l.label for l in LOCATIONS.values()],
            "agregacja": "sumy miesięczne per rok, potem mediana/min/max po latach",
        },
        wejscia={loc.key: config.raw_series_path(loc, BAZA)
                 for loc in LOCATIONS.values()},
        wyniki=[sciezka_m, sciezka_r],
    )

    print("=" * 78)
    print("Zasoby słoneczne — %s, lata %d-%d" % (BAZA, *okres))
    print("=" * 78)
    for nazwa, grupa in staty.groupby("lokalizacja"):
        mediana_roczna = roczne.loc[roczne.lokalizacja == nazwa,
                                    "ghi_roczne_kwh_m2"].median()
        czerwiec = float(grupa.loc[grupa.miesiac == 6, "median"].iloc[0])
        grudzien = float(grupa.loc[grupa.miesiac == 12, "median"].iloc[0])
        print("  %-8s mediana roczna %7.1f kWh/m2 | VI %6.1f | XII %5.1f | "
              "stosunek %4.1f" % (nazwa, mediana_roczna, czerwiec, grudzien,
                                  czerwiec / grudzien))
    print()
    print("zapisano:", sciezka_m.name, "oraz", sciezka_r.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
