# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: grid_shift_check.py
Opis: wrażliwość sumy rocznej na położenie punktu w siatce. Po co. W
      rozdz. 4.5 stało twierdzenie, że niedokładność współrzędnych
      stacji wpływa na sumę roczną „poniżej 0,5 %". Nie było ono niczym
      poparte. Skrypt zastępuje je pomiarem: pobiera szereg SARAH3 dla
      punktu przesuniętego o jedno oczko siatki (0,05 stopnia, czyli ok.
      5,6 km na południku i ok. 3,2 km na równoleżniku dla tej
      szerokości) i porównuje sumy roczne z punktem odniesienia.

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
import urllib.request
from pathlib import Path

import pandas as pd

import config
import weather_qc
from config import LOCATIONS

# Rozdzielczość siatki bazy SARAH3 (CM SAF), stopnie.
OCZKO_SIATKI_DEG = 0.05
KATALOG = config.RAW_DIR / "grid_shift"


def pobierz(lat: float, lon: float, cel: Path) -> Path:
    """Pobiera szereg SARAH3 dla zadanego punktu, jeśli pliku jeszcze nie ma."""
    if cel.exists():
        print("  [pomijam] %s już istnieje" % cel.name)
        return cel
    url = (f"{config.PVGIS_BASE}/seriescalc"
           f"?lat={lat}&lon={lon}&raddatabase=PVGIS-SARAH3"
           f"&startyear={config.SERIES_START_YEAR}"
           f"&endyear={config.SERIES_END_YEAR}"
           f"&angle=0&aspect=0&components=1&outputformat=csv&browser=0")
    print("  GET", url)
    cel.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=config.HTTP_TIMEOUT_S) as odp:
        cel.write_bytes(odp.read())
    print("  -> %s: %d B" % (cel.name, cel.stat().st_size))
    return cel


def sumy_roczne(sciezka: Path, loc) -> pd.Series:
    """Sumy roczne nasłonecznienia z pliku szeregu, kWh/m²."""
    ramka, _ = weather_qc.parse_seriescalc(sciezka, loc, "SARAH3")
    godziny = ramka.index.floor("h")
    return ramka["GHI"].groupby(godziny.year).sum() / 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location", default="Leba", choices=sorted(LOCATIONS))
    argumenty = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_dirs()

    loc = LOCATIONS[argumenty.location]
    lat2 = round(loc.lat + OCZKO_SIATKI_DEG, 4)
    lon2 = round(loc.lon + OCZKO_SIATKI_DEG, 4)

    print("=" * 78)
    print("Wrażliwość sumy rocznej na położenie punktu — %s" % loc.label)
    print("=" * 78)
    print("  punkt odniesienia : %.3f N, %.3f E" % (loc.lat, loc.lon))
    print("  punkt przesunięty : %.3f N, %.3f E (o %.2f stopnia ukośnie)"
          % (lat2, lon2, OCZKO_SIATKI_DEG))

    plik_odniesienia = config.raw_series_path(loc, "PVGIS-SARAH3")
    if not plik_odniesienia.exists():
        raise SystemExit("Brak szeregu odniesienia: %s" % plik_odniesienia.name)
    plik_przesuniety = pobierz(
        lat2, lon2, KATALOG / f"series_{loc.key}_sarah3_shift_{lat2}_{lon2}.csv")

    a = sumy_roczne(plik_odniesienia, loc)
    b = sumy_roczne(plik_przesuniety, loc)
    tab = pd.DataFrame({"rok": a.index, "ghi_odniesienie_kwh_m2": a.values,
                        "ghi_przesuniety_kwh_m2": b.reindex(a.index).values})
    tab["roznica_kwh_m2"] = (tab["ghi_przesuniety_kwh_m2"]
                             - tab["ghi_odniesienie_kwh_m2"]).round(2)
    tab["roznica_pct"] = (tab["roznica_kwh_m2"]
                          / tab["ghi_odniesienie_kwh_m2"] * 100.0).round(3)
    for kolumna in ("ghi_odniesienie_kwh_m2", "ghi_przesuniety_kwh_m2"):
        tab[kolumna] = tab[kolumna].round(2)

    cel = config.OUT_DIR / "grid_shift_sensitivity.csv"
    tab.to_csv(cel, index=False)

    print()
    print(tab.to_string(index=False))
    print()
    print("  mediana różnicy : %+.3f %%" % tab["roznica_pct"].median())
    print("  wartość skrajna : %+.3f %%"
          % tab.loc[tab["roznica_pct"].abs().idxmax(), "roznica_pct"])
    print("  średnia sum rocznych: %.1f wobec %.1f kWh/m2"
          % (tab["ghi_przesuniety_kwh_m2"].mean(),
             tab["ghi_odniesienie_kwh_m2"].mean()))
    print()
    print("zapisano:", cel.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
