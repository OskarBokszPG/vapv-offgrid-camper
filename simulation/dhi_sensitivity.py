# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: dhi_sensitivity.py
Opis: wpływ podziału promieniowania na składowe na uzysk. Po co. Moduł
      leży poziomo (pochylenie 2 stopnie), więc natężenie na jego
      płaszczyźnie jest praktycznie równe natężeniu na płaszczyźnie
      poziomej. Podział promieniowania całkowitego na składową
      bezpośrednią i rozproszoną wpływa wtedy na wynik tylko przez
      modyfikator kąta padania, który traktuje obie składowe inaczej.
      Skrypt mierzy ten wpływ: zmienia składową rozproszoną o zadany
      odsetek, zachowując sumę całkowitą, i porównuje uzysk roczny.

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
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import config
import make_sam_resource as resource
import run_pv
from config import LOCATIONS

WARIANTY = (-0.20, 0.0, +0.20)


def wariant_zasobu(zrodlo: Path, cel: Path, wspolczynnik: float) -> Path:
    """Zapisuje plik zasobu ze zmienioną składową rozproszoną."""
    naglowek = zrodlo.read_text(encoding="utf-8").splitlines()[:2]
    dane = pd.read_csv(zrodlo, skiprows=2)

    ghi = dane["GHI"].to_numpy(float)
    dni = dane["DNI"].to_numpy(float)
    dhi = dane["DHI"].to_numpy(float)

    dhi_nowe = np.clip(dhi * (1.0 + wspolczynnik), 0.0, ghi)
    bezposrednia = ghi - dhi            # DNI * cos(z)
    mozliwe = (dni > 0.0) & (bezposrednia > 1e-9)
    dni_nowe = dni.copy()
    dni_nowe[mozliwe] = dni[mozliwe] * ((ghi - dhi_nowe)[mozliwe]
                                        / bezposrednia[mozliwe])
    # Tam, gdzie składowej bezpośredniej nie ma, podziału zmienić się nie da.
    dhi_nowe[~mozliwe] = dhi[~mozliwe]

    dane["DHI"] = np.round(dhi_nowe, 1)
    dane["DNI"] = np.round(np.clip(dni_nowe, 0.0, None), 1)

    with open(cel, "w", encoding="utf-8", newline="") as plik:
        plik.write("\n".join(naglowek) + "\n")
        dane.to_csv(plik, index=False, lineterminator="\n")
    return cel


def uzysk(plik: Path) -> dict[str, float]:
    """Uruchamia model i zwraca roczny uzysk stałoprądowy."""
    model = resource._run_model_for_resource(plik)
    out = model.export()["Outputs"]
    poa = np.asarray(out["subarray1_poa_eff"], dtype=float)
    return {
        "dc_gross_kWh": round(float(out["annual_subarray1_dc_gross"]), 2),
        "poa_eff_kWh_m2": round(float(poa.sum()) / 1000.0, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location", default="Suwalki", choices=sorted(LOCATIONS))
    parser.add_argument("--year", type=int, default=2020)
    argumenty = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_dirs()

    loc = LOCATIONS[argumenty.location]
    zrodlo = resource.resource_path(loc, argumenty.year)
    if not zrodlo.exists():
        raise SystemExit("Brak pliku zasobu: %s" % zrodlo.name)

    katalog = Path(tempfile.mkdtemp(prefix="dhi_"))
    wiersze = []
    try:
        for wspolczynnik in WARIANTY:
            plik = (zrodlo if wspolczynnik == 0.0
                    else wariant_zasobu(
                        zrodlo, katalog / ("dhi_%+d.csv" % round(wspolczynnik * 100)),
                        wspolczynnik))
            dane = pd.read_csv(plik, skiprows=2)
            udzial = float(dane["DHI"].sum() / max(dane["GHI"].sum(), 1e-9))
            wiersze.append({
                "wariant": "DHI %+d %%" % round(wspolczynnik * 100),
                "udzial_rozproszonego": round(udzial, 4),
                **uzysk(plik),
            })
    finally:
        shutil.rmtree(katalog, ignore_errors=True)

    tab = pd.DataFrame(wiersze)
    odniesienie = float(tab.loc[tab.wariant == "DHI +0 %", "dc_gross_kWh"].iloc[0])
    tab["zmiana_uzysku_pct"] = ((tab["dc_gross_kWh"] - odniesienie)
                                / odniesienie * 100.0).round(3)
    tab.insert(0, "lokalizacja", loc.label)
    tab.insert(1, "rok", argumenty.year)

    cel = config.OUT_DIR / "dhi_sensitivity.csv"
    tab.to_csv(cel, index=False)

    print("=" * 90)
    print("Wpływ podziału promieniowania na składowe — %s, rok %d"
          % (loc.label, argumenty.year))
    print("=" * 90)
    print(tab.to_string(index=False))
    print()
    print("Największa zmiana uzysku: %+.3f %% przy zmianie składowej "
          "rozproszonej o 20 %%." % tab["zmiana_uzysku_pct"].abs().max())
    print("zapisano:", cel.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
