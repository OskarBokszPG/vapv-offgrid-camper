# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: wariant_pochylenia.py
Opis: wariant z modułem podnoszonym na postoju. Moduł leży na platformie
      przez cały czas jazdy, a na postoju jest podnoszony jedną
      krawędzią do kąta beta. Rozpatrywane są dwa przypadki orientacji:
      ustawienie południowe oraz orientacja przypadkowa, dla której
      wynik uśredniono po ośmiu azymutach co 45 stopni.

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

import battery_balance as bb
import config
import grid_analiza
import manifest
import run_grid
import run_pv
from config import LOCATIONS

PRZEBIEG = "wariant_pochylenia"

POCHYLENIA_DEG = (15.0, 30.0, 45.0)
AZYMUT_POLUDNIE = 180.0
AZYMUTY_PRZYPADKOWE = tuple(float(a) for a in range(0, 360, 45))
ALBEDO = 0.2

MAGAZYN_AH = grid_analiza.OAT_MAGAZYN_AH
LADOWARKA = grid_analiza.OAT_LADOWARKA
SCENARIUSZ = grid_analiza.OAT_SCENARIUSZ      # A-DC
SCENARIUSZ_W1 = "A-AC"                        # scenariusz oceny wymagania W1
POJEMNOSCI_AH = run_grid.POJEMNOSCI_AH


def _generacja(loc, rok, pochylenie: float, azymut: float) -> np.ndarray:
    """Szereg mocy stałoprądowej dla zadanej geometrii montażu, kW."""
    poprzednie = (run_pv.POCHYLENIE_DEG, run_pv.AZYMUT_DEG,
                  getattr(run_pv, "ALBEDO", None))
    run_pv.POCHYLENIE_DEG = pochylenie
    run_pv.AZYMUT_DEG = azymut
    run_pv.ALBEDO = ALBEDO
    run_grid._PAMIEC_PV.clear()
    try:
        moc_kw, _ = run_grid.generacja(loc, rok, run_pv.MODUL_KATALOG)
        return moc_kw.copy()
    finally:
        run_pv.POCHYLENIE_DEG, run_pv.AZYMUT_DEG = poprzednie[0], poprzednie[1]
        if poprzednie[2] is None:
            if hasattr(run_pv, "ALBEDO"):
                delattr(run_pv, "ALBEDO")
        else:
            run_pv.ALBEDO = poprzednie[2]
        run_grid._PAMIEC_PV.clear()


def _bilans(loc, rok, moc_kw: np.ndarray, scenariusz: str,
            pojemnosc_ah: int) -> tuple[pd.DataFrame, dict]:
    """Bilans godzinowy dla zadanego szeregu generacji."""
    par = bb.ParametryUkladu(pojemnosc_kwh=run_grid.pojemnosc_kwh(pojemnosc_ah))
    profil = run_grid.profil_scenariusza(loc, rok, scenariusz)
    _, indeks = run_grid.generacja(loc, rok, run_pv.MODUL_KATALOG)
    bb.sprawdz_zgodnosc_indeksu(indeks, profil.index)
    lokalny = profil.index.tz_convert(config.LOCAL_TZ)
    udzial = run_grid.maska_jazdy(lokalny, scenariusz)
    alternator = run_grid.doplyw_alternatora(udzial, LADOWARKA,
                                             config.ALTERNATOR_PRAD_A)
    t_ogniw = run_grid.temperatura_magazynu(loc, rok, profil, par, udzial)
    dodatek_w = loads_pobor_jazdy()
    obc_dc = profil["DC"].to_numpy(float) / 1000.0 + udzial * dodatek_w / 1000.0
    obc_ac = profil["AC"].to_numpy(float) / 1000.0
    jalowy = profil["pobór falownika z szyny DC"].to_numpy(float) / 1000.0
    jalowy = np.maximum(jalowy - obc_ac, 0.0)
    return bb.bilans_godzinowy(
        moc_kw, par, indeks=indeks,
        obciazenie_dc_kwh=obc_dc,
        obciazenie_ac_kwh=obc_ac if obc_ac.sum() > 0 else None,
        falownik_jalowy_kwh=jalowy if obc_ac.sum() > 0 else None,
        doplyw_alternatora_kwh=alternator,
        temperatura_ogniw_C=t_ogniw,
        rozbieg_godzin=run_grid.ROZBIEG_GODZIN)


def loads_pobor_jazdy() -> float:
    import loads

    return loads.pobory_wlasne_w(jazda=True) - loads.pobory_wlasne_w()


def _na_postoju(moc_pochylona: np.ndarray, moc_plaska: np.ndarray,
                loc, rok, scenariusz: str) -> np.ndarray:
    """Łączy szeregi: pochylenie na postoju, płasko w godzinach jazdy."""
    _, indeks = run_grid.generacja(loc, rok, run_pv.MODUL_KATALOG)
    lokalny = indeks.tz_convert(config.LOCAL_TZ)
    udzial = run_grid.maska_jazdy(lokalny, scenariusz)
    return np.where(udzial > 0.0, moc_plaska, moc_pochylona)


def przebieg_wariantu(loc, rok: int, pochylenie: float, azymut: float,
                      moc_plaska: np.ndarray, pojemnosc_ah: int,
                      scenariusz: str) -> dict:
    """Wskaźniki jednego wariantu geometrii dla jednego roku."""
    moc = _na_postoju(_generacja(loc, rok, pochylenie, azymut),
                      moc_plaska, loc, rok, scenariusz)
    przebiegi, w = _bilans(loc, rok, moc, scenariusz, pojemnosc_ah)
    oceniane = przebiegi.iloc[run_grid.ROZBIEG_GODZIN:]
    miesieczne = bb.wskazniki_miesieczne(oceniane)
    return {
        "E_DC_kWh": float(przebiegi["P_pv_kW"].sum()),
        "E_pokryte_kWh": w["E_pokryte_kWh"],
        "E_curt_kWh": w["E_curt_kWh"],
        "LPSP": w["LPSP"],
        "SSR_PV": w["SSR_PV"],
        "miesieczne": miesieczne,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_dirs()
    lata = list(range(config.SERIES_START_YEAR, config.SERIES_END_YEAR + 1))
    rok_odniesienia = int(pd.read_csv(config.OUT_DIR / "grid_oat.csv")
                          ["rok"].iloc[0])

    wiersze, miesiace = [], []
    for loc in LOCATIONS.values():
        # Szereg odniesienia: moduł leżący na platformie.
        plaskie = {rok: run_grid.generacja(loc, rok, run_pv.MODUL_KATALOG)[0]
                   for rok in lata}
        odniesienie = {}
        for rok in lata:
            przebiegi, w = _bilans(loc, rok, plaskie[rok], SCENARIUSZ,
                                   MAGAZYN_AH)
            odniesienie[rok] = {
                "E_DC_kWh": float(przebiegi["P_pv_kW"].sum()),
                "E_pokryte_kWh": w["E_pokryte_kWh"],
                "LPSP": w["LPSP"],
            }
        wiersze.append({
            "lokalizacja": loc.label, "wariant": "poziomo",
            "pochylenie_deg": run_pv.POCHYLENIE_DEG, "azymut": "---",
            "E_DC_kWh": float(np.median([o["E_DC_kWh"]
                                         for o in odniesienie.values()])),
            "E_pokryte_kWh": float(np.median([o["E_pokryte_kWh"]
                                              for o in odniesienie.values()])),
            "LPSP_rok_najgorszy": float(max(o["LPSP"]
                                            for o in odniesienie.values())),
        })

        for beta in POCHYLENIA_DEG:
            # --- orientacja południowa: pełny szereg lat -----------------
            wyniki = [przebieg_wariantu(loc, rok, beta, AZYMUT_POLUDNIE,
                                        plaskie[rok], MAGAZYN_AH, SCENARIUSZ)
                      for rok in lata]
            wiersze.append({
                "lokalizacja": loc.label, "wariant": "południe",
                "pochylenie_deg": beta, "azymut": "180",
                "E_DC_kWh": float(np.median([w["E_DC_kWh"] for w in wyniki])),
                "E_pokryte_kWh": float(np.median([w["E_pokryte_kWh"]
                                                  for w in wyniki])),
                "LPSP_rok_najgorszy": float(max(w["LPSP"] for w in wyniki)),
            })
            for rok, w in zip(lata, wyniki):
                tab = w["miesieczne"].copy()
                tab.insert(0, "pochylenie_deg", beta)
                tab.insert(0, "wariant", "południe")
                tab.insert(0, "rok", rok)
                tab.insert(0, "lokalizacja", loc.label)
                miesiace.append(tab)

            # --- orientacja przypadkowa: rok odniesienia, osiem azymutów --
            losowe = [przebieg_wariantu(loc, rok_odniesienia, beta, azymut,
                                        plaskie[rok_odniesienia], MAGAZYN_AH,
                                        SCENARIUSZ)
                      for azymut in AZYMUTY_PRZYPADKOWE]
            wiersze.append({
                "lokalizacja": loc.label, "wariant": "przypadkowa",
                "pochylenie_deg": beta,
                "azymut": "średnia z 8",
                "E_DC_kWh": float(np.mean([w["E_DC_kWh"] for w in losowe])),
                "E_pokryte_kWh": float(np.mean([w["E_pokryte_kWh"]
                                                for w in losowe])),
                "LPSP_rok_najgorszy": float(np.mean([w["LPSP"]
                                                     for w in losowe])),
            })
        print("  %s: gotowe" % loc.label, flush=True)

    tab = pd.DataFrame(wiersze)
    tab.to_csv(config.OUT_DIR / "wariant_pochylenia.csv", index=False)
    pd.concat(miesiace, ignore_index=True).to_csv(
        config.OUT_DIR / "wariant_pochylenia_miesiace.csv", index=False)

    manifest.zapisz(
        PRZEBIEG,
        skrypt="simulation/wariant_pochylenia.py",
        konfiguracja={
            "pochylenia_deg": list(POCHYLENIA_DEG),
            "azymut_poludnie": AZYMUT_POLUDNIE,
            "azymuty_przypadkowe": list(AZYMUTY_PRZYPADKOWE),
            "albedo": ALBEDO, "magazyn_Ah": MAGAZYN_AH,
            "ladowarka": LADOWARKA, "scenariusz": SCENARIUSZ,
            "rok_orientacji_przypadkowej": rok_odniesienia,
        },
        wejscia={"grid_results": config.OUT_DIR / "grid_results.csv"},
        wyniki=[config.OUT_DIR / "wariant_pochylenia.csv",
                config.OUT_DIR / "wariant_pochylenia_miesiace.csv"])

    print(tab.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
