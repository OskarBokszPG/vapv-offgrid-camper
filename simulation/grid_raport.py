# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: grid_raport.py
Opis: postprocessing siatki: łańcuch energii, progi, warunek początkowy.

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
import manifest
import loads
import run_grid
import run_pv
from config import LOCATIONS

# Konfiguracja zalecana (B11): moduł bazowy, zasilanie laptopa z przetwornicy,
# ładowarka o większym prądzie. Pojemność wyznaczana z oceny wymagań.
PRZEBIEG = "raport_siatki"

MODUL_ZALECANY = "AIKO 470 Wp"
SCENARIUSZ_ZALECANY = "A-DC"
LADOWARKA_ZALECANA = "Orion XS 50"

PROGI_W1 = (0.03, 0.05, 0.07)
PROGI_W3 = (0.75, 1.0, 1.5)


def pojemnosc_zalecana(wymagania: pd.DataFrame) -> str:
    """Najmniejsza pojemność LFP spełniająca wymagania we wszystkich lokalizacjach."""
    lfp = wymagania[wymagania.magazyn.str.startswith("LFP")].copy()
    lfp["ah"] = lfp.magazyn.str.extract(r"(\d+)").astype(int)
    for kryterium in ("spelnia_W1_W4", "W1_W2_W4"):
        if kryterium not in lfp.columns:
            lfp["W1_W2_W4"] = (lfp["W1_spelnione"] & lfp["W2_spelnione"]
                               & lfp["W4_spelnione"])
        zdane = (lfp[lfp[kryterium]].groupby("ah")["lokalizacja"].nunique())
        pelne = [ah for ah, n in zdane.items() if n == lfp.lokalizacja.nunique()]
        if pelne:
            return "LFP %d Ah" % min(pelne)
    return ""


def lancuch_energii(magazyn: str) -> pd.DataFrame:
    """
    Łańcuch przemian energii dla konfiguracji zalecanej, mediana z lat.

    Wszystkie pozycje liczone są w tym samym okresie, w którym liczone są
    wskaźniki, to jest po pominięciu okresu rozbiegowego (run_grid.ROZBIEG_
    GODZIN). Gdyby nasłonecznienie i energia stałoprądowa obejmowały cały
    rok, a pozostałe pozycje tylko okres oceniany, udziały procentowe
    liczone z tej tabeli dotyczyłyby różnych okresów.
    """
    par = bb.ParametryUkladu(
        pojemnosc_kwh=run_grid.pojemnosc_kwh(int(magazyn.split()[1])))
    lata = range(config.SERIES_START_YEAR, config.SERIES_END_YEAR + 1)
    wiersze = []
    for loc in LOCATIONS.values():
        rocznie = []
        for rok in lata:
            przebiegi, w = run_grid.przebieg(
                loc, rok, run_pv.MODUL_KATALOG, par, SCENARIUSZ_ZALECANY,
                LADOWARKA_ZALECANA, config.ALTERNATOR_PRAD_A)
            plik = run_grid.resource.resource_path(loc, rok)
            ghi_godzinowe = pd.read_csv(plik, skiprows=2)["GHI"]
            rozbieg = run_grid.ROZBIEG_GODZIN
            # Nasłonecznienie, uzysk i współczynnik wydajności podaje się
            # dla pełnego roku, żeby były porównywalne z tabelami uzysku
            # właściwego i przenośności. Wskaźniki bilansowe liczone są
            # w okresie ocenianym.
            ghi = float(ghi_godzinowe.sum()) / 1000.0
            oceniane = przebiegi.iloc[rozbieg:]
            e_dc = float(przebiegi["P_pv_kW"].sum())
            kontrola = bb.kontrola_zamkniecia(oceniane, par)
            rocznie.append({
                "H_GHI_kWh_m2": ghi,
                "E_DC_kWh": e_dc,
                "E_MPPT_kWh": e_dc * par.sprawnosc_mppt,
                "E_DC_oceniany_kWh": float(oceniane["P_pv_kW"].sum()),
                "E_wykorzystana_kWh": w["E_pokryte_kWh"],
                "E_curt_kWh": w["E_curt_kWh"],
                "E_def_kWh": w["E_deficyt_kWh"],
                "E_strat_falownika_kWh": float(
                    przebiegi["E_falownik_jalowy_kWh"].sum()),
                "E_strat_magazynu_kWh": (kontrola["E_strat_ladowania_kWh"]
                                         + kontrola["E_strat_rozladowania_kWh"]),
                "E_samorozladowanie_kWh": w["E_samorozladowanie_kWh"],
                "E_pobory_wlasne_kWh": (loads.pobory_wlasne_w()
                                        * len(oceniane) / 1000.0),
                "E_alt_kWh": w["E_alternator_kWh"],
                "E_blokada_pv_kWh": w["E_blokada_pv_kWh"],
                "E_blokada_alt_kWh": w["E_blokada_alternator_kWh"],
                "godziny_blokady": w.get("godziny_blokady", 0),
                "blokada_pv_pct_E_DC": (100.0 * w["E_blokada_pv_kWh"]
                                        / float(oceniane["P_pv_kW"].sum())),
                "PR_DC": e_dc / (ghi * run_pv.MODUL_KATALOG["p_mp"] / 1000.0),
                "SSR_PV": w["SSR_PV"],
                "LPSP": w["LPSP"],
            })
        mediana = pd.DataFrame(rocznie).median().round(2)
        mediana["lokalizacja"] = loc.label
        wiersze.append(mediana)
    tab = pd.DataFrame(wiersze)
    return tab[["lokalizacja"] + [c for c in tab.columns if c != "lokalizacja"]]


def miesieczne_projekt(magazyn: str) -> pd.DataFrame:
    """
    Wskaźniki miesięczne konfiguracji projektowej dla scenariuszy A-DC i C.

    Ocena wymagania W1 prowadzona jest osobno dla każdego miesiąca, więc
    wynikiem jest zbiór miesięcy spełniających próg, a nie jedna liczba
    roczna. Okres rozbiegowy wyłączono tak samo jak we wskaźnikach rocznych.
    """
    par = bb.ParametryUkladu(
        pojemnosc_kwh=run_grid.pojemnosc_kwh(int(magazyn.split()[1])))
    lata = range(config.SERIES_START_YEAR, config.SERIES_END_YEAR + 1)
    czesci = []
    for scen in ("A-AC", "A-DC", "C"):
        for loc in LOCATIONS.values():
            for rok in lata:
                godzin_c = 0.0
                if scen == "C":
                    godzin_c = run_grid.czas_jazdy_min(
                        loc, rok, run_pv.MODUL_KATALOG, par,
                        LADOWARKA_ZALECANA, config.ALTERNATOR_PRAD_A)
                    if godzin_c != godzin_c:
                        godzin_c = 0.0
                przebiegi, _ = run_grid.przebieg(
                    loc, rok, run_pv.MODUL_KATALOG, par, scen,
                    LADOWARKA_ZALECANA, config.ALTERNATOR_PRAD_A,
                    godzin_jazdy_C=godzin_c)
                tab = bb.wskazniki_miesieczne(
                    przebiegi.iloc[run_grid.ROZBIEG_GODZIN:], par)
                tab.insert(0, "scenariusz", scen)
                tab.insert(0, "rok", rok)
                tab.insert(0, "lokalizacja", loc.label)
                czesci.append(tab)
    return pd.concat(czesci, ignore_index=True)


def przebieg_tygodnia(magazyn: str) -> pd.DataFrame:
    """
    Przebieg godzinowy okna siedmiodobowego o najniższym stanie naładowania.

    Okno wybierane jest w scenariuszu zimowym, w lokalizacji i roku, dla
    których stan naładowania schodzi najniżej. Zwracany jest przebieg
    godzinowy tego okna wraz z kolumną czasu, żeby rysunek nie musiał
    powtarzać wyboru.
    """
    par = bb.ParametryUkladu(
        pojemnosc_kwh=run_grid.pojemnosc_kwh(int(magazyn.split()[1])))
    lata = range(config.SERIES_START_YEAR, config.SERIES_END_YEAR + 1)
    najnizszy, wybrane = 2.0, None
    for loc in LOCATIONS.values():
        for rok in lata:
            godzin_c = run_grid.czas_jazdy_min(
                loc, rok, run_pv.MODUL_KATALOG, par, LADOWARKA_ZALECANA,
                config.ALTERNATOR_PRAD_A)
            if godzin_c != godzin_c:
                godzin_c = 0.0
            przebiegi, _ = run_grid.przebieg(
                loc, rok, run_pv.MODUL_KATALOG, par, "C", LADOWARKA_ZALECANA,
                config.ALTERNATOR_PRAD_A, godzin_jazdy_C=godzin_c)
            oceniane = przebiegi.iloc[run_grid.ROZBIEG_GODZIN:]
            okno = oceniane["SoC_wzgledny"].rolling(168).min()
            if float(okno.min()) < najnizszy:
                najnizszy = float(okno.min())
                koniec = int(okno.argmin())
                wybrane = (oceniane.iloc[max(0, koniec - 167):koniec + 1]
                           .copy())
                wybrane.insert(0, "t_jazdy_h", godzin_c)
                wybrane.insert(0, "rok", rok)
                wybrane.insert(0, "lokalizacja", loc.label)
    if wybrane is None:
        raise RuntimeError("nie znaleziono okna siedmiodobowego")
    wybrane = wybrane.reset_index().rename(columns={"index": "czas"})
    return wybrane


def bilans_przykladowy(magazyn: str, lokalizacja: str, rok: int) -> pd.DataFrame:
    """
    Bilans jednego przebiegu: pozycje równania kontrolnego.

    Tabela służy pokazaniu, z czego składa się bilans godzinowy po zsumowaniu
    do roku, i zamknięciu rachunku: reszta równania kontrolnego ma być rzędu
    błędu maszynowego. Przebieg jest ten sam, na którym prowadzona jest
    analiza wrażliwości.
    """
    par = bb.ParametryUkladu(
        pojemnosc_kwh=run_grid.pojemnosc_kwh(int(magazyn.split()[1])))
    loc = next(v for v in LOCATIONS.values() if v.label == lokalizacja)
    przebiegi, w = run_grid.przebieg(
        loc, rok, run_pv.MODUL_KATALOG, par, SCENARIUSZ_ZALECANY,
        LADOWARKA_ZALECANA, config.ALTERNATOR_PRAD_A)
    oceniane = przebiegi.iloc[run_grid.ROZBIEG_GODZIN:]
    # Stan na początku okresu ocenianego, a nie na początku przebiegu:
    # inaczej reszta równania kontrolnego zawierałaby różnicę stanów.
    stan_przed = float(przebiegi["SoC_kWh"].iloc[run_grid.ROZBIEG_GODZIN - 1])
    kontrola = bb.kontrola_zamkniecia(oceniane, par, stan_przed)
    stan_poczatkowy = stan_przed
    stan_koncowy = float(oceniane["SoC_kWh"].iloc[-1])
    pozycje = [
        ("energia magazynu na początku okresu", stan_poczatkowy),
        ("energia magazynu na końcu okresu", stan_koncowy),
        ("dopływ z generatora za regulatorem",
         kontrola["E_pv_dostarczona_kWh"]),
        ("dopływ zaoferowany przez alternator",
         kontrola["E_alt_ofiarowana_kWh"]),
        ("energia dostarczona odbiornikom", w["E_pokryte_kWh"]),
        ("straty ładowania magazynu", kontrola["E_strat_ladowania_kWh"]),
        ("straty rozładowania magazynu", kontrola["E_strat_rozladowania_kWh"]),
        ("samorozładowanie magazynu", w["E_samorozladowanie_kWh"]),
        ("energia nieprzyjęta (pełny lub zimny magazyn)", w["E_curt_kWh"]),
        ("energia niedostarczona odbiornikom", w["E_deficyt_kWh"]),
        ("reszta równania kontrolnego", kontrola["reszta_kWh"]),
    ]
    tab = pd.DataFrame(pozycje, columns=["pozycja", "energia_kWh"])
    tab.insert(0, "rok", rok)
    tab.insert(0, "lokalizacja", lokalizacja)
    return tab


def warunek_poczatkowy(magazyn: str) -> pd.DataFrame:
    """Porównanie metryk pierwszego roku dla dwóch stanów początkowych."""
    pojemnosc = run_grid.pojemnosc_kwh(int(magazyn.split()[1]))
    rok = config.SERIES_START_YEAR
    wiersze = []
    for loc in LOCATIONS.values():
        for opis, soc0 in (("SOC_max", None), ("SOC_min", "min")):
            par = bb.ParametryUkladu(pojemnosc_kwh=pojemnosc)
            if soc0 == "min":
                par = bb.ParametryUkladu(
                    pojemnosc_kwh=pojemnosc,
                    soc_poczatkowy_wzgl=par.soc_min_wzgl)
            _, w = run_grid.przebieg(loc, rok, run_pv.MODUL_KATALOG, par,
                                     SCENARIUSZ_ZALECANY, LADOWARKA_ZALECANA,
                                     config.ALTERNATOR_PRAD_A)
            wiersze.append({"lokalizacja": loc.label, "stan_poczatkowy": opis,
                            "LPSP": round(w["LPSP"], 5),
                            "SSR_PV": round(w["SSR_PV"], 4)})
    return pd.DataFrame(wiersze)


def chlodziarka_warianty(magazyn: str) -> pd.DataFrame:
    """Werdykt W1 dla obu postaci modelu chłodziarki."""
    par = bb.ParametryUkladu(
        pojemnosc_kwh=run_grid.pojemnosc_kwh(int(magazyn.split()[1])))
    lata = range(config.SERIES_START_YEAR, config.SERIES_END_YEAR + 1)
    wiersze = []
    for model in loads.MODELE_CHLODZIARKI:
        for loc in LOCATIONS.values():
            najgorszy = 0.0
            for rok in lata:
                _, w = run_grid.przebieg(
                    loc, rok, run_pv.MODUL_KATALOG, par, "A-AC",
                    LADOWARKA_ZALECANA, config.ALTERNATOR_PRAD_A,
                    model_chlodziarki=model)
                najgorszy = max(najgorszy, w["LPSP"])
            wiersze.append({"model_chlodziarki": model,
                            "lokalizacja": loc.label,
                            "LPSP_rok_najgorszy": round(najgorszy, 5),
                            "W1_spelnione": bool(najgorszy <= run_grid.W1_PROG_LPSP)})
    return pd.DataFrame(wiersze)


def wrazliwosc_progow(wymagania: pd.DataFrame) -> pd.DataFrame:
    """Liczba konfiguracji spełniających wymagania przy przesuniętych progach."""
    wiersze = []
    for prog1 in PROGI_W1:
        for prog3 in PROGI_W3:
            w1 = wymagania["W1_LPSP_najgorszy"] <= prog1
            w3 = wymagania["W3_t_jazdy_max_h"] <= prog3
            wiersze.append({
                "prog_W1": prog1, "prog_W3_h": prog3,
                "konfiguracji_W1": int(w1.sum()),
                "konfiguracji_W3": int(w3.sum()),
                "konfiguracji_W1_i_W3": int((w1 & w3).sum()),
                "wszystkich": int(len(wymagania)),
            })
    return pd.DataFrame(wiersze)


def odniesienie_oat() -> tuple[str, int]:
    """Lokalizacja i rok przebiegu, na którym liczona jest analiza OAT."""
    oat = pd.read_csv(config.OUT_DIR / "grid_oat.csv")
    return str(oat["lokalizacja"].iloc[0]), int(oat["rok"].iloc[0])


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_dirs()
    wymagania = pd.read_csv(config.OUT_DIR / "requirements_check.csv")
    wymagania = wymagania[(wymagania.modul == MODUL_ZALECANY)
                          & (wymagania.ladowarka == LADOWARKA_ZALECANA)]

    magazyn = pojemnosc_zalecana(wymagania)
    if not magazyn:
        magazyn = "LFP 300 Ah"
    print("Magazyn projektowy:", magazyn)

    tabele = {
        "grid_miesieczne_projekt": miesieczne_projekt(magazyn),
        "grid_bilans_przykladowy": bilans_przykladowy(
            magazyn, *odniesienie_oat()),
        "grid_tydzien_krytyczny": przebieg_tygodnia(magazyn),
        "grid_lancuch_energii": lancuch_energii(magazyn),
        "grid_warunek_poczatkowy": warunek_poczatkowy(magazyn),
        "grid_chlodziarka": chlodziarka_warianty(magazyn),
        "grid_progi": wrazliwosc_progow(wymagania),
    }
    for nazwa, tab in tabele.items():
        tab.to_csv(config.OUT_DIR / (nazwa + ".csv"), index=False)
        print("  %s: %d wierszy" % (nazwa, len(tab)))

    manifest.zapisz(
        PRZEBIEG,
        skrypt="simulation/grid_raport.py",
        konfiguracja={
            "modul": MODUL_ZALECANY, "scenariusz": SCENARIUSZ_ZALECANY,
            "ladowarka": LADOWARKA_ZALECANA, "magazyn": magazyn,
            "progi_W1": list(PROGI_W1), "progi_W3": list(PROGI_W3),
            "rozbieg_godzin": run_grid.ROZBIEG_GODZIN,
        },
        wejscia={"grid_results": config.OUT_DIR / "grid_results.csv",
                 "requirements_check": (config.OUT_DIR
                                        / "requirements_check.csv")},
        wyniki=[config.OUT_DIR / (n + ".csv") for n in tabele])
    print("manifest:", manifest.sciezka_manifestu(PRZEBIEG).name)

    print()
    print(tabele["grid_lancuch_energii"][
        ["lokalizacja", "H_GHI_kWh_m2", "E_DC_kWh", "PR_DC", "E_curt_kWh",
         "E_alt_kWh", "E_blokada_pv_kWh", "blokada_pv_pct_E_DC",
         "godziny_blokady", "SSR_PV", "LPSP"]].to_string(index=False))
    print()
    print(tabele["grid_warunek_poczatkowy"].to_string(index=False))
    print()
    print(tabele["grid_chlodziarka"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
