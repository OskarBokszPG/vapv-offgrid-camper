# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: grid_analiza.py
Opis: ocena wymagań, analiza wrażliwości, porównania. Czyta wyniki
      siatki (``out/grid_results.csv``) i uzupełnia je o trzy rachunki,
      których nie da się zrobić bez dodatkowych przebiegów:

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

import copy
import sys

import numpy as np
import pandas as pd

import battery_balance as bb
import config
import fmt
import loads
import manifest
import run_grid
import run_pv
from config import LOCATIONS

# Konfiguracja odniesienia analizy wrażliwości.
# Konfiguracja odniesienia analizy wrażliwości to konfiguracja projektowa
# (rozdz. 5): moduł bazowy, magazyn dobrany z oceny wymagań, ładowarka
# o większym prądzie i zalecany wariant zasilania laptopa. Lokalizację
# i rok wybiera się tą samą definicją co w ocenie wymagań: lokalizacja
# o największym wskaźniku W1a, a w niej rok najgorszy.
# Pojemnosc konfiguracji odniesienia analizy wrazliwosci. Ustawiana
# w main() na pojemnosc projektowa, czyli najmniejsza spelniajaca
# wymagania W1a-W4: analiza musi dotyczyc tej konfiguracji, na ktorej
# oparto dobor (komentarz K96).
OAT_MAGAZYN_AH = 200
# Wariant projektowy: zasilanie laptopa przez falownik. Jest on
# niekorzystniejszy od wariantu z przetwornicą o straty falownika,
# a dobór pojemności magazynu oparto właśnie na nim, więc analiza
# wrażliwości musi dotyczyć tego samego wariantu (komentarz K96).
OAT_SCENARIUSZ = "A-AC"
OAT_LADOWARKA = "Orion XS 50"
OAT_MODUL = "AIKO 470 Wp"

# Niepewność względna danych o natężeniu promieniowania w skali miesięcznej,
# wyznaczona w rozdziale 6 z opublikowanej walidacji zbioru SARAH-3
# (stosunek średniej odchyłki bezwzględnej do średniego natężenia własnych
# szeregów). Wartość nadpisuje grid_tex przez plik out/niepewnosc_ghi.json.
OAT_MNOZNIK_GHI_U = 0.042


def _wczytaj_niepewnosc_ghi() -> float:
    """Niepewność z przebiegu ``niepewnosc_ghi.py``, jeśli jest policzona."""
    plik = config.OUT_DIR / "niepewnosc_ghi.json"
    if not plik.exists():
        return OAT_MNOZNIK_GHI_U
    import json

    return float(json.loads(plik.read_text(encoding="utf-8"))["u_mnoznik_ghi"])

# Pojazdy do oceny przenośności: pole montażowe dachu, m².
# Wartości katalogowe wymiarów nadwozia; pole użytkowe przyjęto jako iloczyn
# długości i szerokości dachu pomniejszony o 15 % na relingi i wyposażenie.
POJAZDY = {
    "VW Caddy Maxi (pojazd bazowy)": 2.05,
    "VW Transporter T6.1 L2H1": 4.40,
    "VW Crafter L3H2": 7.60,
}
PASMO_PRZENOSNOSCI = 0.15


# ===========================================================================
# 1. Wymagania W1-W5
# ===========================================================================

def ocena_wymagan(wyniki: pd.DataFrame) -> pd.DataFrame:
    """Ocena wymagań W1-W4 dla każdej konfiguracji magazyn-moduł-lokalizacja."""
    wiersze = []
    # Ocena prowadzona osobno dla obu ładowarek: wariant ładowarki zmienia
    # wyłącznie wymaganie W3, ale zmienia je na tyle, że rozstrzyga o doborze.
    for (lok, modul, magazyn, ladowarka), grupa in wyniki.groupby(
            ["lokalizacja", "modul", "magazyn", "ladowarka"]):
        # --- W1: scenariusz A, rok najgorszy, miesiące III-X ---------------
        a = grupa[grupa.scenariusz == "A-AC"]
        rok_najgorszy = (a.loc[a["LPSP"].idxmax(), "rok"]
                         if not a.empty else np.nan)
        lpsp_najgorszy = float(a["LPSP"].max()) if not a.empty else np.nan

        # --- W3: scenariusz C, najdłuższy wymagany czas jazdy -------------
        c = grupa[grupa.scenariusz == "C"]
        t_jazdy = float(c["t_jazdy_min_h"].max()) if not c.empty else np.nan

        # --- W2: scenariusz B ---------------------------------------------
        b = grupa[grupa.scenariusz == "B"]
        lata_bez_deficytu = (int((b["E_deficyt_kWh"] <= 1e-9).sum())
                             if not b.empty else 0)
        udzial_b = lata_bez_deficytu / max(len(b), 1)

        wiersze.append({
            "lokalizacja": lok, "modul": modul, "magazyn": magazyn,
            "ladowarka": ladowarka,
            "W1_rok_najgorszy": rok_najgorszy,
            "W1_LPSP_najgorszy": round(lpsp_najgorszy, 4),
            "W1_spelnione": bool(lpsp_najgorszy <= run_grid.W1_PROG_LPSP),
            "W2_udzial_lat_bez_deficytu": round(udzial_b, 3),
            "W2_spelnione": bool(udzial_b >= run_grid.W2_PROG_UDZIAL),
            "W3_t_jazdy_max_h": round(t_jazdy, 2) if t_jazdy == t_jazdy else np.nan,
            "W3_spelnione": bool(t_jazdy <= run_grid.W3_PROG_H)
                            if t_jazdy == t_jazdy else False,
        })

    tab = pd.DataFrame(wiersze)

    # --- W4: masa zabudowy dachowej (nie zależy od lokalizacji ani roku) ---
    masa_platformy = config.PLATFORM_MASS_KG
    masy_modulu = {"AIKO 470 Wp": 20.6, "Victron 285 W": 14.32}
    tab["W4_masa_kg"] = tab["modul"].map(
        lambda m: round(masa_platformy + masy_modulu[m], 2))
    tab["W4_spelnione"] = tab["W4_masa_kg"] <= run_grid.W4_PROG_KG
    # Masa mocowań pozostaje nieustalona; kolumna zapisuje ten brak wprost.
    tab["W4_uwaga"] = "bez masy mocowań (DO UZUPEŁNIENIA)"
    # W5 jest listą kontrolną wymagań normatywnych, bez progu liczbowego.
    tab["W5_ocena"] = "lista kontrolna, rozdz. 3 — poza siatką"
    tab["spelnia_W1_W4"] = (tab["W1_spelnione"] & tab["W2_spelnione"]
                            & tab["W3_spelnione"] & tab["W4_spelnione"])
    return tab


# ===========================================================================
# 2. Analiza wrażliwości OAT
# ===========================================================================

def _odniesienie_oat(wyniki: pd.DataFrame) -> tuple[str, int]:
    """
    Lokalizacja i rok konfiguracji odniesienia analizy wrażliwości.

    Lokalizację wybiera wskaźnik W1a, czyli ten sam, którym ocenia się
    wymaganie; rokiem odniesienia jest rok najgorszy tej lokalizacji
    w scenariuszu analizy. Dzięki temu analiza prowadzona jest w warunkach,
    w których rozstrzyga się spełnienie wymagania.
    """
    magazyn = "LFP %d Ah" % OAT_MAGAZYN_AH
    wspolne = ((wyniki.modul == OAT_MODUL) & (wyniki.ladowarka == OAT_LADOWARKA)
               & (wyniki.magazyn == magazyn))
    w1a = wyniki[wspolne & (wyniki.scenariusz == "A-AC")]
    etykieta = w1a.groupby("lokalizacja")["LPSP"].max().idxmax()
    klucz = next(k for k, v in LOCATIONS.items() if v.label == etykieta)
    w_scen = wyniki[wspolne & (wyniki.scenariusz == OAT_SCENARIUSZ)
                    & (wyniki.lokalizacja == etykieta)]
    rok = int(w_scen.loc[w_scen["LPSP"].idxmax(), "rok"])
    return klucz, rok


def _jeden_przebieg_oat(klucz_lok: str, rok: int,
                        **nadpisania) -> dict[str, float]:
    """Jeden przebieg roczny konfiguracji odniesienia z podmianą parametru."""
    loc = LOCATIONS[klucz_lok]
    par = bb.ParametryUkladu(
        pojemnosc_kwh=run_grid.pojemnosc_kwh(OAT_MAGAZYN_AH),
        **{k: v for k, v in nadpisania.items()
           if k in bb.ParametryUkladu.__dataclass_fields__})
    _, w = run_grid.przebieg(
        loc, rok, run_pv.MODUL_KATALOG, par, OAT_SCENARIUSZ,
        OAT_LADOWARKA, nadpisania.get("prad_alternatora_A",
                                      config.ALTERNATOR_PRAD_A),
        dostepnosc_alternatora=nadpisania.get("dostepnosc", 1.0),
        model_chlodziarki=nadpisania.get("model_chlodziarki",
                                         loads.MODEL_CHLODZIARKI),
        podbicie_K=nadpisania.get("podbicie_K", loads.KABINA_PODBICIE_BAZOWE_K))
    return {"LPSP": w["LPSP"], "SSR_PV": w["SSR_PV"],
            "E_deficyt_kWh": w["E_deficyt_kWh"],
            "godziny_deficytu": w["godziny_deficytu"],
            "najdluzsza_przerwa_h": w["najdluzsza_przerwa_h"]}


def _przebieg_oat(klucz_lok: str, rok: int, **nadpisania) -> dict[str, float]:
    """
    Przypadek najgorszy dla jednego zestawu założeń.

    Wymaganie W1a ocenia rok najgorszy, a rok najgorszy może się zmieniać
    wraz z parametrem (komentarz K96). Dlatego każdy wariant liczony jest
    dla wszystkich lokalizacji i wszystkich lat szeregu, a do zestawienia
    wchodzi przebieg o największym wskaźniku niedoboru. Argumenty
    ``klucz_lok`` i ``rok`` pozostają jako wartości domyślne na wypadek
    przebiegu pojedynczego.
    """
    lata = range(config.SERIES_START_YEAR, config.SERIES_END_YEAR + 1)
    najgorszy: dict[str, float] | None = None
    for klucz in LOCATIONS:
        for r in lata:
            wynik = _jeden_przebieg_oat(klucz, r, **nadpisania)
            if najgorszy is None or wynik["LPSP"] > najgorszy["LPSP"]:
                wynik["lokalizacja_najgorsza"] = LOCATIONS[klucz].label
                wynik["rok_najgorszy"] = r
                najgorszy = wynik
    assert najgorszy is not None
    return najgorszy


def _wiersz(parametr: str, odniesienie: str, zmieniona: str,
            wynik: dict[str, float]) -> dict[str, object]:
    return {"parametr": parametr, "wartosc_odniesienia": odniesienie,
            "wartosc_zmieniona": zmieniona, **wynik}


def analiza_oat(wyniki: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analiza wrażliwości metodą jednego parametru naraz.

    Zwraca dwie tabele: wrażliwość pojedynczych parametrów oraz scenariusze
    łączne, w których zmieniono kilka założeń naraz. Konfiguracją odniesienia
    jest konfiguracja projektowa, a nie konfiguracja pomocnicza, więc każdy
    wiersz pierwszej tabeli różni się od odniesienia dokładnie jednym
    parametrem.
    """
    klucz_lok, rok = _odniesienie_oat(wyniki)
    domyslne = bb.ParametryUkladu(
        pojemnosc_kwh=run_grid.pojemnosc_kwh(OAT_MAGAZYN_AH))
    baza = _przebieg_oat(klucz_lok, rok)
    odniesienie = baza["LPSP"]
    wiersze = [_wiersz("konfiguracja odniesienia", "---", "---", baza)]

    # --- parametry bilansu -------------------------------------------------
    proste = [
        ("sprawność obiegu magazynu",
         fmt.pl(domyslne.sprawnosc_obiegu, 2), "0,90", {"sprawnosc_obiegu": 0.90}),
        ("sprawność obiegu magazynu",
         fmt.pl(domyslne.sprawnosc_obiegu, 2), "0,94", {"sprawnosc_obiegu": 0.94}),
        ("sprawność regulatora",
         fmt.pl(domyslne.sprawnosc_mppt, 2), "0,97", {"sprawnosc_mppt": 0.97}),
        ("sprawność regulatora",
         fmt.pl(domyslne.sprawnosc_mppt, 2), "0,98", {"sprawnosc_mppt": 0.98}),
        ("stała czasowa temperatury magazynu",
         fmt.pl(domyslne.tau_bat_h, 0) + " h", "1 h", {"tau_bat_h": 1.0}),
        ("stała czasowa temperatury magazynu",
         fmt.pl(domyslne.tau_bat_h, 0) + " h", "6 h", {"tau_bat_h": 6.0}),
        ("samorozładowanie magazynu",
         fmt.pl(domyslne.samorozladowanie_pct_mies, 0) + " %/mies.",
         "0 %/mies.", {"samorozladowanie_pct_mies": 0.0}),
        ("dostępność alternatora", "100 %", "50 %", {"dostepnosc": 0.5}),
        ("prąd znamionowy alternatora",
         fmt.pl(config.ALTERNATOR_PRAD_A, 0) + " A", "180 A",
         {"prad_alternatora_A": 180.0}),
        ("podbicie temperatury w zabudowie",
         fmt.pl(loads.KABINA_PODBICIE_BAZOWE_K, 0) + " K", "5 K",
         {"podbicie_K": 5.0}),
        ("podbicie temperatury w zabudowie",
         fmt.pl(loads.KABINA_PODBICIE_BAZOWE_K, 0) + " K", "10 K",
         {"podbicie_K": 10.0}),
        ("postać modelu chłodziarki", "liniowy", "stały pobór",
         {"model_chlodziarki": "staly"}),
    ]
    for parametr, odn, zm, nadpisania in proste:
        if odn == zm:
            continue    # wariant tożsamy z odniesieniem nie wnosi informacji
        wiersze.append(_wiersz(parametr, odn, zm,
                               _przebieg_oat(klucz_lok, rok, **nadpisania)))

    # --- parametry odbiorników --------------------------------------------
    odbiorniki = [
        ("pobór łącza satelitarnego", "STARLINK_MOC_W", 25.0, "W", 0),
        ("pobór łącza satelitarnego", "STARLINK_MOC_W", 40.0, "W", 0),
        ("pobór laptopa", "LAPTOP_MOC_W", 7.0, "W", 0),
        ("pobór laptopa", "LAPTOP_MOC_W", 50.0, "W", 0),
        ("oświetlenie i gniazda USB", "LED_USB_WH_D", 50.0, "Wh/d", 0),
        ("oświetlenie i gniazda USB", "LED_USB_WH_D", 100.0, "Wh/d", 0),
        ("pobór chłodziarki", "CHLODZIARKA_AH_H_REF",
         loads.CHLODZIARKA_AH_H_REF * 0.8, "Ah/h", 2),
        ("pobór chłodziarki", "CHLODZIARKA_AH_H_REF",
         loads.CHLODZIARKA_AH_H_REF * 1.2, "Ah/h", 2),
    ]
    for parametr, nazwa, wartosc, jednostka, miejsca in odbiorniki:
        poprzednia = getattr(loads, nazwa)
        setattr(loads, nazwa, wartosc)
        try:
            wynik = _przebieg_oat(klucz_lok, rok)
        finally:
            setattr(loads, nazwa, poprzednia)
        wiersze.append(_wiersz(
            parametr, "%s %s" % (fmt.pl(poprzednia, miejsca), jednostka),
            "%s %s" % (fmt.pl(wartosc, miejsca), jednostka), wynik))

    # --- parametry generacji (wymagają ponownego przebiegu Pvsamv1) -------
    generacja = [
        ("straty zabrudzenia", "STRATA_ZABRUDZENIE_PCT", 2.0, "%"),
        ("straty zabrudzenia", "STRATA_ZABRUDZENIE_PCT", 5.0, "%"),
        ("straty zacienienia", "STRATA_ZACIENIENIE_PCT", 5.0, "%"),
        ("straty zacienienia", "STRATA_ZACIENIENIE_PCT", 10.0, "%"),
    ]
    for parametr, nazwa, wartosc, jednostka in generacja:
        poprzednia = getattr(run_pv, nazwa)
        setattr(run_pv, nazwa, wartosc)
        run_grid._PAMIEC_PV.clear()
        try:
            wynik = _przebieg_oat(klucz_lok, rok)
        finally:
            setattr(run_pv, nazwa, poprzednia)
            run_grid._PAMIEC_PV.clear()
        wiersze.append(_wiersz(
            parametr, "%s %s" % (fmt.pl(poprzednia, 0), jednostka),
            "%s %s" % (fmt.pl(wartosc, 0), jednostka), wynik))

    # --- parametry modułu --------------------------------------------------
    for parametr, klucz, wartosc, jednostka, miejsca in (
            ("nominalna temperatura pracy ogniwa", "t_noct", 45.0, "°C", 1),
            ("nominalna temperatura pracy ogniwa", "t_noct", 51.0, "°C", 1),
            ("spadek początkowy mocy", "lid_pct", 1.5, "%", 1)):
        modul = copy.deepcopy(run_pv.MODUL_KATALOG)
        poprzednia = modul[klucz]
        modul[klucz] = wartosc
        oryginal = run_pv.MODUL_KATALOG
        run_pv.MODUL_KATALOG = modul
        run_grid._PAMIEC_PV.clear()
        try:
            wynik = _przebieg_oat(klucz_lok, rok)
        finally:
            run_pv.MODUL_KATALOG = oryginal
            run_grid._PAMIEC_PV.clear()
        wiersze.append(_wiersz(
            parametr, "%s %s" % (fmt.pl(poprzednia, miejsca), jednostka),
            "%s %s" % (fmt.pl(wartosc, miejsca), jednostka), wynik))

    # --- niepewność danych o promieniowaniu --------------------------------
    u_ghi = _wczytaj_niepewnosc_ghi()
    for mnoznik in (1.0 - u_ghi, 1.0 + u_ghi):
        run_grid.MNOZNIK_PROMIENIOWANIA = mnoznik
        try:
            wynik = _przebieg_oat(klucz_lok, rok)
        finally:
            run_grid.MNOZNIK_PROMIENIOWANIA = 1.0
        wiersze.append(_wiersz("mnożnik danych o promieniowaniu", "1,00",
                               fmt.pl(mnoznik, 3), wynik))

    tab = pd.DataFrame(wiersze)

    # --- scenariusze łączne -------------------------------------------------
    laczne = []
    kombinacje = [
        ("założenia o odbiornikach niekorzystne",
         "chłodziarka: stały pobór; laptop 50 W; łącze satelitarne 40 W",
         {"model_chlodziarki": "staly"},
         [("LAPTOP_MOC_W", 50.0), ("STARLINK_MOC_W", 40.0)]),
        ("założenia o odbiornikach korzystne",
         "chłodziarka −20 %; laptop 7 W; łącze satelitarne 25 W", {},
         [("CHLODZIARKA_AH_H_REF", loads.CHLODZIARKA_AH_H_REF * 0.8),
          ("LAPTOP_MOC_W", 7.0), ("STARLINK_MOC_W", 25.0)]),
        ("zabudowa cieplejsza i zacieniona",
         "podbicie temperatury 10 K; zacienienie 10 %",
         {"podbicie_K": 10.0}, []),
    ]
    for nazwa, opis, nadpisania, podmiany in kombinacje:
        poprzednie = [(n, getattr(loads, n)) for n, _ in podmiany]
        for n, wartosc in podmiany:
            setattr(loads, n, wartosc)
        zacienienie = run_pv.STRATA_ZACIENIENIE_PCT
        if nazwa.startswith("zabudowa"):
            run_pv.STRATA_ZACIENIENIE_PCT = 10.0
            run_grid._PAMIEC_PV.clear()
        try:
            wynik = _przebieg_oat(klucz_lok, rok, **nadpisania)
        finally:
            for n, wartosc in poprzednie:
                setattr(loads, n, wartosc)
            run_pv.STRATA_ZACIENIENIE_PCT = zacienienie
            run_grid._PAMIEC_PV.clear()
        laczne.append({"scenariusz": nazwa, "zmienione parametry": opis,
                       **wynik})
    tab_laczne = pd.DataFrame(laczne)

    for ramka in (tab, tab_laczne):
        ramka["zmiana_LPSP_pp"] = ((ramka["LPSP"] - odniesienie) * 100.0).round(3)
        ramka["LPSP"] = ramka["LPSP"].round(5)
        ramka["SSR_PV"] = ramka["SSR_PV"].round(4)
        ramka["E_deficyt_kWh"] = ramka["E_deficyt_kWh"].round(3)
        ramka.insert(0, "lokalizacja", LOCATIONS[klucz_lok].label)
        ramka.insert(1, "rok", rok)
        ramka.insert(2, "scenariusz_bazowy", OAT_SCENARIUSZ)
    return tab, tab_laczne


# ===========================================================================
# 3. Porównanie technologii i przenośność
# ===========================================================================

def porownanie_technologii(wyniki: pd.DataFrame) -> pd.DataFrame:
    """Porównanie modułów w przeliczeniu na powierzchnię i na moc znamionową."""
    wiersze = []
    for nazwa, modul in run_grid.moduly():
        for loc in LOCATIONS.values():
            uzyski = []
            for rok in range(config.SERIES_START_YEAR, config.SERIES_END_YEAR + 1):
                moc_kw, _ = run_grid.generacja(loc, rok, modul)
                uzyski.append(float(moc_kw.sum()))
            mediana = float(np.median(uzyski))
            wiersze.append({
                "modul": nazwa, "lokalizacja": loc.label,
                "p_mp_W": modul["p_mp"],
                "powierzchnia_m2": modul["powierzchnia"],
                "uzysk_mediana_kWh": round(mediana, 1),
                "uzysk_na_powierzchnie_kWh_m2": round(
                    mediana / modul["powierzchnia"], 1),
                "uzysk_wlasciwy_kWh_kWp": round(
                    mediana / (modul["p_mp"] / 1000.0), 1),
                "gestosc_mocy_W_m2": round(modul["p_mp"] / modul["powierzchnia"], 1),
            })
    return pd.DataFrame(wiersze)


def przenosnosc(technologie: pd.DataFrame) -> pd.DataFrame:
    """
    Przeniesienie wyniku na pojazdy o większym dachu.

    Skalowanie wyłącznie powierzchnią pola montażowego, przy niezmienionym
    profilu obciążeń i niezmienionej geometrii montażu. Pasmo +/-15 % obejmuje
    niepewność pola użytkowego dachu, które zależy od wyposażenia egzemplarza.
    """
    bazowa = technologie[technologie.modul == "AIKO 470 Wp"]
    wiersze = []
    for _, r in bazowa.iterrows():
        for pojazd, pole in POJAZDY.items():
            krotnosc = pole / POJAZDY["VW Caddy Maxi (pojazd bazowy)"]
            uzysk = r["uzysk_mediana_kWh"] * krotnosc
            wiersze.append({
                "pojazd": pojazd, "lokalizacja": r["lokalizacja"],
                "pole_montazowe_m2": pole,
                "krotnosc_pola": round(krotnosc, 2),
                "uzysk_mediana_kWh": round(uzysk, 1),
                "uzysk_dolny_kWh": round(uzysk * (1 - PASMO_PRZENOSNOSCI), 1),
                "uzysk_gorny_kWh": round(uzysk * (1 + PASMO_PRZENOSNOSCI), 1),
            })
    return pd.DataFrame(wiersze)


def pojemnosc_projektowa_ah(wymagania: pd.DataFrame) -> int:
    """Najmniejsza pojemnosc LFP spelniajaca W1a-W4 we wszystkich lokalizacjach."""
    lfp = wymagania[(wymagania.modul == OAT_MODUL)
                    & (wymagania.ladowarka == OAT_LADOWARKA)
                    & wymagania.magazyn.str.startswith("LFP")].copy()
    lfp["ah"] = lfp.magazyn.str.extract(r"(\d+)").astype(int)
    zdane = lfp[lfp["spelnia_W1_W4"]].groupby("ah")["lokalizacja"].nunique()
    pelne = [ah for ah, n in zdane.items() if n == lfp.lokalizacja.nunique()]
    if not pelne:
        raise RuntimeError("żadna pojemność LFP nie spełnia W1a-W4")
    return int(min(pelne))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_dirs()
    sciezka = config.OUT_DIR / "grid_results.csv"
    if not sciezka.exists():
        raise SystemExit("Brak %s — uruchom najpierw run_grid.py" % sciezka.name)
    wyniki = pd.read_csv(sciezka)

    print("=" * 78)
    print("Analiza siatki")
    print("=" * 78)

    wymagania = ocena_wymagan(wyniki)
    wymagania.to_csv(config.OUT_DIR / "requirements_check.csv", index=False)
    print("  requirements_check.csv: %d konfiguracji, %d spełnia W1-W4"
          % (len(wymagania), int(wymagania["spelnia_W1_W4"].sum())))

    # Konfiguracja odniesienia analizy wrazliwosci: pojemnosc projektowa.
    global OAT_MAGAZYN_AH
    OAT_MAGAZYN_AH = pojemnosc_projektowa_ah(wymagania)
    print("  konfiguracja odniesienia OAT: LFP %d Ah, scenariusz %s"
          % (OAT_MAGAZYN_AH, OAT_SCENARIUSZ))

    technologie = porownanie_technologii(wyniki)
    technologie.to_csv(config.OUT_DIR / "grid_technologie.csv", index=False)
    print("  grid_technologie.csv: %d wierszy" % len(technologie))

    przen = przenosnosc(technologie)
    przen.to_csv(config.OUT_DIR / "grid_przenosnosc.csv", index=False)
    print("  grid_przenosnosc.csv: %d wierszy" % len(przen))

    oat, laczne = analiza_oat(wyniki)
    oat.to_csv(config.OUT_DIR / "grid_oat.csv", index=False)
    laczne.to_csv(config.OUT_DIR / "grid_oat_laczne.csv", index=False)
    print("  grid_oat.csv: %d wierszy (odniesienie: %s, rok %d)"
          % (len(oat), oat["lokalizacja"].iloc[0], int(oat["rok"].iloc[0])))
    print("  grid_oat_laczne.csv: %d scenariuszy" % len(laczne))

    print()
    print("Parametry o największym wpływie na energetyczny wskaźnik LPSP:")
    naj = oat.reindex(oat["zmiana_LPSP_pp"].abs().sort_values(ascending=False).index)
    print(naj.head(6)[["parametr", "wartosc_zmieniona", "LPSP",
                       "zmiana_LPSP_pp"]].to_string(index=False))

    manifest.zapisz(
        "analiza_siatki",
        skrypt="simulation/grid_analiza.py",
        konfiguracja={
            "oat_lokalizacja": str(oat["lokalizacja"].iloc[0]),
            "oat_rok": int(oat["rok"].iloc[0]),
            "oat_magazyn_Ah": OAT_MAGAZYN_AH, "oat_scenariusz": OAT_SCENARIUSZ,
            "oat_ladowarka": OAT_LADOWARKA, "oat_modul": OAT_MODUL,
            "oat_mnoznik_ghi_u": OAT_MNOZNIK_GHI_U,
            "pojazdy_przenosnosc": POJAZDY,
            "pasmo_przenosnosci": PASMO_PRZENOSNOSCI,
        },
        wejscia={"grid_results": sciezka},
        wyniki=[config.OUT_DIR / n for n in
                ("requirements_check.csv", "grid_technologie.csv",
                 "grid_przenosnosc.csv", "grid_oat.csv",
                 "grid_oat_laczne.csv")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
