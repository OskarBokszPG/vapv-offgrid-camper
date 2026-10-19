# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: run_grid.py
Opis: siatka symulacji: lokalizacje x lata x moduły x magazyny x
      scenariusze.

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
import copy
import itertools
import sys
import time

import numpy as np
import pandas as pd

import battery_balance as bb
import config
import loads
import make_sam_resource as resource
import manifest
import run_pv
from config import LOCATIONS, Location

PRZEBIEG = "siatka_symulacji"

# --- magazyny --------------------------------------------------------------
# Napięcie znamionowe magazynu litowego, V. Pojemności w amperogodzinach
# przeliczane są na energię tym napięciem.
U_MAGAZYNU_V = 12.8
POJEMNOSCI_AH = (50, 100, 150, 200, 300)

# Stan zastany: Banner AGM 92 Ah. Użyteczna głębokość rozładowania przyjęta
# 50 % (zakres analizy wrażliwości 40-60 %), czyli okno 25-75 %.
AGM_AH = 92
# Napiecie znamionowe ogniwa kwasowo-olowiowego to 2 V, wiec akumulator
# szesciokomorowy ma 12 V. Magazyn LiFePO4 ma 12,8 V (4 x 3,2 V) i energia
# obu liczona jest z ich wlasnych napiec znamionowych.
AGM_NAPIECIE_V = 12.0
AGM_SOC = (0.25, 0.75)
AGM_SPRAWNOSC_OBIEGU = 0.80     # przyjęta; AGM ma niższą sprawność niż LFP

# --- regulator ładowania (B6) ---------------------------------------------
# SmartSolar MPPT 100/30: prąd ładowania 30 A, nominalna moc PV przy
# magistrali 12 V równa 440 W (instrukcja s. 65). Moduł 470 Wp przekracza tę
# moc o 7 %, więc regulator obcina moc wyjściową przy najwyższym natężeniu
# promieniowania. Ograniczenie zadane jest prądem wyjściowym przy napięciu
# podtrzymania 13,8 V z karty, a energia obcięta raportowana osobno.
MPPT_PRAD_WYJSCIOWY_A = 30.0
MPPT_NAPIECIE_PODTRZYMANIA_V = 13.8
MPPT_MOC_WYJSCIOWA_W = MPPT_PRAD_WYJSCIOWY_A * MPPT_NAPIECIE_PODTRZYMANIA_V

# --- ładowarki stałoprądowe ------------------------------------------------
# Moc ładowania przyjęta jako iloczyn prądu wyjściowego i napięcia magazynu.
# Orion-Tr 12/12-18: 18 A; Orion XS 12/12-50: 50 A.
# Prąd wyjściowy ładowarki, A. Orion XS ma nastawny zakres prądu wejściowego
# i wyjściowego 1-50 A (karta s. 2), więc wariant z ograniczeniem 30 A na
# wejściu daje się zrealizować nastawą, bez zmiany sprzętu. Wariant ten liczy
# się wyłącznie dla konfiguracji zalecanej, nie dla całej siatki.
LADOWARKI = {
    "Orion-Tr 18": 18.0,
    "Orion XS 50": 50.0,
}
LADOWARKA_OGRANICZONA = ("Orion XS 50 (30 A na wejściu)", 30.0)
# Prąd alternatora ogranicza dopływ wyłącznie wtedy, gdy ładowarka mogłaby
# pobrać więcej, niż alternator oddaje po pokryciu odbiorów pokładowych.
# Przyjęto, że na ładowanie magazynu bytowego dostępne jest 25 % prądu
# znamionowego alternatora (wartość PRZYJĘTA — reszta zasila odbiory pojazdu).
UDZIAL_ALTERNATORA = 0.25

# --- profile jazdy (rozdz. 4.3) -------------------------------------------
# Godziny jazdy w czasie lokalnym, zadane jako funkcja indeksu dobowego.
PROFILE_JAZDY = {
    "A": {"opis": "dojazd 3 h w dniu pierwszym, potem 0,5 h co 3 dni"},
    "B": {"opis": "2 h w czwartek wieczorem i 2 h w niedzielę po południu"},
    "C": {"opis": "t_jazdy,min wyznaczane bisekcją"},
}

SCENARIUSZE = ("A-DC", "A-AC", "B", "C")

# --- wymagania projektowe (4.6) -------------------------------------------
# Okres rozbiegowy wyłączony z oceny: pierwszy miesiąc roku symulacyjnego.
# Powód: stan naładowania w pierwszej godzinie jest założeniem, a przy
# starcie od dolnej granicy okna wskaźnik samowystarczalności różnił się
# o blisko punkt procentowy od wariantu ze startem od granicy górnej.
ROZBIEG_GODZIN = 744

W1_PROG_LPSP = 0.05
W1_MIESIACE = tuple(range(3, 11))
W2_PROG_UDZIAL = 0.95
W3_PROG_H = 1.0
W4_PROG_KG = 100.0


def pojemnosc_kwh(ah: float, napiecie: float = U_MAGAZYNU_V) -> float:
    return ah * napiecie / 1000.0


# ===========================================================================
# 1. Generacja PV — liczona raz na (lokalizacja, rok, moduł)
# ===========================================================================

_PAMIEC_PV: dict[tuple, tuple[np.ndarray, pd.DatetimeIndex]] = {}

# Mnożnik danych o promieniowaniu. Wartość różna od jedności skaluje wszystkie
# składowe promieniowania w pliku zasobu i służy wyłącznie analizie
# wrażliwości na niepewność danych wejściowych (rozdz. 6).
MNOZNIK_PROMIENIOWANIA = 1.0
# Energia obcięta przez ograniczenie prądu wyjściowego regulatora, kWh/rok.
_OBCIECIE_MPPT: dict[tuple[str, int, str], float] = {}


def plik_zasobu_skalowany(plik, mnoznik: float):
    """Kopia pliku zasobu z przeskalowanymi składowymi promieniowania.

    Skalowanie jest jednakowe dla promieniowania całkowitego, bezpośredniego
    i rozproszonego, więc nie zmienia podziału na składowe, a wyłącznie
    poziom natężenia.
    """
    import shutil

    katalog = config.OUT_DIR / "zasob_skalowany"
    katalog.mkdir(parents=True, exist_ok=True)
    cel = katalog / ("%s_x%s.csv" % (plik.stem, ("%.4f" % mnoznik).replace(".", "p")))
    if cel.exists():
        return cel
    wiersze = plik.read_text(encoding="utf-8").splitlines()
    naglowek = wiersze[:2]
    blok = pd.read_csv(plik, skiprows=2)
    for kolumna in ("GHI", "DNI", "DHI"):
        if kolumna in blok.columns:
            blok[kolumna] = (blok[kolumna] * mnoznik).round(3)
    with cel.open("w", encoding="utf-8", newline="\n") as plik_wy:
        plik_wy.write("\n".join(naglowek) + "\n")
        blok.to_csv(plik_wy, index=False, lineterminator="\n")
    del shutil
    return cel


def generacja(loc: Location, rok: int, modul: dict) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Godzinowy szereg mocy stałoprądowej generatora, kW, oraz indeks UTC."""
    klucz = (loc.key, rok, modul["nazwa"], round(MNOZNIK_PROMIENIOWANIA, 4))
    if klucz in _PAMIEC_PV:
        return _PAMIEC_PV[klucz]

    plik = resource.resource_path(loc, rok)
    if abs(MNOZNIK_PROMIENIOWANIA - 1.0) > 1e-9:
        plik = plik_zasobu_skalowany(plik, MNOZNIK_PROMIENIOWANIA)
    oryginal = run_pv.MODUL_KATALOG
    run_pv.MODUL_KATALOG = modul
    try:
        model = resource._run_model_for_resource(plik)
    finally:
        run_pv.MODUL_KATALOG = oryginal

    out = model.export()["Outputs"]
    # Zacienienie wchodzi jako stały uszczerbek mocy stałoprądowej (patrz
    # komentarz przy run_pv.STRATA_ZACIENIENIE_PCT), a nie jako geometria.
    moc_kw = (np.asarray(out["dc_net"], dtype=float)
              * (1.0 - run_pv.STRATA_ZACIENIENIE_PCT / 100.0))
    # Obcięcie mocą wyjściową regulatora (B6). Nadwyżka nie wchodzi do
    # bilansu i jest raportowana jako energia obcięta przez regulator.
    # Regulator ogranicza prad wyjsciowy, wiec progiem dla mocy
    # stalopradowej jest iloczyn prądu znamionowego i napięcia podtrzymania
    # podzielony przez sprawność regulatora: moc wejściowa przechodzi na
    # stronę wyjściową przez tę sprawność.
    limit_kw = (MPPT_MOC_WYJSCIOWA_W / 1000.0
                / bb.ParametryUkladu().sprawnosc_mppt)
    obciete_kwh = float(np.maximum(moc_kw - limit_kw, 0.0).sum())
    moc_kw = np.minimum(moc_kw, limit_kw)
    _OBCIECIE_MPPT[klucz] = round(obciete_kwh, 3)
    blok = pd.read_csv(plik, skiprows=2)
    znaczniki = pd.to_datetime(blok[["Year", "Month", "Day", "Hour", "Minute"]])
    indeks = pd.DatetimeIndex(znaczniki).tz_localize("UTC")
    if moc_kw.size != len(indeks):
        raise ValueError("Szereg generacji (%d) i indeks (%d) różnią się długością."
                         % (moc_kw.size, len(indeks)))
    _PAMIEC_PV[klucz] = (moc_kw, indeks)
    return _PAMIEC_PV[klucz]


# ===========================================================================
# 2. Obciążenia, jazda i temperatura magazynu
# ===========================================================================

def profil_scenariusza(loc: Location, rok: int, scenariusz: str,
                       model_chlodziarki: str = loads.MODEL_CHLODZIARKI,
                       podbicie_K: float = loads.KABINA_PODBICIE_BAZOWE_K
                       ) -> pd.DataFrame:
    """Profil obciążeń dla scenariusza siatki (A-DC i A-AC to warianty A)."""
    bazowy = "A" if scenariusz.startswith("A") else scenariusz
    wariant = "DC" if scenariusz == "A-DC" else "AC"
    poprzedni = loads.MODEL_CHLODZIARKI
    loads.MODEL_CHLODZIARKI = model_chlodziarki
    try:
        return loads.zbuduj_profil(loc, rok, bazowy, wariant_laptopa=wariant,
                                   podbicie_K=podbicie_K)
    finally:
        loads.MODEL_CHLODZIARKI = poprzedni


def maska_jazdy(indeks_lokalny: pd.DatetimeIndex, scenariusz: str,
                godzin_na_dobe: float = 0.0) -> np.ndarray:
    """
    Udział godziny przeznaczony na jazdę, w zakresie 0-1.

    Profile odpowiadają opisom z podrozdziału 4.3. Dla scenariusza C czas
    jazdy jest wielkością wynikową i zadaje się go argumentem.
    """
    godzina = indeks_lokalny.hour.to_numpy()
    dzien_roku = indeks_lokalny.dayofyear.to_numpy()
    dzien_tygodnia = indeks_lokalny.dayofweek.to_numpy()
    udzial = np.zeros(len(indeks_lokalny), dtype=float)

    bazowy = scenariusz[0]
    if bazowy == "A":
        # Dojazd 3 h pierwszego dnia okresu, potem 0,5 h co trzy dni.
        pierwszy = dzien_roku == int(dzien_roku.min())
        udzial[pierwszy & np.isin(godzina, (9, 10, 11))] = 1.0
        co_trzy = (dzien_roku % 3 == 0) & ~pierwszy
        udzial[co_trzy & (godzina == 11)] = 0.5
    elif bazowy == "B":
        # Czwartek 18:00-20:00 (wyjazd) i niedziela 16:00-18:00 (powrót).
        udzial[(dzien_tygodnia == 3) & np.isin(godzina, (18, 19))] = 1.0
        udzial[(dzien_tygodnia == 6) & np.isin(godzina, (16, 17))] = 1.0
    else:
        # Scenariusz C: zadany dzienny czas jazdy rozłożony na godziny 10-12.
        if godzin_na_dobe > 0.0:
            pelne = int(godzin_na_dobe)
            reszta = godzin_na_dobe - pelne
            for i in range(pelne):
                udzial[godzina == (10 + i) % 24] = 1.0
            if reszta > 0.0:
                udzial[godzina == (10 + pelne) % 24] = reszta
    return udzial


def doplyw_alternatora(udzial_jazdy: np.ndarray, ladowarka: str,
                       prad_alternatora_A: float,
                       dostepnosc: float = 1.0) -> np.ndarray:
    """
    Energia oddawana przez ładowarkę stałoprądową w godzinie, kWh.

    Moc ładowarki ograniczona jest prądem dostępnym z alternatora: przyjęto,
    że na ładowanie magazynu bytowego przypada ``UDZIAL_ALTERNATORA`` prądu
    znamionowego. ``dostepnosc`` odwzorowuje wariant, w którym alternator
    sterowany nie pracuje przez cały czas jazdy.
    """
    moc_ladowarki_w = LADOWARKI[ladowarka] * U_MAGAZYNU_V
    moc_alternatora_w = prad_alternatora_A * UDZIAL_ALTERNATORA * U_MAGAZYNU_V
    moc_w = min(moc_ladowarki_w, moc_alternatora_w)
    return udzial_jazdy * dostepnosc * moc_w / 1000.0


def temperatura_magazynu(loc: Location, rok: int, profil: pd.DataFrame,
                         parametry: bb.ParametryUkladu,
                         udzial_jazdy: np.ndarray | None = None) -> np.ndarray:
    """
    Temperatura ogniw z temperatury powietrza, ogrzewania i godzin jazdy.

    Magazyn stoi we wnęce pod siedzeniem kierowcy, czyli wewnątrz
    przestrzeni ogrzewanej. Podnoszą ją dwa zjawiska: ogrzewanie postojowe
    (scenariusz C) oraz praca silnika podczas jazdy. Pominięcie drugiego
    z nich daje wynik jakościowo błędny w scenariuszu zimowym: ogniwa
    zostają poniżej progu blokady, więc ładowanie z alternatora nie dochodzi
    do skutku niezależnie od czasu jazdy, a najmniejszy dzienny czas jazdy
    wychodzi nieosiągalny przy dowolnej jego wartości.
    """
    pogoda = loads.wczytaj_pogode(loc, rok)
    ogrzewanie = profil["ogrzewanie postojowe"].to_numpy(float) > 0.0
    if udzial_jazdy is not None:
        ogrzewanie = ogrzewanie | (np.asarray(udzial_jazdy, dtype=float) > 0.0)
    return bb.temperatura_ogniw(pogoda["t2m"].to_numpy(float), ogrzewanie,
                                parametry)


# ===========================================================================
# 3. Pojedynczy przebieg
# ===========================================================================

def przebieg(loc: Location, rok: int, modul: dict, parametry: bb.ParametryUkladu,
             scenariusz: str, ladowarka: str, prad_alternatora_A: float,
             dostepnosc_alternatora: float = 1.0,
             godzin_jazdy_C: float = 0.0,
             model_chlodziarki: str = loads.MODEL_CHLODZIARKI,
             podbicie_K: float = loads.KABINA_PODBICIE_BAZOWE_K,
             ) -> tuple[pd.DataFrame, dict[str, float]]:
    """Wykonuje jeden bilans roczny i zwraca (przebiegi, wskaźniki)."""
    moc_kw, indeks = generacja(loc, rok, modul)
    profil = profil_scenariusza(loc, rok, scenariusz, model_chlodziarki, podbicie_K)
    bb.sprawdz_zgodnosc_indeksu(indeks, profil.index)

    lokalny = profil.index.tz_convert(config.LOCAL_TZ)
    udzial = maska_jazdy(lokalny, scenariusz, godzin_jazdy_C)
    alternator = doplyw_alternatora(udzial, ladowarka, prad_alternatora_A,
                                    dostepnosc_alternatora)
    t_ogniw = temperatura_magazynu(loc, rok, profil, parametry, udzial)

    # W godzinach jazdy pobór własny ładowarki jest wyższy; różnicę dodaje się
    # do obciążenia szyny stałoprądowej.
    dodatek_w = loads.pobory_wlasne_w(jazda=True) - loads.pobory_wlasne_w()
    obc_dc = profil["DC"].to_numpy(float) / 1000.0 + udzial * dodatek_w / 1000.0
    obc_ac = profil["AC"].to_numpy(float) / 1000.0
    jalowy = profil["pobór falownika z szyny DC"].to_numpy(float) / 1000.0
    jalowy = np.maximum(jalowy - obc_ac, 0.0)

    return bb.bilans_godzinowy(
        moc_kw, parametry, indeks=indeks,
        obciazenie_dc_kwh=obc_dc,
        obciazenie_ac_kwh=obc_ac if obc_ac.sum() > 0 else None,
        falownik_jalowy_kwh=jalowy if obc_ac.sum() > 0 else None,
        doplyw_alternatora_kwh=alternator,
        temperatura_ogniw_C=t_ogniw,
        rozbieg_godzin=ROZBIEG_GODZIN)


def czas_jazdy_min(loc: Location, rok: int, modul: dict,
                   parametry: bb.ParametryUkladu, ladowarka: str,
                   prad_alternatora_A: float, tolerancja: float = 0.05
                   ) -> float:
    """
    Najmniejszy dzienny czas jazdy utrzymujący stan naładowania (scenariusz C).

    Wyznaczany bisekcją na przedziale 0-8 h. Kryterium: zerowy deficyt
    energetyczny w całym okresie obowiązywania scenariusza.
    """
    def deficyt(godzin: float) -> float:
        _, w = przebieg(loc, rok, modul, parametry, "C", ladowarka,
                        prad_alternatora_A, godzin_jazdy_C=godzin)
        return w["E_deficyt_kWh"]

    if deficyt(0.0) <= 1e-9:
        return 0.0
    gora = 8.0
    if deficyt(gora) > 1e-9:
        return float("nan")      # nawet 8 h jazdy nie wystarcza
    dol = 0.0
    while gora - dol > tolerancja:
        srodek = (dol + gora) / 2.0
        if deficyt(srodek) > 1e-9:
            dol = srodek
        else:
            gora = srodek
    return round(gora, 2)


# ===========================================================================
# 4. Siatka
# ===========================================================================

def warianty_magazynu() -> list[tuple[str, bb.ParametryUkladu]]:
    """Lista wariantów magazynu: LFP w pięciu pojemnościach oraz AGM."""
    lista = [("LFP %d Ah" % ah,
              bb.ParametryUkladu(pojemnosc_kwh=pojemnosc_kwh(ah)))
             for ah in POJEMNOSCI_AH]
    lista.append(("AGM %d Ah" % AGM_AH, bb.ParametryUkladu(
        pojemnosc_kwh=pojemnosc_kwh(AGM_AH, AGM_NAPIECIE_V),
        soc_min_wzgl=AGM_SOC[0], soc_max_wzgl=AGM_SOC[1],
        sprawnosc_obiegu=AGM_SPRAWNOSC_OBIEGU,
        samorozladowanie_pct_mies=3.0)))
    return lista


def moduly() -> list[tuple[str, dict]]:
    return [("AIKO 470 Wp", run_pv.MODUL_KATALOG),
            ("Victron 285 W", run_pv.MODUL_POROWNAWCZY)]


def siatka(lata: range, bez_oat: bool = False) -> dict[str, pd.DataFrame]:
    """Wykonuje całą siatkę i zwraca zestaw tabel wynikowych."""
    wiersze: list[dict[str, object]] = []
    miesieczne: list[dict[str, object]] = []
    magazyny = warianty_magazynu()
    start = time.monotonic()
    liczba = 0

    for loc, rok, (nazwa_modulu, modul), (nazwa_mag, par), scen, ladowarka in \
            itertools.product(LOCATIONS.values(), lata, moduly(), magazyny,
                              SCENARIUSZE, LADOWARKI):
        # Wariant alternatora zmienia wynik tylko tam, gdzie jazda występuje;
        # dla uproszczenia siatki prąd alternatora bierzemy bazowy, a wariant
        # 180 A wchodzi do analizy wrażliwości.
        godzin_C = 0.0
        if scen == "C":
            godzin_C = czas_jazdy_min(loc, rok, modul, par, ladowarka,
                                      config.ALTERNATOR_PRAD_A)
        _, w = przebieg(loc, rok, modul, par, scen, ladowarka,
                        config.ALTERNATOR_PRAD_A,
                        godzin_jazdy_C=0.0 if np.isnan(godzin_C) else godzin_C)
        liczba += 1
        wiersze.append({
            "lokalizacja": loc.label, "rok": rok, "modul": nazwa_modulu,
            "magazyn": nazwa_mag, "scenariusz": scen, "ladowarka": ladowarka,
            "pojemnosc_kWh": round(par.pojemnosc_kwh, 3),
            "SSR_PV": round(w["SSR_PV"], 4),
            "LPSP": round(w["LPSP"], 5),
            "E_obciazenie_kWh": round(w["E_obciazenie_kWh"], 2),
            "E_deficyt_kWh": round(w["E_deficyt_kWh"], 3),
            "E_curt_kWh": round(w["E_curt_kWh"], 2),
            "E_alternator_kWh": round(w["E_alternator_kWh"], 3),
            "E_blokada_pv_kWh": round(w["E_blokada_pv_kWh"], 3),
            "E_blokada_alternator_kWh": round(w["E_blokada_alternator_kWh"], 3),
            "godziny_deficytu": w["godziny_deficytu"],
            "godziny_blokady": w.get("godziny_blokady", 0),
            "E_obciete_mppt_kWh": _OBCIECIE_MPPT.get(
                (loc.key, rok, nazwa_modulu,
                 round(MNOZNIK_PROMIENIOWANIA, 4)), 0.0),
            "autonomia_d": round(bb.autonomia_doby(par, w["srednie_dobowe_kWh"]), 2),
            "t_jazdy_min_h": godzin_C if scen == "C" else np.nan,
            "SoC_min": round(w["SoC_min_wzgledny"], 4),
        })

        # Wskaźniki miesięczne tylko dla konfiguracji odniesienia.
        if (nazwa_modulu == "AIKO 470 Wp" and ladowarka == "Orion-Tr 18"
                and nazwa_mag == "LFP 100 Ah"):
            przebiegi, _ = przebieg(loc, rok, modul, par, scen, ladowarka,
                                    config.ALTERNATOR_PRAD_A,
                                    godzin_jazdy_C=0.0 if np.isnan(godzin_C) else godzin_C)
            tab = bb.wskazniki_miesieczne(przebiegi, par)
            tab.insert(0, "scenariusz", scen)
            tab.insert(0, "rok", rok)
            tab.insert(0, "lokalizacja", loc.label)
            miesieczne.append(tab)

        if liczba % 50 == 0:
            print("  przebiegów: %d, czas %.0f s" % (liczba, time.monotonic() - start))

    print("  razem przebiegów: %d w %.0f s" % (liczba, time.monotonic() - start))
    return {
        "grid_results": pd.DataFrame(wiersze),
        "grid_monthly": (pd.concat(miesieczne, ignore_index=True)
                         if miesieczne else pd.DataFrame()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lata", nargs=2, type=int,
                        default=[config.SERIES_START_YEAR, config.SERIES_END_YEAR])
    parser.add_argument("--bez-oat", action="store_true")
    argumenty = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_dirs()

    lata = range(argumenty.lata[0], argumenty.lata[1] + 1)
    print("=" * 78)
    print("Siatka symulacji — lata %d-%d" % (lata.start, lata.stop - 1))
    print("=" * 78)

    tabele = siatka(lata, bez_oat=argumenty.bez_oat)
    for nazwa, tab in tabele.items():
        if tab.empty:
            continue
        cel = config.OUT_DIR / (nazwa + ".csv")
        tab.to_csv(cel, index=False)
        print("  zapisano %s (%d wierszy)" % (cel.name, len(tab)))

    manifest.zapisz(
        PRZEBIEG,
        skrypt="simulation/run_grid.py",
        konfiguracja={
            "lata": [lata.start, lata.stop - 1],
            "lokalizacje": [l.label for l in LOCATIONS.values()],
            "moduly": [n for n, _ in moduly()],
            "magazyny_Ah": list(POJEMNOSCI_AH) + [AGM_AH],
            "ladowarki": list(LADOWARKI),
            "alternator_A": config.ALTERNATOR_PRAD_A,
            "udzial_alternatora": UDZIAL_ALTERNATORA,
            "scenariusze": list(SCENARIUSZE),
            "profile_jazdy": {k: v["opis"] for k, v in PROFILE_JAZDY.items()},
        },
        wejscia={loc.key: resource.resource_path(loc, lata.start)
                 for loc in LOCATIONS.values()},
        wyniki=[config.OUT_DIR / (n + ".csv") for n in tabele if not tabele[n].empty],
    )
    print("manifest:", manifest.sciezka_manifestu(PRZEBIEG).name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
