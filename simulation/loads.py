# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: loads.py
Opis: profile obciążeń bytowych, krok godzinowy. Profile buduje się od
      dołu, z danych katalogowych pojedynczych odbiorników, a nie z
      ryczałtu dobowego. Każdy odbiór ma przypisaną szynę: ``DC`` dla
      odbiorników zasilanych wprost z magistrali 12 V i ``AC`` dla
      odbiorników za falownikiem. Rozdział szyn jest potrzebny, bo
      sprawność falownika obciąża wyłącznie odbiory prądu przemiennego
      (patrz ``battery_balance``).

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
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
import manifest
from config import LOCATIONS, Location

# ---------------------------------------------------------------------------
# Stałe modelu
# ---------------------------------------------------------------------------
# --- chłodziarka Dometic CFX3 45 (katalogowa) ---
# Karta podaje jeden punkt pracy: 1,03 Ah/h przy 12 V DC, temperaturze
# otoczenia 32 stC i temperaturze wnętrza 4 stC. Drugiego punktu nie ma, więc
# prostej dopasować się nie da — model jest jednopunktowy.
CHLODZIARKA_AH_H_REF = 1.03
CHLODZIARKA_U_NOM_V = 12.0        # napięcie odniesienia z karty
CHLODZIARKA_T_OTOCZ_REF_C = 32.0
CHLODZIARKA_T_WNETRZA_C = 4.0
CHLODZIARKA_OAT = (-0.20, 0.20)   # +/- 20 % energii dobowej
# Postać modelu jest osobnym wariantem analizy wrażliwości, nie parametrem:
#   "liniowy" — pobór proporcjonalny do różnicy temperatur, obcięty do zera,
#   "staly"   — pobór równy punktowi katalogowemu niezależnie od temperatury.
# Wariant stały jest GÓRNYM oszacowaniem: zakłada pracę sprężarki w takim
# samym rytmie jak przy 32 stC otoczenia. Wariant liniowy jest DOLNYM:
# przy temperaturze otoczenia bliskiej nastawie wnętrza schodzi do zera,
# a instrukcja chłodziarki nie podaje poboru układu sterowania w czuwaniu
# (sprawdzone w instrukcji i w karcie produktu — wartości nie ma).
MODEL_CHLODZIARKI = "liniowy"
MODELE_CHLODZIARKI = ("liniowy", "staly")
# Producent deklaruje pracę od 16 stC otoczenia; poniżej model jest
# ekstrapolowany. Ograniczenie opisane w 4.4.
CHLODZIARKA_T_MIN_KARTA_C = 16.0

# --- temperatura kabiny (przyjęta) ---
# Wariant bazowy przyjmuje temperaturę w zabudowie równą temperaturze powietrza
# na wysokości 2 m, czyli podbicie zerowe. Analiza wrażliwości obejmuje
# podbicie 0, +5 i +10 K w godzinach ze słońcem.
#
# BRAK ŹRÓDŁA. Dla zabudowy kamperowej nie odnaleziono zweryfikowanego pomiaru
# przyrostu temperatury wnętrza nad temperaturę powietrza. Publikacje dotyczące
# nagrzewania wnętrz samochodowych opisują kabiny osobowe z dużą powierzchnią
# przeszkleń, więc ich wartości nie przenoszą się na zabudowę z małymi okienkami
# i izolacją. Cały zakres należy do klasy wartości PRZYJĘTYCH przez autora.
KABINA_PODBICIE_BAZOWE_K = 0.0
KABINA_PODBICIE_OAT_K = (0.0, 5.0, 10.0)
KABINA_PROG_SLONCA_W_M2 = 120.0     # próg uznania godziny za słoneczną
KABINA_NASTAWA_WEBASTO_C = 18.0

# --- Starlink Mini (katalogowa: zakres 25-40 W; przyjęta: wartość) ---
STARLINK_MOC_W = 30.0
STARLINK_OAT_W = (25.0, 40.0)
STARLINK_GODZINY = (9, 17)          # czas lokalny, [od, do)

# --- laptop HP Pavilion Gaming 15-ec1000 ---
# Wartość przyjęta (30 W po stronie prądu przemiennego). Weryfikację wykona
# autor: energia z raportu baterii (powercfg /batteryreport) podczas pracy na
# baterii, przeliczona na stronę przemienną przez przyjętą sprawność zasilacza.
# Wynik takiego pomiaru będzie dolnym oszacowaniem, ponieważ laptop na baterii
# ogranicza wydajność układu.
# Moc laptopa odniesiona do jego wejscia stalopradowego, czyli do
# zaciskow odbiornika. Ta sama wartosc obowiazuje w obu wariantach
# zasilania, wiec porownanie dotyczy tej samej uslugi energetycznej:
# w wariancie AC dochodzi zasilacz sieciowy i falownik, w wariancie DC
# przetwornica 12 -> 19,5 V. Straty obu torow sa osobnymi pozycjami.
LAPTOP_MOC_W = 30.0
# Dolna granica zakresu analizy wrażliwości jest OBLICZONA z danych
# producenta: pojemność baterii 52,5 Wh i deklarowany czas pracy do
# 8 h 15 min dają ok. 6,4 W po stronie baterii, co po przeliczeniu przez
# przyjętą sprawność zasilacza 0,9 daje ok. 7 W po stronie przemiennej.
# Jest to granica dolna w ścisłym sensie: komputer zasilany z baterii
# ogranicza wydajność układu. Górna granica odpowiada pracy obciążonej.
LAPTOP_OAT_W = (7.0, 50.0)
LAPTOP_GODZINY = (9, 17)
# Wariant stałoprądowy: przetwornica 12 -> 19,5 V. Sprawność przyjęta,
# bez zweryfikowanego źródła.
LAPTOP_DC_SPRAWNOSC = 0.90        # przetwornica 12 -> 19,5 V
LAPTOP_AC_SPRAWNOSC = 0.90        # zasilacz sieciowy laptopa
LAPTOP_DC_OAT = (0.85, 0.93)

# --- falownik Xantrex 1800i ---
# Model strat dopasowany do krzywej sprawności z instrukcji (odczyt z rysunku,
# niepewność +/- 2 pkt proc.). P_strat = P0 + a*P + b*P^2.
# P0 = 22 W to wartość MAKSYMALNA poboru jałowego z karty — założenie
# zachowawcze. Zapis P_in = P0 + P_out/eta liczyłby P0 dwa razy, bo krzywa
# sprawności zawiera już straty jałowe.
FALOWNIK_P0_W = 22.0
FALOWNIK_A = 0.057019
FALOWNIK_B = 3.920e-5
FALOWNIK_ZAKRES_POMIAROWY_W = (223.0, 1800.0)

# Stany pracy falownika i pobór w każdym z nich, W. Model musi je rozróżniać,
# bo pobór jałowy jest duży w porównaniu z obciążeniem: 22 W wobec 30 W
# pobieranych przez laptop.
#   wyłączony   — rozłączony wyłącznikiem, pobór zerowy,
#   wyszukiwanie— tryb wyszukiwania obciążenia (karta: < 1,5 W),
#   praca       — załączony; pobór to P_0 plus straty zależne od obciążenia.
FALOWNIK_STANY_W = {"wyłączony": 0.0, "wyszukiwanie": 1.5, "praca": FALOWNIK_P0_W}

# Harmonogram stanów w scenariuszach. Falownik jest załączany wyłącznie na
# czas pracy odbiornika prądu przemiennego; poza tym użytkownik wyłącza go
# wyłącznikiem. Wariant analizy wrażliwości: pozostawienie falownika w trybie
# wyszukiwania przez całą dobę (FALOWNIK_POZA_PRACA = "wyszukiwanie").
FALOWNIK_GODZINY_PRACY = {"A": LAPTOP_GODZINY, "B": None, "C": None}
FALOWNIK_POZA_PRACA = "wyłączony"

# --- pobory własne podzespołów (katalogowa) ---
# Regulator, ładowarka DC-DC, układ zarządzania magazynem i bocznik pomiarowy
# pobierają prąd nieprzerwanie, niezależnie od użytkowania pojazdu. Pozycja
# wchodzi do bilansu przez cały czas trwania scenariusza, także w dobach, gdy
# pojazd stoi nieużywany.
#
# Napięcie odniesienia 12,8 V to napięcie znamionowe magazynu LiFePO4
# (4 ogniwa po 3,2 V) — pobory podane są w kartach jako prąd, nie jako moc.
POBORY_WLASNE_U_V = 12.8

# Pobór zależy od STANU POJAZDU. Ładowarka stałoprądowa pobiera prąd jałowy
# tylko wtedy, gdy jest załączona, czyli w czasie jazdy; po wykryciu postoju
# (albo po rozwarciu wejścia zdalnego) przechodzi w czuwanie o poborze
# pomijalnym. Regulator ładowania pobiera prąd nieprzerwanie.
#
# Orion-Tr Smart 12/12-18 (instrukcja s. 29): prąd jałowy < 80 mA, prąd
# czuwania < 1 mA. Wariant Orion XS 12/12-50 (karta s. 2): < 100 mA i < 1,5 mA
# — zmiana wariantu podnosi obie pozycje o kilkanaście procent.
POBORY_WLASNE_POSTOJ_MA = {
    "regulator MPPT": 30.0,            # instrukcja MPPT s. 65
    "ładowarka DC-DC (czuwanie)": 1.0,  # instrukcja Orion-Tr s. 29
}
POBORY_WLASNE_JAZDA_MA = {
    "regulator MPPT": 30.0,
    "ładowarka DC-DC (bieg jałowy)": 80.0,
}
# Pobór układu zarządzania magazynem i bocznika pomiarowego: kart tych
# elementów nie ma w katalogu kart, a dobór należy do rozdz. 5. Dopóki stoi
# tu None, pozycja NIE jest podstawiana wartością domyślną — suma poborów
# własnych jest wtedy dolnym oszacowaniem i tak jest opisana w 4.4.
POBORY_WLASNE_DO_UZUPELNIENIA = ("BMS", "bocznik pomiarowy")
POBORY_WLASNE_BMS_MA = None         # DO UZUPEŁNIENIA: karta układu BMS
POBORY_WLASNE_BOCZNIK_MA = None     # DO UZUPEŁNIENIA: karta bocznika

# Samorozładowanie magazynu nie jest odbiorem na szynie i nie wchodzi do tego
# modułu — jest ubytkiem energii zmagazynowanej, liczonym w battery_balance
# (karta ogniwa LF280K, s. 5: <= 3 %/miesiąc).


# --- oświetlenie LED i ładowanie drobnej elektroniki (przyjęta) ---
LED_USB_WH_D = 70.0
LED_USB_OAT_WH_D = (50.0, 100.0)
LED_USB_GODZINY = (18, 23)

# --- ogrzewanie postojowe Webasto Air Top 2000 STC ---
# Karta podaje 13-30 W w zakresie roboczym; przyjęto 20 W.
WEBASTO_MOC_W = 20.0
# Zakres regulacji poboru mocy: 14-29 W według instrukcji warsztatowej
# (s. 19, „Rated power consumption, control range"). Karta katalogowa podaje
# 13-30 W dla tej samej wielkości (s. 2); przyjęto zakres z instrukcji
# warsztatowej jako dokumentu szczegółowego. Wartość bazowa 20 W leży
# w obu, więc rozstrzygnięcie nie zmienia wyniku bazowego.
WEBASTO_OAT_W = (14.0, 29.0)
WEBASTO_GODZINY = (22, 8)           # przez północ

# Energia rozruchu. Instrukcja warsztatowa Air Top 2000 STC (s. 14) opisuje
# przebieg: świeca żarowa załącza się przy starcie i pracuje impulsowo
# („clocked"), pompa paliwa rusza po ok. 40 s, a świeca wyłącza się po
# kolejnych 110 s dla oleju napędowego — faza rozruchu trwa ok. 150 s.
# Rezystancja świecy w 25 stC: 0,263-0,323 om (s. 20). Napięcia świecy
# sterownik NIE reguluje: świeca jest przewidziana na 12 V (s. 18), a dane
# techniczne obowiązują przy napięciu znamionowym, więc o energii rozstrzyga
# napięcie magazynu. Przy napięciu roboczym magazynu LiFePO4 (13,2 V):
#     E = U^2 / R_min * t = 13,2^2 / 0,263 * 150 s = 99,4 kJ = 27,6 Wh
# Jest to granica GÓRNA z dwóch powodów: praca impulsowa świecy i wzrost jej
# rezystancji po nagrzaniu. Przy napięciu znamionowym 12 V ta sama granica
# wynosi 22,8 Wh i stanowi dolny kraniec zakresu analizy wrażliwości.
WEBASTO_U_SWIECY_V = 13.2
WEBASTO_R_SWIECY_OHM = (0.263, 0.323)
WEBASTO_CZAS_ROZRUCHU_S = 150.0
WEBASTO_ENERGIA_STARTU_WH = 27.6
WEBASTO_ENERGIA_STARTU_OAT_WH = (22.8, 27.6)
# Instrukcja warsztatowa (s. 14): praca grzewcza jest przerywana co 8 h tak
# samo jak przy pauzie regulacyjnej, więc przy 10 h pracy na dobę przypadają
# co najmniej dwa rozruchy.
WEBASTO_ROZRUCHY_NA_DOBE = 2
WEBASTO_ROZRUCHY_OAT = (1, 4)

# Lokalizacja odniesienia dla profili dobowych i zestawień poglądowych.
# Profil obciążeń zależy od lokalizacji wyłącznie przez temperaturę
# powietrza wchodzącą do modelu chłodziarki.
LOKALIZACJA_ODNIESIENIA = "Leba"

# --- scenariusze ---
SCENARIUSZE = {
    "A": {"miesiace": range(3, 11), "opis": "praca zdalna stacjonarna"},
    "B": {"miesiace": range(4, 10), "opis": "wyjazd 3-dobowy, cykl tygodniowy"},
    "C": {"miesiace": [11, 12, 1, 2], "opis": "postój zimowy z ogrzewaniem"},
}

# Scenariusz B: wyjazd od czwartku wieczorem do niedzieli po południu.
B_START_DZIEN, B_START_GODZ = 3, 18   # czwartek = 3 (pon = 0)
B_KONIEC_DZIEN, B_KONIEC_GODZ = 6, 18  # niedziela


@dataclass(frozen=True)
class Odbior:
    """Opis odbioru na potrzeby zestawień i rysunku."""

    nazwa: str
    szyna: str     # "DC" albo "AC"
    klasa: str     # klasyfikacja wartości wg CLAUDE.md D


ODBIORY = (
    Odbior("chłodziarka", "DC", "katalogowa + obliczona"),
    Odbior("Starlink", "DC", "katalogowa (zakres), przyjęta (wartość)"),
    Odbior("oświetlenie LED i USB", "DC", "przyjęta"),
    Odbior("ogrzewanie postojowe", "DC", "przyjęta w granicach karty"),
    Odbior("laptop", "AC albo DC", "przyjęta"),
    Odbior("pobory własne", "DC", "katalogowa (dolne oszacowanie)"),
)


# ---------------------------------------------------------------------------
# Dane pogodowe
# ---------------------------------------------------------------------------

_PAMIEC_POGODY: dict[tuple[str, int], pd.DataFrame] = {}


def wczytaj_pogode(loc: Location, year: int) -> pd.DataFrame:
    """
    Czyta plik zasobu słonecznego i zwraca ramkę z osią UTC.

    Kolumny: ``t2m`` (temperatura powietrza, stC) oraz ``ghi``
    (natężenie promieniowania na płaszczyznę poziomą, W/m²).
    """
    import make_sam_resource as resource

    # Plik zasobu czytany jest raz na parę (lokalizacja, rok). Bez tego
    # bisekcja czasu jazdy w siatce symulacji wczytywałaby ten sam plik
    # kilkanaście razy na jedną konfigurację, co dominowało czas przebiegu.
    klucz = (loc.key, year)
    if klucz in _PAMIEC_POGODY:
        return _PAMIEC_POGODY[klucz]

    sciezka = resource.resource_path(loc, year)
    block = pd.read_csv(sciezka, skiprows=2)
    stamps = pd.to_datetime(block[["Year", "Month", "Day", "Hour", "Minute"]])
    idx = pd.DatetimeIndex(stamps).tz_localize("UTC")
    ramka = pd.DataFrame({"t2m": block["Tdry"].to_numpy(float),
                          "ghi": block["GHI"].to_numpy(float)}, index=idx)
    _PAMIEC_POGODY[klucz] = ramka
    return ramka


def _czas_lokalny(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Przelicza oś UTC na czas lokalny z obsługą czasu letniego."""
    return idx.tz_convert(config.LOCAL_TZ)


def _w_godzinach(lokalny: pd.DatetimeIndex, od: int, do: int) -> np.ndarray:
    """Maska godzin w czasie lokalnym; obsługuje przedział przez północ."""
    h = lokalny.hour.to_numpy()
    return (h >= od) & (h < do) if od < do else (h >= od) | (h < do)


# ---------------------------------------------------------------------------
# Modele odbiorów — wszystkie zwracają energię godzinową w Wh
# ---------------------------------------------------------------------------

def pobory_wlasne_w(jazda: bool = False) -> float:
    """
    Suma poborów własnych podzespołów w watach, przy napięciu znamionowym.

    ``jazda`` wybiera stan pojazdu: postój (domyślnie) albo jazda, w której
    ładowarka stałoprądowa jest załączona. Profil obciążeń budowany jest dla
    postoju — godziny jazdy wchodzą do modelu bilansu razem z dopływem
    z alternatora, więc tam uwzględniany jest też wyższy pobór ładowarki.

    Pozycje bez karty katalogowej (BMS, bocznik) nie są podstawiane wartością
    domyślną — wynik jest wtedy dolnym oszacowaniem. Funkcja zwraca sumę tych
    pozycji, które mają pokrycie w kartach.
    """
    prad_ma = sum((POBORY_WLASNE_JAZDA_MA if jazda
                   else POBORY_WLASNE_POSTOJ_MA).values())
    for brakujacy in (POBORY_WLASNE_BMS_MA, POBORY_WLASNE_BOCZNIK_MA):
        if brakujacy is not None:
            prad_ma += float(brakujacy)
    return prad_ma / 1000.0 * POBORY_WLASNE_U_V


def temperatura_kabiny(pogoda: pd.DataFrame, webasto_zal: np.ndarray,
                       podbicie_K: float = KABINA_PODBICIE_BAZOWE_K) -> np.ndarray:
    """
    Temperatura w zabudowie.

    Przyjęto równość z temperaturą powietrza na wysokości 2 m. Jest to
    założenie zaniżające pobór chłodziarki przy nasłonecznieniu, bo pomija
    nagrzewanie wnętrza przez przeszklenia i poszycie; warianty analizy
    wrażliwości podnoszą temperaturę o 0, 5 albo 10 K w godzinach ze słońcem
    (KABINA_PODBICIE_OAT_K). W godzinach pracy ogrzewania postojowego
    przyjmuje się nastawę termostatu.
    """
    t = pogoda["t2m"].to_numpy(float).copy()
    if podbicie_K:
        t = np.where(pogoda["ghi"].to_numpy(float) > KABINA_PROG_SLONCA_W_M2,
                     t + float(podbicie_K), t)
    return np.where(webasto_zal, np.maximum(t, KABINA_NASTAWA_WEBASTO_C), t)


def chlodziarka_wh(t_kabiny: np.ndarray, czynna: np.ndarray,
                   model: str | None = None) -> np.ndarray:
    """
    Pobór chłodziarki w godzinie, Wh.

    Wariant ``liniowy`` (domyślny) przyjmuje pobór proporcjonalny do różnicy
    temperatur, obcięty do zera:

        E(T_a) = E_ref * (T_a - T_in) / (T_a,ref - T_in)

    Liniowość jest założeniem: karta podaje jeden punkt pracy, więc nachylenia
    nie da się wyznaczyć z danych producenta.

    Wariant ``staly`` przyjmuje pobór równy punktowi katalogowemu niezależnie
    od temperatury otoczenia. Oba warianty ograniczają wynik od dołu i od góry
    — rzeczywisty pobór leży między nimi.
    """
    # Wartość domyślna czytana przy wywołaniu, a nie przy definicji funkcji:
    # wariant modelu podmienia się w analizie wrażliwości przez stałą modułu.
    model = model if model is not None else MODEL_CHLODZIARKI
    if model not in MODELE_CHLODZIARKI:
        raise ValueError("model chłodziarki: %s" % ", ".join(MODELE_CHLODZIARKI))
    e_ref_wh = CHLODZIARKA_AH_H_REF * CHLODZIARKA_U_NOM_V
    if model == "staly":
        return np.where(czynna, e_ref_wh, 0.0)
    skala = ((t_kabiny - CHLODZIARKA_T_WNETRZA_C)
             / (CHLODZIARKA_T_OTOCZ_REF_C - CHLODZIARKA_T_WNETRZA_C))
    return np.where(czynna, e_ref_wh * np.clip(skala, 0.0, None), 0.0)


def falownik_straty_w(p_ac_w: np.ndarray | float) -> np.ndarray:
    """
    Straty falownika przy zadanej mocy wyjściowej, W.

        P_strat = P_0 + a * P_out + b * P_out^2

    Człon stały P_0 odpowiada poborowi jałowemu przy załączonym falowniku,
    człony a i b opisują straty PONAD P_0. Współczynniki a i b dopasowano
    do krzywej sprawności z instrukcji przy P_0 ustalonym na 22 W, więc
    zapis nie liczy poboru jałowego dwa razy.
    """
    p = np.asarray(p_ac_w, dtype=float)
    return FALOWNIK_P0_W + FALOWNIK_A * p + FALOWNIK_B * p ** 2


def falownik_sprawnosc(p_ac_w: np.ndarray | float) -> np.ndarray:
    """Sprawność falownika wynikająca z modelu strat, do kontroli z krzywą."""
    p = np.asarray(p_ac_w, dtype=float)
    return np.where(p > 0.0, p / (p + falownik_straty_w(p)), 0.0)


def falownik_pobor_dc_wh(p_ac_w: np.ndarray, stan: np.ndarray) -> np.ndarray:
    """
    Energia pobierana przez falownik z szyny stałoprądowej, Wh na godzinę.

    ``stan`` to tablica nazw stanów pracy (patrz FALOWNIK_STANY_W). W stanie
    „praca" pobór to moc wyjściowa powiększona o straty; w pozostałych stanach
    pobór nie zależy od obciążenia. Poniżej 223 W model strat jest
    ekstrapolacją poza zakres pomiarowy — tam rozstrzyga człon stały.
    """
    stan = np.asarray(stan)
    pracuje = stan == "praca"
    pobor = np.zeros(stan.shape, dtype=float)
    pobor[pracuje] = (p_ac_w[pracuje] + falownik_straty_w(p_ac_w[pracuje]))
    for nazwa, moc in FALOWNIK_STANY_W.items():
        if nazwa == "praca":
            continue
        pobor[stan == nazwa] = moc
    return pobor


def falownik_stany(scenariusz: str, lokalny, uzytkowanie: np.ndarray,
                   poza_praca: str = FALOWNIK_POZA_PRACA) -> np.ndarray:
    """Buduje godzinowy przebieg stanów falownika dla scenariusza."""
    stany = np.full(len(uzytkowanie), "wyłączony", dtype=object)
    godziny = FALOWNIK_GODZINY_PRACY.get(scenariusz)
    if godziny is None:
        return stany
    stany[uzytkowanie] = poza_praca
    stany[uzytkowanie & _w_godzinach(lokalny, *godziny)] = "praca"
    return stany


def zbuduj_profil(loc: Location, year: int, scenariusz: str,
                  wariant_laptopa: str = "AC",
                  falownik_calodobowo: bool = False,
                  podbicie_K: float = KABINA_PODBICIE_BAZOWE_K,
                  starlink_godzin: int | None = None) -> pd.DataFrame:
    """
    Buduje godzinowy profil obciążeń dla lokalizacji, roku i scenariusza.

    Zwraca ramkę o indeksie UTC z kolumnami nazwanymi jak odbiory oraz
    kolumnami zbiorczymi ``DC`` i ``AC``.
    """
    if wariant_laptopa not in ("AC", "DC"):
        raise ValueError("wariant_laptopa musi być 'AC' albo 'DC'")

    pogoda = wczytaj_pogode(loc, year)
    idx = pogoda.index
    lok = _czas_lokalny(idx)

    miesiac_czynny = np.isin(lok.month.to_numpy(), list(SCENARIUSZE[scenariusz]["miesiace"]))

    # Scenariusz B: pojazd użytkowany tylko podczas wyjazdu; poza wyjazdem
    # chłodziarka jest wyłączona, a pozostałe odbiory nie występują.
    if scenariusz == "B":
        dow = lok.dayofweek.to_numpy()
        godz = lok.hour.to_numpy()
        po_starcie = (dow > B_START_DZIEN) | ((dow == B_START_DZIEN) & (godz >= B_START_GODZ))
        przed_koncem = (dow < B_KONIEC_DZIEN) | ((dow == B_KONIEC_DZIEN) & (godz < B_KONIEC_GODZ))
        uzytkowanie = miesiac_czynny & po_starcie & przed_koncem
    else:
        uzytkowanie = miesiac_czynny

    # --- ogrzewanie postojowe (tylko scenariusz C) ---
    if scenariusz == "C":
        webasto_zal = uzytkowanie & _w_godzinach(lok, *WEBASTO_GODZINY)
        webasto = np.where(webasto_zal, WEBASTO_MOC_W, 0.0)
        # Energia rozruchów dodawana jest do godziny załączenia nagrzewnicy
        # i do godziny przerwy regulacyjnej po ok. 8 h pracy.
        if WEBASTO_ENERGIA_STARTU_WH:
            godzina_lok = lok.hour.to_numpy()
            poczatek = WEBASTO_GODZINY[0]
            godziny_rozruchu = [(poczatek + 8 * i) % 24
                                for i in range(WEBASTO_ROZRUCHY_NA_DOBE)]
            maska_rozruchu = webasto_zal & np.isin(godzina_lok, godziny_rozruchu)
            webasto = webasto + np.where(maska_rozruchu,
                                         WEBASTO_ENERGIA_STARTU_WH, 0.0)
    else:
        webasto_zal = np.zeros(len(idx), dtype=bool)
        webasto = np.zeros(len(idx))

    # --- chłodziarka ---
    t_kabiny = temperatura_kabiny(pogoda, webasto_zal, podbicie_K)
    chlodziarka = chlodziarka_wh(t_kabiny, uzytkowanie)

    # --- Starlink i laptop: tylko scenariusz A ---
    if scenariusz == "A":
        godzin = starlink_godzin if starlink_godzin is not None else (
            STARLINK_GODZINY[1] - STARLINK_GODZINY[0])
        maska_sl = uzytkowanie & _w_godzinach(
            lok, STARLINK_GODZINY[0], STARLINK_GODZINY[0] + godzin)
        starlink = np.where(maska_sl, STARLINK_MOC_W, 0.0)
        maska_lap = uzytkowanie & _w_godzinach(lok, *LAPTOP_GODZINY)
    else:
        starlink = np.zeros(len(idx))
        maska_lap = np.zeros(len(idx), dtype=bool)

    # --- oświetlenie i drobna elektronika ---
    maska_led = uzytkowanie & _w_godzinach(lok, *LED_USB_GODZINY)
    godzin_led = LED_USB_GODZINY[1] - LED_USB_GODZINY[0]
    led = np.where(maska_led, LED_USB_WH_D / godzin_led, 0.0)

    # --- laptop w wariancie AC albo DC ---
    laptop_ac = np.zeros(len(idx))
    laptop_dc = np.zeros(len(idx))
    straty_dc = np.zeros(len(idx))
    if wariant_laptopa == "AC":
        # Moc po stronie 230 V to moc na zaciskach laptopa podzielona przez
        # sprawnosc zasilacza sieciowego; falownik widzi dopiero tę moc.
        p_ac = np.where(maska_lap,
                        LAPTOP_MOC_W / LAPTOP_AC_SPRAWNOSC, 0.0)
        stany = falownik_stany(
            scenariusz, lok, uzytkowanie,
            poza_praca="wyszukiwanie" if falownik_calodobowo else FALOWNIK_POZA_PRACA)
        laptop_ac = falownik_pobor_dc_wh(p_ac, stany)
    else:
        # Zapotrzebowanie laptopa i strata przetwornicy jako dwie
        # pozycje: pierwsza wchodzi do energii zadanej, druga do strat.
        laptop_dc = np.where(maska_lap, LAPTOP_MOC_W, 0.0)
        straty_dc = np.where(
            maska_lap,
            LAPTOP_MOC_W * (1.0 / LAPTOP_DC_SPRAWNOSC - 1.0), 0.0)

    # --- pobory własne podzespołów -------------------------------------
    # Nieprzerwanie przez cały czas trwania scenariusza, także wtedy, gdy
    # pojazd stoi nieużywany: regulator i ładowarka pozostają podłączone.
    wlasne = np.where(miesiac_czynny, pobory_wlasne_w(), 0.0)

    ramka = pd.DataFrame({
        "chłodziarka": chlodziarka,
        "Starlink": starlink,
        "oświetlenie LED i USB": led,
        "ogrzewanie postojowe": webasto,
        "pobory własne": wlasne,
        "laptop (DC)": laptop_dc,
        "straty przetwornicy laptopa": straty_dc,
        "laptop przez falownik (AC)": laptop_ac,
    }, index=idx)
    # Kolumna „laptop przez falownik" opisuje energię pobraną z szyny
    # stałoprądowej przez falownik, więc w bilansie wchodzi po stronie DC.
    # Rozdział szyn na potrzeby battery_balance: energia użyteczna odbiorników
    # AC podawana jest osobno.
    # Energia zadana laptopa w wariancie AC: tyle samo, ile w wariancie DC,
    # bo odnosi sie do zaciskow odbiornika, a nie do strony 230 V.
    ramka["energia użyteczna AC"] = np.where(
        maska_lap & (wariant_laptopa == "AC"), LAPTOP_MOC_W, 0.0)
    # Kolumny "DC" i "AC" opisuja energie zadana na zaciskach odbiornikow,
    # a kolumna "straty toru" - energie trat w zasilaczu, przetwornicy
    # i falowniku, pobierana z szyny ponad energie zadana.
    ramka["DC"] = ramka[["chłodziarka", "Starlink", "oświetlenie LED i USB",
                         "ogrzewanie postojowe", "pobory własne",
                         "laptop (DC)"]].sum(axis=1)
    ramka["AC"] = ramka["energia użyteczna AC"]
    ramka["straty toru"] = (
        ramka["straty przetwornicy laptopa"]
        + np.maximum(ramka["laptop przez falownik (AC)"]
                     - ramka["energia użyteczna AC"], 0.0))
    ramka["pobór falownika z szyny DC"] = ramka["laptop przez falownik (AC)"]
    # Godziny obowiazywania scenariusza: potrzebne do normowania srednich
    # dobowych. Bez tego srednia dzieli energie przez daty kalendarzowe,
    # a wyjazd trzydobowy wypada na czterech datach i srednia jest zanizona.
    ramka["godziny użytkowania"] = uzytkowanie.astype(float)
    ramka["godziny miesiąca czynnego"] = miesiac_czynny.astype(float)
    return ramka


def energia_dobowa(ramka: pd.DataFrame) -> pd.DataFrame:
    """Sumy dobowe w czasie lokalnym (doba kalendarzowa użytkownika)."""
    lok = _czas_lokalny(ramka.index)
    dobowe = ramka.groupby(lok.date).sum()
    dobowe.index = pd.to_datetime(dobowe.index)
    return dobowe


def zestawienie(year: int | list[int], warianty: bool = True) -> pd.DataFrame:
    """
    Zestawienie energii dobowej według odbioru, scenariusza i miesiąca.

    Argument ``year`` przyjmuje jeden rok albo listę lat. Przy liście wynik
    jest średnią po latach, liczoną osobno dla każdego miesiąca: energia
    dobowa zależy od temperatury powietrza, więc pojedynczy rok opisywałby
    warunki tego roku, a nie typowe obciążenie.
    """
    lata = [year] if isinstance(year, int) else list(year)
    wiersze = []
    kolumny_odbiorow = ["chłodziarka", "Starlink", "oświetlenie LED i USB",
                        "ogrzewanie postojowe", "pobory własne", "laptop (DC)",
                        "laptop przez falownik (AC)"]
    # Doba „czynna" rozpoznawana jest po odbiorach użytkowych, a nie po sumie
    # wszystkich pozycji: pobory własne płyną także w dobach bez użytkowania,
    # więc włączone do kryterium uznałyby za czynną każdą dobę scenariusza.
    kolumny_uzytkowe = [k for k in kolumny_odbiorow if k != "pobory własne"]
    for key, loc in LOCATIONS.items():
        for scen in SCENARIUSZE:
            for wariant in (("AC", "DC") if (warianty and scen == "A") else ("AC",)):
                for rok in lata:
                    ramka = zbuduj_profil(loc, rok, scen, wariant_laptopa=wariant)
                    dob = energia_dobowa(ramka)
                    czynne = dob[dob[kolumny_uzytkowe].sum(axis=1) > 0]
                    if czynne.empty:
                        continue
                    for miesiac, grupa in czynne.groupby(czynne.index.month):
                        # Doba to 24 godziny obowiazywania scenariusza, nie
                        # data kalendarzowa: wyjazd trzydobowy rozklada sie na
                        # cztery daty, a ogrzewanie pracuje przez polnoc.
                        doby_uzytk = max(
                            float(grupa["godziny użytkowania"].sum()) / 24.0,
                            1e-9)
                        doby_miesiaca = max(
                            float(grupa["godziny miesiąca czynnego"].sum())
                            / 24.0, 1e-9)
                        for kol in kolumny_odbiorow:
                            doby = (doby_miesiaca if kol == "pobory własne"
                                    else doby_uzytk)
                            wiersze.append({
                                "lokalizacja": loc.label,
                                "scenariusz": scen,
                                "wariant_laptopa": wariant,
                                "rok": rok,
                                "miesiac": miesiac,
                                "odbior": kol,
                                "szyna": ("DC (za falownikiem)"
                                          if kol == "laptop przez falownik (AC)" else "DC"),
                                "energia_dobowa_wh": round(
                                    float(grupa[kol].sum()) / doby, 1),
                                "dni_czynnych": int(round(doby)),
                            })
    tab = pd.DataFrame(wiersze)
    if len(lata) > 1:
        tab = (tab.groupby(["lokalizacja", "scenariusz", "wariant_laptopa",
                            "miesiac", "odbior", "szyna"], as_index=False)
               .agg(energia_dobowa_wh=("energia_dobowa_wh", "mean"),
                    dni_czynnych=("dni_czynnych", "mean")))
        tab["energia_dobowa_wh"] = tab["energia_dobowa_wh"].round(1)
        tab["dni_czynnych"] = tab["dni_czynnych"].round(0).astype(int)
    return tab


def zestawienie_oat_kabiny(year: int, loc: Location | None = None) -> pd.DataFrame:
    """
    Energia dobowa scenariuszy przy podbiciu temperatury w zabudowie.

    Zakres podbicia (0, +5, +10 K w godzinach ze słońcem) jest wariantem
    analizy wrażliwości; wartości należą do klasy przyjętych przez autora.
    Wielkością, na którą podbicie działa, jest wyłącznie pobór chłodziarki.
    """
    loc = loc if loc is not None else LOCATIONS[LOKALIZACJA_ODNIESIENIA]
    kolumny = ["chłodziarka", "Starlink", "oświetlenie LED i USB",
               "ogrzewanie postojowe", "pobory własne", "laptop (DC)",
               "laptop przez falownik (AC)"]
    uzytkowe = [k for k in kolumny if k != "pobory własne"]
    wiersze = []
    for scen in SCENARIUSZE:
        for podbicie in KABINA_PODBICIE_OAT_K:
            ramka = zbuduj_profil(loc, year, scen, podbicie_K=podbicie)
            dob = energia_dobowa(ramka)
            czynne = dob[dob[uzytkowe].sum(axis=1) > 0]
            miesieczne = czynne[kolumny].groupby(czynne.index.month).mean()
            wiersze.append({
                "scenariusz": scen,
                "podbicie_K": podbicie,
                "energia_dobowa_wh": round(float(miesieczne.sum(axis=1).mean()), 1),
                "chlodziarka_wh": round(float(miesieczne["chłodziarka"].mean()), 1),
            })
    return pd.DataFrame(wiersze)


def profil_dobowy_sredni(loc: Location, year: int | list[int], scenariusz: str,
                         wariant_laptopa: str = "AC") -> pd.DataFrame:
    """
    Średni profil dobowy w czasie lokalnym: 24 wiersze, kolumny to odbiory.

    Uśrednianie obejmuje wyłącznie doby, w których układ jest użytkowany —
    dla scenariusza B są to doby wyjazdu, a nie wszystkie doby w miesiącu.
    """
    lata = [year] if isinstance(year, int) else list(year)
    if len(lata) > 1:
        czesci = [profil_dobowy_sredni(loc, r, scenariusz, wariant_laptopa)
                  for r in lata]
        return (pd.concat(czesci).groupby(level=0).mean().round(2))

    ramka = zbuduj_profil(loc, lata[0], scenariusz, wariant_laptopa=wariant_laptopa)
    lok = _czas_lokalny(ramka.index)
    kolumny = ["chłodziarka", "Starlink", "oświetlenie LED i USB",
               "ogrzewanie postojowe", "pobory własne", "laptop (DC)",
               "laptop przez falownik (AC)"]
    uzytkowe = [k for k in kolumny if k != "pobory własne"]
    praca = ramka[uzytkowe].sum(axis=1) > 0
    doby_czynne = pd.Series(praca.to_numpy(), index=lok.date).groupby(level=0).any()
    maska = pd.Series(lok.date, index=ramka.index).map(doby_czynne).to_numpy()
    wybrane = ramka.loc[maska, kolumny]
    godziny = pd.Index(lok[maska].hour, name="godzina")
    return wybrane.groupby(godziny).mean().round(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2020,
                        help="rok, gdy policzyć profil dla jednego roku")
    parser.add_argument("--lata", nargs=2, type=int, default=None,
                        help="zakres lat do uśrednienia, np. --lata 2014 2023")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_dirs()

    lata = (list(range(args.lata[0], args.lata[1] + 1))
            if args.lata else [args.year])
    tab = zestawienie(lata if len(lata) > 1 else lata[0])
    sciezka = config.OUT_DIR / "load_profiles_daily.csv"
    tab.to_csv(sciezka, index=False)

    # Lokalizacja odniesienia dla rysunku i tabeli: ta o największym
    # zapotrzebowaniu w scenariuszu A, bo w niej rozstrzyga się wymaganie W1.
    suma_a = (tab[(tab.scenariusz == "A") & (tab.wariant_laptopa == "AC")]
              .groupby(["lokalizacja", "miesiac"])["energia_dobowa_wh"].sum()
              .groupby("lokalizacja").mean())
    etykieta_odniesienia = str(suma_a.idxmax())
    loc_odniesienia = next(l for l in LOCATIONS.values()
                           if l.label == etykieta_odniesienia)
    print("\nLokalizacja odniesienia (największe zapotrzebowanie w A): %s"
          % etykieta_odniesienia)
    print(suma_a.round(1).to_string())

    oat = zestawienie_oat_kabiny(lata[-1], loc_odniesienia)
    sciezka_oat = config.OUT_DIR / "load_profiles_oat_kabina.csv"
    oat.to_csv(sciezka_oat, index=False)

    print("=" * 78)
    print("Profile obciążeń — rok %d" % args.year)
    print("=" * 78)
    # Najpierw suma odbiorów w miesiącu, potem średnia po miesiącach —
    # odwrotna kolejność sumowałaby energię kolejnych miesięcy.
    miesieczne = (tab.groupby(["scenariusz", "wariant_laptopa", "lokalizacja",
                               "miesiac"])["energia_dobowa_wh"].sum())
    suma = (miesieczne.groupby(["scenariusz", "wariant_laptopa", "lokalizacja"])
            .mean().round(0).unstack())
    print("\nŚrednia energia dobowa [Wh] — suma odbiorów, średnia po miesiącach:")
    print(suma.to_string())

    etykieta = LOCATIONS[LOKALIZACJA_ODNIESIENIA].label
    print("\nUdział odbiorów, scenariusz A, wariant AC, %s:" % etykieta)
    a = tab[(tab.scenariusz == "A") & (tab.wariant_laptopa == "AC")
            & (tab.lokalizacja == etykieta)]
    print(a.groupby("odbior")["energia_dobowa_wh"].mean().round(1).to_string())

    print("\nPobory własne podzespołów: %.2f W (%.1f Wh/d) — %s"
          % (pobory_wlasne_w(), pobory_wlasne_w() * 24.0,
             ", ".join("%s %.0f mA" % (k, v) for k, v in POBORY_WLASNE_POSTOJ_MA.items())))
    print("  DO UZUPEŁNIENIA: %s — suma jest dolnym oszacowaniem"
          % ", ".join(POBORY_WLASNE_DO_UZUPELNIENIA))

    print("\nWariant OAT: podbicie temperatury w zabudowie [K]")
    print(oat.pivot(index="scenariusz", columns="podbicie_K",
                    values="energia_dobowa_wh").to_string())

    # Profile dobowe do rysunku — lokalizacja odniesienia.
    pliki_profili = []
    for scen in SCENARIUSZE:
        prof = profil_dobowy_sredni(loc_odniesienia,
                                    lata if len(lata) > 1 else lata[0], scen)
        cel = config.OUT_DIR / ("load_profile_hourly_%s.csv" % scen)
        prof.to_csv(cel)
        pliki_profili.append(cel)

    import make_sam_resource as resource
    manifest.zapisz(
        "profile_obciazen",
        skrypt="simulation/loads.py",
        konfiguracja={
            "lata": lata,
            "lokalizacja_profili_dobowych": etykieta_odniesienia,
            "scenariusze": {k: v["opis"] for k, v in SCENARIUSZE.items()},
            "pobory_wlasne_postoj_mA": POBORY_WLASNE_POSTOJ_MA,
            "pobory_wlasne_jazda_mA": POBORY_WLASNE_JAZDA_MA,
            "pobory_wlasne_do_uzupelnienia": list(POBORY_WLASNE_DO_UZUPELNIENIA),
            "pobory_wlasne_W": round(pobory_wlasne_w(), 3),
            "oat_podbicie_kabiny_K": list(KABINA_PODBICIE_OAT_K),
            "oat_chlodziarka": list(CHLODZIARKA_OAT),
            "oat_starlink_W": list(STARLINK_OAT_W),
            "oat_laptop_W": list(LAPTOP_OAT_W),
            "oat_webasto_W": list(WEBASTO_OAT_W),
            "oat_led_usb_Wh": list(LED_USB_OAT_WH_D),
            "falownik": {"P0_W": FALOWNIK_P0_W, "a": FALOWNIK_A, "b": FALOWNIK_B,
                         "zakres_pomiarowy_W": list(FALOWNIK_ZAKRES_POMIAROWY_W)},
        },
        wejscia={"%s_%d" % (loc.key, rok): resource.resource_path(loc, rok)
                 for loc in LOCATIONS.values() for rok in lata},
        wyniki=[sciezka, sciezka_oat] + pliki_profili,
    )

    print("\nzapisano:", sciezka)
    print("profile dobowe:", config.OUT_DIR / "load_profile_hourly_<A|B|C>.csv")
    print("manifest:", manifest.sciezka_manifestu("profile_obciazen").name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
