# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: battery_balance.py
Opis: godzinowy bilans energetyczny magazynu. Moduł realizuje drugą
      część podziału odpowiedzialności przyjętego w rozdz. 6.1: logika
      dyspozycji magazynu oraz wskaźniki oceny liczone są poza SAM, po
      stronie Pythona. Uzasadnienie: model magazynu w SAM projektowany
      jest pod systemy przyłączone do sieci, a wskaźnik LPSP nie należy
      do jego wielkości wyjściowych.

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

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

# Krok czasowy modelu. Energia w kWh jest liczbowo równa mocy w kW przy kroku
# jednogodzinnym, co upraszcza zapis wszystkich równań bilansu.
KROK_H = 1.0

# Średnia długość miesiąca w godzinach (365,25 / 12 * 24). Potrzebna do
# przeliczenia samorozładowania podanego w procentach na miesiąc na ubytek
# godzinowy.
GODZIN_W_MIESIACU = 365.25 / 12.0 * 24.0


@dataclass
class ParametryUkladu:
    """
    Parametry układu zasilania przyjęte w rozdz. 4.7 pracy.

    Wartości sprawności pochodzą z kart katalogowych podzespołów, a nie
    z literatury algorytmicznej — dotyczy to zwłaszcza sprawności układu MPPT,
    dla której źródła algorytmiczne podają wartości rzędu 99 %, nieosiągalne
    dla gotowych regulatorów.
    """

    pojemnosc_kwh: float = 1.28     # magazyn LiFePO4 100 Ah / 12,8 V
    # Okno stanu naładowania wprost z karty ogniwa EVE LF280K (s. 4): zalecany
    # zakres pracy 10-90 %. Szerokość okna, czyli głębokość rozładowania 0,8,
    # jest tej pary WYNIKIEM, a nie osobnym założeniem.
    #
    # Dlaczego położenie okna zmienia wynik przy tej samej szerokości: energia
    # użyteczna jest ta sama (0,8 pojemności), ale okno 10-90 % ma sufit niższy
    # o 10 % pojemności, więc w okresach nadwyżki wcześniej zaczyna się
    # ograniczanie generacji, a stan początkowy (= sufit) jest w ujęciu
    # bezwzględnym niższy. Oba skutki podnoszą energetyczny wskaźnik LPSP
    # względem okna 20-100 %, mimo identycznej głębokości rozładowania.
    soc_min_wzgl: float = 0.10
    soc_max_wzgl: float = 0.90
    sprawnosc_obiegu: float = 0.90  # sprawność cyklu ładowanie-rozładowanie
    sprawnosc_falownika: float = 0.92
    sprawnosc_mppt: float = 0.97
    zapotrzebowanie_dobowe_kwh: float = 1.0   # scenariusz A, rozdz. 4.3
    # Samorozładowanie magazynu, % pojemności znamionowej na miesiąc.
    # Karta ogniwa EVE LF280K (s. 5) podaje <= 3 %/miesiąc przy 25 +/- 2 stC
    # i stanie naładowania 30-50 %. Przyjęcie wartości maksymalnej jest
    # zachowawcze podwójnie: w scenariuszu zimowym temperatura ogniw jest
    # znacznie niższa od warunków pomiaru, a samorozładowanie maleje
    # z temperaturą. Zakres analizy wrażliwości: 0-3 %/miesiąc.
    samorozladowanie_pct_mies: float = 3.0

    # --- blokada ładowania w niskiej temperaturze (D.5) -------------------
    # Karta ogniwa dopuszcza ładowanie od 0 stC. Blokadę realizuje układ
    # zarządzania magazynem; liczona jest osobno dla energii z generatora
    # i z alternatora, bo w rozpatrywanych wariantach odpowiada za nią inny
    # element układu.
    temperatura_blokady_C: float = 0.0
    # Magazyn z samogrzaniem: grzałka załącza się poniżej progu blokady
    # i pobiera energię z magazynu. Moc grzałki pochodzi z karty produktu,
    # który zostanie dobrany w rozdz. 5 — do tego czasu wariant jest wyłączony.
    samogrzanie: bool = False
    moc_grzalki_w: float | None = None   # DO UZUPEŁNIENIA: karta magazynu
    # Model temperatury ogniw: stała czasowa i nastawa ogrzewania postojowego.
    tau_bat_h: float = 3.0
    t_kabiny_ogrzewanie_C: float = 18.0
    # Stan naładowania w pierwszej godzinie, jako ułamek pojemności
    # znamionowej. Brak wartości oznacza start od górnej granicy okna.
    soc_poczatkowy_wzgl: float | None = None

    def moc_grzalka_w_lub_blad(self) -> float:
        """Moc grzałki magazynu albo czytelny błąd, gdy jej nie podano."""
        if self.moc_grzalki_w is None:
            raise ValueError(
                "Włączono wariant z samogrzaniem, ale nie podano mocy grzałki "
                "(moc_grzalki_w). Wartość pochodzi z karty magazynu dobranego "
                "w rozdz. 5 — do tego czasu wariant pozostaje wyłączony.")
        return float(self.moc_grzalki_w)

    @property
    def samorozladowanie_kwh_h(self) -> float:
        """Ubytek energii zmagazynowanej w jednej godzinie, kWh."""
        return (self.pojemnosc_kwh * self.samorozladowanie_pct_mies / 100.0
                / GODZIN_W_MIESIACU * KROK_H)

    @property
    def soc_max_kwh(self) -> float:
        """Górna granica stanu naładowania, kWh."""
        return self.pojemnosc_kwh * self.soc_max_wzgl

    @property
    def soc_min_kwh(self) -> float:
        """Dolna granica stanu naładowania, kWh."""
        return self.pojemnosc_kwh * self.soc_min_wzgl

    @property
    def dod_max(self) -> float:
        """Głębokość rozładowania wynikająca z okna stanu naładowania."""
        return self.soc_max_wzgl - self.soc_min_wzgl

    @property
    def pojemnosc_uzyteczna_kwh(self) -> float:
        """Energia użyteczna magazynu w przyjętym oknie stanu naładowania."""
        return self.pojemnosc_kwh * self.dod_max

    @property
    def sprawnosc_ladowania(self) -> float:
        """
        Sprawność jednego kierunku przepływu energii przez magazyn.

        Sprawność obiegu rozdzielono symetrycznie na ładowanie i rozładowanie
        (pierwiastek kwadratowy), zgodnie z konwencją stosowaną w modelach
        bilansowych układów off-grid.
        """
        return float(np.sqrt(self.sprawnosc_obiegu))

    @property
    def sprawnosc_rozladowania(self) -> float:
        """Sprawność rozładowania — patrz komentarz przy sprawnosc_ladowania."""
        return float(np.sqrt(self.sprawnosc_obiegu))


def temperatura_ogniw(t_powietrza_C: np.ndarray,
                      ogrzewanie_zal: np.ndarray | None = None,
                      parametry: ParametryUkladu | None = None,
                      t_poczatkowa_C: float | None = None) -> np.ndarray:
    """
    Temperatura ogniw magazynu, model inercyjny pierwszego rzędu.

        tau * dT_bat/dt = T_kab - T_bat

rozwiązany dokładnie przy stałej temperaturze otoczenia w kroku:

        T_bat(t) = T_kab(t) + [T_bat(t-1) - T_kab(t)] * exp(-dt / tau)

    Postać wykładnicza zastąpiła schemat jawny: przy kroku godzinowym
    porównywalnym ze stałą czasową schemat jawny daje przebieg zgrubny,
    a przy dt > tau prowadziłby do przeregulowania.

    Temperatura otoczenia magazynu (T_kab) to temperatura powietrza, a w
    godzinach pracy ogrzewania postojowego — nastawa termostatu, bo magazyn
    stoi we wnęce pod siedzeniem kierowcy, wewnątrz ogrzewanej przestrzeni.
    W scenariuszu C oznacza to, że ogrzewanie podnosi temperaturę ogniw i tym
    samym skraca czas obowiązywania blokady ładowania — skutek działający na
    korzyść układu i dlatego wart osobnego wykazania.

    Stała czasowa rzędu godzin odpowiada masie magazynu i osłonięciu wnęki;
    jest wartością przyjętą, objętą analizą wrażliwości.
    """
    par = parametry if parametry is not None else ParametryUkladu()
    t_powietrza_C = np.asarray(t_powietrza_C, dtype=float)
    n = t_powietrza_C.size
    t_kabiny = t_powietrza_C.copy()
    if ogrzewanie_zal is not None:
        ogrzewanie_zal = np.asarray(ogrzewanie_zal, dtype=bool)
        t_kabiny = np.where(ogrzewanie_zal,
                            np.maximum(t_kabiny, par.t_kabiny_ogrzewanie_C),
                            t_kabiny)

    wynik = np.empty(n, dtype=float)
    stan = float(t_poczatkowa_C if t_poczatkowa_C is not None else t_kabiny[0])
    zanik = float(np.exp(-KROK_H / par.tau_bat_h))
    for i in range(n):
        stan = t_kabiny[i] + (stan - t_kabiny[i]) * zanik
        wynik[i] = stan
    return wynik


def sprawdz_zgodnosc_indeksu(indeks_pv, indeks_obciazenia) -> None:
    """
    Sprawdza, że szereg generacji i profil obciążeń mają ten sam indeks czasu.

    Powód: pliki zasobu słonecznego nie zawierają 29 lutego (lata 2016 i 2020
    są skracane do 8760 godzin, aby wszystkie lata miały tę samą długość).
    Gdyby profil obciążeń zbudowano na pełnym roku kalendarzowym, oba szeregi
    rozjechałyby się o dobę od marca wzwyż — bez żadnego komunikatu, bo obie
    tablice byłyby „prawie" tej samej długości. Niezgodność ma przerwać
    obliczenie, a nie zostać wyrównana po cichu.
    """
    if len(indeks_pv) != len(indeks_obciazenia):
        raise ValueError(
            "Szereg generacji ma %d godzin, a profil obciążeń %d. Najczęstsza "
            "przyczyna: 29 lutego usunięty w jednym szeregu, a obecny w drugim "
            "(lata przestępne 2016 i 2020)."
            % (len(indeks_pv), len(indeks_obciazenia)))
    roznice = (pd.DatetimeIndex(indeks_pv) != pd.DatetimeIndex(indeks_obciazenia))
    if bool(roznice.any()):
        pierwsza = int(np.argmax(roznice))
        raise ValueError(
            "Indeksy czasu rozchodzą się od pozycji %d: generacja %s, "
            "obciążenie %s." % (pierwsza, indeks_pv[pierwsza],
                                indeks_obciazenia[pierwsza]))


def profil_obciazenia_stale(
    liczba_godzin: int, zapotrzebowanie_dobowe_kwh: float
) -> np.ndarray:
    """
    Zwraca płaski profil obciążenia — zapotrzebowanie dobowe rozłożone na 24 h.

    Profil stały jest uproszczeniem przyjętym na etapie weryfikacji potoku.
    Rzeczywisty profil scenariusza A ma wyraźne maksimum wieczorne, co pogarsza
    bilans (energia potrzebna jest wtedy, gdy generacja już nie występuje);
    zastąpienie profilu odbywa się przez podanie własnej tablicy do funkcji
    ``bilans_godzinowy``.
    """
    return np.full(liczba_godzin, zapotrzebowanie_dobowe_kwh / 24.0, dtype=float)


def profil_dobowy_wieczorny(zapotrzebowanie_dobowe_kwh: float) -> np.ndarray:
    """
    Zwraca przykładowy profil dobowy z maksimum wieczornym, w czasie lokalnym.

    Profil służy wyłącznie demonstracji wpływu przeliczenia strefy czasowej —
    rzeczywisty profil scenariusza A zostanie wyznaczony z kart katalogowych
    odbiorników w rozdz. 4.3. Udziały godzinowe sumują się do jedności.
    """
    weights = np.array([
        # 0-5: tylko odbiory stałe (chłodnia, czujniki)
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        # 6-8: poranek
        4.0, 5.0, 4.0,
        # 9-16: praca zdalna, komputer i ładowanie
        5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
        # 17-22: maksimum wieczorne, oświetlenie i gotowanie
        7.0, 8.0, 9.0, 8.0, 6.0, 4.0,
        # 23: noc
        2.0,
    ], dtype=float)
    return zapotrzebowanie_dobowe_kwh * weights / weights.sum()


def load_profile_utc(
    index_utc: pd.DatetimeIndex,
    daily_profile_local: np.ndarray | None = None,
    zapotrzebowanie_dobowe_kwh: float = 1.0,
    tz: str | None = None,
) -> np.ndarray:
    """
    Odwzorowuje profil dobowy zadany w czasie lokalnym na oś czasu UTC.

    Argumenty:
        index_utc            -- indeks czasu UTC szeregu generacji (ze strefą),
        daily_profile_local  -- 24 wartości energii [kWh] przypisane kolejnym
                                godzinom czasu LOKALNEGO; domyślnie profil płaski,
        zapotrzebowanie_dobowe_kwh -- suma dobowa, gdy profil nie został podany,
        tz                   -- strefa czasowa profilu; domyślnie Europe/Warsaw.

    Zwraca tablicę o długości indeksu, z wartością profilu przypisaną według
    godziny czasu lokalnego odpowiadającej każdemu znacznikowi UTC.

    Przeliczenie wykonuje biblioteka strefowa, nie stałe przesunięcie, dlatego
    zmiana czasu letniego jest uwzględniona automatycznie: ta sama godzina
    lokalna odpowiada innej godzinie UTC w styczniu i w lipcu. Doba zmiany
    czasu ma 23 lub 25 godzin lokalnych, więc suma dobowa w takiej dobie
    odbiega od zadanej — jest to właściwość kalendarza, nie błąd odwzorowania.
    """
    timezone = tz if tz is not None else config.LOCAL_TZ

    if daily_profile_local is None:
        daily_profile_local = profil_obciazenia_stale(
            24, zapotrzebowanie_dobowe_kwh
        )
    daily_profile_local = np.asarray(daily_profile_local, dtype=float)
    if daily_profile_local.size != 24:
        raise ValueError(
            f"Profil dobowy musi mieć 24 wartości, podano "
            f"{daily_profile_local.size}."
        )

    if index_utc.tz is None:
        raise ValueError(
            "Indeks musi mieć zadeklarowaną strefę czasową (UTC). "
            "Indeks bez strefy nie pozwala wykonać przeliczenia na czas lokalny."
        )

    local_hours = index_utc.tz_convert(timezone).hour
    return daily_profile_local[np.asarray(local_hours)]


def bilans_godzinowy(
    moc_pv_kw: np.ndarray,
    parametry: ParametryUkladu | None = None,
    obciazenie_kwh: np.ndarray | None = None,
    indeks: pd.DatetimeIndex | None = None,
    profil_dobowy_local: np.ndarray | None = None,
    obciazenie_dc_kwh: np.ndarray | None = None,
    obciazenie_ac_kwh: np.ndarray | None = None,
    straty_toru_kwh: np.ndarray | None = None,
    doplyw_alternatora_kwh: np.ndarray | None = None,
    temperatura_ogniw_C: np.ndarray | None = None,
    rozbieg_godzin: int = 0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Wykonuje godzinowy bilans energetyczny układu PV + magazyn.

    Argumenty:
        moc_pv_kw      -- godzinowy szereg mocy stałoprądowej generatora, kW
        parametry      -- parametry układu; domyślnie ParametryUkladu()
        obciazenie_kwh -- gotowy godzinowy profil zapotrzebowania odbiorników
                          prądu przemiennego, kWh (zapis zgodny z wcześniejszą
                          wersją modułu; równoważny obciazenie_ac_kwh)
        indeks         -- indeks czasowy wyniku; jeśli ma zadeklarowaną strefę
                          czasową, profil dobowy jest przeliczany z czasu
                          lokalnego na UTC
        profil_dobowy_local -- 24 wartości profilu w czasie LOKALNYM
        obciazenie_dc_kwh -- odbiory zasilane wprost z szyny stałoprądowej, kWh
        obciazenie_ac_kwh -- odbiory za falownikiem, kWh
        straty_toru_kwh -- straty torów zasilania odniesione do szyny
                          stałoprądowej, kWh; podawany osobno, ponieważ zależy
                          od godzin załączenia falownika, a nie od energii
                          wydanej odbiornikom

    Zwraca krotkę (ramka z przebiegami godzinowymi, słownik wskaźników).

    KOLEJNOŚĆ OPERACJI W KROKU GODZINOWYM (rozdz. 4.7):

      1. generacja modułu -> regulator (sprawność śledzenia i przetwarzania),
      2. szyna stałoprądowa,
      3. pokrycie odbiorów stałoprądowych i poboru falownika wprost
         z generatora,
      4. nadwyżka -> magazyn, niedobór -> z magazynu,
      5. dopływ z alternatora -> magazyn (po PV, bo generator ma pierwszeństwo),
      6. samorozładowanie.

    Energia, której magazyn nie przyjmie — bo jest pełny albo bo ładowanie
    blokuje niska temperatura ogniw — trafia do licznika ``E_curt_kWh``
    (energia niewykorzystana). Bilans zamyka się z tym licznikiem.

    Konwencja bilansu: wszystkie przepływy sprowadzane są do szyny
    stałoprądowej. Sprawność falownika obciąża **wyłącznie** odbiory prądu
    przemiennego; odbiory stałoprądowe (chłodziarka, oświetlenie, ogrzewanie
    postojowe, łącze satelitarne) pobierają energię z szyny bez przetwarzania.
    Zapotrzebowanie i niedobory raportowane są po stronie odbiorników, czyli
    jako energia użyteczna, niezależnie od szyny.
    """
    par = parametry if parametry is not None else ParametryUkladu()
    moc_pv_kw = np.asarray(moc_pv_kw, dtype=float)
    n = moc_pv_kw.size

    obciazenie_kwh_podane = obciazenie_kwh is not None
    szyny_podane = obciazenie_dc_kwh is not None or obciazenie_ac_kwh is not None

    if obciazenie_kwh is None and szyny_podane:
        # Profil rozdzielony na szyny zastępuje profil zbiorczy; syntezę
        # profilu zastępczego trzeba wtedy pominąć.
        obciazenie_kwh = np.zeros(n)
    elif obciazenie_kwh is None:
        # Profil zadany w czasie lokalnym wymaga przeliczenia na oś UTC, na
        # której wyznaczona jest generacja. Bez indeksu ze strefą czasową
        # przeliczenie jest niewykonalne — stosowany jest wtedy profil płaski,
        # dla którego przesunięcie godzinowe nie zmienia wyniku.
        if indeks is not None and getattr(indeks, "tz", None) is not None:
            obciazenie_kwh = load_profile_utc(
                indeks,
                daily_profile_local=profil_dobowy_local,
                zapotrzebowanie_dobowe_kwh=par.zapotrzebowanie_dobowe_kwh,
            )
        else:
            if profil_dobowy_local is not None:
                raise ValueError(
                    "Podano profil dobowy w czasie lokalnym, ale indeks nie ma "
                    "strefy czasowej — przeliczenie na UTC jest niewykonalne. "
                    "Przekaż indeks UTC ze strefą (patrz weather_qc.load_all)."
                )
            obciazenie_kwh = profil_obciazenia_stale(
                n, par.zapotrzebowanie_dobowe_kwh
            )
    obciazenie_kwh = np.asarray(obciazenie_kwh, dtype=float)

    if obciazenie_kwh.size != n:
        raise ValueError(
            f"Profil obciążenia ma {obciazenie_kwh.size} wartości, "
            f"a szereg generacji {n}. Długości muszą być równe."
        )

    # --- rozdział obciążeń na szynę stałoprądową i przemienną --------------
    # Zapis z jednym profilem (obciazenie_kwh) jest zachowany: cały profil
    # traktowany jest wtedy jako odbiory prądu przemiennego, tak jak
    # w pierwszej wersji modułu. Profil rozdzielony podaje się przez
    # obciazenie_dc_kwh i obciazenie_ac_kwh — zob. simulation/loads.py.
    if obciazenie_dc_kwh is None and obciazenie_ac_kwh is None:
        obc_dc = np.zeros(n)
        obc_ac = obciazenie_kwh
    else:
        if obciazenie_kwh_podane:
            raise ValueError(
                "Podano jednocześnie obciazenie_kwh oraz profil rozdzielony "
                "na szyny. Wybierz jeden zapis: albo pojedynczy profil AC, "
                "albo parę obciazenie_dc_kwh / obciazenie_ac_kwh."
            )
        obc_dc = (np.zeros(n) if obciazenie_dc_kwh is None
                  else np.asarray(obciazenie_dc_kwh, dtype=float))
        obc_ac = (np.zeros(n) if obciazenie_ac_kwh is None
                  else np.asarray(obciazenie_ac_kwh, dtype=float))
        for nazwa, tablica in (("obciazenie_dc_kwh", obc_dc),
                               ("obciazenie_ac_kwh", obc_ac)):
            if tablica.size != n:
                raise ValueError(
                    f"{nazwa} ma {tablica.size} wartości, a szereg generacji "
                    f"{n}. Długości muszą być równe."
                )

    jalowy_podany = straty_toru_kwh is not None
    jalowy = (np.zeros(n) if straty_toru_kwh is None
              else np.asarray(straty_toru_kwh, dtype=float))
    if jalowy.size != n:
        raise ValueError(
            f"straty_toru_kwh ma {jalowy.size} wartości, a szereg "
            f"generacji {n}. Długości muszą być równe."
        )
    if np.any(jalowy < 0.0):
        raise ValueError("straty_toru_kwh zawiera wartości ujemne.")

    # --- dopływ z alternatora i temperatura ogniw -----------------------
    alternator = (np.zeros(n) if doplyw_alternatora_kwh is None
                  else np.asarray(doplyw_alternatora_kwh, dtype=float))
    if alternator.size != n:
        raise ValueError(
            f"doplyw_alternatora_kwh ma {alternator.size} wartości, a szereg "
            f"generacji {n}. Długości muszą być równe.")

    if temperatura_ogniw_C is None:
        # Bez podanej temperatury blokada nie działa — model opisuje wtedy
        # magazyn pracujący stale powyżej progu. Wariant ten służy wyłącznie
        # testom; w obliczeniach projektowych temperatura jest podawana.
        t_ogniw = np.full(n, par.temperatura_blokady_C + 1.0)
    else:
        t_ogniw = np.asarray(temperatura_ogniw_C, dtype=float)
        if t_ogniw.size != n:
            raise ValueError(
                f"temperatura_ogniw_C ma {t_ogniw.size} wartości, a szereg "
                f"generacji {n}. Długości muszą być równe.")

    # Energia użyteczna wydana odbiornikom, niezależnie od szyny.
    obciazenie_kwh = obc_dc + obc_ac

    # Tablice wynikowe — alokowane z góry dla czytelności pętli.
    soc = np.zeros(n)                 # stan naładowania na koniec godziny, kWh
    e_bezposrednie = np.zeros(n)      # energia PV zużyta wprost przez odbiorniki
    e_do_magazynu = np.zeros(n)       # energia zmagazynowana (przyrost SoC)
    e_z_magazynu = np.zeros(n)        # energia pobrana z magazynu, strona DC
    e_deficyt = np.zeros(n)           # energia niedostarczona, strona odbiorników
    e_odrzucona = np.zeros(n)         # nadwyżka niewykorzystana (magazyn pełny)
    e_curt = np.zeros(n)              # energia nieprzyjęta: pełny lub zimny magazyn
    e_alternator = np.zeros(n)        # energia przyjęta z alternatora
    e_alt_oferowana = np.zeros(n)     # energia zaoferowana przez alternator
    e_blok_pv = np.zeros(n)           # energia PV odrzucona przez blokadę
    e_blok_alt = np.zeros(n)          # energia alternatora odrzucona przez blokadę
    e_grzalka = np.zeros(n)           # pobór grzałki magazynu
    e_samorozladowanie = np.zeros(n)  # ubytek energii zmagazynowanej
    wsp_uzyteczne = np.zeros(n)       # energia użyteczna na jednostkę energii szyny

    # Symulację rozpoczynamy od magazynu naładowanego. Przy rocznym horyzoncie
    # wpływ warunku początkowego na wskaźniki jest pomijalny.
    stan = (par.soc_max_kwh if par.soc_poczatkowy_wzgl is None
            else par.pojemnosc_kwh * par.soc_poczatkowy_wzgl)

    for t in range(n):
        # Energia dostępna na szynie DC po przetwarzaniu w regulatorze MPPT.
        e_pv_dc = moc_pv_kw[t] * KROK_H * par.sprawnosc_mppt

        # Zapotrzebowanie sprowadzone na stronę stałoprądową. Odbiory
        # stałoprądowe wchodzą bez przetwarzania. Dla odbiorów prądu
        # przemiennego są dwa zapisy i NIE WOLNO ich łączyć:
        #   - podany jawny człon strat toru (straty_toru_kwh):
        #     zapotrzebowanie to energia użyteczna POWIĘKSZONA o ten człon,
        #     bo zawiera on już pobór jałowy i straty zależne od obciążenia
        #     (model P_strat = P_0 + a*P + b*P^2 z loads.py),
        #   - brak członu strat: zapotrzebowanie liczone sprawnością stałą.
        # Użycie obu naraz liczyłoby straty dwa razy.
        if jalowy_podany:
            e_potrzebne_dc = obc_dc[t] + obc_ac[t] + jalowy[t]
        else:
            e_potrzebne_dc = obc_dc[t] + obc_ac[t] / par.sprawnosc_falownika

        # Współczynnik przeliczenia energii z szyny na energię użyteczną
        # odbiorników. Przy ograniczaniu odbiorów przyjęto ograniczanie
        # proporcjonalne na obu szynach, więc niepokryta energia szyny
        # przelicza się średnim współczynnikiem danej godziny.
        wsp_uzyteczny = (
            obciazenie_kwh[t] / e_potrzebne_dc if e_potrzebne_dc > 0.0 else 0.0
        )

        # --- 1. Pokrycie bezpośrednie ------------------------------------
        # Energia z generatora zasila odbiorniki z pominięciem magazynu.
        bezposrednio = min(e_pv_dc, e_potrzebne_dc)
        e_bezposrednie[t] = bezposrednio

        nadwyzka = e_pv_dc - bezposrednio
        niedobor = e_potrzebne_dc - bezposrednio

        # Blokada ładowania: ogniwo wolno ładować dopiero powyżej progu.
        # Rozładowanie dozwolone jest w całym zakresie temperatur pracy.
        ladowanie_dozwolone = t_ogniw[t] > par.temperatura_blokady_C

        # --- 2. Ładowanie magazynu nadwyżką ------------------------------
        if nadwyzka > 0.0:
            if not ladowanie_dozwolone:
                # Energii nie da się przyjąć; regulator ogranicza pobór mocy
                # z generatora. Pozycja raportowana osobno, bo jej przyczyną
                # jest temperatura, a nie stan naładowania.
                e_curt[t] += nadwyzka
                e_blok_pv[t] = nadwyzka
            else:
                wolne_miejsce = par.soc_max_kwh - stan
                przyrost = min(nadwyzka * par.sprawnosc_ladowania, wolne_miejsce)
                stan += przyrost
                e_do_magazynu[t] = przyrost
                # Nadwyżka, której nie da się zmagazynować, jest tracona.
                nieprzyjeta = nadwyzka - przyrost / par.sprawnosc_ladowania
                e_odrzucona[t] = nieprzyjeta
                e_curt[t] += nieprzyjeta

        # --- 3. Rozładowanie magazynu na pokrycie niedoboru ---------------
        if niedobor > 0.0:
            zapas_uzyteczny = (stan - par.soc_min_kwh) * par.sprawnosc_rozladowania
            oddane = min(niedobor, max(zapas_uzyteczny, 0.0))
            stan -= oddane / par.sprawnosc_rozladowania
            e_z_magazynu[t] = oddane
            # Niedobór przeliczony z powrotem na stronę odbiorników.
            e_deficyt[t] = (niedobor - oddane) * wsp_uzyteczny
        wsp_uzyteczne[t] = wsp_uzyteczny

        # --- 4. Dopływ z alternatora -------------------------------------
        # Alternator ładuje magazyn po generatorze: pierwszeństwo ma energia
        # darmowa. Blokada niskotemperaturowa obowiązuje tak samo, ale jest
        # liczona osobno, bo w wariantach projektowych realizuje ją inny
        # element układu niż przy ładowaniu z generatora.
        e_alt_oferowana[t] = alternator[t]
        if alternator[t] > 0.0:
            if not ladowanie_dozwolone:
                e_curt[t] += alternator[t]
                e_blok_alt[t] = alternator[t]
            else:
                wolne_miejsce = par.soc_max_kwh - stan
                przyrost = min(alternator[t] * par.sprawnosc_ladowania,
                               wolne_miejsce)
                stan += przyrost
                e_alternator[t] = przyrost
                nieprzyjeta = alternator[t] - przyrost / par.sprawnosc_ladowania
                e_curt[t] += nieprzyjeta

        # --- 5. Samogrzanie magazynu -------------------------------------
        # Grzałka załącza się poniżej progu blokady i pobiera energię
        # z magazynu, czyli kosztem energii użytecznej. Wariant wyłączony,
        # dopóki nie zostanie dobrany produkt (moc grzałki z jego karty).
        if par.samogrzanie and t_ogniw[t] <= par.temperatura_blokady_C:
            if par.moc_grzalka_w_lub_blad() > 0.0:
                pobor = min(par.moc_grzalka_w_lub_blad() * KROK_H / 1000.0,
                            max(stan - par.soc_min_kwh, 0.0))
                stan -= pobor
                e_grzalka[t] = pobor

        # --- 6. Samorozładowanie -----------------------------------------
        # Ubytek energii zmagazynowanej, niezależny od przepływów. Dolną
        # granicą jest zero, a nie SOC_min: granica SOC_min jest nastawą
        # układu zarządzania magazynem, a samorozładowanie jest zjawiskiem
        # fizycznym i działa także poniżej niej.
        ubytek = min(par.samorozladowanie_kwh_h, stan)
        stan -= ubytek
        e_samorozladowanie[t] = ubytek

        soc[t] = stan

    # Energia faktycznie dostarczona odbiornikom, po obu szynach.
    e_pokryte = (e_bezposrednie + e_z_magazynu) * wsp_uzyteczne
    # Rozdzielenie na pokrycie bezpośrednie i z magazynu: potrzebne do
    # przypisania energii dostarczonej odbiornikom do źródeł (SSR_PV).
    e_pokryte_bezp = e_bezposrednie * wsp_uzyteczne
    e_pokryte_mag = e_z_magazynu * wsp_uzyteczne

    przebiegi = pd.DataFrame(
        {
            "P_pv_kW": moc_pv_kw,
            "E_obciazenie_kWh": obciazenie_kwh,
            "E_obciazenie_dc_kWh": obc_dc,
            "E_obciazenie_ac_kWh": obc_ac,
            "E_falownik_jalowy_kWh": jalowy,
            "E_bezposrednie_kWh": e_bezposrednie,
            "E_do_magazynu_kWh": e_do_magazynu,
            "E_z_magazynu_kWh": e_z_magazynu,
            "E_pokryte_kWh": e_pokryte,
            "E_pokryte_bezposrednio_kWh": e_pokryte_bezp,
            "E_pokryte_z_magazynu_kWh": e_pokryte_mag,
            "E_deficyt_kWh": e_deficyt,
            "E_odrzucona_kWh": e_odrzucona,
            "E_curt_kWh": e_curt,
            "E_alternator_kWh": e_alternator,
            "E_alternator_oferowana_kWh": e_alt_oferowana,
            "E_blokada_pv_kWh": e_blok_pv,
            "E_blokada_alternator_kWh": e_blok_alt,
            "E_grzalka_kWh": e_grzalka,
            "E_samorozladowanie_kWh": e_samorozladowanie,
            "T_ogniw_C": t_ogniw,
            "SoC_kWh": soc,
            "SoC_wzgledny": soc / par.pojemnosc_kwh,
        },
        index=indeks if indeks is not None else pd.RangeIndex(n, name="godzina"),
    )

    # Okres rozbiegowy wyłączony z oceny. Stan naładowania w pierwszej
    # godzinie jest założeniem, a nie wynikiem, więc metryki liczone od
    # początku roku niosą ślad tego założenia. Przebiegi zwracane są
    # w całości; pominięcie dotyczy wyłącznie wskaźników.
    oceniane = przebiegi.iloc[rozbieg_godzin:] if rozbieg_godzin else przebiegi
    wskazniki = oblicz_wskazniki(oceniane, par)
    wskazniki["rozbieg_godzin"] = int(rozbieg_godzin)
    return przebiegi, wskazniki


def oblicz_wskazniki(przebiegi: pd.DataFrame,
                     parametry: ParametryUkladu | None = None) -> dict[str, float]:
    """
    Wyznacza wskaźniki oceny układu na podstawie przebiegów godzinowych.

    Wskaźniki liczone są w UJĘCIU ENERGETYCZNYM, to jest jako iloraz energii,
    a nie liczby godzin. W całej pracy wielkość ta nosi nazwę „energetyczny
    wskaźnik niedoboru zasilania" (LPSP_E), żeby odróżnić ją od ujęcia
    czasowego spotykanego w literaturze:

        LPSP_E = suma(E_deficyt) / suma(E_obciazenie)

    Przy wspólnym magazynie energii z generatora i z alternatora nie da się
    rozdzielić po stronie odbiorników: obie wchodzą do tego samego zasobu.
    Przyjęto regułę przypisania proporcjonalną do udziału źródeł
    w ładowaniu magazynu:

        SSR_PV = (E_bezp,odb + u_PV * E_mag,odb) / E_żąd
        u_PV   = E_lad,PV / (E_lad,PV + E_lad,alt)

    gdzie E_bezp,odb to energia dostarczona odbiornikom bezpośrednio
    z generatora, E_mag,odb — energia dostarczona z magazynu, a E_żąd —
    energia wymagana na zaciskach odbiorników. Wskaźnik mierzony jest po
    stronie odbiorników, więc spełnia SSR_PV <= 1 - LPSP_E; udział
    generatora w samym dopływie na szynę raportowany jest osobno jako
    wielkość pomocnicza.

    Autonomia magazynu wyrażona jest w dobach: energia użyteczna magazynu
    odniesiona do średniego dobowego zapotrzebowania odbiorników.
    """
    e_obciazenie = float(przebiegi["E_obciazenie_kWh"].sum())
    e_pokryte = float(przebiegi["E_pokryte_kWh"].sum())
    e_deficyt = float(przebiegi["E_deficyt_kWh"].sum())

    if e_obciazenie <= 0.0:
        raise ValueError("Sumaryczne zapotrzebowanie wynosi zero — brak podstawy "
                         "do wyznaczenia wskaźników SSR i LPSP.")

    godziny_deficytu = int((przebiegi["E_deficyt_kWh"] > 1e-9).sum())

    def suma(kolumna: str) -> float:
        return (float(przebiegi[kolumna].sum())
                if kolumna in przebiegi.columns else 0.0)

    e_alternator = suma("E_alternator_kWh")
    doby = max(len(przebiegi) * KROK_H / 24.0, 1e-9)
    srednie_dobowe = e_obciazenie / doby

    # Dopływ na szynę, osobno z generatora i z alternatora. Energia
    # zmagazynowana przeliczana jest z powrotem na stronę szyny przez
    # sprawność ładowania, żeby obie składowe były mierzone w tym samym
    # miejscu układu.
    wsp = parametry.sprawnosc_ladowania if parametry is not None else 1.0
    e_pv_szyna = (float(przebiegi["E_bezposrednie_kWh"].sum())
                  + float(przebiegi["E_do_magazynu_kWh"].sum()) / wsp)
    e_alt_szyna = e_alternator / wsp
    doplyw = e_pv_szyna + e_alt_szyna

    # Przypisanie energii oddanej z magazynu do źródeł: proporcjonalnie do
    # ich udziału w ładowaniu, bo magazyn nie rozróżnia pochodzenia energii.
    # Pokrycie bezpośrednie pochodzi w całości z generatora.
    e_lad_pv = float(przebiegi["E_do_magazynu_kWh"].sum())
    e_lad_alt = e_alternator
    udzial_pv_mag = (e_lad_pv / (e_lad_pv + e_lad_alt)
                     if (e_lad_pv + e_lad_alt) > 0.0 else 0.0)
    e_pokryte_pv = (float(przebiegi["E_pokryte_bezposrednio_kWh"].sum())
                    + udzial_pv_mag
                    * float(przebiegi["E_pokryte_z_magazynu_kWh"].sum()))

    # Najdłuższa nieprzerwana przerwa w pokryciu obciążenia, h. Wskaźnik
    # energetyczny nie opisuje ciągłości zasilania, więc liczony jest osobno.
    deficyt_h = (przebiegi["E_deficyt_kWh"] > 1e-9).to_numpy()
    najdluzsza, biezaca = 0, 0
    for jest in deficyt_h:
        biezaca = biezaca + 1 if jest else 0
        najdluzsza = max(najdluzsza, biezaca)

    wynik = {
        "SSR_PV": (e_pokryte_pv / e_obciazenie) if e_obciazenie > 0.0 else 0.0,
        "udzial_PV_w_dopływie": (e_pv_szyna / doplyw) if doplyw > 0.0 else 0.0,
        "udzial_PV_w_ladowaniu": udzial_pv_mag,
        "E_pokryte_PV_kWh": e_pokryte_pv,
        "najdluzsza_przerwa_h": int(najdluzsza),
        "E_pv_szyna_kWh": e_pv_szyna,
        "E_alt_szyna_kWh": e_alt_szyna,
        "SSR": e_pokryte / e_obciazenie,
        "LPSP": e_deficyt / e_obciazenie,
        "E_obciazenie_kWh": e_obciazenie,
        "E_pokryte_kWh": e_pokryte,
        "E_deficyt_kWh": e_deficyt,
        "E_odrzucona_kWh": suma("E_odrzucona_kWh"),
        "E_curt_kWh": suma("E_curt_kWh"),
        "E_alternator_kWh": e_alternator,
        "E_blokada_pv_kWh": suma("E_blokada_pv_kWh"),
        "E_blokada_alternator_kWh": suma("E_blokada_alternator_kWh"),
        "E_samorozladowanie_kWh": suma("E_samorozladowanie_kWh"),
        "E_grzalka_kWh": suma("E_grzalka_kWh"),
        "E_pv_kWh": float(przebiegi["P_pv_kW"].sum()),
        "godziny_deficytu": godziny_deficytu,
        "SoC_koncowy_wzgledny": (float(przebiegi["SoC_wzgledny"].iloc[-1])
                                 if "SoC_wzgledny" in przebiegi.columns
                                 and len(przebiegi) else 0.0),
        "SoC_min_wzgledny": float(przebiegi["SoC_wzgledny"].min()),
        "srednie_dobowe_kWh": srednie_dobowe,
    }
    if "T_ogniw_C" in przebiegi.columns:
        wynik["godziny_blokady"] = int((przebiegi["T_ogniw_C"] <= 0.0).sum())
    return wynik


def kontrola_tozsamosci(przebiegi: pd.DataFrame,
                        parametry: ParametryUkladu | None = None,
                        wskazniki: dict | None = None) -> dict[str, float]:
    """
    Tożsamość przypisania energii dostarczonej odbiornikom.

        E_PV,odb + E_alt,odb + E_def = E_żąd

    gdzie E_żąd to energia wymagana na zaciskach odbiorników, a pierwsze
    dwa człony są jej częściami pokrytymi z generatora i z alternatora.
    Energia oddana z magazynu dzieli się między źródła proporcjonalnie do
    ich udziału w ładowaniu; energia zgromadzona przed początkiem okresu
    podlega temu samemu podziałowi, co jest przyjętym uproszczeniem reguły.

    Zwraca składniki i resztę bezwzględną oraz względną.
    """
    w = (wskazniki if wskazniki is not None
         else oblicz_wskazniki(przebiegi, parametry))
    zadana = float(w["E_obciazenie_kWh"])
    pv = float(w["E_pokryte_PV_kWh"])
    alt = float(w["E_pokryte_kWh"]) - pv
    deficyt = float(w["E_deficyt_kWh"])
    reszta = pv + alt + deficyt - zadana
    return {
        "E_zadana_kWh": zadana,
        "E_pokryte_PV_kWh": pv,
        "E_pokryte_alternator_kWh": alt,
        "E_deficyt_kWh": deficyt,
        "reszta_kWh": reszta,
        "reszta_wzgledna": reszta / zadana if zadana > 0.0 else 0.0,
    }


def kontrola_zamkniecia(przebiegi: pd.DataFrame,
                        parametry: ParametryUkladu,
                        stan_poczatkowy_kwh: float | None = None
                        ) -> dict[str, float]:
    """
    Sprawdza zamknięcie bilansu energii na szynie stałoprądowej.

    Tożsamość, którą musi spełniać każdy przebieg:

        E_PV,dost + E_alt = (E_bezp + E_z_magazynu)
                          + E_str,lad + E_str,roz + E_sd + E_grz
                          + E_curt + dE_mag

    gdzie E_PV,dost to energia generatora po regulatorze, E_alt — energia
    zaoferowana przez alternator, E_str,lad i E_str,roz — straty obiegu
    magazynu, E_sd — samorozładowanie, E_grz — pobór grzałki, E_curt —
    energia nieprzyjęta, a dE_mag — przyrost energii zmagazynowanej.
    Zwraca składniki i resztę; reszta ma być rzędu błędu maszynowego.
    """
    def suma(kolumna: str) -> float:
        return (float(przebiegi[kolumna].sum())
                if kolumna in przebiegi.columns else 0.0)

    e_pv_dost = float((przebiegi["P_pv_kW"] * KROK_H).sum()) * parametry.sprawnosc_mppt
    e_alt_ofiarowana = suma("E_alternator_oferowana_kWh")
    przyrost = suma("E_do_magazynu_kWh") + suma("E_alternator_kWh")
    oddane = suma("E_z_magazynu_kWh")
    straty_lad = przyrost * (1.0 / parametry.sprawnosc_ladowania - 1.0)
    straty_roz = oddane * (1.0 / parametry.sprawnosc_rozladowania - 1.0)
    # Stan początkowy: podany jawnie (gdy sprawdza się wycinek przebiegu,
    # np. okres oceniany) albo wynikający z parametrów całego przebiegu.
    if stan_poczatkowy_kwh is not None:
        stan_poczatkowy = float(stan_poczatkowy_kwh)
    else:
        stan_poczatkowy = (parametry.soc_max_kwh
                           if parametry.soc_poczatkowy_wzgl is None
                           else parametry.pojemnosc_kwh
                           * parametry.soc_poczatkowy_wzgl)
    zmiana_mag = (float(przebiegi["SoC_kWh"].iloc[-1]) - stan_poczatkowy)

    lewa = e_pv_dost + e_alt_ofiarowana
    prawa = (float(przebiegi["E_bezposrednie_kWh"].sum()) + oddane
             + straty_lad + straty_roz + suma("E_samorozladowanie_kWh")
             + suma("E_grzalka_kWh") + suma("E_curt_kWh") + zmiana_mag)
    return {
        "E_pv_dostarczona_kWh": e_pv_dost,
        "E_alt_ofiarowana_kWh": e_alt_ofiarowana,
        "E_strat_ladowania_kWh": straty_lad,
        "E_strat_rozladowania_kWh": straty_roz,
        "zmiana_magazynu_kWh": zmiana_mag,
        "reszta_kWh": lewa - prawa,
    }


def autonomia_doby(parametry: ParametryUkladu,
                   srednie_dobowe_kwh: float) -> float:
    """
    Autonomia magazynu w dobach.

    Iloraz energii użytecznej magazynu i średniego dobowego zapotrzebowania
    odbiorników. Wielkość opisuje magazyn, nie układ: nie uwzględnia dopływu
    z generatora ani z alternatora.
    """
    if srednie_dobowe_kwh <= 0.0:
        raise ValueError("Średnie zapotrzebowanie dobowe wynosi zero.")
    return parametry.pojemnosc_uzyteczna_kwh / srednie_dobowe_kwh


def wskazniki_miesieczne(przebiegi: pd.DataFrame,
                         parametry: ParametryUkladu | None = None) -> pd.DataFrame:
    """
    Energetyczny wskaźnik LPSP i SSR_PV w podziale na miesiące.

    Ocena wymagania W1 prowadzona jest osobno dla każdego miesiąca, bo deficyt
    skupiony w jednym miesiącu daje inny wniosek projektowy niż ten sam deficyt
    rozłożony na cały rok. Agregacja miesięczna prowadzona jest po indeksie
    czasu przebiegów (czas uniwersalny).
    """
    if not isinstance(przebiegi.index, pd.DatetimeIndex):
        raise ValueError("Przebiegi muszą mieć indeks czasu, aby podzielić je "
                         "na miesiące.")
    grupy = przebiegi.groupby(przebiegi.index.month)
    wiersze = []
    for miesiac, grupa in grupy:
        obciazenie = float(grupa["E_obciazenie_kWh"].sum())
        if obciazenie <= 0.0:
            continue
        alternator = (float(grupa["E_alternator_kWh"].sum())
                      if "E_alternator_kWh" in grupa.columns else 0.0)
        wsp = parametry.sprawnosc_ladowania if parametry is not None else 1.0
        pv_szyna = (float(grupa["E_bezposrednie_kWh"].sum())
                    + float(grupa["E_do_magazynu_kWh"].sum()) / wsp)
        alt_szyna = alternator / wsp
        # Korekta o zmianę stanu magazynu w miesiącu: energia, która w nim
        # przybyła albo ubyła, nie trafiła do odbiorników w tym miesiącu.
        zmiana_mag = (float(grupa["SoC_kWh"].iloc[-1])
                      - float(grupa["SoC_kWh"].iloc[0]))
        doplyw = max(pv_szyna + alt_szyna - max(zmiana_mag, 0.0) / wsp, 1e-12)
        wiersze.append({
            "miesiac": int(miesiac),
            "E_obciazenie_kWh": round(obciazenie, 4),
            "E_deficyt_kWh": round(float(grupa["E_deficyt_kWh"].sum()), 4),
            "LPSP": round(float(grupa["E_deficyt_kWh"].sum()) / obciazenie, 5),
            "SSR_PV": round(min(pv_szyna / doplyw, 1.0), 5),
            "zmiana_magazynu_kWh": round(zmiana_mag, 4),
            "godziny_deficytu": int((grupa["E_deficyt_kWh"] > 1e-9).sum()),
        })
    return pd.DataFrame(wiersze)


def wypisz_wskazniki(wskazniki: dict[str, float], etykieta: str = "") -> None:
    """Wypisuje wskaźniki oceny w formie czytelnej dla protokołu obliczeń."""
    naglowek = f"Wskaźniki oceny{f' — {etykieta}' if etykieta else ''}"
    print(f"\n  {naglowek}")
    print(f"    energia wyprodukowana (DC)  : {wskazniki['E_pv_kWh']:8.1f} kWh")
    print(f"    zapotrzebowanie odbiorników : {wskazniki['E_obciazenie_kWh']:8.1f} kWh")
    print(f"    energia dostarczona         : {wskazniki['E_pokryte_kWh']:8.1f} kWh")
    print(f"    energia niedostarczona      : {wskazniki['E_deficyt_kWh']:8.1f} kWh")
    print(f"    nadwyżka niewykorzystana    : {wskazniki['E_odrzucona_kWh']:8.1f} kWh")
    print(f"    SSR_PV                      : {wskazniki['SSR_PV']:8.3f}")
    print(f"    LPSP (energetyczny)         : {wskazniki['LPSP']:8.3f}")
    print(f"    energia nieprzyjęta E_curt  : {wskazniki['E_curt_kWh']:8.1f} kWh")
    print(f"    godziny z deficytem         : {wskazniki['godziny_deficytu']:8d} h")
    print(f"    minimalny stan naładowania  : {wskazniki['SoC_min_wzgledny']:8.3f}")


# ===========================================================================
# Test samodzielny na syntetycznym przebiegu generacji
# ===========================================================================

def _syntetyczna_generacja(moc_szczytowa_kw: float = 0.30) -> np.ndarray:
    """
    Buduje uproszczony roczny przebieg mocy generatora do testu modelu.

    Przebieg jest iloczynem dobowej półfali sinusoidalnej i rocznej modulacji
    kosinusoidalnej odwzorowującej zmienność sezonową w szerokości geograficznej
    Polski. NIE jest to model fizyczny — służy wyłącznie sprawdzeniu poprawności
    pętli bilansowej bez udziału PySAM.
    """
    godziny = np.arange(8760)
    godzina_doby = godziny % 24
    dzien_roku = godziny // 24

    # Dobowa półfala: generacja od 6:00 do 18:00.
    dobowy = np.sin(np.pi * (godzina_doby - 6) / 12.0)
    dobowy = np.clip(dobowy, 0.0, None)

    # Modulacja roczna: maksimum w przesileniu letnim (ok. 172 dnia roku).
    roczny = 0.55 + 0.45 * np.cos(2 * np.pi * (dzien_roku - 172) / 365.0)

    return moc_szczytowa_kw * dobowy * roczny


def main() -> int:
    print("=" * 72)
    print("Test modelu bilansu na syntetycznym przebiegu generacji")
    print("(przebieg zastępczy — nie pochodzi z PySAM)")
    print("=" * 72)

    parametry = ParametryUkladu()
    print(f"  pojemność magazynu      : {parametry.pojemnosc_kwh:.2f} kWh")
    print(f"  pojemność użyteczna     : {parametry.pojemnosc_uzyteczna_kwh:.2f} kWh")
    print(f"  DoD maks.               : {parametry.dod_max:.2f}")
    print(f"  sprawność obiegu        : {parametry.sprawnosc_obiegu:.2f}")
    print(f"  sprawność falownika     : {parametry.sprawnosc_falownika:.2f}")
    print(f"  sprawność MPPT          : {parametry.sprawnosc_mppt:.2f}")

    moc_pv = _syntetyczna_generacja()
    przebiegi, wskazniki = bilans_godzinowy(moc_pv, parametry)

    wypisz_wskazniki(wskazniki, "przebieg syntetyczny")

    # Kontrola zamknięcia bilansu — energia dostarczona i niedostarczona muszą
    # sumować się do zapotrzebowania z dokładnością do błędu numerycznego.
    reszta = abs(
        wskazniki["E_pokryte_kWh"] + wskazniki["E_deficyt_kWh"]
        - wskazniki["E_obciazenie_kWh"]
    )
    print(f"\n  kontrola zamknięcia bilansu : reszta = {reszta:.2e} kWh")
    print(f"  kontrola tożsamości SSR+LPSP: "
          f"{wskazniki['SSR'] + wskazniki['LPSP']:.6f}")

    # --- wpływ przeliczenia strefy czasowej profilu obciążenia ------------
    # Test wykazuje, że przeliczenie czasu lokalnego na UTC nie jest formalnością.
    # Dla profilu płaskiego wynik jest identyczny; dla profilu z maksimum
    # wieczornym przesunięcie o strefę zmienia wskaźniki.
    print("\n" + "-" * 72)
    print("Wpływ bazy czasu profilu obciążenia")
    print("-" * 72)

    indeks_utc = pd.date_range(
        "2019-01-01 00:00", periods=8760, freq="h", tz="UTC"
    )
    profil_wieczorny = profil_dobowy_wieczorny(parametry.zapotrzebowanie_dobowe_kwh)

    for etykieta, profil in [
        ("płaski", profil_obciazenia_stale(24, parametry.zapotrzebowanie_dobowe_kwh)),
        ("z maksimum wieczornym", profil_wieczorny),
    ]:
        # Wariant poprawny: profil zadany lokalnie, przeliczony na UTC.
        _, w_utc = bilans_godzinowy(
            moc_pv, parametry, indeks=indeks_utc, profil_dobowy_local=profil
        )
        # Wariant błędny: ten sam profil przyłożony wprost do osi UTC,
        # bez przeliczenia strefy — pomyłka, którą ten test ma wychwycić.
        bledny = np.tile(profil, 365)
        _, w_bez = bilans_godzinowy(
            moc_pv, parametry, obciazenie_kwh=bledny, indeks=indeks_utc
        )
        print(f"  profil {etykieta}:")
        print(f"    z przeliczeniem na UTC : SSR={w_utc['SSR']:.4f}  "
              f"LPSP={w_utc['LPSP']:.4f}")
        print(f"    bez przeliczenia       : SSR={w_bez['SSR']:.4f}  "
              f"LPSP={w_bez['LPSP']:.4f}")
        print(f"    różnica LPSP           : "
              f"{abs(w_utc['LPSP'] - w_bez['LPSP']):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
