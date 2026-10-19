# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: tests_bilans.py
Opis: testy modelu bilansu, profili obciążeń i formatowania liczb. Testy
      sprawdzają własności, które łatwo zepsuć niezauważenie: rozdział
      szyn prądu stałego i przemiennego, zamknięcie bilansu energii
      magazynu wraz z samorozładowaniem, zgodność sum godzinowych z
      dobowymi przy zmianie czasu oraz zapis liczb według konwencji
      polskiej.

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
import traceback

import numpy as np
import pandas as pd

import battery_balance as bb
import config
import fmt
import loads
from config import LOCATIONS

ROK = 2020
TESTY = []


def test(opis: str):
    """Rejestruje funkcję jako test."""
    def dekorator(f):
        TESTY.append((opis, f))
        return f
    return dekorator


def blisko(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# ===========================================================================
# Bilans magazynu
# ===========================================================================

def _generacja(n: int = 8760, szczyt_kw: float = 0.30) -> np.ndarray:
    """Powtarzalny, deterministyczny szereg generacji do testów."""
    godziny = np.arange(n) % 24
    dobowy = np.clip(np.sin((godziny - 6) / 12.0 * np.pi), 0.0, None)
    sezon = 0.55 + 0.45 * np.cos((np.arange(n) / n * 2 - 1) * np.pi)
    return szczyt_kw * dobowy * sezon


@test("profil zbiorczy AC daje ten sam wynik co profil rozdzielony na szyny")
def t_zgodnosc_zapisow():
    moc = _generacja()
    obc = np.full(moc.size, 1.0 / 24.0)
    stary, w_stary = bb.bilans_godzinowy(moc, obciazenie_kwh=obc)
    nowy, w_nowy = bb.bilans_godzinowy(moc, obciazenie_ac_kwh=obc)
    assert blisko(w_stary["LPSP"], w_nowy["LPSP"], 1e-12), (
        w_stary["LPSP"], w_nowy["LPSP"])


@test("odbiory stałoprądowe dają niższe LPSP niż te same odbiory za falownikiem")
def t_szyna_dc_lepsza():
    moc = _generacja()
    obc = np.full(moc.size, 1.0 / 24.0)
    _, w_ac = bb.bilans_godzinowy(moc, obciazenie_ac_kwh=obc)
    _, w_dc = bb.bilans_godzinowy(moc, obciazenie_dc_kwh=obc)
    assert w_dc["LPSP"] < w_ac["LPSP"], (w_dc["LPSP"], w_ac["LPSP"])


@test("pobór jałowy falownika podnosi LPSP")
def t_jalowy_podnosi_lpsp():
    moc = _generacja()
    obc = np.full(moc.size, 1.0 / 24.0)
    jalowy = np.full(moc.size, 0.022)
    _, bez = bb.bilans_godzinowy(moc, obciazenie_ac_kwh=obc)
    _, z = bb.bilans_godzinowy(moc, obciazenie_ac_kwh=obc,
                               falownik_jalowy_kwh=jalowy)
    assert z["LPSP"] > bez["LPSP"], (z["LPSP"], bez["LPSP"])


@test("podanie obu zapisów profilu naraz jest błędem")
def t_podwojny_zapis():
    moc = _generacja(48)
    obc = np.full(48, 0.04)
    try:
        bb.bilans_godzinowy(moc, obciazenie_kwh=obc, obciazenie_dc_kwh=obc)
    except ValueError:
        return
    raise AssertionError("oczekiwano ValueError przy podwójnym zapisie profilu")


@test("bilans odbiorników zamyka się: pokryte + deficyt = zapotrzebowanie")
def t_zamkniecie_odbiorniki():
    moc = _generacja()
    obc = np.full(moc.size, 1.0 / 24.0)
    _, w = bb.bilans_godzinowy(moc, obciazenie_dc_kwh=obc * 0.4,
                               obciazenie_ac_kwh=obc * 0.6)
    reszta = abs(w["E_pokryte_kWh"] + w["E_deficyt_kWh"] - w["E_obciazenie_kWh"])
    assert reszta < 1e-6, reszta


@test("bilans magazynu zamyka się wraz z samorozładowaniem")
def t_zamkniecie_magazynu():
    moc = _generacja()
    obc = np.full(moc.size, 1.0 / 24.0)
    par = bb.ParametryUkladu()
    przebiegi, _ = bb.bilans_godzinowy(moc, par, obciazenie_dc_kwh=obc)
    # Przyrost stanu naładowania = energia zmagazynowana - energia oddana
    # (z uwzględnieniem sprawności rozładowania) - samorozładowanie.
    zmiana = float(przebiegi["SoC_kWh"].iloc[-1]) - par.soc_max_kwh
    oczekiwana = (przebiegi["E_do_magazynu_kWh"].sum()
                  - przebiegi["E_z_magazynu_kWh"].sum() / par.sprawnosc_rozladowania
                  - przebiegi["E_samorozladowanie_kWh"].sum())
    assert blisko(zmiana, oczekiwana, 1e-6), (zmiana, oczekiwana)


@test("samorozładowanie zgadza się z wartością z karty ogniwa")
def t_samorozladowanie():
    par = bb.ParametryUkladu()
    n = int(round(bb.GODZIN_W_MIESIACU))
    # Bez generacji i praktycznie bez odbiorów ubytek po miesiącu ma
    # odpowiadać deklarowanemu procentowi pojemności znamionowej. Odbiór
    # zerowy nie jest dopuszczalny (wskaźniki SSR i LPSP nie miałyby
    # mianownika), więc przyjęto pobór pomijalny wobec samorozładowania.
    znikomy = np.full(n, 1e-9)
    przebiegi, _ = bb.bilans_godzinowy(np.zeros(n), par,
                                       obciazenie_dc_kwh=znikomy)
    ubytek = par.soc_max_kwh - float(przebiegi["SoC_kWh"].iloc[-1])
    oczekiwany = par.pojemnosc_kwh * par.samorozladowanie_pct_mies / 100.0
    assert abs(ubytek - oczekiwany) / oczekiwany < 0.01, (ubytek, oczekiwany)


@test("D.2: energia nieprzyjęta trafia do E_curt, a bilans magazynu się domyka")
def t_curt_domkniecie():
    # Mała pojemność i duża generacja wymuszają ograniczanie nadwyżki.
    par = bb.ParametryUkladu(pojemnosc_kwh=0.2)
    moc = _generacja()
    obc = np.full(moc.size, 0.2 / 24.0)
    przebiegi, wskazniki = bb.bilans_godzinowy(moc, par, obciazenie_dc_kwh=obc)
    assert wskazniki["E_curt_kWh"] > 0.0, wskazniki["E_curt_kWh"]
    # Energia, która weszła na szynę, musi się rozejść bez reszty:
    # pokrycie bezpośrednie + do magazynu (po sprawności) + nieprzyjęta.
    e_pv_dc = float((przebiegi["P_pv_kW"] * bb.KROK_H).sum()) * par.sprawnosc_mppt
    rozchod = (float(przebiegi["E_bezposrednie_kWh"].sum())
               + float(przebiegi["E_do_magazynu_kWh"].sum()) / par.sprawnosc_ladowania
               + float(przebiegi["E_curt_kWh"].sum()))
    assert blisko(e_pv_dc, rozchod, 1e-6), (e_pv_dc, rozchod)


@test("D.3: stan naładowania nie schodzi poniżej SOC_min")
def t_soc_min():
    par = bb.ParametryUkladu()
    # Obciążenie bez pokrycia: magazyn ma się zatrzymać na dolnej granicy.
    n = 24 * 30
    przebiegi, _ = bb.bilans_godzinowy(
        np.zeros(n), par, obciazenie_dc_kwh=np.full(n, 0.05))
    minimum = float(przebiegi["SoC_kWh"].min())
    # Poniżej granicy schodzi wyłącznie samorozładowanie, nie pobór odbiorników.
    ubytek = float(przebiegi["E_samorozladowanie_kWh"].sum())
    assert minimum >= par.soc_min_kwh - ubytek - 1e-9, (minimum, par.soc_min_kwh)
    assert float(przebiegi["E_deficyt_kWh"].sum()) > 0.0


@test("D.5: poniżej zera ładowanie jest zablokowane osobno dla PV i alternatora")
def t_blokada_temperaturowa():
    par = bb.ParametryUkladu()
    n = 48
    zimno = np.full(n, -5.0)
    ciepło = np.full(n, 10.0)

    # (a) Nadwyżka z generatora przy zimnym magazynie.
    _, pv_zimno = bb.bilans_godzinowy(
        np.full(n, 0.1), par, obciazenie_dc_kwh=np.full(n, 0.001),
        temperatura_ogniw_C=zimno)
    assert pv_zimno["E_blokada_pv_kWh"] > 0.0
    assert blisko(float(pv_zimno["E_curt_kWh"]), float(pv_zimno["E_blokada_pv_kWh"]),
                  1e-9)

    # (b) Dopływ z alternatora przy magazynie rozładowywanym obciążeniem.
    # Magazyn startuje pełny, więc bez poboru nie byłoby wolnego miejsca
    # i test mierzyłby zapełnienie zamiast blokady.
    obc = np.full(n, 0.2)
    alt = np.full(n, 0.05)
    _, alt_zimno = bb.bilans_godzinowy(
        np.zeros(n), par, obciazenie_dc_kwh=obc,
        doplyw_alternatora_kwh=alt, temperatura_ogniw_C=zimno)
    _, alt_cieplo = bb.bilans_godzinowy(
        np.zeros(n), par, obciazenie_dc_kwh=obc,
        doplyw_alternatora_kwh=alt, temperatura_ogniw_C=ciepło)

    assert alt_zimno["E_blokada_alternator_kWh"] > 0.0
    assert blisko(alt_zimno["E_alternator_kWh"], 0.0, 1e-12)
    assert alt_cieplo["E_blokada_alternator_kWh"] == 0.0
    assert alt_cieplo["E_alternator_kWh"] > 0.0
    # Blokada przy alternatorze podnosi deficyt: energia, której nie przyjęto,
    # nie pokryje odbiorników.
    assert alt_zimno["LPSP"] > alt_cieplo["LPSP"]


@test("D.6: temperatura ogniw dąży do temperatury otoczenia ze stałą czasową")
def t_temperatura_ogniw():
    par = bb.ParametryUkladu(tau_bat_h=3.0)
    n = 48
    powietrze = np.full(n, 10.0)
    wynik = bb.temperatura_ogniw(powietrze, parametry=par, t_poczatkowa_C=0.0)
    assert wynik[0] < wynik[1] < wynik[-1], wynik[:3]
    assert abs(wynik[-1] - 10.0) < 0.1, wynik[-1]
    # Po jednej stałej czasowej różnica maleje do ok. 1/e wartości początkowej.
    po_tau = wynik[2]            # 3 godziny przy kroku 1 h
    assert 0.25 < (10.0 - po_tau) / 10.0 < 0.35, po_tau

    # Ogrzewanie postojowe podnosi temperaturę w zabudowie do nastawy.
    zimno = np.full(n, -10.0)
    bez = bb.temperatura_ogniw(zimno, parametry=par, t_poczatkowa_C=-10.0)
    z_ogrzewaniem = bb.temperatura_ogniw(
        zimno, ogrzewanie_zal=np.ones(n, dtype=bool), parametry=par,
        t_poczatkowa_C=-10.0)
    assert z_ogrzewaniem[-1] > bez[-1] + 15.0, (z_ogrzewaniem[-1], bez[-1])


@test("D.7: energetyczny wskaźnik LPSP liczony miesięcznie zgadza się z rocznym")
def t_lpsp_miesieczny():
    par = bb.ParametryUkladu()
    indeks = pd.date_range("2019-01-01", periods=8760, freq="h", tz="UTC")
    moc = _generacja()
    obc = np.full(moc.size, 1.0 / 24.0)
    przebiegi, wskazniki = bb.bilans_godzinowy(
        moc, par, obciazenie_dc_kwh=obc, indeks=indeks)
    miesieczne = bb.wskazniki_miesieczne(przebiegi)
    assert len(miesieczne) == 12, len(miesieczne)
    # Suma energii z miesięcy musi dać energię roczną.
    assert blisko(float(miesieczne["E_deficyt_kWh"].sum()),
                  wskazniki["E_deficyt_kWh"], 1e-3)
    # Miesiąc najgorszy ma wskaźnik nie mniejszy od rocznego.
    assert float(miesieczne["LPSP"].max()) >= wskazniki["LPSP"]


@test("D.8: przesunięcie okna SOC przy tej samej szerokości nie zmienia wyniku")
def t_okno_soc():
    moc = _generacja()
    obc = np.full(moc.size, 1.0 / 24.0)
    _, karta = bb.bilans_godzinowy(
        moc, bb.ParametryUkladu(soc_min_wzgl=0.10, soc_max_wzgl=0.90),
        obciazenie_dc_kwh=obc)
    _, dawne = bb.bilans_godzinowy(
        moc, bb.ParametryUkladu(soc_min_wzgl=0.20, soc_max_wzgl=1.00),
        obciazenie_dc_kwh=obc)
    assert blisko(bb.ParametryUkladu(soc_min_wzgl=0.10, soc_max_wzgl=0.90).dod_max,
                  bb.ParametryUkladu(soc_min_wzgl=0.20, soc_max_wzgl=1.00).dod_max,
                  1e-12)
    # Model zależy wyłącznie od ZAPASU nad dolną granicą i od WOLNEGO MIEJSCA
    # pod górną. Oba są przy przesunięciu okna identyczne, więc wskaźniki też.
    # Zapisane jako test, bo wcześniej różnicę przypisywano położeniu okna —
    # a brała się ona z czego innego (dodania samorozładowania).
    assert blisko(karta["LPSP"], dawne["LPSP"], 1e-12), (karta["LPSP"], dawne["LPSP"])
    assert blisko(karta["E_curt_kWh"], dawne["E_curt_kWh"], 1e-9)
    # Różnicę daje dopiero zmiana SZEROKOŚCI okna.
    _, wezsze = bb.bilans_godzinowy(
        moc, bb.ParametryUkladu(soc_min_wzgl=0.20, soc_max_wzgl=0.90),
        obciazenie_dc_kwh=obc)
    assert wezsze["LPSP"] > karta["LPSP"], (wezsze["LPSP"], karta["LPSP"])


@test("D.9: niezgodny indeks czasu przerywa obliczenie, zamiast je wyrównać")
def t_indeks_29_lutego():
    pelny = pd.date_range("2020-01-01", periods=8784, freq="h", tz="UTC")
    bez_29 = pelny[~((pelny.month == 2) & (pelny.day == 29))]
    bb.sprawdz_zgodnosc_indeksu(bez_29, bez_29)      # zgodne — bez wyjątku
    for a, b in ((pelny, bez_29), (bez_29, pelny)):
        try:
            bb.sprawdz_zgodnosc_indeksu(a, b)
        except ValueError:
            continue
        raise AssertionError("oczekiwano ValueError przy niezgodnym indeksie")
    # Ta sama długość, inne znaczniki: też ma przerwać.
    przesuniety = bez_29 + pd.Timedelta(hours=1)
    try:
        bb.sprawdz_zgodnosc_indeksu(bez_29, przesuniety)
    except ValueError:
        return
    raise AssertionError("oczekiwano ValueError przy przesuniętym indeksie")


@test("C5: model strat odtwarza krzywą sprawności z instrukcji falownika")
def t_krzywa_falownika():
    # Punkty odczytane z krzywej w instrukcji (moc wyjściowa, sprawność).
    punkty = ((223.0, 0.849), (400.0, 0.878), (900.0, 0.895), (1800.0, 0.875))
    for moc, sprawnosc in punkty:
        model = float(loads.falownik_sprawnosc(moc))
        assert abs(model - sprawnosc) < 0.02, (moc, model, sprawnosc)
    # Poniżej zakresu pomiarowego rozstrzyga człon stały.
    assert abs(float(loads.falownik_straty_w(30.0)) - 23.7) < 0.2


@test("B2: SSR_PV to udział generatora w dopływie na szynę")
def t_ssr_proporcjonalny():
    par = bb.ParametryUkladu()
    moc = _generacja()
    obc = np.full(moc.size, 1.0 / 24.0)
    alt = np.zeros(moc.size)
    alt[::72] = 0.2

    _, bez_alt = bb.bilans_godzinowy(moc, par, obciazenie_dc_kwh=obc)
    _, z_alt = bb.bilans_godzinowy(moc, par, obciazenie_dc_kwh=obc,
                                   doplyw_alternatora_kwh=alt)
    # Bez alternatora cały dopływ pochodzi z generatora.
    assert blisko(bez_alt["SSR_PV"], 1.0, 1e-9), bez_alt["SSR_PV"]
    # Z alternatorem udział maleje i równa się ilorazowi energii na szynie.
    oczekiwany = (z_alt["E_pv_szyna_kWh"]
                  / (z_alt["E_pv_szyna_kWh"] + z_alt["E_alt_szyna_kWh"]))
    assert blisko(z_alt["SSR_PV"], oczekiwany, 1e-12)
    assert z_alt["SSR_PV"] < 1.0


@test("B4: bilans energii na szynie zamyka się ze wszystkimi członami")
def t_zamkniecie_pelne():
    par = bb.ParametryUkladu()
    moc = _generacja()
    obc = np.full(moc.size, 1.0 / 24.0)
    alt = np.zeros(moc.size)
    alt[::72] = 0.2
    zimno = np.where(np.arange(moc.size) % 8760 < 1500, -5.0, 10.0)
    przebiegi, _ = bb.bilans_godzinowy(
        moc, par, obciazenie_dc_kwh=obc, doplyw_alternatora_kwh=alt,
        temperatura_ogniw_C=zimno)
    kontrola = bb.kontrola_zamkniecia(przebiegi, par)
    # Tolerancja 1e-9 kWh, czyli mikrowatogodziny — rząd błędu maszynowego
    # przy sumowaniu 8760 wyrazów.
    assert abs(kontrola["reszta_kWh"]) < 1e-9, kontrola["reszta_kWh"]


# ===========================================================================
# Profile obciążeń
# ===========================================================================

def _profil(scenariusz: str = "A"):
    return loads.zbuduj_profil(LOCATIONS[loads.LOKALIZACJA_ODNIESIENIA],
                               ROK, scenariusz)


@test("suma energii godzinowej równa się sumie energii dobowej")
def t_suma_godzinowa_dobowa():
    for scen in ("A", "B", "C"):
        ramka = _profil(scen)
        kolumny = [k for k in ramka.columns
                   if k not in ("DC", "AC", "energia użyteczna AC",
                                "pobór falownika z szyny DC")]
        godzinowa = float(ramka[kolumny].to_numpy().sum())
        dobowa = float(loads.energia_dobowa(ramka)[kolumny].to_numpy().sum())
        assert blisko(godzinowa, dobowa, 1e-6), (scen, godzinowa, dobowa)


@test("zmiana czasu: doba 23-godzinna i 25-godzinna liczone poprawnie")
def t_zmiana_czasu():
    ramka = _profil("A")
    lokalny = ramka.index.tz_convert(config.LOCAL_TZ)
    liczba_godzin = pd.Series(1, index=lokalny.date).groupby(level=0).sum()
    doby = {str(d): int(n) for d, n in liczba_godzin.items()}
    # Doba przejścia na czas letni ma 23 godziny, powrotu na zimowy -- 25.
    # Dwie doby brzegowe roku też mają 23 godziny, bo szereg zaczyna się
    # o północy czasu uniwersalnego, czyli o pierwszej czasu lokalnego.
    assert doby["2020-03-29"] == 23, doby["2020-03-29"]
    assert doby["2020-10-25"] == 25, doby["2020-10-25"]
    # Pozostałe doby niepełne mają znane przyczyny i nie wolno ich pomylić
    # ze skutkiem zmiany czasu:
    #   - brzegi roku: szereg zaczyna się o północy czasu uniwersalnego,
    #     więc pierwsza doba lokalna ma 23 godziny, a ostatnia przechodzi
    #     jedną godziną na 1 stycznia następnego roku,
    #   - 29 lutego usuwany jest przy budowie pliku zasobu, aby każdy rok
    #     miał 8760 wierszy (make_sam_resource.write_resource_file).
    znane = {"2020-01-01", "2021-01-01", "2020-02-29", "2020-03-01",
             "2020-03-29", "2020-10-25"}
    nieoczekiwane = {d for d, n in doby.items() if n != 24 and d not in znane}
    assert not nieoczekiwane, nieoczekiwane
    assert sum(doby.values()) == 8760, sum(doby.values())
    krotka, dluga = pd.Index(["2020-03-29"]), pd.Index(["2020-10-25"])
    # Doba krótsza ma mniej godzin poboru stałego (pobory własne), dłuższa
    # więcej — energia dobowa nie jest stała, i tak ma być.
    wlasne = ramka["pobory własne"].groupby(lokalny.date.astype(str)).sum()
    jednostka = loads.pobory_wlasne_w()
    assert blisko(float(wlasne[krotka[0]]), 23 * jednostka, 1e-6)
    assert blisko(float(wlasne[dluga[0]]), 25 * jednostka, 1e-6)


@test("chłodziarka nie pobiera energii, gdy otoczenie nie jest cieplejsze od wnętrza")
def t_chlodziarka_zero():
    czynna = np.ones(5, dtype=bool)
    t_a = np.array([-10.0, 0.0, 3.9, 4.0, 4.1])
    wynik = loads.chlodziarka_wh(t_a, czynna)
    assert np.all(wynik[:4] == 0.0), wynik
    assert wynik[4] > 0.0, wynik


@test("pobory własne płyną całą dobę, także w dobach bez użytkowania")
def t_pobory_wlasne():
    ramka = _profil("B")
    lokalny = ramka.index.tz_convert(config.LOCAL_TZ)
    czynne_miesiace = np.isin(lokalny.month.to_numpy(),
                              list(loads.SCENARIUSZE["B"]["miesiace"]))
    wlasne = ramka["pobory własne"].to_numpy()
    assert np.all(wlasne[czynne_miesiace] > 0.0)
    assert np.all(wlasne[~czynne_miesiace] == 0.0)
    # Doba bez wyjazdu ma mieć sam pobór własny, a nie zero.
    dobowe = ramka.loc[czynne_miesiace].groupby(
        lokalny[czynne_miesiace].date).sum()
    uzytkowe = dobowe[["chłodziarka", "oświetlenie LED i USB"]].sum(axis=1)
    bez_wyjazdu = dobowe.loc[uzytkowe == 0.0]
    assert not bez_wyjazdu.empty
    assert np.all(bez_wyjazdu["pobory własne"].to_numpy() > 0.0)


# ===========================================================================
# Formatowanie liczb
# ===========================================================================

@test("formater liczb: przecinek dziesiętny i spacja jako separator tysięcy")
def t_formater():
    assert fmt.pl(1105.5, 1) == "1105,5", fmt.pl(1105.5, 1)
    assert fmt.pl(12345.67, 2) == "12 345,67", fmt.pl(12345.67, 2)
    assert fmt.pl(1234567, 0) == "1 234 567", fmt.pl(1234567, 0)
    assert fmt.pl(-0.26, 2) == "−0,26", fmt.pl(-0.26, 2)
    assert fmt.pl(871, 0) == "871"
    assert fmt.pl_tex(12345.6, 1) == r"12\,345,6", fmt.pl_tex(12345.6, 1)
    # Na wejściu makr siunitx separatorem dziesiętnym jest kropka.
    assert fmt.qty(1105.5, r"\kilo\watt\hour", 1) == r"\qty{1105.5}{\kilo\watt\hour}"
    assert fmt.num(0.9, 2) == r"\num{0.90}"
    try:
        fmt.pl(float("nan"))
    except ValueError:
        return
    raise AssertionError("oczekiwano ValueError dla wartości brakującej")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 78)
    print("Testy modelu bilansu, profili obciążeń i formatowania liczb")
    print("=" * 78)
    bledy = 0
    for opis, funkcja in TESTY:
        try:
            funkcja()
            print("  [OK]    %s" % opis)
        except Exception:
            bledy += 1
            print("  [BŁĄD]  %s" % opis)
            traceback.print_exc()
    print("-" * 78)
    print("przeszło %d z %d" % (len(TESTY) - bledy, len(TESTY)))
    return 1 if bledy else 0


if __name__ == "__main__":
    sys.exit(main())
