# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: run_pv.py
Opis: model generacji fotowoltaicznej w module PySAM Pvsamv1. Skrypt
      realizuje pierwszą część podziału odpowiedzialności przyjętego w
      rozdz. 6.1: PySAM liczy **wyłącznie generację** (transpozycja
      promieniowania na płaszczyznę modułu, temperatura ogniwa,
      modyfikator kąta padania, straty stałoprądowe), natomiast bilans
      magazynu, wkład alternatora oraz wskaźniki SSR i LPSP wyznaczane
      są po stronie Pythona (moduł ``battery_balance``).

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
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

# Import PySAM jest opakowany, ponieważ pakiet nrel-pysam nie jest dostępny dla
# każdej wersji interpretera (patrz requirements.txt). Dzięki temu pozostałe
# moduły potoku dają się zaimportować i przetestować także bez PySAM.
try:
    import PySAM
    import PySAM.Pvsamv1 as Pvsamv1

    PYSAM_DOSTEPNY = True
    _BLAD_IMPORTU: Exception | None = None
except ImportError as e:  # brak pakietu albo brak koła dla tej wersji Pythona
    PYSAM_DOSTEPNY = False
    _BLAD_IMPORTU = e


# ===========================================================================
# 1. Parametry projektowe generatora PV
# ===========================================================================

# Obwiednię montażową wyznacza platforma bagażowa Front Runner KRVC006T
# (1762 x 1165 mm), a nie rozstaw relingów. Moduł bazowy 1757 x 1134 mm mieści
# się w niej z zapasem 5 mm na długości i 31 mm na szerokości; zapas długości
# jest na tyle mały, że wymaga sprawdzenia pomiarem (patrz config.py).
# Pole PV tworzy jeden moduł.
LICZBA_MODULOW_W_LANCUCHU = 1
LICZBA_LANCUCHOW = 1

# Kąt pochylenia równy 2 stopnie odpowiada praktycznie płaskiemu montażowi na
# dachu pojazdu; niewielkie pochylenie zapewnia spływ wody i ogranicza osadzanie
# się zanieczyszczeń. Azymut 180 stopni = kierunek południowy (konwencja SAM).
POCHYLENIE_DEG = 2.0
AZYMUT_DEG = 180.0

# Model temperatury ogniwa — montaż na relingach ze szczeliną wentylacyjną.
# Enumeracja SAM dla parametru "standoff" (odstęp modułu od poszycia):
#   0 = montaż na gruncie lub na stelażu wolnostojącym
#   1 = powyżej 3,5 cala          2 = 2,5-3,5 cala
#   3 = 1,5-2,5 cala              4 = 0,5-1,5 cala
#   5 = poniżej 0,5 cala          6 = zabudowa zintegrowana (BIPV)
# Szczelina pod modułem mocowanym na relingach wynosi ok. 40-60 mm, tj. ok.
# 1,6-2,4 cala, co odpowiada wartości 3. Parametr jest jednym z kandydatów do
# analizy wrażliwości OAT w rozdz. 6.4 — przy zabudowie bez szczeliny (wartość
# 5) temperatura ogniwa rośnie o kilkanaście kelwinów, a uzysk maleje.
STANDOFF_MODEL_CIEPLNY = 1

# Wysokość montażu w enumeracji SAM: 0 = jedna kondygnacja (do 3 m nad gruntem).
WYSOKOSC_MONTAZU = 0


# ---------------------------------------------------------------------------
# Straty stałoprądowe przyjęte przez autora
# ---------------------------------------------------------------------------
# Wszystkie trzy wartości należą do klasy "przyjęta" (CLAUDE.md D) i wchodzą do
# analizy wrażliwości OAT w rozdz. 6.4.
#
# Niedopasowanie. Pole tworzy jeden moduł, więc niedopasowanie MIĘDZYmodułowe
# nie występuje. Przyjęte 2 % obejmuje niedopasowanie ogniw w module, tolerancję
# mocy i nierównomierne zabrudzenie powierzchni — składniki, których model
# jednodiodowy pojedynczego modułu nie odwzorowuje. Dla pola wielomodułowego
# (klasa 400-430 Wp z dwóch modułów) wartość trzeba będzie przeliczyć.
STRATA_NIEDOPASOWANIE_PCT = 2.0

# Spadek początkowy mocy (LID — degradacja indukowana światłem),
# nieodwracalny, zachodzi w pierwszych tygodniach eksploatacji. Wartość
# zależy od technologii ogniwa i jest przypisana OSOBNO KAŻDEMU MODUŁOWI
# (klucz „lid_pct" w słowniku modułu), a nie jedna dla całego potoku:
# moduł bazowy ma ogniwa typu n (0,5 %), moduł porównawczy ogniwa typu p
# z pasywacją tylną (1,5 %). Zakres analizy wrażliwości: 0,5-1,5 %.
#
# Spadek początkowy to NIE jest starzenie modułu. Degradacji rocznej model
# nie obejmuje — opisuje układ nowy (ograniczenie w 4.7).
STRATA_LID_OAT_PCT = (0.5, 1.5)


def strata_lid_pct(modul: dict | None = None) -> float:
    """Spadek początkowy mocy dla zadanego modułu, w procentach."""
    m = modul if modul is not None else MODUL_KATALOG
    return float(m["lid_pct"])

# Zabrudzenie. Dach pojazdu w trasie brudzi się szybciej niż instalacja
# stacjonarna, a pochylenie 2 stopnie prawie nie wspomaga samooczyszczania.
# Przyjęto 3 % przez cały rok; zakres OAT 2-5 %.
STRATA_ZABRUDZENIE_PCT = 3.0
STRATA_ZABRUDZENIE_OAT_PCT = (2.0, 5.0)

# Zacienienie. Wariant bazowy zakłada postój w miejscu nasłonecznionym,
# a moduł leży na platformie nad relingami i anteną, więc cień własny nie
# pada na ogniwa w godzinach o istotnej generacji. Stąd wartość bazowa zero.
# Zakres analizy wrażliwości (5 i 10 %) to wartości PRZYJĘTE: odpowiadają
# postojowi w częściowym cieniu, którego geometrii model nie odwzorowuje.
# Wielkość wchodzi jako stały uszczerbek mocy stałoprądowej, a nie jako
# geometria cienia — model Pvsamv1 liczy zacienienie własne pola, którego
# tu nie ma (pole tworzy jeden moduł).
# Współczynnik odbicia podłoża (wartość przyjęta; typowa dla terenu
# trawiastego i betonu). Wchodzi do rachunku wyłącznie przy module
# pochylonym.
ALBEDO = 0.2

STRATA_ZACIENIENIE_PCT = 0.0
STRATA_ZACIENIENIE_OAT_PCT = (0.0, 5.0, 10.0)


# ---------------------------------------------------------------------------
# Dane katalogowe modułu odniesienia
# ---------------------------------------------------------------------------
# Wszystkie wartości pochodzą z karty katalogowej producenta (klasa: katalogowa).
# Moduł generyczny, używany w pierwszym przebiegu potoku, został usunięty:
# jego współczynnik temperaturowy nie miał pokrycia w żadnej karcie.
#
# Przy danych katalogowych SAM sam rozwiązuje układ równań modelu
# jednodiodowego (tzw. model 6-parametrowy, De Soto i in. 2006), zamiast
# odczytywać gotowe współczynniki — stąd dwie ścieżki konfiguracji poniżej.
#
# Moduł bazowy: AIKO Neostar 2P AIKO-A470-MAH54Mw, karta V4.1_202510.
# Ogniwa typu n ABC (tylnokontaktowe), 108 ogniw połowkowych w układzie 6 x 18,
# 3 diody obejściowe. Do modelu jednodiodowego wchodzi liczba ogniw POŁĄCZONYCH
# SZEREGOWO, czyli 54: moduł połowkowy tworzą dwie równoległe połowy po 54
# ogniwa (41,18 V / 54 = 0,76 V na ogniwo, wartość typowa dla ABC; przy 108
# ogniwach szeregowo wyszłoby 0,38 V, co jest fizycznie niemożliwe).
MODUL_KATALOG = {
    "nazwa": "AIKO Neostar 2P AIKO-A470-MAH54Mw",
    "p_mp": 470.0,      # moc w punkcie mocy maksymalnej, W (STC)
    "v_mp": 34.74,      # napięcie w punkcie mocy maksymalnej, V
    "i_mp": 13.54,      # prąd w punkcie mocy maksymalnej, A
    "v_oc": 41.18,      # napięcie obwodu otwartego, V
    "i_sc": 14.32,      # prąd zwarcia, A
    "n_ogniw": 54,      # liczba ogniw połączonych szeregowo (108 połowkowych)
    "powierzchnia": 1.9924,  # powierzchnia modułu, m^2 (1757 x 1134 mm)
    # NOCT nie jest podana w karcie wprost — karta podaje tylko warunki NOCT
    # (800 W/m^2, 20 stC, 1 m/s) i moc w tych warunkach. Wartość odtworzono
    # z P(NOCT) = P_STC * 0,8 * (1 + gamma * (T - 25)) i wynosi 47,5 stC
    # (klasa: OBLICZONA, nie katalogowa). Kontrola z napięcia jałowego daje
    # 50,3 stC — różnica bierze się z logarytmicznej zależności Uoc od
    # natężenia, pominiętej w tym rachunku. Zakres OAT: 45-51 stC, z górną
    # granicą obejmującą tę drugą estymatę.
    "t_noct": 47.5,
    # Współczynniki temperaturowe. SAM oczekuje alfa w A/K, beta w V/K,
    # gamma w %/K — przeliczenie z wartości względnych z karty
    # (+0,05 %/stC dla Isc, -0,22 %/stC dla Uoc, -0,26 %/stC dla Pmax).
    "alpha_isc": 0.0005 * 14.32,    # A/K
    "beta_voc": -0.0022 * 41.18,    # V/K
    "gamma_pmp": -0.26,             # %/K
    "technologia": 0,   # 0 = monokrystaliczne (enumeracja SAM)
    # Ogniwa typu n (ABC) — spadek początkowy mniejszy niż dla ogniw typu p.
    "lid_pct": 0.5,
}

# Moduł porównawczy: Victron BlueSolar SPM042855603, 285 W, 1550 x 880 x 35 mm,
# ogniwa monokrystaliczne typu p, 56 ogniw pełnych połączonych szeregowo.
# Karta „current models" (Datasheet-BlueSolar-Monocrystalline-Panels-current-
# models-EN-.pdf, s. 1).
#
# NOCT — WARTOŚĆ PRZYJĘTA, NIE KATALOGOWA. Karta Victrona nie podaje ani
# nominalnej temperatury pracy ogniwa, ani danych w warunkach NOCT, więc nie
# da się jej nawet odtworzyć rachunkiem (inaczej niż dla modułu AIKO).
# Do porównania technologii przyjęto tę samą wartość co dla modułu bazowego:
# wspólne NOCT izoluje różnice parametrów elektrycznych obu modułów od różnicy
# w modelu cieplnym, której nie da się udokumentować. Zakres OAT jest wspólny
# dla obu modułów (45-51 stC).
NOCT_WSPOLNE_C = 47.5
NOCT_OAT_C = (45.0, 51.0)

MODUL_POROWNAWCZY = {
    "nazwa": "Victron BlueSolar SPM042855603 (wariant porównawczy)",
    "p_mp": 285.0,
    "v_mp": 33.4,
    "i_mp": 8.63,
    "v_oc": 38.9,
    "i_sc": 9.08,
    "n_ogniw": 56,          # ogniwa pełne, nie połowkowe
    "powierzchnia": 1.364,  # 1550 x 880 mm
    "t_noct": NOCT_WSPOLNE_C,   # PRZYJĘTA — patrz komentarz wyżej
    "alpha_isc": 0.0004 * 9.08,     # A/K (+0,04 %/stC)
    "beta_voc": -0.0035 * 38.9,     # V/K (-0,35 %/stC)
    "gamma_pmp": -0.45,             # %/K
    "technologia": 0,   # 0 = monokrystaliczne (enumeracja SAM)
    # Ogniwa typu p z pasywacją tylną — spadek początkowy większy.
    "lid_pct": 1.5,
}


# ---------------------------------------------------------------------------
# Falownik zastępczy
# ---------------------------------------------------------------------------
# Pvsamv1 jest modelem systemu przyłączonego do sieci i zawsze wymaga falownika.
# W badanym układzie off-grid rolę tę pełni przetwornica Xantrex Sine Wave 1800i
# modelowana po stronie Pythona, dlatego falownik zadany w SAM jest wyłącznie
# elementem formalnym: dobrano go z zapasem mocy, aby nie ograniczał (nie
# obcinał) mocy stałoprądowej, która jest jedyną wielkością odczytywaną z SAM.
FALOWNIK_ZASTEPCZY = {
    "moc_ac": 600.0,        # W — z zapasem względem 470 Wp, brak obcinania
    "sprawnosc": 92.0,      # % — zgodnie z eta_inv przyjętym w rozdz. 4.7
    "pobor_nocny": 0.5,     # W
    "pobor_wlasny": 2.0,    # W
    "u_dc_nom": 32.3,       # V — napięcie MPP pojedynczego modułu
    "u_dc_max": 60.0,       # V
    "mppt_u_min": 12.0,     # V — okno napięciowe MPPT
    "mppt_u_max": 60.0,     # V
}


# Nazwy konfiguracji domyślnych Pvsamv1 próbowane po kolei. Konfiguracja
# domyślna dostarcza komplet kilkuset parametrów modelu; nadpisujemy z niej
# tylko te, które wynikają z projektu. "FlatPlatePVNone" to wariant bez modelu
# finansowego — jedyny potrzebny, skoro analiza kosztowa jest uproszczona
# i prowadzona osobno (rozdz. 7.5).
KONFIGURACJE_DOMYSLNE = (
    "FlatPlatePVNone",
    "FlatPlatePVSingleOwner",
    "FlatPlatePVResidential",
)

# Kandydaci na szereg mocy stałoprądowej w wynikach SSC, wraz ze współczynnikiem
# przeliczenia na kW. Nazwy wyjść bywają zmieniane między wersjami SSC, dlatego
# szukamy pierwszego dostępnego zamiast zakładać jedną nazwę na sztywno.
KANDYDACI_WYJSCIA_DC = (
    ("dc_net", 1.0, "kW"),                    # moc DC na wejściu falownika
    ("subarray1_dc_gross", 0.001, "W"),       # moc DC pola przed stratami wspólnymi
    ("dc_gross", 0.001, "W"),
)


# ===========================================================================
# 2. Informacja o wersjach narzędzi (potrzebna do rozdz. 6.1)
# ===========================================================================

def wersje_narzedzi() -> dict[str, str]:
    """
    Zwraca numery wersji PySAM oraz silnika obliczeniowego SSC.

    Numery te muszą trafić do opisu narzędzi w rozdz. 6.1 — wyniki symulacji
    są odtwarzalne tylko przy podanej wersji SSC, ponieważ kolejne wydania
    zmieniają domyślne wartości parametrów i poprawiają modele składowe.
    """
    wersje = {"PySAM": "nieznana", "SSC": "nieznana"}

    if not PYSAM_DOSTEPNY:
        wersje["PySAM"] = f"NIEDOSTĘPNY ({_BLAD_IMPORTU})"
        return wersje

    # Wersja PySAM — atrybut pakietu, a w razie jego braku metadane dystrybucji.
    try:
        wersje["PySAM"] = str(PySAM.__version__)
    except AttributeError:
        try:
            from importlib.metadata import version
            wersje["PySAM"] = version("nrel-pysam")
        except Exception as e:
            wersje["PySAM"] = f"nieznana ({e})"

    # Wersja SSC — biblioteka natywna, odpytywana przez warstwę PySSC.
    try:
        from PySAM.PySSC import PySSC
        ssc = PySSC()
        wersje["SSC"] = str(ssc.version())
        try:
            wersje["SSC_build"] = str(ssc.build_info())
        except Exception:
            pass
    except Exception as e:
        wersje["SSC"] = f"nieznana ({e})"

    return wersje


def wypisz_wersje() -> None:
    """Wypisuje nagłówek z wersjami narzędzi obliczeniowych."""
    wersje = wersje_narzedzi()
    print("-" * 72)
    print("Wersje narzędzi obliczeniowych (do rozdz. 6.1):")
    print(f"  PySAM : {wersje.get('PySAM')}")
    print(f"  SSC   : {wersje.get('SSC')}")
    if "SSC_build" in wersje:
        print(f"  build : {wersje['SSC_build']}")
    print(f"  Python: {sys.version.split()[0]}")
    print("-" * 72)


# ===========================================================================
# 3. Konfiguracja modelu
# ===========================================================================

def wczytaj_modul_z_biblioteki_cec(sciezka_csv: Path, nazwa_modulu: str) -> dict[str, float]:
    """
    Wczytuje współczynniki modelu jednodiodowego z biblioteki modułów CEC.

    Biblioteka SAM (plik ``CEC Modules.csv``) ma nagłówek w pierwszym wierszu
    i dwa dodatkowe wiersze opisowe, które należy pominąć. Zwracany słownik
    jest gotowy do przekazania do SSC pod nazwami parametrów ``cec_*``.
    """
    with open(sciezka_csv, "r", encoding="utf-8-sig", newline="") as f:
        czytnik = csv.reader(f)
        naglowek = next(czytnik)
        next(czytnik, None)   # wiersz z jednostkami
        next(czytnik, None)   # wiersz z etykietami

        wiersz = None
        for rekord in czytnik:
            if rekord and rekord[0].strip() == nazwa_modulu:
                wiersz = rekord
                break

    if wiersz is None:
        raise KeyError(
            f"Nie znaleziono modułu '{nazwa_modulu}' w pliku {sciezka_csv}. "
            f"Nazwa musi być zgodna co do znaku z kolumną 'Name' biblioteki."
        )

    kolumny = {nazwa.strip(): i for i, nazwa in enumerate(naglowek)}

    def pobierz(nazwa_kolumny: str) -> float:
        if nazwa_kolumny not in kolumny:
            raise KeyError(f"Biblioteka nie zawiera kolumny '{nazwa_kolumny}'.")
        return float(wiersz[kolumny[nazwa_kolumny]])

    # Odwzorowanie kolumn biblioteki CEC na nazwy parametrów wejściowych SSC.
    return {
        "cec_area": pobierz("A_c"),
        "cec_a_ref": pobierz("a_ref"),
        "cec_adjust": pobierz("Adjust"),
        "cec_alpha_sc": pobierz("alpha_sc"),
        "cec_beta_oc": pobierz("beta_oc"),
        # Kolumna gamma_r biblioteki CEC nie ma odpowiednika w parametrach
        # SSC 306: przy gotowych parametrach modelu jednodiodowego współczynnik
        # temperaturowy mocy jest wynikiem modelu, nie jego wejściem.
        "cec_i_l_ref": pobierz("I_L_ref"),
        "cec_i_mp_ref": pobierz("I_mp_ref"),
        "cec_i_o_ref": pobierz("I_o_ref"),
        "cec_i_sc_ref": pobierz("I_sc_ref"),
        "cec_n_s": pobierz("N_s"),
        "cec_r_s": pobierz("R_s"),
        "cec_r_sh_ref": pobierz("R_sh_ref"),
        "cec_t_noct": pobierz("T_NOCT"),
        "cec_v_mp_ref": pobierz("V_mp_ref"),
        "cec_v_oc_ref": pobierz("V_oc_ref"),
    }


def _parametry_modulu(dane_cec: dict[str, float] | None) -> dict[str, object]:
    """
    Buduje zestaw parametrów modułu dla jednej z dwóch ścieżek konfiguracji.

    Ścieżka A (``dane_cec`` podane): model CEC z bazą modułów, ``module_model=1``.
    Ścieżka B (domyślna): model CEC z danymi katalogowymi użytkownika,
    ``module_model=2``; SAM samodzielnie wyznacza parametry modelu jednodiodowego
    z wartości karty katalogowej.
    """
    if dane_cec is not None:
        parametry: dict[str, object] = {"module_model": 1}
        parametry.update(dane_cec)
        # Model temperatury ogniwa wg NOCT z uwzględnieniem sposobu montażu.
        parametry.update({
            "cec_temp_corr_mode": 0,               # 0 = model NOCT
            "cec_standoff": STANDOFF_MODEL_CIEPLNY,
            "cec_height": WYSOKOSC_MONTAZU,
        })
        return parametry

    m = MODUL_KATALOG
    # Przedrostek nazw parametrów to "sixpar_", nie "6par_". PySAM nie
    # dopuszcza identyfikatorów zaczynających się od cyfry, więc nazwy
    # z dokumentacji SSC mają w warstwie pythonowej rozpisaną liczbę słownie.
    # Pomyłka w tym miejscu nie zgłasza błędu sama z siebie — model wykonuje
    # się z parametrami domyślnymi innego modułu, co daje wynik pozornie
    # poprawny. Stąd rygorystyczny tryb zapisu parametrów (patrz _zastosuj).
    return {
        "module_model": 2,               # CEC na podstawie danych katalogowych
        "sixpar_celltech": m["technologia"],
        "sixpar_vmp": m["v_mp"],
        "sixpar_imp": m["i_mp"],
        "sixpar_voc": m["v_oc"],
        "sixpar_isc": m["i_sc"],
        "sixpar_bvoc": m["beta_voc"],
        "sixpar_aisc": m["alpha_isc"],
        "sixpar_gpmp": m["gamma_pmp"],
        "sixpar_nser": m["n_ogniw"],
        "sixpar_area": m["powierzchnia"],
        "sixpar_tnoct": m["t_noct"],
        "sixpar_standoff": STANDOFF_MODEL_CIEPLNY,
        "sixpar_mounting": WYSOKOSC_MONTAZU,
        "sixpar_is_bifacial": 0,
    }


def zbuduj_konfiguracje(epw: Path, dane_cec: dict[str, float] | None) -> dict[str, object]:
    """
    Zwraca płaski słownik parametrów wejściowych modelu Pvsamv1.

    Słownik jest celowo płaski (bez podziału na grupy SSC), ponieważ zapisywany
    jest przez metodę ``value()``, która odnajduje parametr niezależnie od grupy.
    Jest to jedyna droga dla nazw rozpoczynających się od cyfry (``6par_*``),
    niedostępnych jako atrybuty obiektu w Pythonie.
    """
    f = FALOWNIK_ZASTEPCZY
    moc_nominalna_kw = (
        MODUL_KATALOG["p_mp"] * LICZBA_MODULOW_W_LANCUCHU * LICZBA_LANCUCHOW / 1000.0
    )

    konfiguracja: dict[str, object] = {
        # --- plik pogodowy -------------------------------------------------
        "solar_resource_file": str(epw),

        # --- pojedynczy rok symulacji, bez modelu degradacji ----------------
        "system_use_lifetime_output": 0,

        # --- geometria i wielkość pola PV ----------------------------------
        "system_capacity": moc_nominalna_kw,
        "subarray1_tilt": POCHYLENIE_DEG,
        "subarray1_azimuth": AZYMUT_DEG,
        "subarray1_track_mode": 0,          # 0 = konstrukcja nieruchoma
        "subarray1_nstrings": LICZBA_LANCUCHOW,
        "subarray1_modules_per_string": LICZBA_MODULOW_W_LANCUCHU,
        "subarray2_enable": 0,
        "subarray3_enable": 0,
        "subarray4_enable": 0,

        # --- straty stałoprądowe, % ----------------------------------------
        # Wartości i ich uzasadnienie — stałe na początku modułu.
        "subarray1_mismatch_loss": STRATA_NIEDOPASOWANIE_PCT,
        "subarray1_diodeconn_loss": 0.5,
        "subarray1_dcwiring_loss": 1.0,
        "subarray1_tracking_loss": 0.0,
        # Degradacja indukowana światłem wprowadzona jako stały derating mocy
        # znamionowej — Pvsamv1 nie ma osobnego pola na LID.
        "subarray1_nameplate_loss": strata_lid_pct(),
        "acwiring_loss": 1.0,

        # Zabrudzenie zadawane miesięcznie, jedna wartość dla całego roku.
        "subarray1_soiling": [STRATA_ZABRUDZENIE_PCT] * 12,

        # Zacienienie własne wyłączone — pole jednomodułowe nie zacienia się samo.
        # Enumeracja SSC: 0 = brak zacienienia własnego, 1 = model standardowy
        # (nieliniowy), 2 = model dla cienkich warstw. Wartość 1 uruchamia
        # obliczenie zacienienia międzyrzędowego, które dla pola złożonego
        # z jednego modułu kończy się błędem „Number of rows calculated for
        # subarray 1 was zero" — liczba rzędów wychodzi zero przy domyślnym
        # układzie 7 x 2 modułów odziedziczonym z konfiguracji wzorcowej.
        "subarray1_shade_mode": 0,

        # Układ modułów w polu. Konfiguracja domyślna opisuje elektrownię
        # o mocy 100 MW (7 modułów w poziomie, 2 w pionie); tu pole tworzy
        # jeden moduł. Wartości ustawiane jawnie, aby nie zostały po wzorcu.
        "subarray1_nmodx": 1,
        "subarray1_nmody": 1,
        "subarray1_mod_orient": 0,       # 0 = orientacja pionowa (portrait)
        "subarray1_gcr": 0.3,

        # Albedo z pliku pogodowego wyłączone: pliki PVGIS nie zawierają tej
        # kolumny, a SSC zgłasza wtedy 8760 wartości niepoprawnych i i tak
        # przechodzi na wartości miesięczne. Wyłączenie usuwa ostrzeżenie
        # i czyni wybór jawnym.
        "use_wf_albedo": 0,
        # Współczynnik odbicia podłoża. Dla modułu leżącego poziomo
        # nie wpływa na wynik, bo składowa odbita wchodzi przez
        # czynnik (1 - cos beta)/2; znaczenie ma dopiero przy module
        # pochylonym (wariant z rozdz. 7). Wartość przyjęta.
        "albedo": [ALBEDO] * 12,

        # --- falownik zastępczy (patrz komentarz przy FALOWNIK_ZASTEPCZY) ---
        "inverter_model": 1,                # 1 = dane katalogowe
        "inverter_count": 1,
        "inv_ds_paco": f["moc_ac"],
        "inv_ds_eff": f["sprawnosc"],
        "inv_ds_pnt": f["pobor_nocny"],
        "inv_ds_pso": f["pobor_wlasny"],
        "inv_ds_vdco": f["u_dc_nom"],
        "inv_ds_vdcmax": f["u_dc_max"],
        "mppt_low_inverter": f["mppt_u_min"],
        "mppt_hi_inverter": f["mppt_u_max"],
    }

    konfiguracja.update(_parametry_modulu(dane_cec))
    return konfiguracja


def _zastosuj(model, konfiguracja: dict[str, object], rygorystycznie: bool = True) -> list[str]:
    """
    Zapisuje parametry do modelu i zgłasza te, których nie udało się ustawić.

    Tryb rygorystyczny (domyślny) przerywa obliczenia, gdy którykolwiek
    parametr nie został przyjęty. Pierwotna wersja tej funkcji pomijała takie
    parametry po cichu i okazało się to groźne: czternaście parametrów modułu
    zapisanych pod błędnym przedrostkiem (``6par_`` zamiast ``sixpar_``) nie
    zostało ustawionych, a model wykonałby się z parametrami modułu
    odziedziczonymi z konfiguracji wzorcowej — czyli policzyłby inny moduł
    niż opisany w pracy, nie zgłaszając żadnego błędu.

    Łagodny tryb (``rygorystycznie=False``) pozostaje dostępny do badania
    zgodności z inną wersją SSC, gdzie część nazw może nie istnieć.
    """
    pominiete: list[str] = []
    for nazwa, wartosc in konfiguracja.items():
        try:
            model.value(nazwa, wartosc)
        except Exception as e:
            pominiete.append(f"{nazwa} ({type(e).__name__}: {e})")

    if pominiete and rygorystycznie:
        raise RuntimeError(
            "Nie ustawiono "
            f"{len(pominiete)} parametrów modelu; obliczenia przerwane, aby nie "
            "policzyć konfiguracji innej niż opisana. Parametry:\n  "
            + "\n  ".join(pominiete)
        )
    return pominiete


# ===========================================================================
# 4. Wykonanie symulacji
# ===========================================================================

def _odczytaj_szereg_dc(model) -> tuple[np.ndarray, str]:
    """
    Odczytuje godzinowy szereg mocy stałoprądowej i przelicza go na kW.

    Zwraca krotkę (szereg w kW, nazwa użytego wyjścia SSC).
    """
    for nazwa, wspolczynnik, jednostka in KANDYDACI_WYJSCIA_DC:
        try:
            surowe = model.value(nazwa)
        except Exception:
            continue
        if surowe is None:
            continue
        szereg = np.asarray(surowe, dtype=float) * wspolczynnik
        if szereg.size == 0:
            continue
        return szereg, f"{nazwa} [{jednostka}]"

    raise RuntimeError(
        "Nie znaleziono godzinowego szeregu mocy DC w wynikach SSC. "
        f"Sprawdzono wyjścia: {[k[0] for k in KANDYDACI_WYJSCIA_DC]}."
    )


def symuluj_pv(
    epw: Path,
    dane_cec: dict[str, float] | None = None,
    gadatliwie: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Wykonuje roczną symulację generacji PV dla jednego pliku pogodowego.

    Zwraca krotkę (szereg mocy DC w kW o długości 8760, słownik metadanych).
    """
    if not PYSAM_DOSTEPNY:
        raise RuntimeError(
            "Pakiet nrel-pysam nie jest zainstalowany w tym interpreterze. "
            f"Szczegóły importu: {_BLAD_IMPORTU}"
        )
    if not epw.exists():
        raise FileNotFoundError(
            f"Brak pliku pogodowego {epw}. Uruchom najpierw fetch_weather.py."
        )

    # Ostrzeżenie o błędzie bazy czasu. Pliki EPW z PVGIS deklarują w nagłówku
    # strefę czasową 1, mimo że dane są w UTC, a SSC tę deklarację stosuje —
    # sprawdzone bramką w make_sam_resource (MAE kąta zenitalnego 6,7 stopnia,
    # najlepsze dopasowanie przy przesunięciu -1 h, out/qc_ssc_sun_position.csv).
    # Wynik liczony na takim pliku jest obarczony błędem godziny i nie nadaje
    # się do pracy; do wyników służą pliki CSV zasobu z make_sam_resource.
    if epw.suffix.lower() == ".epw" and gadatliwie:
        print("  " + "!" * 68)
        print("  UWAGA: plik EPW z PVGIS. SSC zastosuje strefę czasową")
        print("  z nagłówka (TimeZone = 1) do danych zapisanych w UTC, co daje")
        print("  błąd godziny w geometrii słonecznej. Wynik ma charakter")
        print("  wyłącznie poglądowy — do obliczeń pracy użyj plików zasobu")
        print("  budowanych przez make_sam_resource.py (run_yield.py).")
        print("  " + "!" * 68)

    # Konfiguracja domyślna dostarcza komplet parametrów modelu; nadpisujemy
    # z niej tylko te, które wynikają z projektu instalacji.
    model = None
    uzyta_konfiguracja = None
    bledy_konfiguracji = []
    for nazwa_konfiguracji in KONFIGURACJE_DOMYSLNE:
        try:
            model = Pvsamv1.default(nazwa_konfiguracji)
            uzyta_konfiguracja = nazwa_konfiguracji
            break
        except Exception as e:
            bledy_konfiguracji.append(f"{nazwa_konfiguracji}: {e}")
    if model is None:
        raise RuntimeError(
            "Nie udało się utworzyć modelu Pvsamv1 z żadnej konfiguracji "
            f"domyślnej. Próby: {bledy_konfiguracji}"
        )

    konfiguracja = zbuduj_konfiguracje(epw, dane_cec)
    pominiete = _zastosuj(model, konfiguracja)

    if gadatliwie:
        print(f"  konfiguracja domyślna : {uzyta_konfiguracja}")
        print(f"  plik pogodowy         : {epw.name}")
        print(f"  model modułu          : "
              f"{'baza CEC' if dane_cec else 'CEC z danych katalogowych (6-par)'}")
        print(f"  pochylenie / azymut   : {POCHYLENIE_DEG}° / {AZYMUT_DEG}°")
        print(f"  standoff (model cieplny): {STANDOFF_MODEL_CIEPLNY}")
        if pominiete:
            print(f"  UWAGA — parametry pominięte ({len(pominiete)}):")
            for p in pominiete:
                print(f"      {p}")

    model.execute(0)

    szereg_kw, zrodlo = _odczytaj_szereg_dc(model)

    # Weryfikacja kompletności roku. Przy system_use_lifetime_output = 0 szereg
    # musi mieć dokładnie 8760 wartości; długość 8784 oznaczałaby rok przestępny,
    # a wielokrotność 8760 — niewyłączony tryb wieloletni.
    if szereg_kw.size != 8760:
        raise ValueError(
            f"Szereg mocy DC ma {szereg_kw.size} wartości zamiast 8760 "
            f"(źródło: {zrodlo}). Sprawdź krok czasowy pliku pogodowego "
            f"oraz parametr system_use_lifetime_output."
        )

    metadane: dict[str, object] = {
        "zrodlo_wyjscia": zrodlo,
        "konfiguracja_domyslna": uzyta_konfiguracja,
        "parametry_pominiete": pominiete,
        "plik_epw": str(epw),
    }
    return szereg_kw, metadane


# ===========================================================================
# 5. Podsumowanie wyników
# ===========================================================================

def indeks_godzinowy(rok: int = 2019) -> pd.DatetimeIndex:
    """
    Buduje indeks 8760 godzin dla roku nieprzestępnego.

    Rok jest umowny — dane TMY są rokiem typowym, złożonym z miesięcy
    pochodzących z różnych lat kalendarzowych. Wybór roku nieprzestępnego
    zapewnia zgodność długości szeregu z 8760 wartościami.
    """
    return pd.date_range(f"{rok}-01-01 00:00", periods=8760, freq="h")


def podsumowanie_miesieczne(szereg_kw: np.ndarray) -> pd.Series:
    """
    Agreguje godzinową moc DC [kW] do sum miesięcznych energii [kWh].

    Przy kroku godzinowym wartość mocy w kW jest liczbowo równa energii w kWh,
    więc agregacja sprowadza się do sumowania.
    """
    seria = pd.Series(szereg_kw, index=indeks_godzinowy())
    miesieczne = seria.groupby(seria.index.month).sum()
    miesieczne.index = [
        "sty", "lut", "mar", "kwi", "maj", "cze",
        "lip", "sie", "wrz", "paź", "lis", "gru",
    ]
    return miesieczne


def wypisz_podsumowanie(szereg_kw: np.ndarray, etykieta: str) -> None:
    """Wypisuje roczny uzysk energii oraz rozkład miesięczny."""
    roczny = float(szereg_kw.sum())
    miesieczne = podsumowanie_miesieczne(szereg_kw)
    moc_szczytowa = float(szereg_kw.max())
    godziny_pracy = int((szereg_kw > 0.001).sum())

    print(f"\n  Wyniki dla lokalizacji: {etykieta}")
    print(f"    liczba wartości w szeregu : {szereg_kw.size}")
    print(f"    roczny uzysk energii DC   : {roczny:.1f} kWh")
    print(f"    moc szczytowa             : {moc_szczytowa * 1000:.0f} W")
    print(f"    godziny z generacją       : {godziny_pracy} h")
    print(f"    sumy miesięczne [kWh]:")
    for nazwa, wartosc in miesieczne.items():
        udzial = wartosc / roczny * 100 if roczny > 0 else 0.0
        slupek = "#" * int(round(udzial / 100 * 40))
        print(f"      {nazwa}  {wartosc:7.2f}   {udzial:5.1f} %  {slupek}")


# ===========================================================================
# 6. Uruchomienie samodzielne
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Symulacja generacji PV w module PySAM Pvsamv1."
    )
    parser.add_argument(
        "--epw", type=Path, default=None,
        help="ścieżka do pliku pogodowego EPW (domyślnie wynika z --location)",
    )
    parser.add_argument(
        "--location", type=str, default="Leba", choices=sorted(config.LOCATIONS),
        help="lokalizacja referencyjna z tabeli w config.py",
    )
    parser.add_argument(
        "--cec-csv", type=Path, default=None,
        help="ścieżka do biblioteki modułów CEC z instalacji SAM (CEC Modules.csv)",
    )
    parser.add_argument(
        "--cec-module", type=str, default=None,
        help="nazwa modułu w bibliotece CEC (kolumna 'Name')",
    )
    args = parser.parse_args()

    # Ścieżka pliku pogodowego wyprowadzana jest z tabeli lokalizacji, aby
    # nazwy plików nie były powielane w kilku miejscach potoku.
    if args.epw is None:
        args.epw = config.working_tmy_path(config.LOCATIONS[args.location])

    wypisz_wersje()

    if not PYSAM_DOSTEPNY:
        print("BŁĄD: pakiet nrel-pysam nie jest dostępny w tym interpreterze.")
        print(f"       {_BLAD_IMPORTU}")
        print("       Instalacja: pip install nrel-pysam  (wymaga Pythona 3.9-3.13)")
        return 2

    dane_cec = None
    if args.cec_csv is not None:
        if args.cec_module is None:
            print("BŁĄD: opcja --cec-csv wymaga podania --cec-module.")
            return 2
        try:
            dane_cec = wczytaj_modul_z_biblioteki_cec(args.cec_csv, args.cec_module)
            print(f"Wczytano moduł '{args.cec_module}' z biblioteki CEC.")
        except (OSError, KeyError, ValueError) as e:
            print(f"BŁĄD wczytywania biblioteki CEC: {e}")
            return 2

    print(f"\nSymulacja generacji PV")
    try:
        szereg_kw, metadane = symuluj_pv(args.epw, dane_cec=dane_cec)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"BŁĄD symulacji: {e}")
        return 1

    print(f"  odczytane wyjście SSC : {metadane['zrodlo_wyjscia']}")
    wypisz_podsumowanie(szereg_kw, args.epw.stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
