# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: przewody.py
Opis: dobór przekrojów i zabezpieczeń obwodów instalacji. Dla każdego
      obwodu wyznacza prąd obliczeniowy, przekrój wynikający z kryterium
      spadku napięcia, przekrój przyjęty oraz prąd znamionowy
      zabezpieczenia.

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

import sys
from dataclasses import dataclass

import pandas as pd

import config
import fmt
import loads

GENERATED = config.SIM_DIR.parent / "tex" / "generated"

# Rezystywność miedzi według PN-EN 1648-2, zał. A (50 stC), om*mm^2/m.
RHO_CU = 0.01909

# Przekrój minimalny instalacji bytowej.
PRZEKROJ_MIN_MM2 = 1.5

# Szereg przekrojów znormalizowanych, mm^2.
SZEREG_MM2 = (1.5, 2.5, 4.0, 6.0, 10.0, 16.0, 25.0, 35.0, 50.0, 70.0, 95.0,
              120.0)

# Szereg prądów znamionowych wkładek topikowych, A.
SZEREG_IN_A = (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 200,
               250, 300, 355, 400)

# --- napięcia obliczeniowe -------------------------------------------------
NAPIECIE_MAGISTRALI_V = loads.POBORY_WLASNE_U_V          # 12,8 V
# Napięcie końcowe rozładowania ogniwa (karta LF280K) i liczba ogniw.
NAPIECIE_KONCOWE_OGNIWA_V = 2.5
OGNIW_W_PAKIECIE = 4
NAPIECIE_MIN_V = NAPIECIE_KONCOWE_OGNIWA_V * OGNIW_W_PAKIECIE

# --- falownik --------------------------------------------------------------
FALOWNIK_P_CIAGLA_W = 1800.0
FALOWNIK_P_SZCZYT_W = 2900.0
# Zalecenie producenta dla modelu 1800 W / 12 V: #4/0 AWG.
FALOWNIK_PRZEKROJ_PRODUCENTA_MM2 = 110.0


def prad_falownika_a(p_ac_w: float, napiecie_v: float) -> float:
    """Prąd pobierany z magistrali przy zadanej mocy wyjściowej falownika."""
    return (p_ac_w + float(loads.falownik_straty_w(p_ac_w))) / napiecie_v


PRAD_FALOWNIK_A = round(prad_falownika_a(FALOWNIK_P_CIAGLA_W, NAPIECIE_MIN_V), 1)
PRAD_FALOWNIK_SZCZYT_A = round(
    prad_falownika_a(FALOWNIK_P_SZCZYT_W, NAPIECIE_MIN_V), 1)
PRAD_FALOWNIK_NOMINALNY_A = round(
    prad_falownika_a(FALOWNIK_P_CIAGLA_W, NAPIECIE_MAGISTRALI_V), 1)

# --- ładowarka stałoprądowa ------------------------------------------------
# Nastawa prądu obowiązuje jednocześnie po stronie wejściowej i wyjściowej,
# więc prąd wejściowy nie przekracza nastawy: wcześniejszy rachunek
# I_we = I_wy * U_wy / (eta * U_we) dawał 53,1 A, czyli wartość powyżej
# dopuszczalnej. Moc wyjściową ogranicza mniejszy z dwóch iloczynów, a prąd
# wyjściowy wynika z tej mocy.
LADOWARKA_NASTAWA_A = 50.0
LADOWARKA_SPRAWNOSC = 0.985
LADOWARKA_U_WY_V = 13.8
LADOWARKA_U_WE_V = 13.2
PRAD_LADOWARKI_WE_A = LADOWARKA_NASTAWA_A
LADOWARKA_MOC_WY_W = round(min(
    LADOWARKA_SPRAWNOSC * LADOWARKA_U_WE_V * LADOWARKA_NASTAWA_A,
    LADOWARKA_U_WY_V * LADOWARKA_NASTAWA_A), 1)
LADOWARKA_PRAD_WY_A = round(LADOWARKA_MOC_WY_W / LADOWARKA_U_WY_V, 1)
LADOWARKA_MOC_WY_TRZYDZIESCI_W = round(
    min(LADOWARKA_SPRAWNOSC * LADOWARKA_U_WE_V * 30.0,
        LADOWARKA_U_WY_V * 30.0), 1)

# Udział prądu znamionowego alternatora pobierany przez ładowarkę przy
# nastawie pełnej. Wartość ta opisuje obciążenie alternatora, a nie rezerwę
# dostępną w modelu bilansu (run_grid.UDZIAL_ALTERNATORA).
UDZIAL_ALTERNATORA = PRAD_LADOWARKI_WE_A / config.ALTERNATOR_PRAD_A


@dataclass(frozen=True)
class Obwod:
    nazwa: str
    prad_obliczeniowy_a: float
    dlugosc_m: float                # długość trasy w jedną stronę
    spadek_dop_v: float
    uwaga: str
    przekroj_wymagany_mm2: float = 0.0   # wymaganie producenta, jeśli jest
    zabezpieczenie_a: float = 0.0        # wkładka zalecana przez producenta
    status_zabezpieczenia: str = ""      # pusty = dobór z szeregu


# Długości tras przyjęto z geometrii pojazdu bazowego (wartości przyjęte,
# do weryfikacji pomiarem przy wykonaniu zabudowy).
OBWODY = (
    Obwod("generator PV -- regulator", 17.9, 3.0, 0.8,
          "prąd 1,25 I_sc modułu"),
    Obwod("regulator -- magistrala", 30.0, 1.5, 0.3,
          "wkładka 40 A według instrukcji regulatora",
          zabezpieczenie_a=40.0,
          status_zabezpieczenia="zalecenie producenta"),
    Obwod("magazyn -- magistrala", PRAD_FALOWNIK_A, 1.0, 0.3,
          "prąd falownika przy napięciu odłączenia magazynu",
          FALOWNIK_PRZEKROJ_PRODUCENTA_MM2),
    Obwod("magistrala -- falownik", PRAD_FALOWNIK_A, 1.5, 0.8,
          "jak wyżej; przekrój według zalecenia producenta falownika",
          FALOWNIK_PRZEKROJ_PRODUCENTA_MM2),
    Obwod("akumulator rozruchowy -- ładowarka", PRAD_LADOWARKI_WE_A, 4.0, 0.3,
          "prąd wejściowy ładowarki; przewód ładowania"),
    Obwod("ładowarka -- magistrala", LADOWARKA_PRAD_WY_A, 1.0, 0.3,
          "prąd wyjściowy ładowarki; przewód ładowania"),
    Obwod("magistrala -- ogrzewanie postojowe", 2.4, 3.0, 0.8,
          "pobór ustalony; prąd rozruchu świecy nieznany",
          status_zabezpieczenia="do wykazania"),
    Obwod("magistrala -- chłodziarka", 5.0, 2.5, 0.8,
          "pobór ustalony; prąd rozruchu sprężarki nieznany",
          status_zabezpieczenia="do wykazania"),
    Obwod("magistrala -- obwód odbiorników 12 V", 8.0, 4.0, 0.8,
          "oświetlenie, gniazda USB, łącze satelitarne"),
)


# Tory szeregowe: zbior obwodow, przez ktore przeplywa ten sam prad od
# zrodla do odbiornika, oraz dopuszczalny spadek napiecia calego toru.
# Kryterium normy wyrobu dotyczy drogi przeplywu, a nie pojedynczego
# odcinka, wiec budzet spadku dzieli sie miedzy odcinki proporcjonalnie do
# iloczynu rho*2l*I_B. Tor ladowania z alternatora sklada sie z dwoch
# odcinkow po obu stronach ladowarki (komentarz K71).
TORY = (
    ("tor ładowania z alternatora", (4, 5), 0.3),
)


def budzety_spadku(obwody) -> dict[int, float]:
    """Dopuszczalny spadek przypisany poszczegolnym odcinkom torow."""
    budzety: dict[int, float] = {}
    for _, indeksy, limit in TORY:
        wagi = {i: (RHO_CU * 2.0 * obwody[i].dlugosc_m
                    * obwody[i].prad_obliczeniowy_a) for i in indeksy}
        suma = sum(wagi.values())
        for i in indeksy:
            budzety[i] = limit * wagi[i] / suma
    return budzety


def zaokraglij(wartosc: float, szereg) -> float:
    """Najmniejszy element szeregu nie mniejszy od wartości."""
    for element in szereg:
        if element >= wartosc - 1e-9:
            return element
    return szereg[-1]


def dobierz(obwod: Obwod, oznaczenie: str = "",
            spadek_dop_v: float | None = None) -> dict[str, object]:
    """Dobiera przekrój i zabezpieczenie dla jednego obwodu."""
    # Wzór (A.1) normy wyrobu; L to długość obu żył, czyli dwukrotność trasy.
    limit = (obwod.spadek_dop_v if spadek_dop_v is None else spadek_dop_v)
    s_spadek = (RHO_CU * 2.0 * obwod.dlugosc_m * obwod.prad_obliczeniowy_a
                / limit)
    s_przyjety = zaokraglij(
        max(s_spadek, PRZEKROJ_MIN_MM2, obwod.przekroj_wymagany_mm2),
        SZEREG_MM2)
    spadek = (RHO_CU * 2.0 * obwod.dlugosc_m * obwod.prad_obliczeniowy_a
              / s_przyjety)
    i_n = (obwod.zabezpieczenie_a if obwod.zabezpieczenie_a > 0.0
           else zaokraglij(obwod.prad_obliczeniowy_a, SZEREG_IN_A))
    podstawa = ("wymaganie producenta" if obwod.przekroj_wymagany_mm2 >= s_spadek
                and obwod.przekroj_wymagany_mm2 > 0 else
                ("spadek napięcia" if s_spadek > PRZEKROJ_MIN_MM2
                 else "przekrój minimalny"))
    return {
        "oznaczenie": oznaczenie,
        "obwód": obwod.nazwa,
        "I_B [A]": round(obwod.prad_obliczeniowy_a, 1),
        "L [m]": obwod.dlugosc_m,
        "dU_dop [V]": obwod.spadek_dop_v,
        "dU_dop odcinka [V]": round(limit, 3),
        "S ze spadku [mm2]": round(s_spadek, 2),
        "S wymagany [mm2]": obwod.przekroj_wymagany_mm2 or "",
        "S przyjęte [mm2]": s_przyjety,
        "dU [V]": round(spadek, 3),
        "I_n [A]": i_n,
        "I_B<=I_n": bool(obwod.prad_obliczeniowy_a <= i_n),
        "podstawa doboru": podstawa,
        "status zabezpieczenia": (obwod.status_zabezpieczenia
                                  or "dobór z szeregu"),
        "uwaga": obwod.uwaga,
    }


def tabela_tex(tab: pd.DataFrame) -> str:
    """Składa tabelę doboru do dołączenia w rozdziale 5."""
    wiersze = []
    for _, r in tab.iterrows():
        wiersze.append(
            "        %s & %s & %s & %s & %s & %s & %s & %s \\\\"
            % (r["oznaczenie"],
               r["obwód"],
               fmt.num(r["I_B [A]"], 1),
               fmt.num(r["L [m]"], 1),
               fmt.num(r["dU_dop [V]"], 1),
               fmt.num(r["S przyjęte [mm2]"], 1),
               fmt.num(r["dU [V]"], 3),
               fmt.num(r["I_n [A]"], 0)))
    return (
        "%% WYGENEROWANO AUTOMATYCZNIE: simulation/przewody.py --- nie edytować ręcznie\n"
        "\\begin{table}[!htbp]\n"
        "    \\centering\n"
        "    \\small\n"
        "    \\caption[Dobór przekrojów i zabezpieczeń obwodów]{Dobór przekrojów\n"
        "    i~zabezpieczeń obwodów instalacji: prąd obliczeniowy $I_B$, długość\n"
        "    trasy w~jedną stronę $l$ (długość pętli $L = 2l$), dopuszczalny\n"
        "    spadek napięcia, przekrój przyjęty, spadek napięcia na samym\n"
        "    przewodzie przy przekroju przyjętym oraz prąd znamionowy wkładki.\n"
        "    Przekrój wynika z~kryterium spadku napięcia normy wyrobu,\n"
        "    a~w~torze falownika --- z~zalecenia producenta. Spadek $\\Delta U$\n"
        "    nie obejmuje styków, złączy ani aparatów. Oznaczenia obwodów\n"
        "    odpowiadają oznaczeniom na schemacie blokowym.\n"
        "    Opracowanie własne}\n"
        "    \\label{tbl:przewody}\n"
        "    \\begin{tblr}{%\n"
        "        hlines,%\n"
        "        vlines,%\n"
        "        width=\\linewidth,%\n"
        "        colspec={X[0.5,c]X[2.4,l]X[0.7,c]X[0.6,c]X[0.8,c]"
        "X[0.8,c]X[0.7,c]X[0.6,c]},%\n"
        "        row{1}={font=\\bfseries},%\n"
        "        row{2}={font=\\footnotesize},%\n"
        "    }%\n"
        "        Ozn. & Obwód & $I_B$ & $l$ & $\\Delta U_\\mathrm{dop}$\n"
        "          & $S$ & $\\Delta U$ & $I_n$ \\\\\n"
        "        & & [A] & [m] & [V] & [mm\\textsuperscript{2}] & [V]\n"
        "          & [A] \\\\\n"
        + "\n".join(wiersze)
        + "\n    \\end{tblr}\n\\end{table}\n")


def makra_tex(tab: pd.DataFrame) -> str:
    """Makra liczbowe do prozy rozdziału 5."""
    bs, nl = chr(92), chr(10)

    def mk(nazwa: str, wartosc) -> str:
        miejsca = 0 if float(wartosc) == int(wartosc) else 1
        return (bs + "newcommand{" + bs + nazwa + "}{"
                + fmt.num(wartosc, miejsca) + "}" + nl)

    przekroje = sorted(set(tab["S przyjęte [mm2]"]))
    prady = sorted(set(tab["I_n [A]"]))
    return ("%% WYGENEROWANO AUTOMATYCZNIE: simulation/przewody.py"
            " --- nie edytowac recznie" + nl
            + mk("PradFalownikaCiagly", PRAD_FALOWNIK_A)
            + mk("PradFalownikaSzczyt", PRAD_FALOWNIK_SZCZYT_A)
            + mk("PradFalownikaNominalny", PRAD_FALOWNIK_NOMINALNY_A)
            + mk("NapiecieMinMagistrali", NAPIECIE_MIN_V)
            + mk("NapiecieKoncoweOgniwa", NAPIECIE_KONCOWE_OGNIWA_V)
            + mk("SpadekTorPv", tab.loc[0, "dU [V]"]
                 + tab.loc[1, "dU [V]"])
            + mk("SpadekTorLadowania", tab.loc[4, "dU [V]"]
                 + tab.loc[5, "dU [V]"])
            + mk("SpadekTorFalownika", tab.loc[2, "dU [V]"]
                 + tab.loc[3, "dU [V]"])
            + mk("SpadekFdwa", tab.loc[1, "dU [V]"])
            + mk("SpadekFpiec", tab.loc[4, "dU [V]"])
            + mk("ZabezpieczenieFdwa", tab.loc[1, "I_n [A]"])
            + mk("PradLadowarkiWe", PRAD_LADOWARKI_WE_A)
            + mk("PradLadowarkiWy", LADOWARKA_PRAD_WY_A)
            + mk("MocLadowarkiXs", LADOWARKA_MOC_WY_W)
            + mk("MocLadowarkiXsTrzydziesci",
                 LADOWARKA_MOC_WY_TRZYDZIESCI_W)
            + mk("LadowarkaSprawnosc", LADOWARKA_SPRAWNOSC)
            + mk("UdzialAlternatora", round(100.0 * UDZIAL_ALTERNATORA, 1))
            + mk("PrzekrojMin", min(przekroje))
            + mk("PrzekrojMax", max(przekroje))
            + mk("PrzekrojFalownika", FALOWNIK_PRZEKROJ_PRODUCENTA_MM2)
            + mk("ZabezpieczenieMin", min(prady))
            + mk("ZabezpieczenieMax", max(prady))
            + mk("ObwodowInstalacji", len(tab)))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_dirs()
    budzety = budzety_spadku(OBWODY)
    tab = pd.DataFrame([dobierz(o, "F%d" % (i + 1), budzety.get(i))
                        for i, o in enumerate(OBWODY)])
    cel = config.OUT_DIR / "przewody.csv"
    tab.to_csv(cel, index=False)

    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "tbl-przewody.tex").write_text(
        tabela_tex(tab), encoding="utf-8", newline="\n")
    (GENERATED / "liczby-przewody.tex").write_text(
        makra_tex(tab), encoding="utf-8", newline="\n")

    print("=" * 96)
    print("Dobór przekrojów i zabezpieczeń")
    print("=" * 96)
    print(tab[["oznaczenie", "obwód", "I_B [A]", "L [m]", "dU_dop [V]",
               "S ze spadku [mm2]", "S przyjęte [mm2]", "dU [V]", "I_n [A]",
               "podstawa doboru"]].to_string(index=False))
    niezgodne = tab[~tab["I_B<=I_n"]]
    print()
    if niezgodne.empty:
        print("prąd znamionowy wkładki nie mniejszy od obliczeniowego: "
              "we wszystkich obwodach")
    else:
        print("UWAGA — warunek I_B <= I_n niespełniony w obwodach:")
        print(niezgodne["obwód"].to_string(index=False))
    print("napięcie obliczeniowe toru falownika: %.1f V (%d x %.1f V)"
          % (NAPIECIE_MIN_V, OGNIW_W_PAKIECIE, NAPIECIE_KONCOWE_OGNIWA_V))
    print("zapisano:", cel.name,
          "oraz tex/generated/{tbl-przewody,liczby-przewody}.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
