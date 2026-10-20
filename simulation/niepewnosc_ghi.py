# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: niepewnosc_ghi.py
Opis: względna niepewność danych o promieniowaniu. Przelicza
      opublikowane wyniki walidacji zbioru SARAH-3 wobec stacji
      naziemnych na wartości względne dla lokalizacji rozpatrywanych w
      pracy. Bezwzględne miary błędu podane są w źródle w watach na metr
      kwadratowy, a do analizy wrażliwości potrzebna jest wartość
      odniesiona do natężenia promieniowania występującego w tych
      lokalizacjach.

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

import json
import sys

import pandas as pd

import config

# --- wartości bezwzględne z literatury (Pfeifroth i in. 2024) -------------
# Tabela 4, wiersz SIS (promieniowanie całkowite na płaszczyznę poziomą):
# błąd systematyczny, średnia odchyłka bezwzględna i błąd średniokwadratowy
# w skali miesięcznej (mm) i dobowej (dm), wobec stacji BSRN.
BIAS_MIESIAC_W_M2 = 2.1
BIAS_DOBA_W_M2 = 2.0
MAD_MIESIAC_W_M2 = 5.2
MAD_DOBA_W_M2 = 10.8
RMSE_MIESIAC_W_M2 = 7.0
RMSE_DOBA_W_M2 = 15.9
# Rysunek 14: błąd systematyczny wobec stacji GEBA w Europie.
BIAS_GEBA_W_M2 = 3.0

GODZIN_W_ROKU = 8760
GODZIN_W_MIESIACU = {6: 30 * 24, 12: 31 * 24}


def natezenia_odniesienia() -> pd.DataFrame:
    """Średnie natężenie promieniowania: rok oraz czerwiec i grudzień."""
    roczne = pd.read_csv(config.OUT_DIR / "ghi_annual_sums.csv")
    miesieczne = pd.read_csv(config.OUT_DIR / "ghi_monthly_stats.csv")
    wiersze = []
    for nazwa, grupa in roczne.groupby("lokalizacja"):
        suma = float(grupa["ghi_roczne_kwh_m2"].median())
        wiersz = {"lokalizacja": nazwa,
                  "H_rok_kWh_m2": round(suma, 1),
                  "G_sr_rok_W_m2": round(1000.0 * suma / GODZIN_W_ROKU, 1)}
        for miesiac, godzin in GODZIN_W_MIESIACU.items():
            wybor = miesieczne[(miesieczne.lokalizacja == nazwa)
                               & (miesieczne.miesiac == miesiac)]
            if wybor.empty:
                continue
            suma_m = float(wybor["median"].iloc[0])
            wiersz["G_sr_%02d_W_m2" % miesiac] = round(
                1000.0 * suma_m / godzin, 1)
        wiersze.append(wiersz)
    return pd.DataFrame(wiersze)


def niepewnosci(odniesienia: pd.DataFrame) -> pd.DataFrame:
    """Względne wartości miar błędu dla każdej lokalizacji i okresu."""
    wiersze = []
    for _, r in odniesienia.iterrows():
        for okres, kolumna in (("rok", "G_sr_rok_W_m2"),
                               ("czerwiec", "G_sr_06_W_m2"),
                               ("grudzień", "G_sr_12_W_m2")):
            if kolumna not in r or pd.isna(r[kolumna]):
                continue
            g = float(r[kolumna])
            wiersze.append({
                "lokalizacja": r["lokalizacja"], "okres": okres,
                "G_odniesienia_W_m2": g,
                "bias_mies_pct": round(100.0 * BIAS_MIESIAC_W_M2 / g, 2),
                "MAD_mies_pct": round(100.0 * MAD_MIESIAC_W_M2 / g, 2),
                "RMSE_mies_pct": round(100.0 * RMSE_MIESIAC_W_M2 / g, 2),
                "MAD_doba_pct": round(100.0 * MAD_DOBA_W_M2 / g, 2),
                "RMSE_doba_pct": round(100.0 * RMSE_DOBA_W_M2 / g, 2),
                "bias_GEBA_pct": round(100.0 * BIAS_GEBA_W_M2 / g, 2),
            })
    return pd.DataFrame(wiersze)


def makra_tex(tab: pd.DataFrame, u_mies: float) -> str:
    """Makra liczbowe do podrozdziału o weryfikacji modelu."""
    import fmt

    bs, nl = chr(92), chr(10)

    def mk(nazwa: str, wartosc, miejsca: int = 1) -> str:
        return (bs + "newcommand{" + bs + nazwa + "}{"
                + fmt.pl_tex(wartosc, miejsca) + "}" + nl)

    roczne = tab[tab.okres == "rok"]
    czerwiec = tab[tab.okres == "czerwiec"]
    grudzien = tab[tab.okres == "grudzień"]
    return ("%% WYGENEROWANO AUTOMATYCZNIE: simulation/niepewnosc_ghi.py"
            " --- nie edytowac recznie" + nl
            + mk("MadMiesiac", MAD_MIESIAC_W_M2)
            + mk("MadDoba", MAD_DOBA_W_M2)
            + mk("RmseMiesiac", RMSE_MIESIAC_W_M2)
            + mk("RmseDoba", RMSE_DOBA_W_M2)
            + mk("BiasMiesiac", BIAS_MIESIAC_W_M2)
            + mk("BiasGeba", BIAS_GEBA_W_M2, 0)
            + mk("UGhiRokMin", roczne["MAD_mies_pct"].min())
            + mk("UGhiRokMax", roczne["MAD_mies_pct"].max())
            + mk("UGhiCzerwiecMin", czerwiec["MAD_mies_pct"].min())
            + mk("UGhiCzerwiecMax", czerwiec["MAD_mies_pct"].max())
            + mk("UGhiGrudzienMin", grudzien["MAD_mies_pct"].min())
            + mk("UGhiGrudzienMax", grudzien["MAD_mies_pct"].max())
            + mk("BiasGhiRokMin", roczne["bias_mies_pct"].min(), 2)
            + mk("BiasGhiRokMax", roczne["bias_mies_pct"].max(), 2)
            + mk("MnoznikGhi", 100.0 * u_mies))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_dirs()
    odniesienia = natezenia_odniesienia()
    tab = niepewnosci(odniesienia)
    tab.to_csv(config.OUT_DIR / "niepewnosc_ghi.csv", index=False)

    roczne = tab[tab.okres == "rok"]
    u_mies = float(roczne["MAD_mies_pct"].max()) / 100.0
    dane = {
        "u_mnoznik_ghi": round(u_mies, 4),
        "podstawa": "MAD miesięczny SARAH-3 wobec BSRN do średniego "
                    "rocznego natężenia promieniowania",
        "MAD_mies_W_m2": MAD_MIESIAC_W_M2,
        "bias_mies_W_m2": BIAS_MIESIAC_W_M2,
        "RMSE_mies_W_m2": RMSE_MIESIAC_W_M2,
    }
    (config.OUT_DIR / "niepewnosc_ghi.json").write_text(
        json.dumps(dane, ensure_ascii=False, indent=2), encoding="utf-8")

    generated = config.SIM_DIR.parent / "tex" / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "liczby-niepewnosc.tex").write_text(
        makra_tex(tab, u_mies), encoding="utf-8", newline="\n")

    print(tab.to_string(index=False))
    print()
    print("mnożnik GHI do analizy wrażliwości: 1 ± %.3f" % u_mies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
