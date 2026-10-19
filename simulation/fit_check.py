# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: fit_check.py
Opis: kontrola odporności dopasowania parametrów modułu. Powód
      powstania. Rozkład efektu modułu (module_effect_check.py) wykazał,
      że o znaku tej wielkości rozstrzyga nie współczynnik
      temperaturowy, lecz dopasowanie parametrów modelu jednodiodowego
      do danych katalogowych: przy tym samym gamma zmiana modułu
      przesunęła wynik o -1,8 pkt proc. Skoro rozstrzyga dopasowanie,
      trzeba wiedzieć, ile wynosi jego niepewność.

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
import sys

import numpy as np
import pandas as pd

import config
import make_sam_resource as resource
import manifest
import run_pv
from config import LOCATIONS, Location

T_STC_C = 25.0
PRZEBIEG = "dopasowanie_modulu"

# Nazwa technologii ogniwa w pvlib odpowiadająca enumeracji SAM
# (0 = monokrystaliczne).
TECHNOLOGIA_PVLIB = {0: "monoSi", 1: "multiSi", 2: "cis", 3: "cigs", 4: "cdte"}


def parametry_fit_cec_sam(m: dict) -> dict[str, float]:
    """Parametry modelu CEC wyznaczone solwerem SSC (przez pvlib)."""
    from pvlib.ivtools.sdm import fit_cec_sam

    i_l, i_o, r_s, r_sh, a_ref, adjust = fit_cec_sam(
        celltype=TECHNOLOGIA_PVLIB[m["technologia"]],
        v_mp=m["v_mp"], i_mp=m["i_mp"], v_oc=m["v_oc"], i_sc=m["i_sc"],
        alpha_sc=m["alpha_isc"], beta_voc=m["beta_voc"],
        gamma_pmp=m["gamma_pmp"], cells_in_series=m["n_ogniw"])
    return {"I_L_ref": i_l, "I_o_ref": i_o, "R_s": r_s, "R_sh_ref": r_sh,
            "a_ref": a_ref, "Adjust": adjust}


# Ustawienia solwera dla fit_desoto. Metoda domyślna („hybr") NIE zbiega dla
# modułu bazowego przy żadnym z kilkunastu sprawdzonych punktów startowych —
# układ równań jest dla tego modułu źle uwarunkowany (duży prąd, mała liczba
# ogniw szeregowo, napięcie 0,76 V na ogniwo). Metoda Levenberga-Marquardta
# zbiega, ale część punktów startowych prowadzi do rozwiązania NIEFIZYCZNEGO
# z ujemną rezystancją bocznikową: układ ma więcej niż jeden pierwiastek.
# Punkt startowy pozostaje ten z pracy Duffiego i Beckmana (a_0 = 1,5 k T N_s),
# czyli domyślny w pvlib — niezależny od wyniku solwera SSC.
FIT_DESOTO_USTAWIENIA = {"init_guess": {"Rsh_0": 100.0},
                         "root_kwargs": {"method": "lm"}}


def parametry_fit_desoto(m: dict) -> dict[str, float]:
    """Parametry modelu De Soto wyznaczone implementacją pvlib."""
    from pvlib.ivtools.sdm import fit_desoto

    wynik, _ = fit_desoto(
        v_mp=m["v_mp"], i_mp=m["i_mp"], v_oc=m["v_oc"], i_sc=m["i_sc"],
        alpha_sc=m["alpha_isc"], beta_voc=m["beta_voc"],
        cells_in_series=m["n_ogniw"], **FIT_DESOTO_USTAWIENIA)
    if wynik["R_sh_ref"] <= 0.0 or wynik["I_o_ref"] <= 0.0:
        raise RuntimeError(
            "fit_desoto zbiegł do rozwiązania niefizycznego "
            f"(R_sh = {wynik['R_sh_ref']:.3g} om, I_o = {wynik['I_o_ref']:.3g} A). "
            "Rozwiązania o ujemnej rezystancji bocznikowej nie wolno przenosić "
            "do modelu — zmień punkt startowy.")
    return {"I_L_ref": wynik["I_L_ref"], "I_o_ref": wynik["I_o_ref"],
            "R_s": wynik["R_s"], "R_sh_ref": wynik["R_sh_ref"],
            "a_ref": wynik["a_ref"], "Adjust": 0.0}


def _slownik_cec(m: dict, p: dict[str, float]) -> dict[str, float]:
    """Buduje zestaw parametrów ``cec_*`` dla ścieżki ``module_model=1``."""
    return {
        "cec_area": m["powierzchnia"],
        "cec_a_ref": p["a_ref"],
        "cec_adjust": p["Adjust"],
        "cec_alpha_sc": m["alpha_isc"],
        "cec_beta_oc": m["beta_voc"],
        # Uwaga: w tej ścieżce NIE podaje się współczynnika temperaturowego
        # mocy. SSC 306 nie ma parametru "cec_gamma_r" — przy gotowych
        # parametrach modelu jednodiodowego gamma jest ich WYNIKIEM, a nie
        # wejściem. To ta sama zależność, która w rozkładzie efektu modułu
        # odwróciła znak: gamma z karty nie jest mnożnikiem nałożonym na wynik.
        "cec_i_l_ref": p["I_L_ref"],
        "cec_i_mp_ref": m["i_mp"],
        "cec_i_o_ref": p["I_o_ref"],
        "cec_i_sc_ref": m["i_sc"],
        "cec_n_s": m["n_ogniw"],
        "cec_r_s": p["R_s"],
        "cec_r_sh_ref": p["R_sh_ref"],
        "cec_t_noct": m["t_noct"],
        "cec_v_mp_ref": m["v_mp"],
        "cec_v_oc_ref": m["v_oc"],
    }


def _uruchom(resource_file, dane_cec: dict[str, float] | None):
    """Wykonuje Pvsamv1 dla zadanej ścieżki konfiguracji modułu."""
    import PySAM.Pvsamv1 as Pvsamv1

    model = Pvsamv1.default("FlatPlatePVNone")
    run_pv._zastosuj(model, run_pv.zbuduj_konfiguracje(resource_file, dane_cec))
    model.execute(0)
    return model


def _skladniki(model, gamma_pct_k: float) -> dict[str, float]:
    """Rozkład efektu modułu na składnik temperaturowy i resztę."""
    out = model.export()["Outputs"]
    dc_nominal = float(out["annual_dc_nominal"])
    dc_gross = float(out["annual_subarray1_dc_gross"])
    efekt = (dc_gross - dc_nominal) / dc_nominal * 100.0

    poa = np.asarray(out["subarray1_poa_eff"], dtype=float)
    cell = np.asarray(out["subarray1_celltemp"], dtype=float)
    maska = poa > 0.0
    gamma = gamma_pct_k / 100.0
    skl_temp = (np.sum(poa[maska] * (1.0 + gamma * (cell[maska] - T_STC_C)))
                / np.sum(poa[maska]) - 1.0) * 100.0
    return {
        "dc_nominal_kWh": round(dc_nominal, 1),
        "dc_gross_kWh": round(dc_gross, 1),
        "efekt_modulu_pct": round(efekt, 3),
        "skladnik_temperaturowy_pct": round(skl_temp, 3),
        "reszta_niskie_natezenie_pct": round(efekt - skl_temp, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--location", default="Leba", choices=sorted(LOCATIONS))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_dirs()

    loc: Location = LOCATIONS[args.location]
    plik = resource.resource_path(loc, args.year)
    m = run_pv.MODUL_KATALOG

    # --- 1. Którą ścieżką liczy potok produkcyjny ------------------------
    konfiguracja = run_pv.zbuduj_konfiguracje(plik, None)
    module_model = int(konfiguracja["module_model"])
    wewnetrzne = module_model == 2
    print("=" * 94)
    print("Kontrola odporności dopasowania — %s, rok %d" % (loc.label, args.year))
    print("=" * 94)
    print("Potok produkcyjny: module_model = %d (%s)"
          % (module_model,
             "CEC z danych katalogowych — parametry wyznacza SSC (6parsolve)"
             if wewnetrzne else "CEC z gotowymi parametrami podanymi z zewnątrz"))

    # --- 2. Dwa dopasowania ---------------------------------------------
    metody = {"fit_cec_sam (solwer SSC)": parametry_fit_cec_sam(m),
              "fit_desoto (implementacja pvlib)": parametry_fit_desoto(m)}

    tab_par = pd.DataFrame(metody).T.reset_index(names="metoda")
    sciezka_par = config.OUT_DIR / "module_fit_parameters.csv"
    tab_par.to_csv(sciezka_par, index=False)

    print()
    print("Parametry modelu jednodiodowego dla modułu %s:" % m["nazwa"])
    print(tab_par.to_string(index=False, float_format=lambda v: "%.6g" % v))
    a = metody["fit_cec_sam (solwer SSC)"]
    b = metody["fit_desoto (implementacja pvlib)"]
    print()
    print("Różnice względne (fit_desoto wobec fit_cec_sam):")
    for klucz in ("I_L_ref", "I_o_ref", "R_s", "R_sh_ref", "a_ref"):
        if a[klucz]:
            print("  %-10s %+8.2f %%" % (klucz, (b[klucz] - a[klucz]) / a[klucz] * 100.0))

    # --- 3. Uzysk dla obu zestawów parametrów ---------------------------
    wiersze = []
    wiersze.append({"wariant": "potok produkcyjny (module_model=2, SSC)",
                    **_skladniki(_uruchom(plik, None), m["gamma_pmp"])})
    for nazwa, p in metody.items():
        wiersze.append({"wariant": "module_model=1, %s" % nazwa,
                        **_skladniki(_uruchom(plik, _slownik_cec(m, p)),
                                     m["gamma_pmp"])})
    tab = pd.DataFrame(wiersze)
    sciezka = config.OUT_DIR / "module_fit_comparison.csv"
    tab.to_csv(sciezka, index=False)

    print()
    print(tab.to_string(index=False))

    uzysk = tab["dc_gross_kWh"].tolist()
    efekt = tab["efekt_modulu_pct"].tolist()
    print()
    print("Niepewność dopasowania (fit_desoto wobec fit_cec_sam):")
    print("  uzysk roczny DC : %+.2f kWh (%+.2f %%)"
          % (uzysk[2] - uzysk[1], (uzysk[2] - uzysk[1]) / uzysk[1] * 100.0))
    print("  efekt modułu    : %+.3f pkt proc." % (efekt[2] - efekt[1]))
    print("  składnik temp.  : %+.3f pkt proc."
          % (tab["skladnik_temperaturowy_pct"][2]
             - tab["skladnik_temperaturowy_pct"][1]))
    print("  reszta (światło): %+.3f pkt proc."
          % (tab["reszta_niskie_natezenie_pct"][2]
             - tab["reszta_niskie_natezenie_pct"][1]))
    print()
    print("Kontrola zgodności ścieżek (module_model=2 wobec module_model=1 "
          "z parametrami solwera SSC): %+.3f pkt proc."
          % (efekt[1] - efekt[0]))

    manifest.zapisz(
        PRZEBIEG,
        skrypt="simulation/fit_check.py",
        konfiguracja={
            "modul": m["nazwa"],
            "lokalizacja": loc.label,
            "rok": args.year,
            "module_model_potoku": module_model,
            "parametry_wyznaczane_wewnatrz_ssc": wewnetrzne,
            "metody": list(metody),
            "dane_katalogowe": {k: v for k, v in m.items() if k != "nazwa"},
        },
        wejscia={loc.key: plik},
        wyniki=[sciezka, sciezka_par],
    )
    print("zapisano:", sciezka.name, "oraz", sciezka_par.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
