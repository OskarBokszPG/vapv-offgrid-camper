# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: weather_validation.py
Opis: reprezentatywność TMY i rozrzut międzyzbiorowy. Moduł odpowiada na
      cztery pytania, od których zależy wiarygodność całej dalszej
      części obliczeń:

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
import weather_qc as qc
from config import LOCATIONS, Location

# Miesiące uznane za okres krytyczny dla bilansu energetycznego. Listopad-luty
# obejmuje minimum nasłonecznienia i jednocześnie scenariusz C (postój zimowy).
CRITICAL_MONTHS = (11, 12, 1, 2)
# Miesiące zimy klimatologicznej, użyte do weryfikacji rankingu lokalizacji.
WINTER_MONTHS = (12, 1, 2)
# Progi wysokości Słońca.
DAYLIGHT_ELEVATION_DEG = 0.0
# Temperatura graniczna ładowania ogniw LiFePO4.
LFP_CHARGE_LIMIT_C = 0.0

MONTH_NAMES_PL = [
    "sty", "lut", "mar", "kwi", "maj", "cze",
    "lip", "sie", "wrz", "paź", "lis", "gru",
]


# ===========================================================================
# Wczytanie danych
# ===========================================================================

def load_location(loc: Location) -> dict[str, pd.DataFrame]:
    """Wczytuje trzy zbiory dla jednej lokalizacji."""
    datasets: dict[str, pd.DataFrame] = {}

    tmy_path = config.raw_tmy_path(loc)
    if tmy_path.exists():
        datasets["TMY"], _ = qc.parse_epw(tmy_path, loc)

    for database in config.RADIATION_DATABASES:
        source = database.replace("PVGIS-", "")
        series_path = config.raw_series_path(loc, database)
        if series_path.exists():
            datasets[source], _ = qc.parse_seriescalc(series_path, loc, source)

    return datasets


def hourly_to_daily_kwh(frame: pd.DataFrame) -> pd.Series:
    """
    Agreguje godzinowe GHI [W/m2] do sum dobowych [kWh/m2].

    Przy kroku godzinowym wartość w W/m2 odpowiada energii w Wh/m2, więc suma
    dobowa dzielona przez 1000 daje kWh/m2. Agregacja po dacie UTC.
    """
    daily_wh = frame["GHI"].groupby(frame.index.floor("D")).sum()
    return daily_wh / 1000.0


# ===========================================================================
# Etap 1 — wpływ zmiany współrzędnych
# ===========================================================================

def coordinate_shift_report(legacy_dir=None) -> pd.DataFrame:
    """
    Porównuje roczne GHI dla współrzędnych stacji i centrów miejscowości.

    Pliki dla współrzędnych starych muszą zostać wcześniej pobrane do katalogu
    ``weather/raw/legacy_town_coords/``. Bez nich funkcja zwraca pustą ramkę —
    porównanie jest wtedy pomijane, a nie zgadywane.

    ROZDZIELENIE PRZYCZYN. Porównanie samych plików TMY nie wystarcza do
    orzeczenia, czy zmiana wyniku pochodzi ze zmiany komórki siatki, czy
    z doboru miesięcy przez algorytm TMY. Dlatego zestawiane są dwie pary
    wielkości: roczne GHI z plików TMY oraz średnie wieloletnie SARAH3 w tych
    samych dwóch punktach. Różnica średnich wieloletnich mierzy wyłącznie
    wpływ położenia punktu; różnica TMY zawiera dodatkowo wpływ doboru
    miesięcy. Bez tego rozdzielenia różnica rzędu kilku procent zostałaby
    przypisana geografii, podczas gdy jej źródłem jest algorytm wyboru.
    """
    legacy = legacy_dir if legacy_dir is not None else (
        config.RAW_DIR / "legacy_town_coords"
    )
    rows: list[dict[str, object]] = []

    for loc in LOCATIONS.values():
        station_path = config.raw_tmy_path(loc)
        if not station_path.exists():
            continue
        if loc.key not in config.LEGACY_TOWN_COORDS:
            # Lokalizacja dodana po pierwszym przebiegu potoku (Łeba) nie ma
            # wariantu „centrum miejscowości" — porównania nie da się zrobić
            # i nie jest ono potrzebne: wrażliwość na położenie punktu
            # wyznacza dla niej osobna kontrola przesunięcia o oczko siatki
            # SARAH3 (grid_shift_report).
            continue
        old_lat, old_lon = config.LEGACY_TOWN_COORDS[loc.key]
        town_path = legacy / f"tmy_{loc.key}_town_{old_lat}_{old_lon}.epw"
        if not town_path.exists():
            continue

        station, _ = qc.parse_epw(station_path, loc)
        town, _ = qc.parse_epw(town_path, loc)

        station_annual = float(station["GHI"].sum()) / 1000.0
        town_annual = float(town["GHI"].sum()) / 1000.0

        record: dict[str, object] = {
            "location": loc.key,
            "station_lat": loc.lat,
            "station_lon": loc.lon,
            "town_lat": old_lat,
            "town_lon": old_lon,
            "tmy_ghi_station_kwh_m2": round(station_annual, 1),
            "tmy_ghi_town_kwh_m2": round(town_annual, 1),
            "tmy_difference_pct": round(
                (station_annual - town_annual) / town_annual * 100.0, 3
            ),
        }

        # Średnie wieloletnie w obu punktach — składowa czysto geograficzna.
        station_series_path = config.raw_series_path(loc, "PVGIS-SARAH3")
        town_series_path = legacy / f"series_{loc.key}_town_sarah3_2014_2023.csv"
        years = config.SERIES_END_YEAR - config.SERIES_START_YEAR + 1
        if station_series_path.exists() and town_series_path.exists():
            station_series, _ = qc.parse_seriescalc(
                station_series_path, loc, "SARAH3"
            )
            town_series, _ = qc.parse_seriescalc(town_series_path, loc, "SARAH3")
            station_mean = float(station_series["GHI"].sum()) / 1000.0 / years
            town_mean = float(town_series["GHI"].sum()) / 1000.0 / years
            record.update({
                "sarah3_mean_station_kwh_m2": round(station_mean, 1),
                "sarah3_mean_town_kwh_m2": round(town_mean, 1),
                "sarah3_difference_pct": round(
                    (station_mean - town_mean) / town_mean * 100.0, 3
                ),
                "grid_cell_effect_pct": round(
                    (station_mean - town_mean) / town_mean * 100.0, 3
                ),
                "tmy_selection_effect_pct": round(
                    (station_annual - town_annual) / town_annual * 100.0
                    - (station_mean - town_mean) / town_mean * 100.0,
                    3,
                ),
            })

        rows.append(record)

    return pd.DataFrame(rows)


# ===========================================================================
# Etap 5a — reprezentatywność TMY
# ===========================================================================

def tmy_representativeness(loc: Location, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Porównuje miesięczne i roczne GHI pliku TMY ze średnią wielolecia SARAH3.

    Odchylenie podawane jest w procentach średniej wieloletniej. Wartości
    dodatnie oznaczają, że TMY jest „pogodniejszy" od średniej.
    """
    if "TMY" not in datasets or "SARAH3" not in datasets:
        return pd.DataFrame()

    tmy = datasets["TMY"]
    series = datasets["SARAH3"]

    tmy_monthly = tmy["GHI"].groupby(tmy.index.month).sum() / 1000.0

    # Średnia wieloletnia: suma miesięczna w każdym roku, potem średnia po latach.
    monthly_by_year = (
        series["GHI"].groupby([series.index.year, series.index.month]).sum() / 1000.0
    )
    monthly_by_year.index.names = ["year", "month"]
    series_monthly_mean = monthly_by_year.groupby("month").mean()
    series_monthly_std = monthly_by_year.groupby("month").std()

    rows: list[dict[str, object]] = []
    for month in range(1, 13):
        tmy_value = float(tmy_monthly.get(month, np.nan))
        mean_value = float(series_monthly_mean.get(month, np.nan))
        std_value = float(series_monthly_std.get(month, np.nan))
        rows.append({
            "location": loc.key,
            "month": month,
            "month_name": MONTH_NAMES_PL[month - 1],
            "tmy_ghi_kwh_m2": round(tmy_value, 2),
            "sarah3_mean_ghi_kwh_m2": round(mean_value, 2),
            "sarah3_std_ghi_kwh_m2": round(std_value, 2),
            "deviation_pct": round((tmy_value - mean_value) / mean_value * 100.0, 2),
            # Odchylenie wyrażone w jednostkach rozrzutu międzyrocznego —
            # pokazuje, czy TMY mieści się w naturalnej zmienności miesiąca.
            "deviation_in_sigma": round((tmy_value - mean_value) / std_value, 2)
            if std_value > 0 else np.nan,
            "critical_month": month in CRITICAL_MONTHS,
        })

    annual_tmy = float(tmy_monthly.sum())
    annual_mean = float(monthly_by_year.groupby("year").sum().mean())
    rows.append({
        "location": loc.key,
        "month": 0,
        "month_name": "ROK",
        "tmy_ghi_kwh_m2": round(annual_tmy, 2),
        "sarah3_mean_ghi_kwh_m2": round(annual_mean, 2),
        "sarah3_std_ghi_kwh_m2": round(
            float(monthly_by_year.groupby("year").sum().std()), 2
        ),
        "deviation_pct": round((annual_tmy - annual_mean) / annual_mean * 100.0, 2),
        "deviation_in_sigma": np.nan,
        "critical_month": False,
    })

    return pd.DataFrame(rows)


# ===========================================================================
# Etap 5b — zmienność międzyroczna i okna najniekorzystniejsze
# ===========================================================================

def interannual_variability(loc: Location, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Zestawia roczne i grudniowe sumy GHI dla kolejnych lat wielolecia."""
    if "SARAH3" not in datasets:
        return pd.DataFrame()

    series = datasets["SARAH3"]
    annual = series["GHI"].groupby(series.index.year).sum() / 1000.0
    december = (
        series.loc[series.index.month == 12, "GHI"]
        .groupby(series.loc[series.index.month == 12].index.year).sum() / 1000.0
    )
    critical = series.loc[series.index.month.isin(CRITICAL_MONTHS)]
    critical_sum = critical["GHI"].groupby(critical.index.year).sum() / 1000.0

    rows: list[dict[str, object]] = []
    for year in sorted(annual.index):
        rows.append({
            "location": loc.key,
            "year": int(year),
            "annual_ghi_kwh_m2": round(float(annual[year]), 2),
            "december_ghi_kwh_m2": round(float(december.get(year, np.nan)), 2),
            "nov_feb_ghi_kwh_m2": round(float(critical_sum.get(year, np.nan)), 2),
        })
    return pd.DataFrame(rows)


def variability_statistics(variability: pd.DataFrame) -> pd.DataFrame:
    """Liczy średnią, odchylenie i współczynnik zmienności dla każdej kolumny."""
    rows: list[dict[str, object]] = []
    for location, group in variability.groupby("location"):
        for column, label in [
            ("annual_ghi_kwh_m2", "rok"),
            ("december_ghi_kwh_m2", "grudzień"),
            ("nov_feb_ghi_kwh_m2", "XI-II"),
        ]:
            values = group[column].dropna()
            if values.empty:
                continue
            mean_value = float(values.mean())
            std_value = float(values.std())
            worst_year = int(group.loc[values.idxmin(), "year"])
            rows.append({
                "location": location,
                "period": label,
                "mean_kwh_m2": round(mean_value, 2),
                "std_kwh_m2": round(std_value, 2),
                "cv_pct": round(std_value / mean_value * 100.0, 2),
                "min_kwh_m2": round(float(values.min()), 2),
                "max_kwh_m2": round(float(values.max()), 2),
                "worst_year": worst_year,
                "worst_vs_mean_pct": round(
                    (float(values.min()) - mean_value) / mean_value * 100.0, 2
                ),
            })
    return pd.DataFrame(rows)


def worst_windows(loc: Location, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Wyznacza najniekorzystniejsze okna 7- i 14-dniowe sumy GHI.

    Okna liczone są jako sumy kroczące po sumach dobowych. Raportowane są dwa
    warianty: minimum globalne w całym wieloleciu oraz minimum ograniczone do
    okien w pełni zawartych w miesiącach krytycznych XI-II. Pierwsze odpowiada
    na pytanie o przypadek najgorszy w ogóle, drugie o najgorszy przypadek
    w sezonie, dla którego dobierany jest magazyn.
    """
    if "SARAH3" not in datasets:
        return pd.DataFrame()

    daily = hourly_to_daily_kwh(datasets["SARAH3"])
    rows: list[dict[str, object]] = []

    for window in (7, 14):
        rolling = daily.rolling(window=window, min_periods=window).sum()

        # Wariant globalny.
        end_date = rolling.idxmin()
        rows.append({
            "location": loc.key,
            "window_days": window,
            "scope": "całe wielolecie",
            "window_start": str((end_date - pd.Timedelta(days=window - 1)).date()),
            "window_end": str(end_date.date()),
            "ghi_sum_kwh_m2": round(float(rolling.min()), 3),
            "daily_mean_kwh_m2": round(float(rolling.min()) / window, 4),
        })

        # Wariant ograniczony do okien zawartych w miesiącach krytycznych.
        start_months = (rolling.index - pd.Timedelta(days=window - 1)).month
        inside = pd.Series(
            np.isin(start_months, CRITICAL_MONTHS)
            & np.isin(rolling.index.month, CRITICAL_MONTHS),
            index=rolling.index,
        )
        restricted = rolling.where(inside)
        if restricted.notna().any():
            end_date = restricted.idxmin()
            rows.append({
                "location": loc.key,
                "window_days": window,
                "scope": "okna w XI-II",
                "window_start": str((end_date - pd.Timedelta(days=window - 1)).date()),
                "window_end": str(end_date.date()),
                "ghi_sum_kwh_m2": round(float(restricted.min()), 3),
                "daily_mean_kwh_m2": round(float(restricted.min()) / window, 4),
            })

    return pd.DataFrame(rows)


# ===========================================================================
# Etap 5c — rozrzut międzyzbiorowy SARAH3 vs ERA5
# ===========================================================================

def inter_dataset_spread(loc: Location, datasets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Porównuje godzinowe GHI zbiorów SARAH3 i ERA5 w godzinach dziennych.

    NIE JEST TO WALIDACJA. Oba zbiory są produktami modelowymi; ich różnica
    opisuje rozrzut międzyzbiorowy, nie błąd względem pomiaru. ERA5 traktowany
    jest jako odniesienie wyłącznie arytmetycznie, aby znak różnicy był
    jednoznaczny: MBE > 0 oznacza, że SARAH3 podaje wartości wyższe.

    Zwraca krotkę (statystyki zbiorcze, odchylenie miesięczne).
    """
    if not {"SARAH3", "ERA5"} <= set(datasets):
        return pd.DataFrame(), pd.DataFrame()

    # Znaczniki czasu obu zbiorów różnią się przesunięciem minutowym
    # (SARAH3 :11, ERA5 :30), dlatego łączenie odbywa się po pełnej godzinie,
    # do której rekord się odnosi, a nie po dokładnym znaczniku.
    sarah = datasets["SARAH3"].copy()
    era = datasets["ERA5"].copy()
    sarah["hour_key"] = sarah.index.floor("h")
    era["hour_key"] = era.index.floor("h")

    merged = pd.merge(
        sarah[["hour_key", "GHI", "T2m"]].rename(
            columns={"GHI": "ghi_sarah3", "T2m": "t2m_sarah3"}
        ),
        era[["hour_key", "GHI", "T2m"]].rename(
            columns={"GHI": "ghi_era5", "T2m": "t2m_era5"}
        ),
        on="hour_key",
        how="inner",
    ).set_index("hour_key")

    # Maska dnia liczona raz, dla wspólnej osi czasu.
    with_elevation = qc.add_sun_elevation(merged, loc)
    daylight = with_elevation["sun_elevation"] > DAYLIGHT_ELEVATION_DEG
    day = with_elevation.loc[daylight]

    difference = day["ghi_sarah3"] - day["ghi_era5"]
    mean_reference = float(day["ghi_era5"].mean())

    summary = pd.DataFrame([{
        "location": loc.key,
        "comparison": "SARAH3 minus ERA5",
        "label": "inter-dataset spread (NOT validation against ground truth)",
        "n_hours_daylight": int(len(day)),
        "mean_ghi_sarah3_w_m2": round(float(day["ghi_sarah3"].mean()), 2),
        "mean_ghi_era5_w_m2": round(mean_reference, 2),
        "mbe_w_m2": round(float(difference.mean()), 3),
        "mbe_pct_of_mean": round(float(difference.mean()) / mean_reference * 100.0, 2),
        "rmse_w_m2": round(float(np.sqrt((difference ** 2).mean())), 2),
        "rmse_pct_of_mean": round(
            float(np.sqrt((difference ** 2).mean())) / mean_reference * 100.0, 2
        ),
        "correlation": round(float(day["ghi_sarah3"].corr(day["ghi_era5"])), 4),
        "annual_ghi_sarah3_kwh_m2": round(
            float(datasets["SARAH3"]["GHI"].sum()) / 1000.0 / 10.0, 2
        ),
        "annual_ghi_era5_kwh_m2": round(
            float(datasets["ERA5"]["GHI"].sum()) / 1000.0 / 10.0, 2
        ),
    }])

    # Odchylenie miesięczne — liczone na sumach miesięcznych, bo to one wchodzą
    # do bilansu energetycznego, a nie średnie z wartości chwilowych.
    sarah_monthly = (
        datasets["SARAH3"]["GHI"].groupby(datasets["SARAH3"].index.month).sum()
        / 1000.0 / 10.0
    )
    era_monthly = (
        datasets["ERA5"]["GHI"].groupby(datasets["ERA5"].index.month).sum()
        / 1000.0 / 10.0
    )
    monthly_rows: list[dict[str, object]] = []
    for month in range(1, 13):
        sarah_value = float(sarah_monthly.get(month, np.nan))
        era_value = float(era_monthly.get(month, np.nan))
        monthly_rows.append({
            "location": loc.key,
            "month": month,
            "month_name": MONTH_NAMES_PL[month - 1],
            "sarah3_kwh_m2": round(sarah_value, 2),
            "era5_kwh_m2": round(era_value, 2),
            "bias_kwh_m2": round(sarah_value - era_value, 2),
            "bias_pct": round((sarah_value - era_value) / era_value * 100.0, 2),
        })

    return summary, pd.DataFrame(monthly_rows)


# ===========================================================================
# Etap 5d — temperatura a blokada ładowania LFP
# ===========================================================================

def temperature_statistics(loc: Location, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Zestawia liczbę godzin z temperaturą poniżej zera w podziale na miesiące.

    Godziny poniżej 0 stopni C wyznaczają okres, w którym ładowanie ogniw
    LiFePO4 musi zostać zablokowane. Dla wielolecia podawana jest średnia
    liczba godzin na rok, aby wartości były porównywalne z plikiem TMY.

    UWAGA — normalizacja. Liczbę lat trzeba zadać jawnie, a nie wyprowadzać
    z pola roku w danych: typowy rok meteorologiczny składa się z miesięcy
    pochodzących z różnych lat, więc liczba różnych wartości pola roku wynosi
    dla TMY od 8 do 12, mimo że szereg opisuje jeden rok. Wyprowadzenie
    dzielnika z tego pola zaniżyłoby wynik dla TMY o rząd wielkości.

    UWAGA — źródło temperatury. Parametr ``raddatabase`` interfejsu PVGIS
    przełącza wyłącznie źródło promieniowania. Kolumny T2m i WS10m są
    w obu zbiorach bitowo identyczne (sprawdzone: maksymalna różnica
    0,000000 dla wszystkich 87 648 godzin w każdej lokalizacji), ponieważ
    pochodzą z tej samej reanalizy ERA5. Wiersze SARAH3 i ERA5 w tej tabeli
    opisują więc jeden zbiór temperatur, nie dwa niezależne.
    """
    rows: list[dict[str, object]] = []

    for source in ("TMY", "SARAH3", "ERA5"):
        if source not in datasets:
            continue
        frame = datasets[source]
        # Dzielnik zadany jawnie: TMY to jeden rok, szeregi obejmują
        # zadeklarowany w konfiguracji zakres lat kalendarzowych.
        if source == "TMY":
            years = 1
        else:
            years = config.SERIES_END_YEAR - config.SERIES_START_YEAR + 1
        below = frame.loc[frame["T2m"] < LFP_CHARGE_LIMIT_C]

        hours_by_month = below["T2m"].groupby(below.index.month).count()
        for month in range(1, 13):
            rows.append({
                "location": loc.key,
                "source": source,
                "t2m_source": "ERA5 (wspólny dla obu baz promieniowania)"
                              if source != "TMY" else "TMY",
                "month": month,
                "month_name": MONTH_NAMES_PL[month - 1],
                "years_normalised": years,
                "hours_below_zero_per_year": round(
                    float(hours_by_month.get(month, 0)) / years, 1
                ),
            })
        rows.append({
            "location": loc.key,
            "source": source,
            "t2m_source": "ERA5 (wspólny dla obu baz promieniowania)"
                          if source != "TMY" else "TMY",
            "month": 0,
            "month_name": "ROK",
            "years_normalised": years,
            "hours_below_zero_per_year": round(float(len(below)) / years, 1),
        })

    return pd.DataFrame(rows)


def temperature_extremes(loc: Location, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Porównuje minimalne temperatury w TMY i w wieloleciu.

    Kolumna ``distinct_year_labels`` nie jest długością szeregu: dla TMY jest
    to liczba różnych lat, z których pochodzą miesiące typowego roku, dla
    szeregów wieloletnich liczba lat kalendarzowych. Rozróżnienie zapisano
    jawnie, bo pomyłka w tym miejscu była źródłem błędu normalizacji
    w pierwszym przebiegu.
    """
    rows: list[dict[str, object]] = []
    for source in ("TMY", "SARAH3", "ERA5"):
        if source not in datasets:
            continue
        frame = datasets[source]
        rows.append({
            "location": loc.key,
            "source": source,
            "t2m_min_c": round(float(frame["T2m"].min()), 2),
            "t2m_max_c": round(float(frame["T2m"].max()), 2),
            "t2m_mean_c": round(float(frame["T2m"].mean()), 2),
            "hours_below_zero_total": int((frame["T2m"] < LFP_CHARGE_LIMIT_C).sum()),
            "records": int(len(frame)),
            "distinct_year_labels": int(frame.index.year.nunique()),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Weryfikacja rankingu zimowego
# ===========================================================================

def hours_below_zero_by_year(loc: Location, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Zestawia liczbę godzin z temperaturą poniżej zera w kolejnych latach.

    Rozrzut międzyroczny tej wielkości decyduje o tym, jak długo w roju
    niekorzystnym blokada ładowania LFP pozostaje aktywna, a więc jak
    konserwatywnie trzeba dobrać źródło alternatywne (rozdz. 5.4).
    """
    if "SARAH3" not in datasets:
        return pd.DataFrame()
    frame = datasets["SARAH3"]
    below = frame.loc[frame["T2m"] < LFP_CHARGE_LIMIT_C]
    by_year = below.groupby(below.index.year).size()
    return pd.DataFrame([
        {
            "location": loc.key,
            "year": int(year),
            "hours_below_zero": int(count),
        }
        for year, count in by_year.items()
    ])


def winter_ranking(all_datasets: dict[str, dict[str, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sprawdza na wieloleciu ranking lokalizacji według nasłonecznienia zimowego.

    Wcześniejszy przebieg potoku oparty na jednym roku TMY wskazywał porządek
    Lesko > Suwałki > Łeba w miesiącach XII-II. Twierdzenie oparte na jednym
    roku typowym nie ma mocy klimatologicznej, dlatego jest tu sprawdzane na
    dziesięciu latach danych SARAH3, osobno dla każdego sezonu zimowego.

    Sezon zimowy definiowany jest jako grudzień roku Y oraz styczeń i luty
    roku Y+1, co daje dziewięć pełnych sezonów w okresie 2014-2023 (styczeń
    i luty 2014 oraz grudzień 2023 należą do sezonów niepełnych i są pomijane).

    Zwraca krotkę (sumy sezonowe, podsumowanie częstości rankingu).
    """
    rows: list[dict[str, object]] = []

    for key, datasets in all_datasets.items():
        if "SARAH3" not in datasets:
            continue
        frame = datasets["SARAH3"]
        winter = frame.loc[frame.index.month.isin(WINTER_MONTHS)].copy()
        # Etykieta sezonu: grudzień należy do sezonu rozpoczynającego się
        # w tym samym roku, styczeń i luty do sezonu rozpoczętego rok wcześniej.
        season = np.where(
            winter.index.month == 12, winter.index.year, winter.index.year - 1
        )
        winter["season"] = season
        seasonal = winter["GHI"].groupby(winter["season"]).sum() / 1000.0
        counts = winter["GHI"].groupby(winter["season"]).count()

        for season_year in sorted(seasonal.index):
            # Sezon pełny zawiera grudzień (do 744 h) oraz styczeń i luty.
            if counts[season_year] < 2000:
                continue
            rows.append({
                "location": key,
                "season": f"{season_year}/{str(season_year + 1)[-2:]}",
                "season_start_year": int(season_year),
                "winter_ghi_kwh_m2": round(float(seasonal[season_year]), 2),
            })

    seasonal_frame = pd.DataFrame(rows)
    if seasonal_frame.empty:
        return seasonal_frame, pd.DataFrame()

    # Ranking w każdym sezonie osobno.
    pivot = seasonal_frame.pivot(
        index="season", columns="location", values="winter_ghi_kwh_m2"
    ).dropna()

    ranking_rows: list[dict[str, object]] = []
    expected_order = ["Lesko", "Suwalki", "Leba"]
    available = [name for name in expected_order if name in pivot.columns]

    for season, record in pivot.iterrows():
        order = list(record.sort_values(ascending=False).index)
        ranking_rows.append({
            "season": season,
            "order_descending": " > ".join(order),
            "matches_lesko_suwalki_hel": order == available,
            **{f"ghi_{name}": round(float(record[name]), 2) for name in pivot.columns},
        })

    ranking = pd.DataFrame(ranking_rows)
    return seasonal_frame, ranking


# ===========================================================================
# Wykresy podglądowe
# ===========================================================================

def make_previews(
    representativeness: pd.DataFrame,
    variability: pd.DataFrame,
    monthly_spread: pd.DataFrame,
    seasonal_winter: pd.DataFrame,
) -> list[str]:
    """
    Rysuje wykresy podglądowe. Nie są to rysunki do pracy — służą wyłącznie
    szybkiej kontroli wzrokowej wyników liczbowych.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    config.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. TMY vs średnia wielolecia, miesięcznie.
    monthly = representativeness[representativeness["month"] > 0]
    if not monthly.empty:
        locations = sorted(monthly["location"].unique())
        figure, axes = plt.subplots(1, len(locations), figsize=(5 * len(locations), 4),
                                    sharey=True)
        axes = np.atleast_1d(axes)
        for axis, location in zip(axes, locations):
            group = monthly[monthly["location"] == location].sort_values("month")
            positions = np.arange(12)
            axis.bar(positions - 0.2, group["sarah3_mean_ghi_kwh_m2"], width=0.4,
                     label="SARAH3 średnia 2014-2023")
            axis.bar(positions + 0.2, group["tmy_ghi_kwh_m2"], width=0.4,
                     label="TMY")
            axis.errorbar(positions - 0.2, group["sarah3_mean_ghi_kwh_m2"],
                          yerr=group["sarah3_std_ghi_kwh_m2"], fmt="none",
                          ecolor="black", capsize=2, linewidth=0.8)
            axis.set_title(location)
            axis.set_xticks(positions)
            axis.set_xticklabels(MONTH_NAMES_PL, rotation=45)
            axis.set_xlabel("Miesiąc")
        axes[0].set_ylabel("Nasłonecznienie [kWh/m2]")
        axes[0].legend(fontsize=8)
        figure.tight_layout()
        path = config.PREVIEW_DIR / "tmy_vs_multiyear.png"
        figure.savefig(path, dpi=110)
        plt.close(figure)
        written.append(path.name)

    # 2. Zmienność międzyroczna.
    if not variability.empty:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for location, group in variability.groupby("location"):
            axes[0].plot(group["year"], group["annual_ghi_kwh_m2"],
                         marker="o", label=location)
            axes[1].plot(group["year"], group["nov_feb_ghi_kwh_m2"],
                         marker="s", label=location)
        axes[0].set_title("Roczne GHI (SARAH3)")
        axes[1].set_title("GHI w miesiącach XI-II (SARAH3)")
        for axis in axes:
            axis.set_xlabel("Rok")
            axis.set_ylabel("Nasłonecznienie [kWh/m2]")
            axis.legend(fontsize=8)
            axis.grid(alpha=0.3)
        figure.tight_layout()
        path = config.PREVIEW_DIR / "interannual_variability.png"
        figure.savefig(path, dpi=110)
        plt.close(figure)
        written.append(path.name)

    # 3. Rozrzut międzyzbiorowy, miesięcznie.
    if not monthly_spread.empty:
        figure, axis = plt.subplots(figsize=(8, 4))
        for location, group in monthly_spread.groupby("location"):
            axis.plot(group["month"], group["bias_pct"], marker="o", label=location)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title("Rozrzut międzyzbiorowy SARAH3 - ERA5 (NIE walidacja)")
        axis.set_xlabel("Miesiąc")
        axis.set_ylabel("Różnica [% ERA5]")
        axis.set_xticks(range(1, 13))
        axis.set_xticklabels(MONTH_NAMES_PL)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
        figure.tight_layout()
        path = config.PREVIEW_DIR / "sarah3_vs_era5_monthly.png"
        figure.savefig(path, dpi=110)
        plt.close(figure)
        written.append(path.name)

    # 4. Ranking zimowy w kolejnych sezonach.
    if not seasonal_winter.empty:
        figure, axis = plt.subplots(figsize=(8, 4))
        for location, group in seasonal_winter.groupby("location"):
            axis.plot(group["season"], group["winter_ghi_kwh_m2"],
                      marker="o", label=location)
        axis.set_title("Nasłonecznienie zimowe XII-II w kolejnych sezonach (SARAH3)")
        axis.set_xlabel("Sezon")
        axis.set_ylabel("Nasłonecznienie [kWh/m2]")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
        figure.tight_layout()
        path = config.PREVIEW_DIR / "winter_ranking.png"
        figure.savefig(path, dpi=110)
        plt.close(figure)
        written.append(path.name)

    return written


# ===========================================================================
# Uruchomienie
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reprezentatywność TMY, zmienność i rozrzut międzyzbiorowy."
    )
    parser.add_argument("--no-plots", action="store_true", dest="no_plots",
                        help="pomiń wykresy podglądowe")
    args = parser.parse_args()

    config.ensure_dirs()

    print("=" * 72)
    print("Wczytywanie danych")
    print("=" * 72)
    all_datasets: dict[str, dict[str, pd.DataFrame]] = {}
    for key, loc in LOCATIONS.items():
        datasets = load_location(loc)
        all_datasets[key] = datasets
        print(f"  {key:9s} {', '.join(f'{k}={len(v)}' for k, v in datasets.items())}")

    # --- Etap 1: wpływ zmiany współrzędnych -------------------------------
    shift = coordinate_shift_report()
    if not shift.empty:
        shift.to_csv(config.OUT_DIR / "step1_coordinate_shift.csv", index=False)
        print("\n" + "=" * 72)
        print("Etap 1 — wpływ zmiany współrzędnych na roczne GHI (TMY)")
        print("=" * 72)
        columns = ["location", "tmy_ghi_town_kwh_m2", "tmy_ghi_station_kwh_m2",
                   "tmy_difference_pct"]
        if "sarah3_difference_pct" in shift.columns:
            columns += ["sarah3_mean_town_kwh_m2", "sarah3_mean_station_kwh_m2",
                        "grid_cell_effect_pct", "tmy_selection_effect_pct"]
        print(shift[columns].to_string(index=False))

        worst = shift["tmy_difference_pct"].abs().max()
        print(f"\nNajwiększa zmiana GHI w plikach TMY: {worst:.3f} %  "
              f"({'poniżej' if worst < 1.0 else 'POWYŻEJ'} progu 1 %)")
        if "grid_cell_effect_pct" in shift.columns:
            grid_worst = shift["grid_cell_effect_pct"].abs().max()
            print(f"Z tego wpływ zmiany komórki siatki: do {grid_worst:.3f} % "
                  f"— resztę wnosi dobór miesięcy przez algorytm TMY.")
    else:
        print("\nEtap 1 — pominięty: brak plików dla współrzędnych centrów "
              "miejscowości w weather/raw/legacy_town_coords/.")

    # --- Etap 5a ----------------------------------------------------------
    representativeness = pd.concat(
        [tmy_representativeness(LOCATIONS[k], d) for k, d in all_datasets.items()],
        ignore_index=True,
    )
    representativeness.to_csv(
        config.OUT_DIR / "val_tmy_representativeness.csv", index=False
    )
    print("\n" + "=" * 72)
    print("Etap 5a — reprezentatywność TMY względem średniej 2014-2023 (SARAH3)")
    print("=" * 72)
    for location, group in representativeness.groupby("location"):
        annual = group[group["month"] == 0].iloc[0]
        critical = group[group["critical_month"] & (group["month"] > 0)]
        print(f"\n  {location}:  rok TMY {annual['tmy_ghi_kwh_m2']:.1f} vs "
              f"średnia {annual['sarah3_mean_ghi_kwh_m2']:.1f} kWh/m2  "
              f"({annual['deviation_pct']:+.2f} %)")
        print(f"    miesiące krytyczne XI-II:")
        for _, record in critical.sort_values("month").iterrows():
            print(f"      {record['month_name']}  TMY {record['tmy_ghi_kwh_m2']:6.2f}  "
                  f"średnia {record['sarah3_mean_ghi_kwh_m2']:6.2f}  "
                  f"{record['deviation_pct']:+7.2f} %  "
                  f"({record['deviation_in_sigma']:+.2f} sigma)")

    # --- Etap 5b ----------------------------------------------------------
    variability = pd.concat(
        [interannual_variability(LOCATIONS[k], d) for k, d in all_datasets.items()],
        ignore_index=True,
    )
    variability.to_csv(config.OUT_DIR / "val_interannual.csv", index=False)
    statistics = variability_statistics(variability)
    statistics.to_csv(config.OUT_DIR / "val_variability_stats.csv", index=False)

    print("\n" + "=" * 72)
    print("Etap 5b — zmienność międzyroczna (SARAH3 2014-2023)")
    print("=" * 72)
    print(statistics.to_string(index=False))

    windows = pd.concat(
        [worst_windows(LOCATIONS[k], d) for k, d in all_datasets.items()],
        ignore_index=True,
    )
    windows.to_csv(config.OUT_DIR / "val_worst_windows.csv", index=False)
    print("\n  Okna najniekorzystniejsze:")
    print(windows.to_string(index=False))

    # --- Etap 5c ----------------------------------------------------------
    spread_parts = [inter_dataset_spread(LOCATIONS[k], d) for k, d in all_datasets.items()]
    spread = pd.concat([part[0] for part in spread_parts], ignore_index=True)
    monthly_spread = pd.concat([part[1] for part in spread_parts], ignore_index=True)
    spread.to_csv(config.OUT_DIR / "val_interdataset_spread.csv", index=False)
    monthly_spread.to_csv(config.OUT_DIR / "val_interdataset_monthly.csv", index=False)

    print("\n" + "=" * 72)
    print("Etap 5c — rozrzut międzyzbiorowy SARAH3 vs ERA5")
    print("UWAGA: rozrzut między dwoma zbiorami modelowymi, NIE walidacja.")
    print("=" * 72)
    print(spread[[
        "location", "n_hours_daylight", "mean_ghi_era5_w_m2", "mbe_w_m2",
        "mbe_pct_of_mean", "rmse_w_m2", "rmse_pct_of_mean", "correlation",
        "annual_ghi_sarah3_kwh_m2", "annual_ghi_era5_kwh_m2",
    ]].to_string(index=False))

    # --- Etap 5d ----------------------------------------------------------
    temperature = pd.concat(
        [temperature_statistics(LOCATIONS[k], d) for k, d in all_datasets.items()],
        ignore_index=True,
    )
    temperature.to_csv(config.OUT_DIR / "val_temperature_below_zero.csv", index=False)
    extremes = pd.concat(
        [temperature_extremes(LOCATIONS[k], d) for k, d in all_datasets.items()],
        ignore_index=True,
    )
    extremes.to_csv(config.OUT_DIR / "val_temperature_extremes.csv", index=False)

    below_by_year = pd.concat(
        [hours_below_zero_by_year(LOCATIONS[k], d) for k, d in all_datasets.items()],
        ignore_index=True,
    )
    below_by_year.to_csv(
        config.OUT_DIR / "val_hours_below_zero_by_year.csv", index=False
    )

    print("\n" + "=" * 72)
    print("Etap 5d — temperatura a blokada ładowania LFP")
    print("=" * 72)
    print("  UWAGA: kolumny T2m i WS10m są w zbiorach SARAH3 i ERA5 bitowo")
    print("  identyczne — PVGIS przełącza parametrem raddatabase wyłącznie")
    print("  źródło promieniowania. To jeden zbiór temperatur, nie dwa.")
    annual_hours = temperature[temperature["month"] == 0].pivot(
        index="location", columns="source", values="hours_below_zero_per_year"
    )
    print("\n  Godziny z T2m < 0 st. C na rok:")
    print(annual_hours.to_string())
    if not below_by_year.empty:
        print("\n  Rozrzut międzyroczny godzin T2m < 0 (SARAH3/ERA5):")
        spread_rows = below_by_year.groupby("location")["hours_below_zero"].agg(
            ["mean", "min", "max"]
        ).round(1)
        worst = below_by_year.loc[
            below_by_year.groupby("location")["hours_below_zero"].idxmax()
        ].set_index("location")["year"]
        spread_rows["worst_year"] = worst
        print(spread_rows.to_string())
    print("\n  Temperatury skrajne:")
    print(extremes.to_string(index=False))

    # --- Weryfikacja rankingu zimowego ------------------------------------
    seasonal_winter, ranking = winter_ranking(all_datasets)
    if not ranking.empty:
        seasonal_winter.to_csv(config.OUT_DIR / "val_winter_seasonal.csv", index=False)
        ranking.to_csv(config.OUT_DIR / "val_winter_ranking.csv", index=False)
        print("\n" + "=" * 72)
        print("Weryfikacja rankingu zimowego na wieloleciu")
        print("=" * 72)
        print(ranking.to_string(index=False))
        matches = int(ranking["matches_lesko_suwalki_hel"].sum())
        total = len(ranking)
        print(f"\n  Porządek Lesko > Suwałki > Łeba zachodzi w {matches} "
              f"z {total} pełnych sezonów zimowych.")

    # --- Wykresy podglądowe -----------------------------------------------
    if not args.no_plots:
        written = make_previews(
            representativeness, variability, monthly_spread, seasonal_winter
        )
        print(f"\nWykresy podglądowe: {', '.join(written)}")

    print(f"\nTabele zapisane w {config.OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
