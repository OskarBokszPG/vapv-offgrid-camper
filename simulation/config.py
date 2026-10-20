# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: config.py
Opis: jedyne źródło prawdy o lokalizacjach i ścieżkach potoku. Wszystkie
      skrypty w katalogu ``simulation/`` importują dane lokalizacji
      wyłącznie z tego modułu. Powielanie tabeli współrzędnych w kilku
      plikach było przyczyną rozbieżności między etapem pobierania
      danych a etapem symulacji, dlatego tabela występuje tu dokładnie
      raz.

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
from pathlib import Path

# ---------------------------------------------------------------------------
# Ścieżki
# ---------------------------------------------------------------------------
# Wszystkie ścieżki wyprowadzane są z położenia tego pliku, nie z katalogu
# roboczego — skrypty muszą dawać ten sam wynik niezależnie od miejsca
# uruchomienia.
SIM_DIR = Path(__file__).resolve().parent
WEATHER_DIR = SIM_DIR / "weather"
RAW_DIR = WEATHER_DIR / "raw"               # dane surowe, nigdy nie modyfikowane
PROVIDED_DIR = RAW_DIR / "provided_by_author"  # pliki pobrane ręcznie przez autora
OUT_DIR = SIM_DIR / "out"                   # tabele wynikowe do PGFPlots
PREVIEW_DIR = OUT_DIR / "preview"           # wykresy podglądowe, poza pracą
NOTES_DIR = SIM_DIR.parent / "notes"


def ensure_dirs() -> None:
    """Tworzy katalogi wyjściowe, jeśli nie istnieją."""
    for directory in (WEATHER_DIR, RAW_DIR, PROVIDED_DIR, OUT_DIR, PREVIEW_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Lokalizacje referencyjne
# ---------------------------------------------------------------------------
# UWAGA — współrzędne PROWIZORYCZNE, do zastąpienia dokładnymi wartościami
# z metadanych stacji IMGW (klimat.imgw.pl/pl/meta-dane). Przyjęto położenie
# stacji synoptycznych, a nie centrów miejscowości, ponieważ dane PVGIS będą
# porównywane z pomiarami naziemnymi wykonywanymi właśnie na tych stacjach;
# porównanie w innym punkcie siatki wprowadzałoby błąd przestrzenny
# nierozdzielny od błędu samego modelu satelitarnego.

@dataclass(frozen=True)
class Location:
    """Lokalizacja referencyjna wraz z identyfikatorem stacji pomiarowej."""

    key: str          # identyfikator używany w nazwach plików
    label: str        # nazwa do wydruków i podpisów
    wmo: str          # numer stacji WMO (do zapytań IMGW)
    lat: float        # szerokość geograficzna, stopnie dziesiętne, dodatnia N
    lon: float        # długość geograficzna, stopnie dziesiętne, dodatnia E
    # Wysokość n.p.m. zgłaszana przez PVGIS dla danej komórki siatki (nie
    # wysokość stacji IMGW). Wartość wpisywana do nagłówka pliku zasobu
    # słonecznego SAM; wpływa na masę optyczną atmosfery. Zgodność z nagłówkiem
    # pobranego pliku jest sprawdzana w make_sam_resource.verify_elevation().
    elevation_m: float
    description: str  # uzasadnienie wyboru lokalizacji (rozdz. 4.4)

    @property
    def solar_noon_utc_h(self) -> float:
        """
        Przybliżona godzina południa słonecznego w UTC.

        Zależność uproszczona: 12:00 UTC pomniejszone o przesunięcie wynikające
        z długości geograficznej (15 stopni = 1 h). Pomija równanie czasu,
        którego amplituda nie przekracza ok. 16 min, co dla kontroli osi czasu
        z rozdzielczością godzinową jest wystarczające.
        """
        return 12.0 - self.lon / 15.0


LOCATIONS: dict[str, Location] = {
    # Lokalizacja nadmorska: Łeba, a nie Hel. Powód jest metodyczny, nie
    # klimatyczny — IMGW-PIB udostępnia dla Łeby godzinowe promieniowanie
    # całkowite ORAZ rozproszone, czego dla Helu nie ma. Walidacja naziemna
    # jest w tej pracy warunkiem rozstrzygającym o doborze lokalizacji
    # nadmorskiej. Dane Helu zarchiwizowano w archiwum/hel/ (poza potokiem).
    #
    # Współrzędne: stacja IMGW w Łebie, ul. Rąbka 1a. Wartości za wykazem
    # stacji pomiarowych GIOŚ (stacja należąca do IMGW-PIB; 54,754139 N;
    # 17,534528 E; 2 m n.p.m.), zaokrąglone do trzech miejsc.
    # DO POTWIERDZENIA metadanymi IMGW przy dostawie danych godzinowych.
    "Leba": Location(
        key="Leba",
        label="Łeba",
        wmo="12120",
        lat=54.754,
        lon=17.535,
        elevation_m=2.0,
        description="wybrzeże Bałtyku, stacja z pomiarem promieniowania "
                    "całkowitego i rozproszonego",
    ),
    "Suwalki": Location(
        key="Suwalki",
        label="Suwałki",
        wmo="12195",
        lat=54.13,
        lon=22.95,
        elevation_m=182.0,
        description="klimat kontynentalny chłodny, najtrudniejszy scenariusz zimowy",
    ),
    "Lesko": Location(
        key="Lesko",
        label="Lesko",
        wmo="12690",
        lat=49.47,
        lon=22.34,
        elevation_m=382.0,
        description="Bieszczady, klimat podgórski",
    ),
}

# Współrzędne użyte w pierwszym przebiegu potoku (centra miejscowości).
# Zachowane wyłącznie do porównania wpływu zmiany punktu na uzysk — patrz
# tabela out/step1_coordinate_shift.csv. Nie używać do nowych obliczeń.
LEGACY_TOWN_COORDS: dict[str, tuple[float, float]] = {
    "Suwalki": (54.101, 22.929),
    "Lesko": (49.470, 22.331),
}

# Stacje walidacyjne: nie wchodzą do symulacji, służą wyłącznie porównaniu
# modelu satelitarnego z pomiarem naziemnym (rozdz. 4.5 i 6.2). Mikołajki
# reprezentują region północno-wschodni, dla którego IMGW udostępnia komplet
# składowych promieniowania.
STACJE_WALIDACYJNE: dict[str, dict[str, object]] = {
    "Mikolajki": {"label": "Mikołajki", "wmo": "12280",
                  "rola": "walidacja regionu NE: GHI i DHI"},
}


# ---------------------------------------------------------------------------
# Pojazd bazowy i obwiednia montażowa
# ---------------------------------------------------------------------------
# Wartości zmierzone i katalogowe. Obwiednia modułu wynika z wymiarów dachu
# i platformy bagażowej, a nie z decyzji o szerokości — dawne ograniczenie do
# 1000 mm nie obowiązuje.
ROOF_FIELD_AREA_M2 = 1.90          # pomiar własny, pole między relingami
ROOF_FIELD_LENGTH_MM = 1900.0      # pomiar własny
ROOF_LOAD_LIMIT_KG = 100.0         # ADAC, dopuszczalne obciążenie dachu (W4)

# Platforma bagażowa Front Runner KRVC006T. Wymiary z katalogu producenta —
# DO WERYFIKACJI na stronie producenta przed złożeniem pracy.
PLATFORM_MODEL = "Front Runner KRVC006T"
PLATFORM_LENGTH_MM = 1762.0
PLATFORM_WIDTH_MM = 1165.0
# Masa platformy po montażu, wariant dla Caddy Maxi 2015-2020
# (1165 x 1762 mm). Klasa: dane dystrybutora (4x4downunder.com.au),
# DO WERYFIKACJI u producenta. Masa kompletu mocowań modułu pozostaje
# nieustalona i wchodzi do bilansu masy osobno.
PLATFORM_MASS_KG = 30.39
MODULE_MOUNT_MASS_KG = None        # DO UZUPEŁNIENIA: mocowania modułu (W4)

# --- pomiary, których autor jeszcze nie dostarczył -------------------------
# Dopóki stoi tu None, każdy rachunek, który ich wymaga, ma się zatrzymać,
# a nie podstawiać wartości domyślnej.
# Wysokość szczeliny pod modułem. Pomiar autora: 100 mm między spodem
# belki platformy a poszyciem dachu, więc szczelina pod modułem jest
# nie mniejsza niż ta wartość. Klasa: pomiar własny.
MODULE_GAP_HEIGHT_MM = 100.0
USABLE_WIDTH_ON_BARS_MM = None     # DO UZUPEŁNIENIA: pomiar — szerokość użytkowa na belkach
BAR_SPACING_MM = None              # DO UZUPEŁNIENIA: pomiar — rozstaw belek
# Wnęka pod siedzeniem kierowcy (miejsce magazynu): długość, szerokość, wysokość.
SEAT_BAY_DIMENSIONS_MM: tuple[float, float, float] | None = None  # DO UZUPEŁNIENIA: pomiar


def require_measurement(name: str):
    """
    Zwraca wartość pomiaru albo przerywa obliczenie czytelnym komunikatem.

    Świadomie nie ma tu wartości domyślnej: brak pomiaru ma zatrzymać rachunek,
    a nie wejść do wyniku jako liczba nieoznaczona (CLAUDE.md C).
    """
    value = globals().get(name, None)
    if value is None:
        raise ValueError(
            f"Brak pomiaru: {name}. Uzupełnij w simulation/config.py "
            f"(znacznik DO UZUPEŁNIENIA) albo pomiń obliczenie, które go wymaga."
        )
    return value


# ---------------------------------------------------------------------------
# Alternator pojazdu bazowego
# ---------------------------------------------------------------------------
# Katalog części dla Caddy IV 2.0 TDI wymienia alternatory 140 A i 180 A,
# z regulatorem komunikującym się magistralą COM/LIN; jeden z wariantów
# opisany jest jako przeznaczony do pojazdów z odzyskiem energii przy
# hamowaniu i funkcją Start-Stop. Klasa: katalog części (dystrybutor),
# DO WERYFIKACJI według numeru PR pojazdu.
#
# Wniosek projektowy: alternator jest sterowany, więc napięcie na szynie
# rozruchowej przy pracy silnika bywa niższe od progu 14,0 V, którym
# ładowarka Orion-Tr wykrywa pracę silnika. Ładowarkę trzeba więc
# załączać sygnałem zapłonu podanym na wejście zdalne, a nie progiem
# napięcia (rozdz. 4.1 i 5.4).
ALTERNATOR_PRAD_A = 140.0          # wariant bazowy (zachowawczy)
ALTERNATOR_PRAD_OAT_A = (140.0, 180.0)
ALTERNATOR_STEROWANY = True

# ---------------------------------------------------------------------------
# Parametry zapytań do API PVGIS
# ---------------------------------------------------------------------------
PVGIS_API_VERSION = "v5_3"
PVGIS_BASE = f"https://re.jrc.ec.europa.eu/api/{PVGIS_API_VERSION}"

# Bazy promieniowania. SARAH3 to zbiór satelitarny (Meteosat, CM SAF), ERA5 to
# reanaliza ECMWF. Traktowane jako dwa niezależne zbiory wejściowe — ich
# rozbieżność jest miarą rozrzutu międzyzbiorowego, nie błędu względem pomiaru.
RADIATION_DATABASES = ("PVGIS-SARAH3", "PVGIS-ERA5")

# Okres szeregów wieloletnich. Dolna granica wynika z dostępności SARAH3,
# górna z ostatniego pełnego roku kalendarzowego w bazie.
SERIES_START_YEAR = 2014
SERIES_END_YEAR = 2023

# Strefa czasowa lokalna dla profili obciążenia i raportowania dobowego.
LOCAL_TZ = "Europe/Warsaw"

# Limit czasu zapytania HTTP. Generowanie szeregu 10-letniego po stronie PVGIS
# trwa znacznie dłużej niż pobranie pliku TMY.
HTTP_TIMEOUT_S = 300


def tmy_url(loc: Location) -> str:
    """Adres zapytania o typowy rok meteorologiczny w formacie EPW."""
    return (
        f"{PVGIS_BASE}/tmy"
        f"?lat={loc.lat}&lon={loc.lon}&outputformat=epw&browser=0"
    )


def seriescalc_url(loc: Location, database: str) -> str:
    """
    Adres zapytania o godzinowy szereg czasowy dla płaszczyzny poziomej.

    Parametry: ``angle=0`` i ``aspect=0`` dają płaszczyznę poziomą (promieniowanie
    całkowite na powierzchnię poziomą), ``components=1`` rozdziela promieniowanie
    na składową bezpośrednią, rozproszoną i odbitą, co jest konieczne do
    testów zamknięcia bilansu promieniowania w kontroli jakości.
    """
    return (
        f"{PVGIS_BASE}/seriescalc"
        f"?lat={loc.lat}&lon={loc.lon}"
        f"&raddatabase={database}"
        f"&startyear={SERIES_START_YEAR}&endyear={SERIES_END_YEAR}"
        f"&angle=0&aspect=0&components=1"
        f"&outputformat=csv&browser=0"
    )


# ---------------------------------------------------------------------------
# Nazwy plików surowych
# ---------------------------------------------------------------------------

def raw_tmy_path(loc: Location) -> Path:
    """Ścieżka surowego pliku TMY dla lokalizacji."""
    return RAW_DIR / f"tmy_{loc.key}_{loc.lat}_{loc.lon}.epw"


def raw_series_path(loc: Location, database: str) -> Path:
    """Ścieżka surowego pliku szeregu wieloletniego dla lokalizacji i bazy."""
    tag = database.replace("PVGIS-", "").lower()
    return RAW_DIR / (
        f"series_{loc.key}_{tag}_"
        f"{SERIES_START_YEAR}_{SERIES_END_YEAR}.csv"
    )


def working_tmy_path(loc: Location) -> Path:
    """
    Ścieżka roboczej kopii pliku TMY, czytanej przez model PV.

    Kopia robocza powstaje z pliku surowego i może mieć poprawiony nagłówek
    (nazwa miejscowości); pliki w ``raw/`` pozostają nietknięte.
    """
    return WEATHER_DIR / f"{loc.key}.epw"


# ---------------------------------------------------------------------------
# Atrybucja danych (do podpisów rysunków i tabel w pracy)
# ---------------------------------------------------------------------------
# Formuły wymagane przez regulaminy udostępniania danych. Tekst dla IMGW
# pochodzi z § 5 ust. 3 i 4 regulaminu IMGW-PIB i musi być cytowany dosłownie.
ATTRIBUTION_PVGIS = (
    "PVGIS 5.3, European Commission JRC, SARAH3 / ERA5, data retrieved {date}"
)
ATTRIBUTION_IMGW_SOURCE = (
    "Źródłem pochodzenia danych jest Instytut Meteorologii i Gospodarki "
    "Wodnej – Państwowy Instytut Badawczy"
)
ATTRIBUTION_IMGW_PROCESSED = (
    "Dane Instytutu Meteorologii i Gospodarki Wodnej – Państwowego "
    "Instytutu Badawczego zostały przetworzone"
)
