# MANIFEST — dane wejściowe potoku

Repozytorium zawiera wyłącznie kod. Dane wejściowe trzeba pobrać
osobno; poniżej podano źródło, miejsce w drzewie katalogów
i skrót SHA-256 pliku użytego w pracy. Skróty pozwalają sprawdzić,
czy pobrany plik jest tym samym, na którym policzono wyniki.

## 1. Szeregi godzinowe promieniowania — PVGIS-SARAH3

Źródło: PVGIS, zbiór SARAH-3, szeregi godzinowe 2014--2023,
pobrane przez interfejs <https://re.jrc.ec.europa.eu/pvg_tools/>
(zakładka *Hourly data*, płaszczyzna pozioma, kąt 0°, azymut 0°).
Katalog docelowy: `dane_pvgis/`.

Wymagana formuła o źródle danych w publikacjach:
*Opracowanie własne na podstawie danych PVGIS-SARAH3*.

| Plik | SHA-256 |
|---|---|
| `dane_pvgis/Timeseries_49.470_22.340_SA3_0deg_0deg_2014_2023.csv` | `d65df21831d126beb55acdc61c8bdc944fdea869d5bfe8da1d6dcdabf1540cf0` |
| `dane_pvgis/Timeseries_54.130_22.950_SA3_0deg_0deg_2014_2023.csv` | `36c5c8df00302d320942d0efea09e0b1eaacf86353db686001206416ace9ad5a` |
| `dane_pvgis/Timeseries_54.600_18.810_SA3_0deg_0deg_2014_2023.csv` | `b3964248479a32b3f5d84337e70c8a34335f85b23737a1ffa15bcdb2d4e839d4` |

## 2. Dane naziemne — IMGW-PIB

Źródło: Instytut Meteorologii i Gospodarki Wodnej --- Państwowy
Instytut Badawczy, dane publiczne ze stacji synoptycznych
<https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/>
(dane dobowe i terminowe, lata 2014--2023; stacje Łeba, Suwałki,
Lesko, Mikołajki). Katalog docelowy: `data/imgw/`.

Wymagane formuły atrybucji:
*Dane Instytutu Meteorologii i Gospodarki Wodnej --- Państwowego
Instytutu Badawczego* oraz informacja o przetworzeniu danych:
*dane zostały przetworzone przez autora*.

## 3. Pliki zasobu dla silnika SAM

Pliki zasobu (`simulation/weather/sam_resource/*.csv`) nie wchodzą
do repozytorium: powstają z danych PVGIS skryptem
`simulation/make_sam_resource.py`. Strefa czasowa w nagłówku jest
ustawiana na 0, bo znaczniki czasu PVGIS są w UTC.

| Plik zasobu (rok 2014) | SHA-256 |
|---|---|
| `Leba_sarah3_2014.csv` | `3d0557faee5e0e2b660b4786f51e88a274dbcf884f59fc1257bb084e6b81b9cd` |
| `Lesko_sarah3_2014.csv` | `c38c1ac3957b4d2b851a5eedb84a75190235db026556f5b5e7f4949a6947eb7b` |
| `Suwalki_sarah3_2014.csv` | `4ed537d548db39c80a070cf0a3901bc1107ad37cffea30040ca6262cc789bdd3` |

## 4. Karty katalogowe i normy

Karty katalogowe, instrukcje i teksty norm nie wchodzą do
repozytorium ze względu na prawa wydawców. Wartości z nich
odczytane są wpisane w `simulation/loads.py`, `simulation/run_pv.py`
i `data/zalozenia.csv`, z podaniem klasy wartości i źródła
w załączniku A pracy.

## 5. Dane ankiety

Odpowiedzi ankiety nie wchodzą do repozytorium, również
w postaci zanonimizowanej, podobnie jak kod ich przetwarzania.
Ankieta wchodzi do pracy wyłącznie jako kontekst i nie jest
danymi wejściowymi potoku obliczeniowego.
