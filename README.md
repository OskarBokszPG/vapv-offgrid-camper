# VAPV off-grid camper — potok obliczeniowy pracy magisterskiej

Kod obliczeniowy do pracy magisterskiej *Projekt i analiza efektywności
energetycznej autonomicznego systemu zasilania off-grid typu VAPV dla
pojazdu do celów turystycznych* (Politechnika Gdańska, Wydział
Elektrotechniki i Automatyki, Elektrotechnika II stopnia).

Autor: Oskar Boksz. Promotor: dr inż. Piotr Kołodziejek.

## Cel i kontekst

Praca projektuje autonomiczny układ zasilania odbiorników bytowych
w pojeździe turystycznym (VW Caddy Maxi, zabudowa własna) i ocenia jego
efektywność energetyczną w warunkach krajowych. VAPV oznacza *vehicle-applied
photovoltaics*, czyli fotowoltaikę aplikowaną na pojeździe już
wyprodukowanym, w odróżnieniu od integracji fabrycznej (VIPV). Rozpatrywany
jest podzbiór układów VAPV pracujących w trybie postojowym.

Repozytorium zawiera część obliczeniową: model generacji fotowoltaicznej,
profile obciążeń, godzinowy bilans magazynu, wskaźniki oceny, siatkę
konfiguracji i analizę wrażliwości. Skład dokumentu — generatory tabel,
makr liczbowych i rysunków — pozostaje poza repozytorium.

Repozytorium **nie zawiera** danych pogodowych, kart katalogowych, tekstów
norm, odpowiedzi ankiety ani plików `.tex` pracy. Dane wejściowe pobiera się
według `MANIFEST.md`.

## Struktura katalogów

```
simulation/           moduły potoku obliczeniowego
simulation/run_all.py uruchomienie całego potoku
pyproject.toml        zależności
uv.lock               zamrożone wersje pakietów
MANIFEST.md           źródła danych wejściowych ze skrótami SHA-256
```

Katalogi tworzone przez potok (`simulation/out/`, `simulation/weather/`) są
wyłączone z repozytorium: ich zawartość odtwarza się z danych wejściowych.

## Potok krok po kroku

1. **Dane wejściowe.** Szeregi godzinowe PVGIS-SARAH3 (2014–2023, płaszczyzna
   pozioma) oraz publiczne dane o usłonecznieniu ze stacji IMGW-PIB (kontrola
   spójności); źródła i skróty w `MANIFEST.md`. Kontrolę kompletności
   i zakresów wykonuje `weather_qc.py`.
2. **Zasób dla silnika SAM.** `make_sam_resource.py` przelicza pliki PVGIS na
   format zasobu SSC. Strefa czasowa w nagłówku ustawiana jest na 0, bo
   znaczniki czasu PVGIS są w UTC; przyjęcie deklaracji z nagłówka EPW dawało
   przesunięcie dobowe.
3. **Generacja.** `run_pv.py` konfiguruje model `Pvsamv1` (PySAM): jeden moduł
   z karty katalogowej, model jednodiodowy z wewnętrznym dopasowaniem
   parametrów, model cieplny NOCT, kategoria odstępu odpowiadająca szczelinie
   pod modułem. Z SAM pochodzi wyłącznie moc stałoprądowa.
4. **Profile obciążeń.** `loads.py` składa godzinowe profile trzech
   scenariuszy użytkowania z wartości katalogowych i przyjętych: chłodziarka,
   ogrzewanie postojowe, laptop w dwóch wariantach zasilania, łącze
   satelitarne, oświetlenie, pobory własne podzespołów.
5. **Bilans godzinowy magazynu.** `battery_balance.py` prowadzi dyspozycję
   energii: pokrycie bezpośrednie, ładowanie nadwyżką, rozładowanie na
   niedobór, dopływ z alternatora przez ładowarkę stałoprądową, blokada
   ładowania w niskiej temperaturze, samorozładowanie. Model magazynu
   i wskaźniki liczone są poza SAM, bo SAM nie podaje wskaźnika niedoboru
   zasilania ani nie modeluje alternatora. Cechy modelu istotne dla
   wyników:
   - **granica zapotrzebowania na zaciskach odbiorników** — moc laptopa
     odnosi się w obu wariantach do jego wejścia stałoprądowego, a straty
     zasilacza sieciowego, przetwornicy i falownika są osobnymi pozycjami
     toru, więc oba warianty zasilania opisują tę samą usługę energetyczną;
   - **dwustronne ograniczenie ładowarki** — moc wyjściowa to
     `min(η·U_we·I_we,max; U_wy·I_wy,max)`, dodatkowo ograniczona mocą
     dostępną z alternatora po pokryciu odbiorów pokładowych;
   - **jeden okres rozbiegowy** — dziesięciolecie liczone jest w jednym
     ciągu, ze stanem naładowania przenoszonym z roku na rok, a z oceny
     wyłączony jest tylko pierwszy miesiąc szeregu;
   - **krok cieplny wykładniczy** — temperatura ogniw liczona jako
     `T_kab + (T_bat − T_kab)·exp(−Δt/τ)`, bez przeregulowania schematu
     jawnego;
   - **ograniczenie regulatora** — próg mocy wyjściowej liczony przy
     napięciu znamionowym magazynu, a energia obcięta raportowana osobno.
6. **Wskaźniki.** Z przebiegu liczone są: energetyczny wskaźnik niedoboru
   zasilania (LPSP), udział pokrycia z generatora (SSR), autonomia magazynu
   i najmniejszy dzienny czas jazdy wyznaczany bisekcją. Energię oddaną
   z magazynu przypisuje się źródłom proporcjonalnie do ich udziału
   w ładowaniu, więc udział generatora nie przekracza udziału całej energii
   dostarczonej; tożsamość tego przypisania sprawdza test jednostkowy.
7. **Siatka konfiguracji.** `run_grid.py` przechodzi po kombinacjach modułu,
   magazynu, ładowarki, lokalizacji, roku i scenariusza; `grid_analiza.py`
   ocenia wymagania projektowe i prowadzi analizę wrażliwości metodą zmiany
   jednego parametru; `grid_raport.py` składa zestawienia pochodne.
8. **Wyniki.** Przebiegi zapisują pliki CSV w `simulation/out/`; spinającym
   poleceniem jest `run_all.py`.

Każdy przebieg zapisujący wyniki zapisuje obok manifest (`*.manifest.json`)
z datą, wersjami pakietów, pełną konfiguracją i skrótami SHA-256 plików
wejściowych. Pozwala to sprawdzić, czy wynik policzono na tych danych, które
leżą na dysku.

## Moduły

### Potok główny

- `config.py` — ścieżki, lokalizacje, zakres lat, stałe geometryczne pojazdu
  i platformy dachowej. Jedno miejsce konfiguracji dla wszystkich skryptów.
- `make_sam_resource.py` — budowa plików zasobu dla silnika SSC z szeregów
  PVGIS; obsługa strefy czasowej i roku przestępnego.
- `run_pv.py` — konfiguracja modelu `Pvsamv1` i parametry modułu z karty.
  Zwraca godzinową moc stałoprądową generatora.
- `loads.py` — profile obciążeń trzech scenariuszy, model chłodziarki w dwóch
  postaciach, energia rozruchu ogrzewania, pobory własne.
- `battery_balance.py` — godzinowy bilans magazynu, wskaźniki oceny, kontrola
  zamknięcia bilansu energii.
- `run_grid.py` — siatka konfiguracji i pamięć wyników generacji; ograniczenie
  prądu wyjściowego regulatora.
- `grid_analiza.py` — ocena wymagań W1–W4 i analiza wrażliwości.
- `grid_raport.py` — zestawienia pochodne: łańcuch przemian energii, tygodnie
  krytyczne, przebiegi miesięczne.
- `run_yield.py`, `main.py` — pojedyncze przebiegi kontrolne generacji
  i bilansu, przydatne przy sprawdzaniu konfiguracji.
- `manifest.py` — zapis i sprawdzanie manifestów przebiegów.
- `fmt.py` — konwencje zapisu liczb: przecinek dziesiętny, separator tysięcy,
  miesiące cyframi rzymskimi. Jedno źródło formatowania wyników.
- `run_all.py` — uruchomienie potoku w kolejności zależności.

### Kontrole i warianty

- `tests_bilans.py` — zestaw testów modelu bilansu i profili obciążeń;
  testy wymagające plików zasobu są pomijane z komunikatem, gdy danych nie
  ma.
- `przerwy.py` — godziny z niedoborem i najdłuższa nieprzerwana przerwa
  w pokryciu obciążenia dla konfiguracji projektowej.
- `weather_qc.py`, `weather_validation.py` — kontrola danych pogodowych
  i porównanie z danymi naziemnymi.
- `fit_check.py`, `module_effect_check.py`, `thermal_check.py` — kontrole
  dopasowania parametrów modułu, dekompozycja wpływu modelu i modelu cieplnego.
- `dhi_sensitivity.py`, `ghi_stats.py`, `niepewnosc_ghi.py` — statystyki
  zasobu słonecznego i przeniesienie niepewności danych wejściowych.
- `wariant_pochylenia.py`, `pochylenie_miesiace.py` — wariant z modułem
  podnoszonym na postoju.
- `przewody.py` — dobór przekrojów i zabezpieczeń obwodów stałoprądowych.
- `kontrola_mppt.py` — kontrola ograniczenia prądu wyjściowego regulatora.
- `grid_shift_check.py` — wpływ przesunięcia punktu siatki danych
  satelitarnych na sumy roczne.
- `fetch_weather.py` — pobranie szeregów PVGIS według `MANIFEST.md`.

## Wyniki odniesienia

Wartości, na które powinien trafić odtworzony potok (szereg 2014--2023,
moduł AIKO 470 Wp, ładowarka Orion XS 50, liniowy model poboru
chłodziarki):

| Wielkość | Wartość |
|---|---|
| Najmniejsza pojemność spełniająca W1a–W4 w scenariuszu A-AC | 300 Ah |
| LPSP w roku najgorszym przy 300 Ah (Łeba / Suwałki / Lesko) | 4,2 / 4,1 / 2,1 % |
| Okno miesięczne W1b | IV–IX, IV–IX, III–IX |
| Najmniejszy dzienny czas jazdy W3 | 0,59 / 0,56 / 0,31 h/d |
| Najmniejsza pojemność w scenariuszu A-DC | 50 Ah |

Przebieg kontrolny jednego roku (Łeba 2016, 300 Ah, A-AC) daje
LPSP = 0,0416 i SSR_PV = 0,922.

## Wymagania i instalacja

- Python 3.13 (wersja użyta w pracy: 3.13.16)
- NREL-PySAM 7.1.1.post1 — SAM 2025.4.16 Rev. 2, **SSC 306**
- pandas 3.0.6, numpy 2.5.3, matplotlib 3.11.2, pvlib 0.16.1

Wersje są zamrożone w `uv.lock` i nie należy ich podnosić: wydanie 8.0.0
pakietu (pod nazwą `nlr-pysam`, SSC 308) zmienia wartości domyślne
konfiguracji `Pvsamv1`, co psuje porównywalność z wynikami podanymi w pracy.

```powershell
uv sync --frozen
```

## Dane wejściowe

Dane pobiera się według `MANIFEST.md`, który podaje adresy źródeł, katalogi
docelowe i skróty SHA-256 plików użytych w pracy.

Wymagane formuły o źródle danych:

- PVGIS: *Opracowanie własne na podstawie danych PVGIS-SARAH3*.
- IMGW-PIB: *Dane Instytutu Meteorologii i Gospodarki Wodnej — Państwowego
  Instytutu Badawczego*, wraz z informacją, że dane zostały przetworzone
  przez autora.

## Uruchomienie

Testy modelu bilansu i profili obciążeń:

```powershell
.venv\Scripts\python.exe simulation\tests_bilans.py
```

Bez pobranych danych przechodzi 20 z 23 testów, a trzy pozostałe są
pomijane z komunikatem: liczą pełny przebieg godzinowy i wymagają plików
zasobu zbudowanych według `MANIFEST.md`.

Przebieg kontrolny na jednym pliku zasobu, bez całego katalogu danych:

```powershell
.venv\Scripts\python.exe simulation\run_all.py --kontrola `
    sciezka\do\Leba_sarah3_2016.csv --lokalizacja Łeba --rok 2016 `
    --pojemnosc 300
```

Cały potok, od kontroli danych do plików CSV z wynikami:

```powershell
.venv\Scripts\python.exe simulation\run_all.py
```

Potok wykonuje kolejno: kontrolę danych pogodowych, zasoby słoneczne,
profile obciążeń, uzysk generatora, siatkę przebiegów, ocenę wymagań
W1a–W4 wraz z analizą wrażliwości i scenariuszami łącznymi, raport siatki,
kontrolę ograniczenia regulatora, godziny niedoboru oraz wariant z modułem
podnoszonym. Pełny przebieg zajmuje kilkadziesiąt minut; wyniki trafiają
do `simulation/out/` wraz z manifestami przebiegów.

## Ograniczenia modelu

Krótko, szczegóły w pracy:

- Wyniki opisują zachowanie modelu przy przyjętych danych wejściowych, a nie
  działanie zbudowanego układu — pomiarów na pojeździe nie wykonano
  (rozdz. 6 i 8 pracy).
- Dane satelitarne nie zostały porównane z pomiarami aktynometrycznymi
  na rozpatrywanych stacjach; niepewność danych wejściowych oszacowano
  pośrednio, na podstawie opublikowanych walidacji zbioru (rozdz. 6).
- Model chłodziarki występuje w dwóch postaciach i rozstrzygnięcie o
  spełnieniu wymagania W1a zależy od wyboru postaci (rozdz. 7).
- Pojemność magazynu dobrano dla wariantu z laptopem zasilanym przez
  falownik, czyli niekorzystniejszego z dwóch rozpatrywanych (rozdz. 7).
- Profile obciążeń zbudowano z kart katalogowych i literatury, bez pomiaru
  poboru odbiorników (rozdz. 4).
- Mechanika zabudowy dachowej — drgania, obciążenia od wiatru, mocowania —
  leży poza zakresem pracy.
- Starzenie modułu i magazynu nie wchodzi do modelu; uwzględniony jest
  wyłącznie spadek początkowy mocy modułu.

## Wsparcie AI

Kod źródłowy napisany z wykorzystaniem narzędzia Claude Code (Anthropic) na
podstawie założeń, modeli i decyzji projektowych autora. Autor zweryfikował
działanie kodu testami jednostkowymi i kontrolami bilansu opisanymi w pracy.

## Licencja

MIT — plik `LICENSE`.
