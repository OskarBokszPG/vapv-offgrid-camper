# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: fetch_weather.py
Opis: pobranie surowych danych meteorologicznych z bazy PVGIS. Dla
      każdej lokalizacji referencyjnej pobierane są trzy zbiory:

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
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("BŁĄD: brak pakietu 'requests'. Zainstaluj: pip install requests")
    raise

import config
from config import LOCATIONS, Location

# Plik EPW opisujący pełny rok zawiera 8760 wierszy danych i 8 wierszy nagłówka.
MIN_EPW_LINES = 8000
# Szereg 10-letni: 87 648 wierszy danych (2 lata przestępne) plus nagłówek i stopka.
MIN_SERIES_LINES = 80000


def sha256_of(path: Path) -> str:
    """Zwraca skrót SHA-256 pliku (do zapisu w manifeście)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    """Zwraca liczbę wierszy pliku tekstowego."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def _download(url: str, target: Path, min_lines: int, expect_epw: bool) -> dict | None:
    """
    Pobiera jeden plik i zapisuje go bez żadnej modyfikacji treści.

    Zwraca słownik wpisu do manifestu albo ``None``, jeśli pobranie zawiodło.
    W razie błędu wypisywany jest dokładny adres URL, aby dane można było
    pobrać ręcznie przez interfejs webowy PVGIS.
    """
    print(f"    GET {url}")
    try:
        response = requests.get(url, timeout=config.HTTP_TIMEOUT_S)
    except requests.exceptions.Timeout:
        print(f"      -> przekroczono limit czasu {config.HTTP_TIMEOUT_S} s")
        return None
    except requests.exceptions.RequestException as error:
        print(f"      -> błąd zapytania: {error}")
        return None

    if response.status_code != 200:
        # PVGIS opisuje przyczynę odrzucenia zapytania w treści odpowiedzi.
        print(f"      -> HTTP {response.status_code}: {response.text[:300].strip()}")
        return None

    text = response.text
    if expect_epw and not text.startswith("LOCATION"):
        print("      -> odpowiedź nie jest plikiem EPW (brak wiersza LOCATION)")
        return None

    line_count = len(text.strip().splitlines())
    if line_count < min_lines:
        print(f"      -> plik niekompletny: {line_count} wierszy (< {min_lines})")
        return None

    # Znacznik czasu pobrania w UTC — przed zapisem, żeby odpowiadał momentowi
    # odpowiedzi serwera, a nie momentowi zakończenia operacji dyskowej.
    retrieved_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")

    entry = {
        "file": target.name,
        "url": url,
        "retrieved_utc": retrieved_utc,
        "api_version": config.PVGIS_API_VERSION,
        "sha256": sha256_of(target),
        "lines": line_count,
        "bytes": target.stat().st_size,
    }
    print(f"      -> {target.name}: {entry['bytes']} B, {line_count} wierszy")
    return entry


def fetch_raw(force: bool = False, tmy_only: bool = False) -> list[dict]:
    """
    Pobiera wszystkie zbiory surowe i zwraca listę wpisów do manifestu.

    Pliki już istniejące są pomijane (bez wpisu), chyba że podano ``force``.
    """
    config.ensure_dirs()
    entries: list[dict] = []

    for loc in LOCATIONS.values():
        print(f"\n{loc.label} (WMO {loc.wmo}; {loc.lat} N, {loc.lon} E)")
        print(f"  — {loc.description}")

        # --- a) typowy rok meteorologiczny ---------------------------------
        target = config.raw_tmy_path(loc)
        if target.exists() and not force:
            print(f"    [pomijam] {target.name} już istnieje")
        else:
            entry = _download(
                config.tmy_url(loc), target, MIN_EPW_LINES, expect_epw=True
            )
            if entry:
                entries.append(entry)

        if tmy_only:
            continue

        # --- b, c) szeregi wieloletnie z dwóch niezależnych baz -------------
        for database in config.RADIATION_DATABASES:
            target = config.raw_series_path(loc, database)
            if target.exists() and not force:
                print(f"    [pomijam] {target.name} już istnieje")
                continue
            entry = _download(
                config.seriescalc_url(loc, database),
                target,
                MIN_SERIES_LINES,
                expect_epw=False,
            )
            if entry:
                entries.append(entry)

    return entries


def make_working_copies() -> dict[str, Path]:
    """
    Tworzy kopie robocze plików TMY z poprawionym wierszem LOCATION.

    PVGIS wpisuje w polach opisowych wartość ``unknown``, przez co SAM etykietuje
    wyniki jako lokalizację nieznaną. Podmieniane są wyłącznie nazwa miejscowości
    i kod kraju; współrzędne, strefa czasowa i wysokość n.p.m. pozostają
    dokładnie takie, jakie zwróciło PVGIS. Plik surowy nie jest modyfikowany.
    """
    result: dict[str, Path] = {}
    for loc in LOCATIONS.values():
        source = config.raw_tmy_path(loc)
        if not source.exists():
            continue

        lines = source.read_text(encoding="utf-8").splitlines()
        if lines and lines[0].startswith("LOCATION"):
            fields = lines[0].split(",")
            if len(fields) >= 4:
                fields[1] = loc.key
                fields[3] = "POL"
                lines[0] = ",".join(fields)

        target = config.working_tmy_path(loc)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        result[loc.key] = target
    return result


def register_provided_files() -> list[dict]:
    """
    Rejestruje w manifeście pliki pobrane ręcznie przez autora.

    Pliki pobrane przez interfejs webowy PVGIS nie mają znanego adresu
    zapytania ani znacznika czasu odpowiedzi serwera; zapisywany jest czas
    modyfikacji pliku jako przybliżenie momentu pobrania. Służą do kontroli
    krzyżowej zgodności interfejsu webowego z API.
    """
    entries: list[dict] = []
    if not config.PROVIDED_DIR.exists():
        return entries

    for path in sorted(config.PROVIDED_DIR.iterdir()):
        if not path.is_file():
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        entries.append({
            "file": f"provided_by_author/{path.name}",
            "url": "nieznany — pobrane przez interfejs webowy PVGIS",
            "retrieved_utc": mtime.strftime("%Y-%m-%dT%H:%M:%SZ")
                             + " (czas modyfikacji pliku)",
            "api_version": "interfejs webowy, wersja nieodnotowana",
            "sha256": sha256_of(path),
            "lines": count_lines(path),
            "bytes": path.stat().st_size,
        })
    return entries


def write_manifest(entries: list[dict]) -> Path:
    """
    Zapisuje lub uzupełnia ``weather/raw/MANIFEST.md``.

    Wpisy są scalane po nazwie pliku: ponowne pobranie nadpisuje stary wpis,
    pliki nieobjęte tym przebiegiem pozostają w manifeście bez zmian.
    """
    manifest = config.RAW_DIR / "MANIFEST.md"
    existing: dict[str, dict] = {}

    # Odczyt dotychczasowych wpisów z tabeli Markdown (kolumna pierwsza = plik).
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.startswith("| `"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 7:
                existing[cells[0].strip("`")] = {
                    "file": cells[0].strip("`"),
                    "url": cells[1].strip("`"),
                    "retrieved_utc": cells[2],
                    "api_version": cells[3],
                    "sha256": cells[4].strip("`"),
                    "lines": cells[5],
                    "bytes": cells[6],
                }

    for entry in entries:
        existing[entry["file"]] = entry

    lines = [
        "# MANIFEST — surowe dane wejściowe",
        "",
        "Pliki w tym katalogu są **niezmienne**. Każda obróbka zapisywana jest",
        "poza katalogiem `raw/`. Tabela służy odtworzeniu obliczeń zgodnie",
        "z wymogiem E4 (rozdz. 6.1).",
        "",
        f"Wersja API PVGIS: `{config.PVGIS_API_VERSION}`  ",
        f"Bazy promieniowania: {', '.join(f'`{d}`' for d in config.RADIATION_DATABASES)}  ",
        f"Okres szeregów wieloletnich: {config.SERIES_START_YEAR}–{config.SERIES_END_YEAR}  ",
        f"Manifest zapisany: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "Atrybucja wymagana przy cytowaniu danych w pracy:",
        "",
        f"> {config.ATTRIBUTION_PVGIS.format(date='<data pobrania>')}",
        "",
        "| Plik | URL | Pobrano (UTC) | API | SHA-256 | Wiersze | Bajty |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in sorted(existing):
        entry = existing[name]
        lines.append(
            f"| `{entry['file']}` | `{entry['url']}` | {entry['retrieved_utc']} "
            f"| {entry['api_version']} | `{entry['sha256']}` "
            f"| {entry['lines']} | {entry['bytes']} |"
        )
    lines.append("")

    manifest.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pobranie surowych danych PVGIS (TMY, SARAH3, ERA5)."
    )
    parser.add_argument("--force", action="store_true",
                        help="pobierz ponownie także pliki już istniejące")
    parser.add_argument("--tmy-only", action="store_true", dest="tmy_only",
                        help="pobierz wyłącznie pliki TMY, bez szeregów 10-letnich")
    args = parser.parse_args()

    print("=" * 72)
    print("Pobieranie surowych danych meteorologicznych z bazy PVGIS")
    print("=" * 72)

    entries = fetch_raw(force=args.force, tmy_only=args.tmy_only)
    entries.extend(register_provided_files())

    manifest = write_manifest(entries)
    print(f"\nManifest: {manifest}")

    working = make_working_copies()
    print(f"Kopie robocze TMY: {len(working)} ({', '.join(sorted(working))})")

    expected = len(LOCATIONS) * (1 if args.tmy_only else 3)
    present = sum(
        1 for loc in LOCATIONS.values()
        for path in (
            [config.raw_tmy_path(loc)] if args.tmy_only else
            [config.raw_tmy_path(loc)]
            + [config.raw_series_path(loc, d) for d in config.RADIATION_DATABASES]
        )
        if path.exists()
    )
    print(f"Pliki surowe obecne: {present} z {expected}")
    return 0 if present == expected else 1


if __name__ == "__main__":
    sys.exit(main())
