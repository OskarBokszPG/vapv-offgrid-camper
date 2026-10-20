# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: manifest.py
Opis: opis przebiegu obliczeń zapisywany obok jego wyników. Po co. Pliki
      w ``out/`` nie noszą w sobie informacji o tym, czym i na czym
      zostały policzone. Rysunek wygenerowany ze starej tabeli wygląda
      dokładnie tak samo jak rysunek policzony na nowo, a różnica jest
      nie do wykrycia w gotowym PDF-ie. Manifest zamyka tę lukę:
      przebieg zapisuje obok wyników swój opis, a skrypty rysunków i
      tabel sprawdzają przed wygenerowaniem, czy pliki wejściowe nadal
      mają te same skróty.

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

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import config

ROZSZERZENIE = ".manifest.json"


class NiezgodnoscWejscia(RuntimeError):
    """Skrót pliku wejściowego nie zgadza się ze skrótem z manifestu."""


def sha256_pliku(sciezka: Path, blok: int = 1 << 20) -> str:
    """Skrót SHA-256 pliku, czytanego blokami (pliki pogodowe są duże)."""
    h = hashlib.sha256()
    with open(sciezka, "rb") as f:
        while True:
            dane = f.read(blok)
            if not dane:
                break
            h.update(dane)
    return h.hexdigest()


def _wzgledna(sciezka: Path) -> str:
    """Ścieżka względem katalogu projektu — manifest ma być przenośny."""
    sciezka = Path(sciezka).resolve()
    korzen = config.SIM_DIR.parent
    try:
        return sciezka.relative_to(korzen).as_posix()
    except ValueError:
        return sciezka.as_posix()


def opis_pliku(sciezka: Path) -> dict[str, object]:
    """Skrót, rozmiar i czas modyfikacji jednego pliku."""
    sciezka = Path(sciezka)
    if not sciezka.exists():
        raise FileNotFoundError(f"Brak pliku wejściowego: {sciezka}")
    stat = sciezka.stat()
    return {
        "plik": _wzgledna(sciezka),
        "sha256": sha256_pliku(sciezka),
        "rozmiar_b": stat.st_size,
        "zmodyfikowano": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def wersje() -> dict[str, str]:
    """Wersje narzędzi liczących. PySAM i SSC tylko jeśli dają się zaimportować."""
    w: dict[str, str] = {
        "Python": sys.version.split()[0],
        "platforma": platform.platform(),
    }
    for nazwa, modul in (("pandas", "pandas"), ("numpy", "numpy"),
                         ("pvlib", "pvlib"), ("matplotlib", "matplotlib")):
        try:
            w[nazwa] = __import__(modul).__version__
        except Exception:
            w[nazwa] = "niedostępny"
    try:
        import run_pv
        w.update({k: str(v) for k, v in run_pv.wersje_narzedzi().items()})
    except Exception as e:                                  # pragma: no cover
        w["PySAM"] = f"niedostępny ({e})"
    return w


def sciezka_manifestu(nazwa: str) -> Path:
    """``out/<nazwa>.manifest.json``."""
    return config.OUT_DIR / (nazwa + ROZSZERZENIE)


def zapisz(nazwa: str, *, skrypt: str, konfiguracja: dict[str, object],
           wejscia: dict[str, Path], wyniki: list[Path] | None = None) -> Path:
    """
    Zapisuje manifest przebiegu i zwraca ścieżkę pliku.

    ``nazwa`` jest nazwą przebiegu, nie nazwą pliku wynikowego: jeden przebieg
    może zapisać kilka tabel i wszystkie opisuje jeden manifest.
    """
    config.ensure_dirs()
    dane = {
        "przebieg": nazwa,
        "data": datetime.now().isoformat(timespec="seconds"),
        "skrypt": skrypt,
        "wersje": wersje(),
        "konfiguracja": konfiguracja,
        "wejscia": {etykieta: opis_pliku(Path(p)) for etykieta, p in wejscia.items()},
        "wyniki": [opis_pliku(Path(p)) for p in (wyniki or []) if Path(p).exists()],
    }
    cel = sciezka_manifestu(nazwa)
    cel.write_text(json.dumps(dane, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    return cel


def wczytaj(nazwa: str) -> dict[str, object]:
    """Czyta manifest przebiegu albo przerywa czytelnym komunikatem."""
    cel = sciezka_manifestu(nazwa)
    if not cel.exists():
        raise FileNotFoundError(
            f"Brak manifestu przebiegu „{nazwa}” ({cel.name}). "
            f"Uruchom przebieg, który go zapisuje, zanim wygenerujesz rysunki "
            f"albo tabele z jego wyników."
        )
    return json.loads(cel.read_text(encoding="utf-8"))


def sprawdz(nazwa: str) -> dict[str, object]:
    """
    Sprawdza, czy pliki wejściowe przebiegu nadal mają skróty z manifestu.

    Zwraca manifest, jeżeli wszystko się zgadza. W przeciwnym razie podnosi
    ``NiezgodnoscWejscia`` z wykazem plików, które się rozeszły — rysunek
    wygenerowany po cichu ze starych wyników byłby błędem niewidocznym
    w gotowej pracy.
    """
    dane = wczytaj(nazwa)
    korzen = config.SIM_DIR.parent
    rozbiezne: list[str] = []
    for etykieta, opis in dane.get("wejscia", {}).items():
        sciezka = korzen / str(opis["plik"])
        if not sciezka.exists():
            rozbiezne.append(f"{etykieta}: plik zniknął ({opis['plik']})")
            continue
        if sha256_pliku(sciezka) != opis["sha256"]:
            rozbiezne.append(f"{etykieta}: zmieniona treść ({opis['plik']})")
    if rozbiezne:
        raise NiezgodnoscWejscia(
            "Pliki wejściowe przebiegu „%s” nie zgadzają się z manifestem:\n  - %s\n"
            "Przelicz przebieg (make_thesis_assets.py), zanim wygenerujesz "
            "rysunki i tabele." % (nazwa, "\n  - ".join(rozbiezne))
        )
    return dane
