# -*- coding: utf-8 -*-
"""
Nazwa pliku / Moduł: fmt.py
Opis: formatowanie liczb według konwencji polskiej. Jedno miejsce, w
      którym zapisana jest konwencja zapisu liczb: przecinek dziesiętny
      i spacja jako separator tysięcy. Formatowanie wykonywane jest
      jawnie, a nie przez ustawienia regionalne interpretera —
      ``locale`` zależy od systemu, na którym uruchomiono skrypt, więc
      ten sam skrypt dawałby na dwóch maszynach dwa różne rysunki.

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

import math

# Spacja nierozdzielająca — separator tysięcy w tekście bez składu TeX-owego.
NBSP = " "

# Miesiące cyframi rzymskimi. Konwencja stosowana na wszystkich osiach czasu
# w pracy, zgodna z zapisem w tekście.
MIESIACE_RZYMSKIE = ("I", "II", "III", "IV", "V", "VI",
                     "VII", "VIII", "IX", "X", "XI", "XII")


def _rozdziel(x: float, miejsca: int) -> tuple[str, str, str]:
    """Zwraca (znak, część całkowitą, część dziesiętną) po zaokrągleniu."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise ValueError("Nie formatuje się wartości brakującej — "
                         "brak danych ma być widoczny, a nie zapisany jako 0.")
    znak = "−" if x < 0 else ""      # minus typograficzny, nie dywiz
    tekst = f"{abs(float(x)):.{miejsca}f}"
    calosc, _, dziesietne = tekst.partition(".")
    return znak, calosc, dziesietne


def _grupuj(calosc: str, separator: str) -> str:
    """Wstawia separator tysięcy od prawej strony, grupami po trzy cyfry."""
    if len(calosc) <= 4:
        # Liczby czterocyfrowe zapisuje się bez separatora — tak jak
        # group-minimum-digits=4 w konfiguracji siunitx w preambule.
        return calosc
    grupy = []
    while len(calosc) > 3:
        grupy.insert(0, calosc[-3:])
        calosc = calosc[:-3]
    grupy.insert(0, calosc)
    return separator.join(grupy)


def pl(x: float, miejsca: int = 1, separator: str = NBSP) -> str:
    """Liczba w zapisie polskim: przecinek dziesiętny, spacja w tysiącach."""
    znak, calosc, dziesietne = _rozdziel(x, miejsca)
    wynik = znak + _grupuj(calosc, separator)
    return wynik + ("," + dziesietne if dziesietne else "")


def pl_tex(x: float, miejsca: int = 1) -> str:
    """Jak ``pl``, ale separatorem tysięcy jest ``\\,`` (spacja w TeX-u)."""
    znak, calosc, dziesietne = _rozdziel(x, miejsca)
    znak = "$-$" if znak else ""
    wynik = znak + _grupuj(calosc, r"\,")
    return wynik + ("," + dziesietne if dziesietne else "")


def num(x: float, miejsca: int = 1) -> str:
    """Makro ``\\num`` pakietu siunitx; separator dziesiętny na wejściu: kropka."""
    return r"\num{%s}" % f"{float(x):.{miejsca}f}"


def qty(x: float, jednostka: str, miejsca: int = 1) -> str:
    """Makro ``\\qty`` pakietu siunitx. ``jednostka`` zapisana makrami siunitx."""
    return r"\qty{%s}{%s}" % (f"{float(x):.{miejsca}f}", jednostka)


def qtyrange(a: float, b: float, jednostka: str, miejsca: int = 1) -> str:
    """Makro ``\\qtyrange`` — zakres z jednostką wypisaną raz."""
    return r"\qtyrange{%s}{%s}{%s}" % (
        f"{float(a):.{miejsca}f}", f"{float(b):.{miejsca}f}", jednostka)


def formatter_osi(miejsca: int = 0):
    """
    Zwraca funkcję dla ``matplotlib.ticker.FuncFormatter``.

    Liczba miejsc po przecinku jest argumentem, a nie wynikiem heurystyki:
    jednolitość zapisu na osi ma pierwszeństwo przed skracaniem etykiet.
    """
    def _fmt(wartosc, _pozycja=None):
        return pl(wartosc, miejsca)
    return _fmt
