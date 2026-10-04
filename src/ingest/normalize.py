"""Text normalisation shared by ingestion, entity linking and answer matching.

Three corpus-specific traps are handled here. Each one silently costs accuracy
if you skip it, so they are documented and unit-tested (see tests/test_normalize.py):

1. ``event: Men's 68kg`` -- some infoboxes omit the space between a number and
   its unit, so ``68kg`` must normalise the same way as ``68 kg``.
2. ``Men's +80 kg`` vs ``Men's 80 kg`` -- these are *different* events. Naive
   punctuation stripping merges them, which silently returns the wrong medalist.
3. Unicode: the corpus uses en-dashes in titles and accented medalist names.
"""
from __future__ import annotations

import re
import unicodedata

# en dash, em dash, minus sign, hyphen
DASHES = "–—−-"
_DASH_RE = re.compile(f"[{DASHES}]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_DIGIT_THEN_ALPHA = re.compile(r"(?<=\d)(?=[a-z])")
_ALPHA_THEN_DIGIT = re.compile(r"(?<=[a-z])(?=\d)")


def strip_accents(s: str) -> str:
    """'René Dupont' -> 'Rene Dupont' (names in this corpus are accented)."""
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def norm(s: str | None) -> str:
    """Canonical comparison key.

    >>> norm("Men's 68kg") == norm("Men's 68 kg")
    True
    >>> norm("Men's +80 kg") == norm("Men's 80 kg")
    False
    """
    if not s:
        return ""
    s = strip_accents(s).lower()
    s = _DASH_RE.sub("-", s)
    # Trap 2: keep the '+' distinction alive before punctuation is stripped.
    s = s.replace("+", " plus ")
    s = _NON_ALNUM_RE.sub(" ", s)
    # Trap 1: split number/unit run-ons so '68kg' -> '68 kg'.
    s = _DIGIT_THEN_ALPHA.sub(" ", s)
    s = _ALPHA_THEN_DIGIT.sub(" ", s)
    return " ".join(s.split())


def norm_tokens(s: str | None) -> set[str]:
    return set(norm(s).split())


def to_int(v: str | None) -> int | None:
    """First integer in a field value, or None. Handles '1,024' and '24 (12 boats)'."""
    if not v:
        return None
    m = re.search(r"\d+", v.replace(",", ""))
    return int(m.group()) if m else None


def split_people(v: str | None) -> list[str]:
    """Medalist fields concatenate team members without a separator:

        'Ada LovelaceGrace HopperKaren Jones'  ->  3 people

    Wikipedia's infobox renders one <br> per athlete; the plain-text conversion
    dropped it. We recover the boundary at a lower->upper case transition, which
    is the only signal left. Names that are genuinely one person are unaffected.
    """
    if not v:
        return []
    v = v.strip()
    parts = re.sub(r"(?<=[a-zß-ÿ])(?=[A-ZÀ-Þ])", "\u0000", v).split("\u0000")
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Trap 4: dates. Added after the agent returned ungrounded answers on multi-hop
# questions because "August 12, 2008" would not match an event stored as
# "12 August 2008". See date_keys().
# ---------------------------------------------------------------------------
_MONTHS = {}
for i, m in enumerate(["january","february","march","april","may","june","july",
                       "august","september","october","november","december"], 1):
    _MONTHS[m] = i; _MONTHS[m[:3]] = i
_MONTHS["sept"] = 9
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_PAREN = re.compile(r"\([^)]*\)")
_ISO = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
_DMY = re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b")
_TOKEN = re.compile(rf"(?P<month>{_MONTH_RE})|(?P<num>\d{{1,4}})|(?P<dash>[–—−-]|\bto\b)")

def date_keys(s):
    """Every (month, day) a date string denotes, as {'MM-DD', ...}.

    Infobox dates are wildly irregular -- '7 August 2016', '11-17 August',
    'February 28, 1988', '25 September 2000 (qualification)27 September 2000
    (final)' -- and questions phrase the same day differently again. Comparing
    normalised *strings* therefore fails on word order alone, which is how
    'August 12, 2008' missed an event recorded as '12 August 2008'.
    """
    if not s:
        return set()
    t = _PAREN.sub(" ", s.lower())
    out = set()
    for _y, mo, d in _ISO.findall(t):
        out.add(f"{int(mo):02d}-{int(d):02d}")
    for d, mo, _y in _DMY.findall(t):
        out.add(f"{int(mo):02d}-{int(d):02d}")
    if out:
        return out
    # Bind each day number to the nearest month: the one that follows it if
    # any ('27 July'), otherwise the one before it ('August 12').
    items = [(m.lastgroup, m.group()) for m in _TOKEN.finditer(t)]
    pending, bound, last_month = [], [], None
    for kind, val in items:
        if kind == "month":
            mo = _MONTHS[val]
            for n, dash in pending:
                bound.append((mo, n, dash))
            pending, last_month = [], mo
        elif kind == "num":
            n = int(val)
            if 1 <= n <= 31:
                pending.append((n, False))
            # a 4-digit year is simply dropped
        elif kind == "dash" and pending:
            pending[-1] = (pending[-1][0], True)
        elif kind == "dash" and last_month:
            bound.append((last_month, None, True))
    for n, dash in pending:              # trailing numbers: 'August 12'
        if last_month:
            bound.append((last_month, n, dash))
    prev = None
    for mo, n, dash in bound:
        if n is None:
            continue
        out.add(f"{mo:02d}-{n:02d}")
        if prev and prev[2] and prev[0] == mo and prev[1] < n:
            for d in range(prev[1], n + 1):
                out.add(f"{mo:02d}-{d:02d}")
        prev = (mo, n, dash)
    return out

