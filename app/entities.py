"""
Entity resolution — turning what a user types into a value that exists in the data.

WHY THIS MODULE IS DISPROPORTIONATELY IMPORTANT
-----------------------------------------------
This is unglamorous string matching, and it is the single most likely cause of a
live demo failure -- because it triggers on the FIRST question.

The stored city values are dirty:

    'Mumbai  north'    double space, lowercase 'north'
    'New delhi'        lowercase 'd'
    'Pune city east'   'city' embedded mid-name

A user types "Mumbai". Exact matching returns zero rows. The agent says "no
data". It looks broken, when in fact it is a whitespace bug.

Separately, the brief's sample questions reference stores TBC8, TBF1 and TBC9
which do not exist in the delivered CSV (verified by exhaustive search across
every text column) -- the data was anonymised to STORE_001..STORE_233 after the
brief was written. The panel may still type those codes from their own script.
So an unresolvable entity must produce a helpful "did you mean" response, never
a crash and never a guess.

DESIGN: TIERED MATCHING, FAIL LOUD
-----------------------------------
Resolution walks progressively looser tiers and stops at the first hit. Every
result records WHICH tier matched, so downstream code and the walkthrough can
see exactly how an answer was reached.

The most important rule: WHEN IN DOUBT, DECLINE. A wrong entity match produces a
confidently wrong answer about the wrong store, which is far worse than "I don't
recognise that -- did you mean one of these?".

That principle drives one specific safeguard. Numeric shorthand ("3" -> STORE_003)
is deliberately restricted to inputs that are PURELY numeric or match an explicit
'store N' pattern. It must never fire on arbitrary strings that merely contain a
digit -- otherwise "TBC8" would silently resolve to STORE_008 and the agent would
confidently analyse the wrong store. See _try_numeric().
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher, get_close_matches

import duckdb

import config


# ═══════════════════════════════════════════════════════════════════════
# RESULT TYPE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Resolution:
    """
    Outcome of resolving one entity.

    `method` records which tier matched. It is surfaced in the resolved-context
    panel so the demo can show HOW an entity was understood, not just what it
    resolved to.
    """
    query: str
    value: str | None = None
    matched: bool = False
    method: str | None = None          # exact | normalized | alias | token | numeric | fuzzy
    suggestions: list[str] = field(default_factory=list)
    message: str | None = None         # user-facing text when unmatched

    def __bool__(self) -> bool:
        return self.matched


# ═══════════════════════════════════════════════════════════════════════
# NORMALISATION
# ═══════════════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    """
    Lowercase, collapse internal whitespace, strip.

    Mirrors the city_norm column built in db.py so Python-side and SQL-side
    matching can never drift apart.
    """
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# ═══════════════════════════════════════════════════════════════════════
# ENTITY INDEX
# ═══════════════════════════════════════════════════════════════════════

class EntityIndex:
    """
    In-memory index of every city and store, built once at startup.

    233 stores and 9 cities -- small enough that holding it in Python avoids a
    database round-trip on every turn and makes resolution unit-testable
    without a connection.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection):
        rows = con.execute(
            "SELECT store, city, city_norm FROM dim_store ORDER BY store"
        ).fetchall()

        self.stores: list[str] = [r[0] for r in rows]
        self.store_city: dict[str, str] = {r[0]: r[1] for r in rows}

        self.cities: list[str] = sorted({r[1] for r in rows})
        self.city_norm_to_raw: dict[str, str] = {normalize(r[1]): r[1] for r in rows}

        self.city_stores: dict[str, list[str]] = {}
        for store, city, _ in rows:
            self.city_stores.setdefault(city, []).append(store)

        # store code -> canonical, keyed by an aggressively stripped form
        # ("store_003" / "STORE 003" / "store-003" all collapse to "STORE003")
        self._store_squashed: dict[str, str] = {
            re.sub(r"[^a-z0-9]", "", s.lower()): s for s in self.stores
        }

        # numeric suffix -> store, for the strict numeric shorthand tier
        self._store_by_number: dict[int, str] = {}
        for s in self.stores:
            m = re.search(r"(\d+)$", s)
            if m:
                self._store_by_number[int(m.group(1))] = s

    def stores_in(self, city: str | None) -> list[str]:
        return self.city_stores.get(city, []) if city else self.stores


# ═══════════════════════════════════════════════════════════════════════
# CITY RESOLUTION
# ═══════════════════════════════════════════════════════════════════════

def resolve_city(text: str, index: EntityIndex) -> Resolution:
    """
    Resolve free text to an exact stored city value.

    Tiers: exact -> normalized -> alias -> token-subset -> fuzzy -> decline.
    """
    res = Resolution(query=text)
    if not text or not str(text).strip():
        res.message = "No city specified."
        res.suggestions = index.cities[: config.MAX_SUGGESTIONS]
        return res

    norm = normalize(text)

    # 1. exact against the raw stored value
    if text in index.cities:
        return Resolution(query=text, value=text, matched=True, method="exact")

    # 2. normalized -- catches 'Mumbai north' (single space) vs 'Mumbai  north'
    if norm in index.city_norm_to_raw:
        return Resolution(query=text, value=index.city_norm_to_raw[norm],
                          matched=True, method="normalized")

    # 3. alias -- 'mumbai', 'bengaluru', 'delhi', 'bombay'
    if norm in config.CITY_ALIASES:
        alias_target = config.CITY_ALIASES[norm]
        if alias_target in index.cities:
            return Resolution(query=text, value=alias_target,
                              matched=True, method="alias")

    # 4. token subset -- every word typed appears in the stored name.
    #    Handles "north mumbai", "pune east", "hyderabad city".
    #    Only accepted when exactly one candidate matches; ambiguity declines.
    query_tokens = set(norm.split())
    if query_tokens:
        hits = [
            raw for cnorm, raw in index.city_norm_to_raw.items()
            if query_tokens <= set(cnorm.split())
        ]
        if len(hits) == 1:
            return Resolution(query=text, value=hits[0], matched=True, method="token")

    # 5. fuzzy -- typos like "Bangalor", "Chenai"
    candidates = list(index.city_norm_to_raw) + list(config.CITY_ALIASES)
    close = get_close_matches(norm, candidates, n=1, cutoff=config.FUZZY_MATCH_CUTOFF)
    if close:
        target = index.city_norm_to_raw.get(close[0]) or config.CITY_ALIASES.get(close[0])
        if target in index.cities:
            return Resolution(query=text, value=target, matched=True, method="fuzzy")

    # 6. decline, with the full list -- only 9 cities, so show them all
    res.suggestions = index.cities
    res.message = (
        f"I don't have data for '{text}'. Available cities: "
        + ", ".join(index.cities)
        + "."
    )
    return res


# ═══════════════════════════════════════════════════════════════════════
# STORE RESOLUTION
# ═══════════════════════════════════════════════════════════════════════

_NUMERIC_PATTERNS = [
    re.compile(r"^(\d{1,4})$"),                          # "3", "003"
    re.compile(r"^store[\s_\-#]*(\d{1,4})$"),            # "store 3", "store_003"
    re.compile(r"^#\s*(\d{1,4})$"),                      # "#3"
]


def _try_numeric(norm: str, index: EntityIndex) -> str | None:
    """
    Strict numeric shorthand.

    SAFETY: only fires when the WHOLE input is a number or an explicit
    'store N' form. Deliberately does NOT extract digits from arbitrary text --
    otherwise "TBC8" would resolve to STORE_008 and the agent would confidently
    analyse a store the user never asked about. Declining is the correct
    behaviour there.
    """
    for pattern in _NUMERIC_PATTERNS:
        m = pattern.match(norm)
        if m:
            return index._store_by_number.get(int(m.group(1)))
    return None


def resolve_store(
    text: str,
    index: EntityIndex,
    city_hint: str | None = None,
) -> Resolution:
    """
    Resolve free text to an exact store code.

    city_hint scopes the suggestion list when resolution fails, so the user is
    offered stores from the city already in conversation context.
    """
    res = Resolution(query=text)
    if not text or not str(text).strip():
        res.message = "No store specified."
        res.suggestions = index.stores_in(city_hint)[: config.MAX_SUGGESTIONS]
        return res

    norm = normalize(text)

    # 1. exact
    if text in index.store_city:
        return Resolution(query=text, value=text, matched=True, method="exact")

    # 2. squashed -- "store_003" / "STORE 003" / "store-003"
    squashed = re.sub(r"[^a-z0-9]", "", norm)
    if squashed in index._store_squashed:
        return Resolution(query=text, value=index._store_squashed[squashed],
                          matched=True, method="normalized")

    # 3. strict numeric shorthand
    numeric = _try_numeric(norm, index)
    if numeric:
        return Resolution(query=text, value=numeric, matched=True, method="numeric")

    # 4. fuzzy against squashed store codes
    close = get_close_matches(
        squashed, list(index._store_squashed), n=1, cutoff=config.FUZZY_MATCH_CUTOFF
    )
    if close:
        return Resolution(query=text, value=index._store_squashed[close[0]],
                          matched=True, method="fuzzy")

    # 5. decline with scoped suggestions.
    #    This is the TBC8 path -- the demo-safety net.
    pool = index.stores_in(city_hint)
    ranked = sorted(pool, key=lambda s: _similarity(squashed, s.lower().replace("_", "")),
                    reverse=True)
    res.suggestions = ranked[: config.MAX_SUGGESTIONS]

    scope = f" in {city_hint}" if city_hint else ""
    res.message = (
        f"I don't have a store called '{text}'. "
        f"Store codes in this dataset look like STORE_001 to STORE_233. "
        f"Valid stores{scope} include: " + ", ".join(res.suggestions) + "."
    )
    return res


# ═══════════════════════════════════════════════════════════════════════
# HOUR RESOLUTION
# ═══════════════════════════════════════════════════════════════════════

_HOUR_RANGE = re.compile(r"^(\d{1,2})\s*(?:-|to|until|till)\s*(\d{1,2})$")
_SINGLE_HOUR = re.compile(r"^(?:hour\s*)?(\d{1,2})(?:\s*(?:h|hr|hrs|:00))?$")


def resolve_hours(text: str) -> Resolution:
    """
    Resolve an hour expression to a concrete list of hours.

    Handles named windows ("morning"), explicit hours ("7pm", "hour 19") and
    ranges ("8-11").

    NOTE: the named windows in config.HOUR_LABELS are OUR judgment call -- no
    source document defines them. They are pinned in config so they stay
    consistent and can be defended, rather than being re-invented per query.
    """
    res = Resolution(query=text)
    if not text or not str(text).strip():
        res.message = "No hour or time window specified."
        return res

    norm = normalize(text)

    # named window
    if norm in config.HOUR_LABELS:
        return Resolution(query=text, value=config.HOUR_LABELS[norm],
                          matched=True, method="label")

    # synonym -- "midday", "lunchtime", "overnight"
    if norm in config.HOUR_LABEL_ALIASES:
        target = config.HOUR_LABEL_ALIASES[norm]
        return Resolution(query=text, value=config.HOUR_LABELS[target],
                          matched=True, method="alias")

    # tolerate "morning hours", "the morning", "early morning".
    # Longest label first so "afternoon" is not shadowed by "noon".
    for label in sorted(config.HOUR_LABELS, key=len, reverse=True):
        if re.search(rf"\b{label}\b", norm):
            return Resolution(query=text, value=config.HOUR_LABELS[label],
                              matched=True, method="label")
    for alias in sorted(config.HOUR_LABEL_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", norm):
            target = config.HOUR_LABEL_ALIASES[alias]
            return Resolution(query=text, value=config.HOUR_LABELS[target],
                              matched=True, method="alias")

    # 12-hour clock: "7pm", "8 am"
    m = re.match(r"^(\d{1,2})\s*(am|pm)$", norm)
    if m:
        h = int(m.group(1)) % 12
        if m.group(2) == "pm":
            h += 12
        return Resolution(query=text, value=[h], matched=True, method="clock")

    # range: "8-11", "8 to 11"
    m = _HOUR_RANGE.match(norm)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if 0 <= lo <= 23 and 0 <= hi <= 23 and lo <= hi:
            return Resolution(query=text, value=list(range(lo, hi + 1)),
                              matched=True, method="range")

    # single hour
    m = _SINGLE_HOUR.match(norm)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return Resolution(query=text, value=[h], matched=True, method="hour")

    # fuzzy on the label names
    close = get_close_matches(norm, list(config.HOUR_LABELS),
                              n=1, cutoff=config.FUZZY_MATCH_CUTOFF)
    if close:
        return Resolution(query=text, value=config.HOUR_LABELS[close[0]],
                          matched=True, method="fuzzy")

    res.suggestions = list(config.HOUR_LABELS)
    res.message = (
        f"I couldn't interpret '{text}' as a time window. Try: "
        + ", ".join(config.HOUR_LABELS)
        + ", an hour like '19', or a range like '8-11'."
    )
    return res
