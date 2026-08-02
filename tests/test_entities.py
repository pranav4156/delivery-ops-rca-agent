"""
Entity resolution tests.

This is the module most likely to break a live demo, because it fires on the
FIRST question. The two groups that matter:

  - dirty city values: "mumbai" must reach 'Mumbai  north' (double space)
  - the TBC8 safety net: an unknown store must DECLINE, never silently resolve
    to a different store
"""

import pytest

import config
from app.entities import (
    normalize,
    resolve_city,
    resolve_hours,
    resolve_store,
)


# ═══════════════════════════════════════════════════════════════════════
# INDEX
# ═══════════════════════════════════════════════════════════════════════

def test_index_shape(index):
    assert len(index.stores) == 233
    assert len(index.cities) == 9
    assert len(index.city_stores["Bangalore"]) == 97


def test_python_normalize_matches_sql_city_norm(con):
    """
    Python-side and SQL-side normalisation must never drift apart, or matching
    silently starts failing for some cities and not others.
    """
    rows = con.execute("SELECT DISTINCT city, city_norm FROM fact_store_hour").fetchall()
    for raw, norm in rows:
        assert normalize(raw) == norm


# ═══════════════════════════════════════════════════════════════════════
# CITY
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query,expected", [
    ("Mumbai  north", "Mumbai  north"),   # exact, double space
    ("Mumbai north",  "Mumbai  north"),   # single space
    ("mumbai",        "Mumbai  north"),   # the demo case
    ("Mumbai",        "Mumbai  north"),
    ("bombay",        "Mumbai  north"),
    ("north mumbai",  "Mumbai  north"),
    ("Bangalore",     "Bangalore"),
    ("bengaluru",     "Bangalore"),
    ("BLR",           "Bangalore"),
    ("delhi",         "New delhi"),
    ("New Delhi",     "New delhi"),
    ("pune",          "Pune city east"),
    ("hyderabad",     "Hyderabad city"),
    ("gurugram",      "Gurgaon"),
    ("madras",        "Chennai"),
    ("noida",         "Noida"),
])
def test_city_aliases(query, expected, index):
    r = resolve_city(query, index)
    assert r.matched
    assert r.value == expected


@pytest.mark.parametrize("typo,expected", [
    ("Bangalor", "Bangalore"),
    ("Chenai",   "Chennai"),
    ("noidaa",   "Noida"),
    ("gurgao",   "Gurgaon"),
])
def test_city_fuzzy_matching(typo, expected, index):
    r = resolve_city(typo, index)
    assert r.matched
    assert r.value == expected
    assert r.method == "fuzzy"


def test_unknown_city_declines_with_suggestions(index):
    r = resolve_city("Kolkata", index)
    assert not r.matched
    assert r.value is None
    assert len(r.suggestions) == 9
    assert "Kolkata" in r.message


def test_empty_city_is_safe(index):
    assert not resolve_city("", index).matched
    assert not resolve_city(None, index).matched


# ═══════════════════════════════════════════════════════════════════════
# STORE
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query", [
    "STORE_003", "store_003", "Store 003", "store-003",
    "STORE003", "3", "003", "store 3", "#3",
])
def test_store_format_variants(query, index):
    r = resolve_store(query, index)
    assert r.matched
    assert r.value == "STORE_003"


# ═══════════════════════════════════════════════════════════════════════
# THE TBC8 SAFETY NET
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("code", ["TBC8", "TBF1", "TBC9"])
def test_brief_store_codes_decline_rather_than_guess(code, index):
    """
    The brief's sample questions reference TBC8/TBF1/TBC9, which do not exist in
    the anonymised CSV. The panel may still type them. Declining with
    suggestions is correct; guessing is not.
    """
    r = resolve_store(code, index, city_hint="Bangalore")
    assert not r.matched
    assert r.value is None
    assert r.suggestions
    assert code in r.message


def test_tbc8_must_not_become_store_008(index):
    """
    REGRESSION GUARD. 'TBC8' contains the digit 8. A digit-extracting resolver
    would silently map it to STORE_008 and deliver a complete, confident,
    correct-looking RCA about a store the user never asked about.

    Both halves are asserted: TBC8 declines, but bare '8' still resolves.
    """
    assert resolve_store("TBC8", index).value is None
    assert resolve_store("TBF1", index).value is None
    assert resolve_store("8", index).value == "STORE_008"
    assert resolve_store("1", index).value == "STORE_001"


def test_store_suggestions_scoped_to_city_hint(index):
    r = resolve_store("TBC8", index, city_hint="Noida")
    assert not r.matched
    assert all(index.store_city[s] == "Noida" for s in r.suggestions)
    assert len(r.suggestions) <= config.MAX_SUGGESTIONS


# ═══════════════════════════════════════════════════════════════════════
# HOURS
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query,expected", [
    ("morning",           [6, 7, 8, 9, 10, 11]),
    ("the morning hours", [6, 7, 8, 9, 10, 11]),
    ("afternoon",         [12, 13, 14, 15, 16]),
    ("evening",           [17, 18, 19, 20]),
    ("night",             [21, 22, 0, 1]),
    ("7pm",               [19]),
    ("19",                [19]),
    ("hour 19",           [19]),
    ("8-11",              [8, 9, 10, 11]),
    ("8 to 11",           [8, 9, 10, 11]),
])
def test_hour_expressions(query, expected, index):
    r = resolve_hours(query)
    assert r.matched
    assert r.value == expected


def test_unparseable_hour_declines(index):
    r = resolve_hours("banana")
    assert not r.matched
    assert r.suggestions
