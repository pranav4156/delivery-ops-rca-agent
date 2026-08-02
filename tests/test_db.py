"""
Load layer and validation tests.

The central assertion of this file is that the load layer is ADDITIVE: it casts
types and adds derived columns, and it never mutates, imputes, caps or drops a
value. Several tests exist specifically to fail if someone later "cleans" the
data.
"""

import pytest

import config
from app import db, tables
from app.validation import ValidationError, validate


# ═══════════════════════════════════════════════════════════════════════
# SHAPE
# ═══════════════════════════════════════════════════════════════════════

def test_row_and_entity_counts(con):
    assert db.row_count(con) == config.EXPECTED_ROW_COUNT == 3813
    assert con.execute("SELECT COUNT(DISTINCT store) FROM fact_store_hour").fetchone()[0] == 233
    assert con.execute("SELECT COUNT(DISTINCT city) FROM fact_store_hour").fetchone()[0] == 9


def test_single_date(con):
    dates = [str(r[0]) for r in
             con.execute("SELECT DISTINCT charge_date FROM fact_store_hour").fetchall()]
    assert dates == [config.DATA_DATE]


def test_hour_is_integer_not_float(con):
    """CSV stores hour as 19.0; float keys break joins and dict lookups."""
    assert db.column_types(con)["hour"] == "INTEGER"


# ═══════════════════════════════════════════════════════════════════════
# NOTHING WAS MUTATED
# ═══════════════════════════════════════════════════════════════════════

def test_null_or2a_preserved_not_imputed(con):
    """
    2 rows have NULL avg_or2a because those store-hours were 100% cancelled --
    no order ever reached assignment. Filling with 0 would report perfect
    performance for a total failure.
    """
    n = con.execute("SELECT COUNT(*) FROM fact_store_hour WHERE avg_or2a IS NULL").fetchone()[0]
    assert n == config.EXPECTED_NULL_OR2A_ROWS == 2


def test_null_man_hour_preserved_not_imputed(con):
    """139 NULL man_hour rows are the worst supply failures, not missing data."""
    n = con.execute("SELECT COUNT(*) FROM fact_store_hour WHERE man_hour IS NULL").fetchone()[0]
    assert n == config.EXPECTED_NULL_MANHOUR_ROWS == 139


def test_outliers_not_capped(con):
    """Max OR2A is 2,488 min. Capping would hide STORE_210, the worst store."""
    mx = con.execute("SELECT MAX(avg_or2a) FROM fact_store_hour").fetchone()[0]
    assert round(mx, 1) == 2488.2


def test_misspelled_columns_preserved(con):
    """
    The schema doc says use `orginal_size` as-is -- it is the real production
    column name. Renaming breaks queries against the client's warehouse.
    """
    cols = db.column_types(con)
    assert "orginal_size" in cols
    assert "orginal_capacity_booked" in cols
    assert "original_size" not in cols


def test_raw_city_untouched(con):
    """'Mumbai  north' keeps its double space -- it is the ops team's own label."""
    n = con.execute("SELECT COUNT(*) FROM fact_store_hour WHERE city = 'Mumbai  north'").fetchone()[0]
    assert n == 847


# ═══════════════════════════════════════════════════════════════════════
# DERIVED COLUMNS
# ═══════════════════════════════════════════════════════════════════════

def test_city_norm_collapses_whitespace(con):
    v = con.execute(
        "SELECT DISTINCT city_norm FROM fact_store_hour WHERE city = 'Mumbai  north'"
    ).fetchone()[0]
    assert v == "mumbai north"


def test_no_supply_flag_matches_null_man_hour(con):
    """
    The flag must be exactly equivalent to man_hour IS NULL. If they ever
    diverge, the utilization check's NO_SUPPLY branch is testing the wrong
    condition.
    """
    n = con.execute(
        "SELECT COUNT(*) FROM fact_store_hour WHERE (no_supply_flag=1) <> (man_hour IS NULL)"
    ).fetchone()[0]
    assert n == 0


def test_minute_to_hour_conversion(con):
    """Source columns are in MINUTES despite being named *_per_hour."""
    n = con.execute("""
        SELECT COUNT(*) FROM fact_store_hour
        WHERE booked_hours_per_hour > 0
          AND abs(booked_hours - booked_hours_per_hour/60.0) > 1e-9
    """).fetchone()[0]
    assert n == 0


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def test_validation_passes_on_real_data(con):
    results = validate(con, raise_on_fail=False)
    failed = [r.name for r in results if not r.passed]
    assert failed == [], f"invariants violated: {failed}"


@pytest.mark.parametrize("expected_check,corruption", [
    ("grain_unique",
     "INSERT INTO fact_store_hour SELECT * FROM fact_store_hour WHERE store='STORE_003' AND hour=8"),
    ("breached_rate_consistent",
     "UPDATE fact_store_hour SET breached_rate=0.5 WHERE store='STORE_003' AND hour=8"),
    ("order_outcomes_sum",
     "UPDATE fact_store_hour SET cancelled_order_count=999 WHERE store='STORE_003' AND hour=8"),
    ("pileup_flag_consistent",
     "UPDATE fact_store_hour SET pileup_flag=1 WHERE store='STORE_001' AND hour=22"),
    ("man_hour_is_ratio",
     "UPDATE fact_store_hour SET man_hour=7.5 WHERE store='STORE_003' AND hour=8"),
    ("store_maps_to_one_city",
     "UPDATE fact_store_hour SET city='Chennai' WHERE store='STORE_003' AND hour=8"),
    ("supply_constant_within_slot",
     "UPDATE fact_store_hour SET booked_size=999 WHERE store='STORE_001' AND hour=18"),
    ("no_zero_denominators",
     "UPDATE fact_store_hour SET order_projection=0 WHERE store='STORE_003' AND hour=8"),
])
def test_each_assertion_actually_fires(fresh_con, expected_check, corruption):
    """
    An assertion that never fails is indistinguishable from one that CANNOT
    fail. Each invariant is verified against a deliberate corruption.
    """
    fresh_con.execute(corruption)
    results = validate(fresh_con, raise_on_fail=False)
    fired = {r.name for r in results if not r.passed and r.tier == "structural"}
    assert expected_check in fired


def test_validation_error_includes_offending_rows(fresh_con):
    """A bare 'assertion failed' is undebuggable; the message must show data."""
    fresh_con.execute(
        "UPDATE fact_store_hour SET breached_count=9999 WHERE store='STORE_003' AND hour=8")
    with pytest.raises(ValidationError) as e:
        validate(fresh_con)
    assert "STORE_003" in str(e.value)
    assert "offending" in str(e.value)
