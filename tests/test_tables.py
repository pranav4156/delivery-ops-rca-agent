"""
Derived table tests.

Two of these tables exist purely to prevent wrong answers, and the tests that
matter most are the ones that pin the WRONG value alongside the right one --
so that a later "simplification" fails loudly rather than silently regressing.
"""

import config


# ═══════════════════════════════════════════════════════════════════════
# SHAPE
# ═══════════════════════════════════════════════════════════════════════

def test_table_row_counts(con):
    assert con.execute("SELECT COUNT(*) FROM dim_store").fetchone()[0] == 233
    assert con.execute("SELECT COUNT(*) FROM agg_store_day").fetchone()[0] == 233
    assert con.execute("SELECT COUNT(*) FROM agg_city_day").fetchone()[0] == 9


def test_slot_supply_is_smaller_than_fact(con):
    """Multi-hour slots must collapse; equal counts would mean dedup did nothing."""
    slots = con.execute("SELECT COUNT(*) FROM slot_supply").fetchone()[0]
    assert slots == 2280
    assert slots < 3813


# ═══════════════════════════════════════════════════════════════════════
# THE DEDUP FIX  — the single most important table
# ═══════════════════════════════════════════════════════════════════════

def test_slot_dedup_prevents_double_counting(con):
    """
    Rider-supply columns are slot-level values replicated onto every hour of the
    slot. Summing them from the fact table double-counts every multi-hour slot.

    Both values are asserted deliberately: 249 is correct, 426 is the bug. If
    someone later points agg_store_day at fact_store_hour, this test fails.
    """
    naive = con.execute(
        "SELECT SUM(booked_size) FROM fact_store_hour WHERE store='STORE_003'").fetchone()[0]
    deduped = con.execute(
        "SELECT SUM(booked_size) FROM slot_supply WHERE store='STORE_003'").fetchone()[0]

    assert naive == 426, "the known-wrong value"
    assert deduped == 249, "the correct value"
    assert naive / deduped > 1.7


def test_agg_store_day_uses_deduped_supply(con):
    v = con.execute(
        "SELECT booked_size FROM agg_store_day WHERE store='STORE_003'").fetchone()[0]
    assert v == 249


def test_multi_hour_slot_collapses_to_one_row(con):
    hours_covered = con.execute("""
        SELECT hours_covered FROM slot_supply
        WHERE store='STORE_001' AND slot_name='Night Slot 1'
    """).fetchone()[0]
    assert hours_covered == 2


# ═══════════════════════════════════════════════════════════════════════
# WEIGHTED ROLLUPS
# ═══════════════════════════════════════════════════════════════════════

def test_store_day_golden_fixture(con):
    r = con.execute("""
        SELECT total_orders, breached_count, w_breached_rate, w_avg_or2a,
               problem_hours, hours_present
        FROM agg_store_day WHERE store='STORE_003'
    """).fetchone()
    assert r[0] == 991
    assert r[1] == 760
    assert abs(r[2] - 0.766902) < 1e-6
    assert abs(r[3] - 317.6698) < 1e-3
    assert r[4] == 16
    assert r[5] == 16


def test_city_day_golden_fixture(con):
    r = con.execute("""
        SELECT store_count, total_orders, breached_count, w_breached_rate, w_avg_or2a
        FROM agg_city_day WHERE city='Bangalore'
    """).fetchone()
    assert r[0] == 97
    assert r[1] == 88340
    assert r[2] == 10635
    assert abs(r[3] - 0.120387) < 1e-6
    assert abs(r[4] - 33.3561) < 1e-3


def test_weighted_differs_from_naive_average(con):
    """
    Both reference docs warn: "Do not average OR2A across stores without
    weighting by order volume."

    The naive form is what a text-to-SQL agent writes, and it is 29% wrong for
    Bangalore. Asserting the two DIFFER means nobody can quietly replace the
    weighted formula with AVG().
    """
    naive = con.execute(
        "SELECT AVG(avg_or2a) FROM fact_store_hour WHERE city='Bangalore'").fetchone()[0]
    weighted = con.execute(
        "SELECT w_avg_or2a FROM agg_city_day WHERE city='Bangalore'").fetchone()[0]

    assert abs(naive - 43.1354) < 1e-3, "the known-wrong value"
    assert abs(weighted - 33.3561) < 1e-3, "the correct value"
    assert abs(naive - weighted) > 9.0


def test_null_or2a_excluded_from_both_sides(con):
    """
    In SQL, SUM() skips NULLs in the numerator but the denominator would still
    count those rows' total_orders -- silently deflating the result ~4.5%.
    The FILTER must be on BOTH sides.
    """
    ours = con.execute(
        "SELECT w_avg_or2a FROM agg_store_day WHERE store='STORE_008'").fetchone()[0]
    numerator_only = con.execute("""
        SELECT SUM(avg_or2a*total_orders)/SUM(total_orders)
        FROM fact_store_hour WHERE store='STORE_008'
    """).fetchone()[0]

    assert abs(ours - 79.9034) < 1e-3, "the correct value"
    assert abs(numerator_only - 76.3029) < 1e-3, "the known-wrong value"
    assert ours > numerator_only


def test_city_rollup_not_built_from_store_averages(con):
    """
    agg_city_day must aggregate from fact_store_hour, not from agg_store_day --
    averaging store averages would be a second unweighted-average bug one layer
    up.
    """
    correct = con.execute(
        "SELECT w_avg_or2a FROM agg_city_day WHERE city='Bangalore'").fetchone()[0]
    avg_of_store_avgs = con.execute(
        "SELECT AVG(w_avg_or2a) FROM agg_store_day WHERE city='Bangalore'").fetchone()[0]
    assert abs(correct - avg_of_store_avgs) > 1.0


# ═══════════════════════════════════════════════════════════════════════
# NULL HANDLING
# ═══════════════════════════════════════════════════════════════════════

def test_rider_hours_zero_not_null_when_nobody_booked(con):
    """
    STORE_128 books zero riders all day, so SUM(rider_hours) is NULL. That NULL
    means "no rider booked, therefore zero hours worked" -- 0.0 is the correct
    value, and leaving it NULL crashed float() before this was fixed.
    """
    r = con.execute("""
        SELECT rider_hours, booked_hours, booked_size
        FROM agg_store_day WHERE store='STORE_128'
    """).fetchone()
    assert r[0] == 0.0
    assert r[1] == 0.0
    assert r[2] == 0


def test_no_nulls_in_aggregate_hour_columns(con):
    n = con.execute("""
        SELECT COUNT(*) FROM agg_store_day
        WHERE rider_hours IS NULL OR booked_hours IS NULL
    """).fetchone()[0]
    assert n == 0


# ═══════════════════════════════════════════════════════════════════════
# LOOKUPS
# ═══════════════════════════════════════════════════════════════════════

def test_dim_store_maps_city(con):
    assert con.execute(
        "SELECT city FROM dim_store WHERE store='STORE_003'").fetchone()[0] == "Bangalore"
    assert con.execute(
        "SELECT city FROM dim_store WHERE store='STORE_210'").fetchone()[0] == "Noida"
    assert con.execute(
        "SELECT COUNT(*) FROM dim_store WHERE city='Bangalore'").fetchone()[0] == 97
