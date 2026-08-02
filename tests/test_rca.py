"""
RCA engine tests — the four playbook checks, rollups and impact ranking.

This is the correctness core. Every value here was hand-computed from the raw
CSV before the code existed, so these tests verify the implementation against
the data rather than against itself.
"""

import pytest

import config
from app import rca
from app.rollups import NoDataError, all_cities, city_day, store_day


@pytest.fixture(scope="module")
def s003(con):
    """STORE_003 hours, keyed by hour. The primary demo store."""
    hours = rca.fetch_store_hours(con, "STORE_003")
    return hours, {h["hour"]: h for h in hours}


# ═══════════════════════════════════════════════════════════════════════
# CHECK 1 — DEMAND
# ═══════════════════════════════════════════════════════════════════════

def test_demand_spike_store_003_hour_8(s003):
    hours, by_h = s003
    d = rca.check_demand_spike(by_h[8])
    assert d.flag is False
    assert d.actual == 76
    assert d.projected == 103.0
    assert round(d.pct_delta, 1) == -26.2


def test_demand_spike_threshold_is_10_percent(s003):
    """Boundary: exactly 1.10x must NOT fire; above it must."""
    _, by_h = s003
    row = dict(by_h[8])

    row["order_projection"], row["total_orders"] = 100.0, 110
    assert rca.check_demand_spike(row).flag is False

    row["total_orders"] = 111
    assert rca.check_demand_spike(row).flag is True


def test_demand_check_fires_somewhere(con):
    """Guard against a dead check that never returns True."""
    fired = sum(
        rca.check_demand_spike(r).flag
        for s in ["STORE_210", "STORE_091", "STORE_184", "STORE_001", "STORE_046"]
        for r in rca.fetch_store_hours(con, s)
    )
    assert fired > 0


# ═══════════════════════════════════════════════════════════════════════
# CHECK 2 — PILEUP
# ═══════════════════════════════════════════════════════════════════════

def test_pileup_store_003_hour_8_is_sustained(s003):
    hours, by_h = s003
    p = rca.check_pileup(by_h[8], hours)
    assert p.flag is True
    assert p.count == 4
    assert p.run_hours == [7, 8, 9, 10]
    assert p.sustained is True


def test_pileup_hour_20_run_of_two_is_not_sustained(s003):
    hours, by_h = s003
    p = rca.check_pileup(by_h[20], hours)
    assert p.flag is True
    assert p.count == 11
    assert p.run_hours == [19, 20]
    assert p.sustained is False


@pytest.mark.parametrize("hour,run_length,sustained", [
    (12, 1, False),   # isolated
    (14, 2, False),   # run of 2 -- below threshold
    (7,  4, True),    # run of 4
])
def test_sustained_threshold_boundary(hour, run_length, sustained, s003):
    hours, by_h = s003
    p = rca.check_pileup(by_h[hour], hours)
    assert p.run_length == run_length
    assert p.sustained is sustained


def test_store_003_pileup_run_structure(s003):
    hours, _ = s003
    pileup = sorted({h["hour"] for h in hours if h["pileup_flag"] == 1})
    assert pileup == [7, 8, 9, 10, 12, 14, 15, 17, 19, 20, 22]


def test_longest_pileup_run_in_dataset(con):
    hours = rca.fetch_store_hours(con, "STORE_021")
    assert max(rca.check_pileup(r, hours).run_length for r in hours) == 8


def test_strict_adjacency_does_not_bridge_gaps(s003):
    """
    Q8: hours 10 and 12 are both pileup hours but hour 11 is not, so they must
    NOT chain into one run. Gaps are genuine store closures.
    """
    hours, by_h = s003
    assert 10 in rca.check_pileup(by_h[10], hours).run_hours
    assert 12 not in rca.check_pileup(by_h[10], hours).run_hours


# ═══════════════════════════════════════════════════════════════════════
# CHECK 3a — BOOKING
# ═══════════════════════════════════════════════════════════════════════

def test_booking_gap_store_003_hour_8(s003):
    _, by_h = s003
    b = rca.check_booking_gap(by_h[8])
    assert b.flag is True
    assert (b.booked_size, b.current_size) == (32, 59)
    assert b.ratio == 0.49


def test_stored_ratio_disagrees_with_documented_formula(con):
    """
    Q3 REGRESSION GUARD.

    The schema doc says current_capacity_booked = booked_size / current_size.
    That reproduces only 32.7% of rows; the true formula subtracts
    cancelled_count. On STORE_001 hour 19 the two give OPPOSITE verdicts:

        documented  45/49      = 0.918 -> passes the 0.90 gate
        stored                 = 0.88  -> fails it

    We read the stored column so our numbers always match the ops team's SQL.
    This test fails if anyone "fixes" it back to the documented formula.
    """
    row = {h["hour"]: h for h in rca.fetch_store_hours(con, "STORE_001")}[19]
    stored = rca.check_booking_gap(row)
    documented = row["booked_size"] / row["current_size"]

    assert stored.ratio == 0.88
    assert round(documented, 3) == 0.918
    assert stored.flag is True
    assert (documented < config.BOOKING_GAP_THRESHOLD) is False
    assert stored.flag != (documented < config.BOOKING_GAP_THRESHOLD)


# ═══════════════════════════════════════════════════════════════════════
# CHECK 3b — UTILIZATION  (the NaN trap)
# ═══════════════════════════════════════════════════════════════════════

def test_utilization_store_003_hour_8(s003):
    _, by_h = s003
    u = rca.check_utilization(by_h[8])
    assert u.flag is False
    assert round(u.man_hour, 3) == 0.876
    assert u.noshows == 1
    assert u.no_supply is False


def test_zero_booking_flags_no_supply_not_silent_pass(con):
    """
    Q5 REGRESSION GUARD — the highest-value test in this file.

    139 rows have booked_size = 0 -> man_hour NULL. `NULL < 0.85` is FALSE in
    SQL and `NaN < 0.85` is False in pandas, so a naive check reports
    "no utilization problem" on the 139 WORST supply rows in the dataset.

    Both behaviours are asserted: what the naive check would say, and what ours
    does.
    """
    row = {h["hour"]: h for h in rca.fetch_store_hours(con, "STORE_128")}[8]
    assert row["man_hour"] is None
    assert row["booked_size"] == 0

    naive = row["man_hour"] is not None and row["man_hour"] < config.UTILIZATION_GAP_THRESHOLD
    assert naive is False, "the silent-failure behaviour we must avoid"

    u = rca.check_utilization(row)
    assert u.flag is True
    assert u.no_supply is True
    assert u.reason == "NO_SUPPLY"


def test_utilization_threshold_boundary(s003):
    _, by_h = s003
    row = dict(by_h[8])
    row["man_hour"] = 0.85
    assert rca.check_utilization(row).flag is False
    row["man_hour"] = 0.8499
    assert rca.check_utilization(row).flag is True


# ═══════════════════════════════════════════════════════════════════════
# COMPOUNDING
# ═══════════════════════════════════════════════════════════════════════

def test_all_four_checks_always_run(s003):
    """
    "All three checks run independently -- root causes compound, they are not
     mutually exclusive." So this is never an if/elif cascade.
    """
    hours, by_h = s003
    c = rca.run_all_checks(by_h[8], hours)
    for check in (c.demand, c.pileup, c.booking, c.utilization):
        assert check is not None
        assert isinstance(check.flag, bool)


def test_triggered_names_every_flag(s003):
    hours, by_h = s003
    assert rca.run_all_checks(by_h[8], hours).triggered == ["sustained pileup", "booking gap"]
    assert rca.run_all_checks(by_h[20], hours).triggered == ["pileup", "booking gap"]


def test_zero_flag_problem_hour_exists(con):
    """9.9% of problem hours trigger nothing. Task 2.4 needs a template for it."""
    hours = rca.fetch_store_hours(con, "STORE_001")
    c = rca.run_all_checks({h["hour"]: h for h in hours}[22], hours)
    assert c.is_problem_hour is True
    assert c.triggered == []
    assert c.any_flag is False


# ═══════════════════════════════════════════════════════════════════════
# ROLLUPS
# ═══════════════════════════════════════════════════════════════════════

def test_store_day_golden_fixture(con):
    s = store_day(con, "STORE_003")
    assert s.total_orders == 991
    assert s.breached_count == 760
    assert abs(s.w_breached_rate - 0.766902) < 1e-6
    assert abs(s.w_avg_or2a - 317.6698) < 1e-3
    assert s.problem_hours == 16
    assert s.booked_size == 249            # deduped, not 426
    assert s.provenance == config.PROVENANCE_VALIDATED


def test_city_day_golden_fixture(con):
    c = city_day(con, "Bangalore")
    assert c.store_count == 97
    assert c.total_orders == 88340
    assert abs(c.w_avg_or2a - 33.3561) < 1e-3


def test_city_top_stores_order(con):
    c = city_day(con, "Bangalore")
    assert [s.store for s in c.top_stores] == [
        "STORE_091", "STORE_054", "STORE_003", "STORE_046", "STORE_080"]
    assert abs(c.top_stores[0].w_avg_or2a - 873.166) < 1e-3


def test_worst_city_is_noida(con):
    cities = all_cities(con)
    assert len(cities) == 9
    assert cities[0].city == "Noida"


def test_double_space_city_key_works(con):
    assert city_day(con, "Mumbai  north").store_count == 51


def test_store_with_no_riders_all_day(con):
    s = store_day(con, "STORE_128")
    assert s.booked_size == 0
    assert s.rider_hours == 0.0
    assert s.no_supply_hours == s.hours_present
    assert s.total_orders > 0


@pytest.mark.parametrize("call,exc", [
    (lambda c: store_day(c, "STORE_999"), NoDataError),
    (lambda c: city_day(c, "Kolkata"), NoDataError),
    (lambda c: store_day(c, "STORE_003", date="2026-04-23"), NoDataError),
    (lambda c: city_day(c, "Bangalore", rank_metric="1; DROP TABLE dim_store"), ValueError),
])
def test_rollup_error_handling(call, exc, con):
    with pytest.raises(exc):
        call(con)


def test_all_233_stores_roll_up(con):
    stores = [r[0] for r in con.execute("SELECT store FROM dim_store").fetchall()]
    for s in stores:
        store_day(con, s)


# ═══════════════════════════════════════════════════════════════════════
# IMPACT RANKING
# ═══════════════════════════════════════════════════════════════════════

def test_ranking_keeps_every_problem_hour(con):
    """Option 2: rank the presentation, never filter the analysis."""
    r = rca.rank_problem_hours(con, "STORE_003")
    assert r.problem_hour_count == 16
    assert len(r.top) == config.TOP_N_PROBLEM_HOURS == 3
    assert r.remainder_count == 13
    assert len(r.top) + r.remainder_count == r.problem_hour_count


def test_top_three_hours(con):
    r = rca.rank_problem_hours(con, "STORE_003")
    assert [c.hour for c in r.top] == [8, 10, 20]
    assert [c.breached_count for c in r.top] == [76, 76, 74]


def test_tiebreak_by_severity(con):
    """Hours 8 and 10 both have breached_count=76; avg_or2a breaks the tie."""
    r = rca.rank_problem_hours(con, "STORE_003")
    a, b = r.problem_ranked[0], r.problem_ranked[1]
    assert a.breached_count == b.breached_count == 76
    assert a.hour == 8
    assert a.avg_or2a > b.avg_or2a


def test_ranking_is_deterministic(con):
    """A live demo must produce identical output on repeat questions."""
    seqs = {
        tuple(c.hour for c in rca.rank_problem_hours(con, "STORE_003").problem_ranked)
        for _ in range(5)
    }
    assert len(seqs) == 1


def test_show_all_hours(con):
    r = rca.rank_problem_hours(con, "STORE_003", top_n=None)
    assert len(r.top) == 16
    assert r.remainder_count == 0
    assert r.truncated is False


def test_hour_window_scoping(con):
    r = rca.rank_problem_hours(con, "STORE_003", hours=config.HOUR_LABELS["morning"])
    assert set(r.scope_hours) <= set(config.HOUR_LABELS["morning"])
    assert [c.hour for c in r.window] == sorted(c.hour for c in r.window)
    assert [c.hour for c in r.problem_ranked][:2] == [8, 10]


def test_pileup_run_uses_whole_day_not_just_window(con):
    """
    Scoping to the morning must not truncate run detection -- a run can start
    before the window opens. Filtering rows before detecting the run would
    under-report SUSTAINED, a bug that never surfaces in casual testing.
    """
    r = rca.rank_problem_hours(con, "STORE_003", hours=config.HOUR_LABELS["morning"])
    h7 = next(c for c in r.window if c.hour == 7)
    assert h7.pileup.run_hours == [7, 8, 9, 10]
    assert h7.pileup.sustained is True


def test_ranking_edge_cases(con):
    assert rca.rank_problem_hours(con, "STORE_003", hours=[3]).hours_in_scope == 0
    assert rca.rank_problem_hours(con, "STORE_999").hours_in_scope == 0


def test_all_233_stores_rank(con):
    stores = [r[0] for r in con.execute("SELECT store FROM dim_store").fetchall()]
    for s in stores:
        rca.rank_problem_hours(con, s)
