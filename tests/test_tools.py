"""
Parameterized query tool tests (Tier 2).

Two groups matter most:

  - the injection boundary: metric and condition names arrive from an LLM and
    must be whitelisted, since column names cannot be parameterized in SQL
  - weighted correctness: these tools exist precisely BECAUSE LLM-written SQL
    would use naive averages, so the tests pin the weighted behaviour
"""

import pytest

import config
from app import tools
from app.tools import (
    METRICS,
    TOOL_CATALOG,
    UnknownConditionError,
    UnknownMetricError,
    compare_entities,
    count_matching,
    hour_profile,
    metric_lookup,
    rank_entities,
    render_tool_result,
)


# ═══════════════════════════════════════════════════════════════════════
# rank_entities
# ═══════════════════════════════════════════════════════════════════════

def test_rank_stores_matches_golden_fixture(con):
    r = rank_entities(con, level="store", scope="Bangalore", metric="w_avg_or2a", n=5)
    assert [x["store"] for x in r.rows] == [
        "STORE_091", "STORE_054", "STORE_003", "STORE_046", "STORE_080"]


def test_rank_cities_by_noshows(con):
    r = rank_entities(con, level="city", metric="noshow_count", n=3)
    assert r.rows[0]["city"] == "Bangalore"


def test_worst_and_best_are_opposite(con):
    worst = rank_entities(con, level="city", metric="w_avg_or2a", n=1, direction="worst")
    best = rank_entities(con, level="city", metric="w_avg_or2a", n=1, direction="best")
    assert worst.rows[0]["city"] == "Noida"
    assert best.rows[0]["city"] == "Faridabad"


def test_direction_respects_metric_polarity(con):
    """
    High breach rate is bad; high order volume is not. "Best by volume" must
    therefore sort DESC while "best by OR2A" sorts ASC.
    """
    r = rank_entities(con, level="city", metric="total_orders", n=1, direction="best")
    assert r.rows[0]["city"] == "Bangalore"


def test_rank_pluralisation(con):
    assert "cities" in rank_entities(con, level="city", n=2).title
    assert "stores" in rank_entities(con, level="store", n=2).title


# ═══════════════════════════════════════════════════════════════════════
# compare_entities
# ═══════════════════════════════════════════════════════════════════════

def test_compare_stores(con):
    r = compare_entities(con, ["STORE_003", "STORE_091"])
    or2a = next(row for row in r.rows if row["metric"] == "Avg OR2A")
    assert or2a["STORE_003"] == "317.7 min"
    assert or2a["STORE_091"] == "873.2 min"


def test_compare_cities(con):
    r = compare_entities(con, ["Bangalore", "Noida"], level="city")
    or2a = next(row for row in r.rows if row["metric"] == "Avg OR2A")
    assert or2a["Bangalore"] == "33.4 min"
    assert or2a["Noida"] == "115.8 min"


def test_compare_with_missing_entity_notes_it(con):
    """A missing entity must be reported, never silently dropped."""
    r = compare_entities(con, ["STORE_003", "TBC8"])
    assert "TBC8" in r.note
    assert r.rows


def test_compare_requires_two(con):
    with pytest.raises(ValueError):
        compare_entities(con, ["STORE_003"])


# ═══════════════════════════════════════════════════════════════════════
# hour_profile
# ═══════════════════════════════════════════════════════════════════════

def test_hour_profile_single_hour(con):
    r = hour_profile(con, hours=[19], city="Bangalore")
    assert len(r.rows) == 1
    assert r.rows[0]["hour"] == 19


def test_hour_profile_uses_weighted_average(con):
    """
    The whole reason this tool exists rather than LLM-written SQL. Both values
    are asserted so a regression to AVG() fails loudly.
    """
    weighted = hour_profile(con, hours=[19], city="Bangalore").rows[0]["w_avg_or2a"]
    naive = con.execute(
        "SELECT AVG(avg_or2a) FROM fact_store_hour WHERE city='Bangalore' AND hour=19"
    ).fetchone()[0]
    assert abs(weighted - 30.0107) < 1e-3
    assert abs(naive - 44.8864) < 1e-3
    assert abs(weighted - naive) > 10


def test_hour_profile_ordering_modes(con):
    chrono = hour_profile(con, city="Bangalore", order="hour")
    severity = hour_profile(con, city="Bangalore", order="severity")
    assert [r["hour"] for r in chrono.rows] == sorted(r["hour"] for r in chrono.rows)
    assert severity.rows[0]["w_avg_or2a"] >= severity.rows[1]["w_avg_or2a"]


def test_hour_profile_bad_order_rejected(con):
    with pytest.raises(ValueError):
        hour_profile(con, order="; DROP TABLE dim_store")


# ═══════════════════════════════════════════════════════════════════════
# count_matching
# ═══════════════════════════════════════════════════════════════════════

def test_sustained_pileup_count(con):
    """
    Delegates to rca.check_pileup rather than reimplementing "3+ consecutive"
    in SQL, so the tool and the RCA path can never disagree.
    """
    r = count_matching(con, "sustained_pileup", level="store")
    assert len(r.rows) == 40
    assert r.rows[0]["store"] == "STORE_210"
    assert r.rows[0]["longest_run"] == 16
    assert next(x for x in r.rows if x["store"] == "STORE_021")["longest_run"] == 8


@pytest.mark.parametrize("condition,expected_count", [
    ("problem_hour", 2202),
    ("no_supply", 139),
    ("pileup", 497),
    ("demand_spike", 929),
    ("utilization_gap", 1863),
    ("booking_gap", 2199),
])
def test_condition_counts(condition, expected_count, con):
    r = count_matching(con, condition, level="hour")
    assert f"{expected_count:,} of 3,813" in r.title


@pytest.mark.parametrize("condition", sorted(tools._CONDITIONS))
@pytest.mark.parametrize("level", ["store", "hour"])
def test_every_condition_at_every_level(condition, level, con):
    count_matching(con, condition, level=level)


# ═══════════════════════════════════════════════════════════════════════
# metric_lookup
# ═══════════════════════════════════════════════════════════════════════

def test_metric_lookup_formats(con):
    r = metric_lookup(con, "STORE_003", metrics=["w_breached_rate", "w_avg_or2a"])
    assert r.rows[0]["value"] == "76.7%"
    assert r.rows[1]["value"] == "317.7 min"


def test_metric_lookup_null_safe(con):
    """STORE_008 has two NULL avg_or2a hours; no NaN may reach the output."""
    r = metric_lookup(con, "STORE_008", metrics=["w_avg_or2a"])
    assert r.rows[0]["value"] == "79.9 min"
    assert "nan" not in str(r.rows).lower()


def test_metric_lookup_unknown_entity(con):
    r = metric_lookup(con, "TBC8")
    assert r.empty
    assert "TBC8" in r.note


# ═══════════════════════════════════════════════════════════════════════
# INJECTION BOUNDARY
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("call,exc", [
    (lambda c: rank_entities(c, metric="1; DROP TABLE dim_store"), UnknownMetricError),
    (lambda c: count_matching(c, "'; DELETE FROM fact_store_hour--"), UnknownConditionError),
    (lambda c: rank_entities(c, level="'; DROP--"), ValueError),
    (lambda c: rank_entities(c, direction="hax"), ValueError),
    (lambda c: hour_profile(c, order="; DROP"), ValueError),
    (lambda c: rank_entities(c, level="store", metric="store_count"), UnknownMetricError),
    (lambda c: metric_lookup(c, "STORE_003", metrics=["; DROP"]), UnknownMetricError),
])
def test_malicious_input_rejected(call, exc, con):
    """
    Metric and condition names come from an LLM. Column names cannot be
    parameterized in SQL, so the closed registry IS the security boundary.
    """
    with pytest.raises(exc):
        call(con)


def test_tables_survive_injection_attempts(con):
    assert con.execute("SELECT COUNT(*) FROM dim_store").fetchone()[0] == 233
    assert con.execute("SELECT COUNT(*) FROM fact_store_hour").fetchone()[0] == 3813


# ═══════════════════════════════════════════════════════════════════════
# COVERAGE + PROVENANCE
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("metric", sorted(METRICS))
def test_every_metric_ranks_at_its_declared_levels(metric, con):
    for level in METRICS[metric].levels:
        rank_entities(con, level=level, metric=metric, n=1)


def test_catalog_matches_implementations():
    """The LLM-facing catalog must not drift from the actual functions."""
    assert len(TOOL_CATALOG) == 5
    for name in TOOL_CATALOG:
        assert callable(getattr(tools, name))


def test_tier2_always_reports_validated_provenance(con):
    results = [
        rank_entities(con),
        compare_entities(con, ["STORE_003", "STORE_091"]),
        hour_profile(con, hours=[19]),
        count_matching(con, "no_supply"),
        metric_lookup(con, "STORE_003"),
    ]
    for r in results:
        assert r.provenance == config.PROVENANCE_VALIDATED
        assert r.provenance != config.PROVENANCE_EXPLORATORY


def test_results_render_as_markdown_tables(con):
    out = render_tool_result(rank_entities(con, level="city", n=3))
    assert out.startswith("**")
    assert "|---" in out
    assert out.count("\n|") >= 4


def test_empty_result_renders_gracefully(con):
    out = render_tool_result(metric_lookup(con, "TBC8"))
    assert "TBC8" in out
