"""
Load-time data validation.

WHY THIS EXISTS
---------------
Every assertion below was verified true against the delivered CSV. So on today's
data this module does nothing -- it passes silently and adds ~50ms to startup.

That is the point. These are not defensive checks against data we distrust; they
are a CONTRACT. The RCA engine is built on specific structural guarantees:
that (store, hour) is unique, that breached_rate really equals
breached_count/total_orders, that supply columns are constant within a slot.

If the client ever swaps in a different day's extract and one of those
guarantees quietly stops holding, the agent would not crash. It would return
confidently wrong numbers -- which is far worse. These assertions convert a
silent correctness failure into a loud startup failure.

TWO TIERS
---------
STRUCTURAL checks must hold for ANY valid extract of this table. A violation
means the data is malformed or our understanding of it is wrong. These raise.

SHAPE checks are specific to the delivered file (3,813 rows, 233 stores, one
day). A different day would legitimately differ, so these only warn unless
config.STRICT_VALIDATION is on. They exist to catch "wrong file loaded".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb

import config


class ValidationError(Exception):
    """Raised when a structural invariant is violated."""


@dataclass
class CheckResult:
    name: str
    passed: bool
    violations: int
    detail: str
    tier: str = "structural"
    sample: list = field(default_factory=list)

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.detail}"


# ═══════════════════════════════════════════════════════════════════════
# STRUCTURAL INVARIANTS
#
# Each entry: (name, description, counting SQL, sampling SQL for the error msg)
# The counting SQL must return a single integer -- the number of violating rows.
# ═══════════════════════════════════════════════════════════════════════

_STRUCTURAL = [
    (
        "grain_unique",
        "(charge_date, store, hour) is unique — the declared grain",
        """SELECT COUNT(*) FROM (
               SELECT charge_date, store, hour FROM fact_store_hour
               GROUP BY 1,2,3 HAVING COUNT(*) > 1)""",
        """SELECT charge_date, store, hour, COUNT(*) n FROM fact_store_hour
           GROUP BY 1,2,3 HAVING COUNT(*) > 1 LIMIT 5""",
    ),
    (
        "breached_within_total",
        "breached_count <= total_orders",
        "SELECT COUNT(*) FROM fact_store_hour WHERE breached_count > total_orders",
        """SELECT store, hour, breached_count, total_orders FROM fact_store_hour
           WHERE breached_count > total_orders LIMIT 5""",
    ),
    (
        "breached_rate_consistent",
        "breached_rate == breached_count / total_orders",
        """SELECT COUNT(*) FROM fact_store_hour
           WHERE abs(breached_rate - CAST(breached_count AS DOUBLE)/total_orders) > 1e-9""",
        """SELECT store, hour, breached_rate,
                  CAST(breached_count AS DOUBLE)/total_orders AS computed
           FROM fact_store_hour
           WHERE abs(breached_rate - CAST(breached_count AS DOUBLE)/total_orders) > 1e-9 LIMIT 5""",
    ),
    (
        "order_outcomes_sum",
        "completed + cancelled + rto == total_orders",
        """SELECT COUNT(*) FROM fact_store_hour
           WHERE completed_order_count + cancelled_order_count + rto_order_count <> total_orders""",
        """SELECT store, hour, completed_order_count, cancelled_order_count,
                  rto_order_count, total_orders FROM fact_store_hour
           WHERE completed_order_count + cancelled_order_count + rto_order_count <> total_orders
           LIMIT 5""",
    ),
    (
        "pileup_flag_consistent",
        "pileup_flag = 1 if and only if pileup_count > 0",
        """SELECT COUNT(*) FROM fact_store_hour
           WHERE (pileup_flag = 1) <> (pileup_count > 0)""",
        """SELECT store, hour, pileup_flag, pileup_count FROM fact_store_hour
           WHERE (pileup_flag = 1) <> (pileup_count > 0) LIMIT 5""",
    ),
    (
        "flags_binary",
        "is_problem_hour and pileup_flag are in {0, 1}",
        """SELECT COUNT(*) FROM fact_store_hour
           WHERE is_problem_hour NOT IN (0,1) OR pileup_flag NOT IN (0,1)""",
        """SELECT store, hour, is_problem_hour, pileup_flag FROM fact_store_hour
           WHERE is_problem_hour NOT IN (0,1) OR pileup_flag NOT IN (0,1) LIMIT 5""",
    ),
    (
        "no_zero_denominators",
        "total_orders > 0 and order_projection > 0 — no division-by-zero anywhere",
        """SELECT COUNT(*) FROM fact_store_hour
           WHERE total_orders <= 0 OR order_projection <= 0 OR order_projection IS NULL""",
        """SELECT store, hour, total_orders, order_projection FROM fact_store_hour
           WHERE total_orders <= 0 OR order_projection <= 0 OR order_projection IS NULL LIMIT 5""",
    ),
    (
        "man_hour_is_ratio",
        "man_hour is in [0, 1] or NULL — it is a ratio, not an hour count",
        """SELECT COUNT(*) FROM fact_store_hour
           WHERE man_hour IS NOT NULL AND (man_hour < 0 OR man_hour > 1)""",
        """SELECT store, hour, man_hour, rider_hours_per_hour, booked_hours_per_hour
           FROM fact_store_hour
           WHERE man_hour IS NOT NULL AND (man_hour < 0 OR man_hour > 1) LIMIT 5""",
    ),
    (
        "store_maps_to_one_city",
        "each store belongs to exactly one city",
        """SELECT COUNT(*) FROM (
               SELECT store FROM fact_store_hour GROUP BY store
               HAVING COUNT(DISTINCT city) > 1)""",
        """SELECT store, COUNT(DISTINCT city) n FROM fact_store_hour
           GROUP BY store HAVING COUNT(DISTINCT city) > 1 LIMIT 5""",
    ),
    (
        # The most important assertion in this file. The whole reason
        # slot_supply exists (Task 1.4) is that these columns are slot-level
        # values replicated across the slot's hours. If a future extract ever
        # made them genuinely per-hour, the dedup in slot_supply would start
        # silently discarding real data.
        "supply_constant_within_slot",
        "rider-supply columns are constant within (store, slot_name)",
        """SELECT COUNT(*) FROM (
               SELECT store, slot_name FROM fact_store_hour
               GROUP BY store, slot_name
               HAVING COUNT(DISTINCT orginal_size)            > 1
                   OR COUNT(DISTINCT current_size)            > 1
                   OR COUNT(DISTINCT booked_size)             > 1
                   OR COUNT(DISTINCT booked_hours_per_hour)   > 1
                   OR COUNT(DISTINCT current_capacity_booked) > 1)""",
        """SELECT store, slot_name,
                  COUNT(DISTINCT booked_size)   AS distinct_booked,
                  COUNT(DISTINCT current_size)  AS distinct_capacity
           FROM fact_store_hour GROUP BY store, slot_name
           HAVING COUNT(DISTINCT booked_size) > 1 OR COUNT(DISTINCT current_size) > 1
           LIMIT 5""",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# SHAPE CHECKS  — specific to the delivered file, not to the table in general
# ═══════════════════════════════════════════════════════════════════════

def _shape_checks(con) -> list[CheckResult]:
    def n(sql: str) -> int:
        return con.execute(sql).fetchone()[0]

    expectations = [
        ("row_count",         "SELECT COUNT(*) FROM fact_store_hour",                          config.EXPECTED_ROW_COUNT),
        ("store_count",       "SELECT COUNT(DISTINCT store) FROM fact_store_hour",             config.EXPECTED_STORE_COUNT),
        ("city_count",        "SELECT COUNT(DISTINCT city) FROM fact_store_hour",              config.EXPECTED_CITY_COUNT),
        ("null_or2a_rows",    "SELECT COUNT(*) FROM fact_store_hour WHERE avg_or2a IS NULL",   config.EXPECTED_NULL_OR2A_ROWS),
        ("null_manhour_rows", "SELECT COUNT(*) FROM fact_store_hour WHERE man_hour IS NULL",   config.EXPECTED_NULL_MANHOUR_ROWS),
        ("problem_hours",     "SELECT COUNT(*) FROM fact_store_hour WHERE is_problem_hour=1",  config.EXPECTED_PROBLEM_HOURS),
    ]

    results = []
    for name, sql, expected in expectations:
        actual = n(sql)
        results.append(CheckResult(
            name=name,
            passed=(actual == expected),
            violations=abs(actual - expected),
            detail=f"expected {expected:,}, got {actual:,}",
            tier="shape",
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def validate(
    con: duckdb.DuckDBPyConnection,
    strict: bool | None = None,
    raise_on_fail: bool = True,
) -> list[CheckResult]:
    """
    Run every invariant against the loaded fact table.

    strict         -- include shape checks as hard failures.
                      Defaults to config.STRICT_VALIDATION.
    raise_on_fail  -- raise ValidationError on any failure. Set False to
                      collect results for reporting (used by the checkpoint).

    Returns all CheckResults, passed and failed.
    """
    if strict is None:
        strict = config.STRICT_VALIDATION

    results: list[CheckResult] = []

    for name, description, count_sql, sample_sql in _STRUCTURAL:
        violations = con.execute(count_sql).fetchone()[0]
        passed = violations == 0
        sample = [] if passed else con.execute(sample_sql).fetchall()
        results.append(CheckResult(
            name=name,
            passed=passed,
            violations=violations,
            detail=description if passed else f"{description} — {violations:,} violating row(s)",
            tier="structural",
            sample=sample,
        ))

    results.extend(_shape_checks(con))

    failures = [
        r for r in results
        if not r.passed and (r.tier == "structural" or strict)
    ]

    if failures and raise_on_fail:
        lines = ["Data validation failed:\n"]
        for f in failures:
            lines.append(f"  [{f.tier.upper()}] {f.name}: {f.detail}")
            for row in f.sample[:3]:
                lines.append(f"      offending: {row}")
        lines.append(
            "\nThis means the loaded data violates an assumption the RCA engine "
            "is built on. Fix the data or revisit the assumption — do not bypass "
            "this check, or the agent will return confidently wrong numbers."
        )
        raise ValidationError("\n".join(lines))

    return results
