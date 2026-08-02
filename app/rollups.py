"""
Store-day and city-day rollups.

This module is a thin, typed API over the agg_* tables built in tables.py. The
weighted arithmetic lives in the SQL there -- written once, tested once. Nothing
here recomputes a weighted average, by design: five call sites recomputing the
same fiddly formula is five chances to get it wrong.

WHY WEIGHTING MATTERS (both reference docs warn about this)
-----------------------------------------------------------
    "Do not average OR2A across stores without weighting by order volume."

Concretely, for Bangalore:

    naive   AVG(avg_or2a)  = 43.1354 min     <- what a text-to-SQL agent writes
    correct weighted       = 33.3561 min
                             29.3% error

The naive form lets a 10-order store distort a 90,000-order city. The weighted
form answers the question an ops manager actually has: what did the TYPICAL
CUSTOMER experience?

    weighted breached_rate = SUM(breached_count) / SUM(total_orders)
    weighted avg_or2a      = SUM(avg_or2a * total_orders) / SUM(total_orders)

with the NULL subtlety handled in tables.py: rows where avg_or2a IS NULL must be
excluded from BOTH numerator and denominator, or the result is silently ~4.5%
low.

SUPPLY NUMBERS
--------------
Day-level supply figures (booked_size, current_size, noshow_count, hours) come
from slot_supply, which is deduplicated to slot grain. Summing them straight
from fact_store_hour double-counts every multi-hour slot -- measured at 1.71x on
STORE_003. tables.py already routes this correctly; this module just reads the
result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb

import config


class NoDataError(Exception):
    """Raised when an entity exists in the schema but has no rows for the date."""


# ═══════════════════════════════════════════════════════════════════════
# RESULT TYPES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StoreSummary:
    """One line in a city's worst-stores list."""
    store: str
    city: str
    total_orders: int
    breached_count: int
    w_breached_rate: float
    w_avg_or2a: float | None
    problem_hours: int
    hours_present: int


@dataclass
class StoreDay:
    """Everything known about one store on one day."""
    store: str
    city: str
    date: str

    # order side (weighted)
    total_orders: int
    breached_count: int
    w_breached_rate: float
    w_avg_or2a: float | None
    projected_orders: float
    completed_orders: int
    cancelled_orders: int
    rto_orders: int

    # hour structure
    hours_present: int
    problem_hours: int
    pileup_hours: int
    pileup_orders: int
    no_supply_hours: int
    null_or2a_hours: int
    worst_hour_or2a: float | None

    # supply side — from slot_supply (deduped), NOT summed from fact table
    slot_count: int
    current_size: int
    booked_size: int
    # B1 -- riders BOOK a slot and some later CANCEL, so booked_size can exceed
    # current_size (24.4% of slots do). current_size is the post-cancellation
    # capacity, which is why (booked - cancelled) tracks it. Carried so the
    # render can show the net rather than printing "316 of 311".
    rider_cancelled: int
    riders_completed: int
    noshow_count: int
    booked_hours: float
    rider_hours: float

    provenance: str = config.PROVENANCE_VALIDATED

    @property
    def demand_vs_forecast_pct(self) -> float:
        return (self.total_orders - self.projected_orders) / self.projected_orders * 100.0

    @property
    def healthy_hours(self) -> int:
        return self.hours_present - self.problem_hours


@dataclass
class CityDay:
    """Everything known about one city on one day, plus its worst stores."""
    city: str
    date: str

    store_count: int
    total_orders: int
    breached_count: int
    w_breached_rate: float
    w_avg_or2a: float | None
    projected_orders: float

    store_hours: int
    problem_hours: int
    pileup_hours: int
    no_supply_hours: int

    current_size: int
    booked_size: int
    noshow_count: int
    booked_hours: float
    rider_hours: float

    top_stores: list[StoreSummary] = field(default_factory=list)
    rank_metric: str = config.CITY_RANK_METRIC
    provenance: str = config.PROVENANCE_VALIDATED

    @property
    def demand_vs_forecast_pct(self) -> float:
        return (self.total_orders - self.projected_orders) / self.projected_orders * 100.0


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _one(con: duckdb.DuckDBPyConnection, sql: str, params: list) -> dict | None:
    cur = con.execute(sql, params)
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip([d[0] for d in cur.description], row))


def _f(v) -> float | None:
    return None if v is None else float(v)


# ═══════════════════════════════════════════════════════════════════════
# STORE DAY
# ═══════════════════════════════════════════════════════════════════════

def store_day(
    con: duckdb.DuckDBPyConnection,
    store: str,
    date: str | None = None,
) -> StoreDay:
    """
    Weighted daily rollup for one store.

    `store` must already be resolved to an exact code (see entities.resolve_store).
    Raises NoDataError rather than returning empty, so a caller can never
    silently render zeros as if they were real measurements.
    """
    date = date or config.DATA_DATE
    r = _one(con, "SELECT * FROM agg_store_day WHERE store = ? AND charge_date = ?",
             [store, date])
    if r is None:
        raise NoDataError(f"No data for store {store!r} on {date}.")

    return StoreDay(
        store=r["store"], city=r["city"], date=str(r["charge_date"]),
        total_orders=int(r["total_orders"]),
        breached_count=int(r["breached_count"]),
        w_breached_rate=float(r["w_breached_rate"]),
        w_avg_or2a=_f(r["w_avg_or2a"]),
        projected_orders=float(r["projected_orders"]),
        completed_orders=int(r["completed_orders"]),
        cancelled_orders=int(r["cancelled_orders"]),
        rto_orders=int(r["rto_orders"]),
        hours_present=int(r["hours_present"]),
        problem_hours=int(r["problem_hours"]),
        pileup_hours=int(r["pileup_hours"]),
        pileup_orders=int(r["pileup_orders"]),
        no_supply_hours=int(r["no_supply_hours"]),
        null_or2a_hours=int(r["null_or2a_hours"]),
        worst_hour_or2a=_f(r["worst_hour_or2a"]),
        slot_count=int(r["slot_count"]),
        current_size=int(r["current_size"]),
        booked_size=int(r["booked_size"]),
        rider_cancelled=int(r["rider_cancelled"] or 0),
        riders_completed=int(r["riders_completed"]),
        noshow_count=int(r["noshow_count"]),
        booked_hours=float(r["booked_hours"]),
        rider_hours=float(r["rider_hours"]),
    )


# ═══════════════════════════════════════════════════════════════════════
# CITY DAY
# ═══════════════════════════════════════════════════════════════════════

_TOP_STORES_SQL = """
SELECT store, city, total_orders, breached_count,
       w_breached_rate, w_avg_or2a, problem_hours, hours_present
FROM agg_store_day
WHERE city = ? AND charge_date = ?
ORDER BY {metric} DESC NULLS LAST, total_orders DESC, store ASC
LIMIT ?
"""


def city_day(
    con: duckdb.DuckDBPyConnection,
    city: str,
    date: str | None = None,
    top_n: int | None = None,
    rank_metric: str | None = None,
) -> CityDay:
    """
    Weighted daily rollup for one city, plus its worst-performing stores.

    `city` must be the exact stored value (see entities.resolve_city) -- note
    'Mumbai  north' has a double space.

    Bangalore has 97 stores, so dumping them all is unusable in chat. We return
    the worst N (config.CITY_TOP_N_STORES) and the caller invites a drill-down.

    Ordering is by w_avg_or2a rather than breached_count: across stores,
    breached_count mostly ranks by store size, whereas "which store performed
    worst" is a severity question. breached_count is included on every row so
    volume is still visible. See config.CITY_RANK_METRIC.
    """
    date = date or config.DATA_DATE
    top_n = top_n if top_n is not None else config.CITY_TOP_N_STORES
    metric = rank_metric or config.CITY_RANK_METRIC

    if metric not in {"w_avg_or2a", "w_breached_rate", "breached_count", "total_orders"}:
        raise ValueError(f"Unsupported rank metric {metric!r}")

    r = _one(con, "SELECT * FROM agg_city_day WHERE city = ? AND charge_date = ?",
             [city, date])
    if r is None:
        raise NoDataError(f"No data for city {city!r} on {date}.")

    # metric is validated against a whitelist above, so this format is safe.
    cur = con.execute(_TOP_STORES_SQL.format(metric=metric), [city, date, top_n])
    cols = [d[0] for d in cur.description]
    top = [
        StoreSummary(
            store=d["store"], city=d["city"],
            total_orders=int(d["total_orders"]),
            breached_count=int(d["breached_count"]),
            w_breached_rate=float(d["w_breached_rate"]),
            w_avg_or2a=_f(d["w_avg_or2a"]),
            problem_hours=int(d["problem_hours"]),
            hours_present=int(d["hours_present"]),
        )
        for d in (dict(zip(cols, row)) for row in cur.fetchall())
    ]

    return CityDay(
        city=r["city"], date=str(r["charge_date"]),
        store_count=int(r["store_count"]),
        total_orders=int(r["total_orders"]),
        breached_count=int(r["breached_count"]),
        w_breached_rate=float(r["w_breached_rate"]),
        w_avg_or2a=_f(r["w_avg_or2a"]),
        projected_orders=float(r["projected_orders"]),
        store_hours=int(r["store_hours"]),
        problem_hours=int(r["problem_hours"]),
        pileup_hours=int(r["pileup_hours"]),
        no_supply_hours=int(r["no_supply_hours"]),
        current_size=int(r["current_size"]),
        booked_size=int(r["booked_size"]),
        noshow_count=int(r["noshow_count"]),
        booked_hours=float(r["booked_hours"]),
        rider_hours=float(r["rider_hours"]),
        top_stores=top,
        rank_metric=metric,
    )


def all_cities(con: duckdb.DuckDBPyConnection, date: str | None = None) -> list[CityDay]:
    """Every city, worst first. Backs "which city performed worst" in Task 2.6."""
    date = date or config.DATA_DATE
    cities = [r[0] for r in con.execute(
        "SELECT city FROM agg_city_day WHERE charge_date = ? ORDER BY w_avg_or2a DESC",
        [date]).fetchall()]
    return [city_day(con, c, date) for c in cities]
