"""
Derived tables — built once at startup from fact_store_hour.

Two of these are convenience. Two prevent wrong answers.

  dim_store      convenience — store -> city lookup for entity resolution
  slot_supply    CORRECTNESS — deduplicates slot-level columns to slot grain
  agg_store_day  CORRECTNESS — weighted rollup formula written exactly once
  agg_city_day   CORRECTNESS — same, at city level

WHY slot_supply EXISTS
----------------------
The fact table is one row per store-hour, but the rider-supply columns
(orginal_size, current_size, booked_size, completed_count, noshow_count,
cancelled_count, booked_hours_per_hour, rider_hours_per_hour) are SLOT-level
values replicated onto every hour the slot covers.

  STORE_001 hour 18  Night Slot 1 (18:00-20:00)  booked_size = 45
  STORE_001 hour 19  Night Slot 1 (18:00-20:00)  booked_size = 45
                                                 ^^ one slot, 45 riders, two rows

So SUM(booked_size) across hours double-counts every multi-hour slot. Measured
on STORE_003: naive sum = 426, true value = 249, a 71% overstatement.

  RULE: day-level supply questions read slot_supply.
        hour-level questions read fact_store_hour (duplication is harmless
        when you are only looking at one hour).

validation.supply_constant_within_slot asserts the replication assumption
holds, so this dedup can never silently discard real data.

WHY agg_* EXIST
---------------
The playbook mandates volume-weighted rollups:

    weighted breached_rate = SUM(breached_count) / SUM(total_orders)
    weighted avg_or2a      = SUM(avg_or2a * total_orders) / SUM(total_orders)

with a NULL subtlety that is easy to get wrong: 2 rows have avg_or2a IS NULL
(store-hours that were 100% cancelled, so no order ever reached assignment).
In SQL, SUM() skips NULLs in the numerator but the denominator would still
count those rows' total_orders -- silently deflating the result by ~4.5%.
The FILTER clause must be applied to BOTH sides.

That formula is needed by the city handler, store handler, ranking tool,
comparison tool and hour handler. Writing it in five places is five chances to
get it wrong. Written here once, tested once, read as a plain column everywhere
else.

Naive unweighted AVG(avg_or2a) for Bangalore returns 43.14 min. The correct
weighted value is 33.36 min -- a 29% error, and exactly the mistake both
reference docs warn against.
"""

from __future__ import annotations

import duckdb


# ═══════════════════════════════════════════════════════════════════════
# dim_store — store -> city lookup (233 rows)
# ═══════════════════════════════════════════════════════════════════════

_DIM_STORE = """
CREATE OR REPLACE TABLE dim_store AS
SELECT
    store,
    any_value(city)      AS city,        -- safe: validated 1 city per store
    any_value(city_norm) AS city_norm,
    COUNT(*)             AS hours_present,
    MIN(hour)            AS first_hour,
    MAX(hour)            AS last_hour
FROM fact_store_hour
GROUP BY store
"""


# ═══════════════════════════════════════════════════════════════════════
# slot_supply — one row per (store, slot_name)
# ═══════════════════════════════════════════════════════════════════════

# any_value() is correct here rather than MIN/MAX/SUM precisely because
# validation guarantees these columns do not vary within the group. Using
# any_value() makes that assumption explicit: if it ever stopped holding, the
# validation assertion fires before this table is built.
_SLOT_SUPPLY = """
CREATE OR REPLACE TABLE slot_supply AS
SELECT
    charge_date,
    store,
    any_value(city)                    AS city,
    slot_name,
    any_value(slot_start)              AS slot_start,
    any_value(slot_end)                AS slot_end,
    any_value(slot_type)               AS slot_type,

    -- rider capacity funnel: central plan -> store adjustment -> booked -> showed
    any_value(orginal_size)            AS orginal_size,
    any_value(current_size)            AS current_size,
    any_value(booked_size)             AS booked_size,
    any_value(completed_count)         AS completed_count,
    any_value(incompleted_count)       AS incompleted_count,
    any_value(noshow_count)            AS noshow_count,
    any_value(cancelled_count)         AS cancelled_count,

    -- stored ratios. NOT recomputed: the documented formula
    -- (booked_size / current_size) reproduces only 32.7% of rows; the true
    -- formula is (booked_size - cancelled_count) / current_size. Reading the
    -- stored column guarantees we always agree with the ops team's own SQL.
    any_value(orginal_capacity_booked) AS orginal_capacity_booked,
    any_value(current_capacity_booked) AS current_capacity_booked,

    any_value(booked_hours_per_hour)   AS booked_minutes,   -- honest name
    any_value(rider_hours_per_hour)    AS rider_minutes,
    any_value(booked_hours)            AS booked_hours,
    any_value(rider_hours)             AS rider_hours,
    any_value(man_hour)                AS man_hour,
    any_value(no_supply_flag)          AS no_supply_flag,

    -- how many hours of the day this one slot covers
    COUNT(*)                           AS hours_covered,
    MIN(hour)                          AS first_hour,
    MAX(hour)                          AS last_hour
FROM fact_store_hour
GROUP BY charge_date, store, slot_name
"""


# ═══════════════════════════════════════════════════════════════════════
# agg_store_day — one row per store (233 rows)
# ═══════════════════════════════════════════════════════════════════════

_AGG_STORE_DAY = """
CREATE OR REPLACE TABLE agg_store_day AS
WITH order_side AS (
    SELECT
        charge_date,
        store,
        any_value(city)      AS city,
        any_value(city_norm) AS city_norm,

        SUM(total_orders)          AS total_orders,
        SUM(breached_count)        AS breached_count,
        SUM(completed_order_count) AS completed_orders,
        SUM(cancelled_order_count) AS cancelled_orders,
        SUM(rto_order_count)       AS rto_orders,
        SUM(order_projection)      AS projected_orders,

        -- weighted breached_rate = SUM(breached) / SUM(orders)
        CAST(SUM(breached_count) AS DOUBLE) / NULLIF(SUM(total_orders), 0)
            AS w_breached_rate,

        -- weighted avg_or2a. FILTER on BOTH numerator and denominator:
        -- rows with NULL avg_or2a must be excluded from the weight base too,
        -- or the result is silently deflated.
        SUM(avg_or2a * total_orders) FILTER (WHERE avg_or2a IS NOT NULL)
            / NULLIF(SUM(total_orders) FILTER (WHERE avg_or2a IS NOT NULL), 0)
            AS w_avg_or2a,

        COUNT(*)                                     AS hours_present,
        SUM(is_problem_hour)                         AS problem_hours,
        SUM(pileup_flag)                             AS pileup_hours,
        SUM(pileup_count)                            AS pileup_orders,
        COUNT(*) FILTER (WHERE avg_or2a IS NULL)     AS null_or2a_hours,
        COUNT(*) FILTER (WHERE no_supply_flag = 1)   AS no_supply_hours,
        MAX(avg_or2a)                                AS worst_hour_or2a
    FROM fact_store_hour
    GROUP BY charge_date, store
),
supply_side AS (
    -- read from slot_supply, NOT fact_store_hour: summing supply columns
    -- across hours double-counts every multi-hour slot
    SELECT
        charge_date,
        store,
        COUNT(*)                 AS slot_count,
        SUM(orginal_size)        AS orginal_size,
        SUM(current_size)        AS current_size,
        SUM(booked_size)         AS booked_size,
        SUM(completed_count)     AS riders_completed,
        SUM(noshow_count)        AS noshow_count,
        SUM(cancelled_count)     AS rider_cancelled,

        -- COALESCE to 0, not left NULL.
        --
        -- rider_hours is NULL exactly when booked_hours = 0, which is exactly
        -- when booked_size = 0 (validated: the three always move together). So
        -- for a store like STORE_128, where NO slot was ever booked, SUM()
        -- returns NULL for the whole day.
        --
        -- That NULL does not mean "unknown" -- it means "no rider ever booked,
        -- therefore zero hours were worked". 0.0 is the semantically correct
        -- value and it keeps the no-supply case visible in reports instead of
        -- blanking it out. no_supply_hours already records the cause.
        COALESCE(SUM(booked_hours), 0) AS booked_hours,
        COALESCE(SUM(rider_hours),  0) AS rider_hours
    FROM slot_supply
    GROUP BY charge_date, store
)
SELECT o.*,
       s.slot_count, s.orginal_size, s.current_size, s.booked_size,
       s.riders_completed, s.noshow_count, s.rider_cancelled,
       s.booked_hours, s.rider_hours
FROM order_side o
JOIN supply_side s USING (charge_date, store)
"""


# ═══════════════════════════════════════════════════════════════════════
# agg_city_day — one row per city (9 rows)
# ═══════════════════════════════════════════════════════════════════════

# Aggregated from fact_store_hour, not from agg_store_day: averaging
# store-level averages would be a second unweighted-average bug.
_AGG_CITY_DAY = """
CREATE OR REPLACE TABLE agg_city_day AS
WITH order_side AS (
    SELECT
        charge_date,
        city,
        any_value(city_norm)       AS city_norm,
        COUNT(DISTINCT store)      AS store_count,
        SUM(total_orders)          AS total_orders,
        SUM(breached_count)        AS breached_count,
        SUM(order_projection)      AS projected_orders,

        CAST(SUM(breached_count) AS DOUBLE) / NULLIF(SUM(total_orders), 0)
            AS w_breached_rate,

        SUM(avg_or2a * total_orders) FILTER (WHERE avg_or2a IS NOT NULL)
            / NULLIF(SUM(total_orders) FILTER (WHERE avg_or2a IS NOT NULL), 0)
            AS w_avg_or2a,

        COUNT(*)                                   AS store_hours,
        SUM(is_problem_hour)                       AS problem_hours,
        SUM(pileup_flag)                           AS pileup_hours,
        COUNT(*) FILTER (WHERE no_supply_flag = 1) AS no_supply_hours
    FROM fact_store_hour
    GROUP BY charge_date, city
),
supply_side AS (
    SELECT charge_date, city,
           SUM(current_size)    AS current_size,
           SUM(booked_size)     AS booked_size,
           SUM(noshow_count)    AS noshow_count,
           -- same reasoning as agg_store_day: NULL means no riders booked,
           -- which is genuinely zero hours worked
           COALESCE(SUM(booked_hours), 0) AS booked_hours,
           COALESCE(SUM(rider_hours),  0) AS rider_hours
    FROM slot_supply
    GROUP BY charge_date, city
)
SELECT o.*, s.current_size, s.booked_size, s.noshow_count,
       s.booked_hours, s.rider_hours
FROM order_side o
JOIN supply_side s USING (charge_date, city)
"""


# ═══════════════════════════════════════════════════════════════════════

# Order matters: agg_store_day and agg_city_day both read slot_supply.
# ═══════════════════════════════════════════════════════════════════════
# SAFE VIEWS — the only tables the generated-SQL path may touch
# ═══════════════════════════════════════════════════════════════════════
#
# DON'T WARN ABOUT THE TRAP. REMOVE IT.
#
# A prompt that says "remember to weight OR2A by order count" is advice: the
# model can follow it or not, and nothing detects which happened. These views
# make the wrong query unwritable instead.
#
# TRAP 1 — WEIGHTED ROLLUPS.
#   AVG(w_avg_or2a) across stores is wrong by -40% to +35% at city level and
#   REORDERS the cities: Chennai is 3rd worst correctly and 6th worst naively.
#   Every number in that answer is real and the conclusion is fabricated.
#
#   `or2a_order_minutes` is the pre-multiplied numerator, so ANY grouping the
#   model invents is written the obvious way:
#
#       SUM(or2a_order_minutes) / SUM(total_orders)
#
#   The correct form becomes the natural form. The screen then rejects AVG() on
#   any OR2A column outright, so the wrong form cannot execute even if written.
#
# TRAP 2 — SLOT REPLICATION.
#   Rider supply columns are slot-level values copied onto every hour of the
#   slot. Summing them across hours inflates by 1.75x network-wide -- 128,824
#   rider slots against a true 73,465. So they are ABSENT from the hourly view.
#   Supply questions go to `safe_slot_supply`, which is already deduped. A
#   column that is not in the schema cannot be double-counted.

_SAFE_STORE_HOUR = """
CREATE OR REPLACE VIEW safe_store_hour AS
SELECT store, city, charge_date, hour,
       total_orders, breached_count, cancelled_order_count,
       avg_or2a, order_projection, pileup_count,
       -- pre-multiplied numerator: SUM(this)/SUM(total_orders) is the ONLY
       -- correct way to roll OR2A up over any set of rows
       avg_or2a * total_orders AS or2a_order_minutes
FROM fact_store_hour
"""

_SAFE_SLOT_SUPPLY = """
CREATE OR REPLACE VIEW safe_slot_supply AS
SELECT store, city, charge_date, slot_name, slot_start, slot_end,
       current_size, booked_size, cancelled_count,
       current_capacity_booked, man_hour, noshow_count, completed_count,
       booked_hours, rider_hours, hours_covered
FROM slot_supply
"""

_SAFE_STORE_DAY = """
CREATE OR REPLACE VIEW safe_store_day AS
SELECT store, city, charge_date, total_orders, breached_count,
       w_breached_rate, w_avg_or2a, problem_hours, pileup_hours,
       no_supply_hours, noshow_count, pileup_orders, cancelled_orders,
       w_avg_or2a * total_orders AS or2a_order_minutes
FROM agg_store_day
"""

_SAFE_CITY_DAY = """
CREATE OR REPLACE VIEW safe_city_day AS
SELECT city, charge_date, store_count, total_orders, breached_count,
       w_breached_rate, w_avg_or2a, problem_hours,
       w_avg_or2a * total_orders AS or2a_order_minutes
FROM agg_city_day
"""

# The allowlist the screen enforces. Anything not named here is rejected before
# execution -- including the raw tables these views are built from.
SAFE_VIEWS = ("safe_store_hour", "safe_slot_supply",
              "safe_store_day", "safe_city_day")


_BUILD_ORDER = [
    ("dim_store",     _DIM_STORE),
    ("slot_supply",   _SLOT_SUPPLY),
    ("agg_store_day", _AGG_STORE_DAY),
    ("agg_city_day",  _AGG_CITY_DAY),
]

_SAFE_VIEW_ORDER = [
    ("safe_store_hour",  _SAFE_STORE_HOUR),
    ("safe_slot_supply", _SAFE_SLOT_SUPPLY),
    ("safe_store_day",   _SAFE_STORE_DAY),
    ("safe_city_day",    _SAFE_CITY_DAY),
]


def build_all(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Build every derived table. Returns table name -> row count."""
    counts = {}
    for name, sql in _BUILD_ORDER:
        con.execute(sql)
        counts[name] = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    for name, sql in _SAFE_VIEW_ORDER:
        con.execute(sql)
        counts[name] = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    return counts
