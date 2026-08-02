"""
RCA engine — the four playbook checks.

THIS MODULE CONTAINS NO AI.

Every function here is a pure, deterministic transformation of numbers already
in the database. That is the central architectural decision of this project: the
LLM understands language and writes prose, but it never computes, compares, or
decides a flag. It receives finished facts.

The consequence is that the RCA verdicts cannot be wrong because the model had a
bad day. They can only be wrong if this file is wrong -- and this file is unit
tested against hand-computed values.

THE PLAYBOOK (quick_commerce_rca_logic.md)
------------------------------------------
Three checks, one of which has two independent sub-checks:

    Check 1  Demand    total_orders > order_projection x 1.10
    Check 2  Pileup    pileup_flag = 1;  3+ consecutive hours -> SUSTAINED
    Check 3a Booking   current_capacity_booked < 0.90
    Check 3b Utilization  man_hour < 0.85

CRITICAL RULE, quoted from the playbook:

    "All three checks run independently -- root causes compound, they are not
     mutually exclusive."

So this is never an if/elif chain. All four sub-checks always evaluate, on every
problem hour, and all four always appear in the output -- including when the
answer is NO. The data confirms why that matters: on problem hours, demand fires
28.5%, pileup 21.1%, booking 64.0% and utilization 54.3%. Those sum well past
100% because hours routinely have two or three compounding causes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb

import config


# ═══════════════════════════════════════════════════════════════════════
# CHECK RESULTS
# One dataclass per check. Fields map directly onto the placeholders in the
# playbook's mandated output format, so rendering (Task 2.4) is pure templating
# with no further computation.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DemandCheck:
    """Check 1 — did actual demand exceed the forecast by more than 10%?"""
    flag: bool
    actual: int
    projected: float
    pct_delta: float          # signed; +18.4 means 18.4% above forecast
    threshold_multiplier: float = config.DEMAND_SPIKE_MULTIPLIER

    @property
    def label(self) -> str:
        return "YES" if self.flag else "NO"


@dataclass
class PileupCheck:
    """Check 2 — did unassigned orders spill in from the previous hour?"""
    flag: bool
    count: int
    sustained: bool           # part of a run of 3+ consecutive pileup hours
    run_length: int           # length of the run containing this hour
    run_hours: list[int] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "YES" if self.flag else "NO"


@dataclass
class BookingCheck:
    """Check 3a — were the offered rider slots actually filled?"""
    flag: bool
    booked_size: int
    current_size: int
    ratio: float | None       # stored current_capacity_booked
    pct: float | None         # ratio as a percentage, for display
    cancelled: int = 0        # C3 -- carried ONLY so the render can explain the pct

    # C3 -- WHY THIS FIELD EXISTS.
    #
    # The stored ratio is (booked - cancelled) / current, not booked / current
    # (see check_booking_gap). Rendering only `booked of current (pct%)` shows
    # the reader two numbers that do not produce the third:
    #
    #     "14 of 16 slots booked (81%)"      14/16 = 87.5%, not 81%
    #
    # The percentage is right and the display was incomplete -- the cancelled
    # slot was invisible. Anyone checking the arithmetic concludes the agent
    # cannot divide, which is the worst possible impression for a number the
    # agent is actually being more careful about than the schema doc is.
    #
    # This field is display-only. No check reads it; the verdict still comes
    # from the stored ratio.

    @property
    def label(self) -> str:
        return "YES" if self.flag else "NO"


@dataclass
class UtilizationCheck:
    """Check 3b — were the booked rider hours actually worked on the ground?"""
    flag: bool
    man_hour: float | None
    noshows: int
    no_supply: bool           # booked_size == 0: nobody booked at all
    reason: str | None        # "NO_SUPPLY" | "UTILIZATION_GAP" | "UNKNOWN" | None

    @property
    def label(self) -> str:
        if self.no_supply:
            return "YES (NO SUPPLY)"
        return "YES" if self.flag else "NO"


@dataclass
class HourChecks:
    """All four checks for one store-hour, plus the context needed to render."""
    store: str
    city: str
    hour: int
    avg_or2a: float | None
    total_orders: int
    breached_count: int
    breached_rate: float
    is_problem_hour: bool

    demand: DemandCheck
    pileup: PileupCheck
    booking: BookingCheck
    utilization: UtilizationCheck

    @property
    def triggered(self) -> list[str]:
        """
        Human-readable names of every flag that fired, in playbook order.

        The playbook requires the summary line to "name every contributing
        factor in one sentence", so this list is the summary's raw material.
        """
        out = []
        if self.demand.flag:
            out.append("demand spike")
        if self.pileup.flag:
            out.append("sustained pileup" if self.pileup.sustained else "pileup")
        if self.booking.flag:
            out.append("booking gap")
        if self.utilization.no_supply:
            out.append("no rider supply")
        elif self.utilization.flag:
            out.append("utilization gap")
        return out

    @property
    def any_flag(self) -> bool:
        return bool(self.triggered)


# ═══════════════════════════════════════════════════════════════════════
# CHECK 1 — DEMAND SPIKE
# ═══════════════════════════════════════════════════════════════════════

def check_demand_spike(row: dict) -> DemandCheck:
    """
    "If total_orders > order_projection x 1.10 -> flag DEMAND SPIKE."

    The 1.10 multiplier exists because forecasts are never exact; a 3% miss is
    normal noise, 10%+ is a genuine staffing problem.

    Division is safe without a guard: validation asserts order_projection > 0
    on every row.
    """
    actual = int(row["total_orders"])
    projected = float(row["order_projection"])

    flag = actual > projected * config.DEMAND_SPIKE_MULTIPLIER
    pct_delta = (actual - projected) / projected * 100.0

    return DemandCheck(flag=flag, actual=actual, projected=projected,
                       pct_delta=pct_delta)


# ═══════════════════════════════════════════════════════════════════════
# CHECK 2 — PILEUP  (and SUSTAINED escalation)
# ═══════════════════════════════════════════════════════════════════════

def _pileup_run(hour: int, pileup_hours: set[int], all_hours: list[int]) -> list[int]:
    """
    The maximal run of consecutive pileup hours containing `hour`.

    Q8 DECISION -- what does "consecutive" mean?

    25 stores have non-contiguous hour coverage, and hours 2-5 are absent
    dataset-wide. So pileup at hours 10, 12 and 13 (with 11 missing entirely) is
    ambiguous: is that a run of 3, or runs of 1 and 2?

    config.STRICT_HOUR_ADJACENCY = True means hour values must differ by exactly
    1. That is the plain reading of "3+ consecutive hours", and the gaps are
    almost certainly genuine store closures rather than missing data -- so a
    backlog cannot physically carry across them.

    Setting it False falls back to positional adjacency (3 consecutive rows in
    the store's hour sequence), which tolerates gaps but can call 10:00 and
    12:00 "consecutive" -- physically wrong.
    """
    if hour not in pileup_hours:
        return []

    if config.STRICT_HOUR_ADJACENCY:
        run = [hour]
        h = hour - 1
        while h in pileup_hours:
            run.insert(0, h)
            h -= 1
        h = hour + 1
        while h in pileup_hours:
            run.append(h)
            h += 1
        return run

    # positional fallback
    ordered = sorted(all_hours)
    idx = ordered.index(hour)
    lo = idx
    while lo - 1 >= 0 and ordered[lo - 1] in pileup_hours:
        lo -= 1
    hi = idx
    while hi + 1 < len(ordered) and ordered[hi + 1] in pileup_hours:
        hi += 1
    return ordered[lo: hi + 1]


def check_pileup(row: dict, store_hours: list[dict]) -> PileupCheck:
    """
    "If pileup_flag = 1 -> flag PILEUP.
     Check if pileup is continuous for 3+ consecutive hours -> SUSTAINED PILEUP."

    Needs the store's full day, not just this row, to detect the run.

    DIRECTIONAL SEMANTICS (order_ready_to_assignment.md): "Pileup inflates OR2A
    for the receiving hour, not the hour that caused the backlog." So a pileup
    flag here means THIS hour is the victim -- the failure happened in the
    previous hour. Narration must point backwards, not blame this hour.
    """
    hour = int(row["hour"])
    count = int(row["pileup_count"])
    flag = int(row["pileup_flag"]) == 1

    pileup_hours = {int(r["hour"]) for r in store_hours if int(r["pileup_flag"]) == 1}
    all_hours = [int(r["hour"]) for r in store_hours]

    run = _pileup_run(hour, pileup_hours, all_hours)
    sustained = len(run) >= config.SUSTAINED_PILEUP_HOURS

    return PileupCheck(flag=flag, count=count, sustained=sustained,
                       run_length=len(run), run_hours=run)


# ═══════════════════════════════════════════════════════════════════════
# CHECK 3a — BOOKING GAP
# ═══════════════════════════════════════════════════════════════════════

def check_booking_gap(row: dict) -> BookingCheck:
    """
    "If current_capacity_booked < 0.90 -> flag BOOKING GAP."

    Q3 DECISION -- READ the stored column, never recompute it.

    The schema doc and playbook both state
        current_capacity_booked = booked_size / current_size
    That formula reproduces only 32.7% of rows. The formula that reproduces
    100.0% is
        current_capacity_booked = (booked_size - cancelled_count) / current_size

    Worked example, STORE_001 hour 19: booked=45, cancelled=2, current=49.
        documented:  45/49      = 0.918  -> PASSES the 0.90 gate
        stored:                   0.88   -> FAILS the 0.90 gate
        true formula: (45-2)/49 = 0.878  -> matches stored

    So the two disagree on the verdict itself. We read the stored column because
    it is what the ops team's own SQL returns -- guaranteeing our numbers always
    agree with theirs. Recomputing would make the agent quietly contradict the
    analysts it is meant to assist.
    """
    booked = int(row["booked_size"])
    current = int(row["current_size"])
    ratio = row["current_capacity_booked"]

    if not config.USE_STORED_CAPACITY_RATIO:  # documented-but-wrong path, for comparison
        ratio = booked / current if current else None

    ratio = float(ratio) if ratio is not None else None
    flag = ratio is not None and ratio < config.BOOKING_GAP_THRESHOLD

    return BookingCheck(flag=flag, booked_size=booked, current_size=current,
                        ratio=ratio, pct=(ratio * 100.0) if ratio is not None else None,
                        cancelled=int(row.get("cancelled_count") or 0))


# ═══════════════════════════════════════════════════════════════════════
# CHECK 3b — UTILIZATION GAP  (and the NO SUPPLY case)
# ═══════════════════════════════════════════════════════════════════════

def check_utilization(row: dict) -> UtilizationCheck:
    """
    "If man_hour < 0.85 -> flag UTILIZATION GAP.
     Also check noshow_count to confirm if riders booked but didn't show up."

    Q5 DECISION -- the zero-booking branch MUST come first.

    139 rows have booked_size = 0, which forces booked_hours = 0 and
    man_hour = NULL. These are the most severe supply failures in the dataset:
    capacity was offered and literally nobody booked it, while orders kept
    arriving.

    But NULL < 0.85 is FALSE in SQL, and NaN < 0.85 is False in pandas. So a
    naive implementation reports "Utilization: NO" -- no supply problem -- on
    exactly the 139 worst supply rows in the file. Silent, plausible, wrong.

    We emit a distinct NO_SUPPLY reason rather than folding it into
    UTILIZATION_GAP, because "riders booked but were underused" and "no riders
    at all" are different failures needing different fixes.
    """
    booked = int(row["booked_size"])
    noshows = int(row["noshow_count"])
    man_hour = row["man_hour"]

    # ── the trap branch: check BEFORE any numeric comparison ──
    if booked == 0 and config.ZERO_BOOKING_AS_NO_SUPPLY:
        return UtilizationCheck(flag=True, man_hour=None, noshows=noshows,
                                no_supply=True, reason="NO_SUPPLY")

    if man_hour is None:
        # Defensive: today man_hour is NULL only when booked_size = 0, so this
        # is unreachable. If a future extract breaks that link, report unknown
        # rather than silently passing the check.
        return UtilizationCheck(flag=False, man_hour=None, noshows=noshows,
                                no_supply=False, reason="UNKNOWN")

    man_hour = float(man_hour)
    flag = man_hour < config.UTILIZATION_GAP_THRESHOLD

    return UtilizationCheck(flag=flag, man_hour=man_hour, noshows=noshows,
                            no_supply=False,
                            reason="UTILIZATION_GAP" if flag else None)


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════

def run_all_checks(row: dict, store_hours: list[dict]) -> HourChecks:
    """
    Run all four checks on one store-hour.

    Deliberately NOT an if/elif cascade. The playbook states root causes
    compound, so every check always runs and every result is always reported --
    including the NOs.
    """
    return HourChecks(
        store=row["store"],
        city=row["city"],
        hour=int(row["hour"]),
        avg_or2a=None if row["avg_or2a"] is None else float(row["avg_or2a"]),
        total_orders=int(row["total_orders"]),
        breached_count=int(row["breached_count"]),
        breached_rate=float(row["breached_rate"]),
        is_problem_hour=int(row["is_problem_hour"]) == 1,
        demand=check_demand_spike(row),
        pileup=check_pileup(row, store_hours),
        booking=check_booking_gap(row),
        utilization=check_utilization(row),
    )


# ═══════════════════════════════════════════════════════════════════════
# DATA ACCESS
# ═══════════════════════════════════════════════════════════════════════

_STORE_HOURS_SQL = """
SELECT store, city, hour, total_orders, breached_count, breached_rate,
       is_problem_hour, pileup_count, pileup_flag, avg_or2a, order_projection,
       current_size, booked_size, cancelled_count, current_capacity_booked,
       man_hour, noshow_count, no_supply_flag, slot_name, slot_type
FROM fact_store_hour
WHERE store = ? AND charge_date = ?
ORDER BY hour
"""


def fetch_store_hours(
    con: duckdb.DuckDBPyConnection,
    store: str,
    date: str | None = None,
) -> list[dict]:
    """
    Every hour of one store-day, as dicts.

    Reads fact_store_hour (hour grain) -- correct here because we look at each
    hour individually. Day-level SUPPLY totals must instead come from
    slot_supply, since these columns are slot values replicated across hours.
    """
    date = date or config.DATA_DATE
    cur = con.execute(_STORE_HOURS_SQL, [store, date])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ═══════════════════════════════════════════════════════════════════════
# IMPACT RANKING
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RankedHours:
    """
    Problem hours for a store-day, ranked by impact.

    Two orderings are returned because two different questions need them:

      window          every hour in scope, in HOUR order -- narrative order,
                      used by "walk me through the morning hours"
      problem_ranked  problem hours only, in IMPACT order -- used by
                      "why did STORE_003 underperform"

    `top` is the first N of problem_ranked; `remainder_count` is how many
    problem hours were analysed but not detailed.
    """
    store: str
    city: str
    date: str
    scope_hours: list[int]                     # hours considered
    window: list[HourChecks] = field(default_factory=list)
    problem_ranked: list[HourChecks] = field(default_factory=list)
    top: list[HourChecks] = field(default_factory=list)
    remainder_count: int = 0
    impact_metric: str = config.IMPACT_METRIC
    hour_filter: list[int] | None = None       # set when scoped to a window

    # Every hour the store actually has, regardless of the window filter. Lets
    # an empty result explain itself: "no data for hours 12-12; this store has
    # hours 7, 10, 13, 15, 21, 22".
    available_hours: list[int] = field(default_factory=list)

    @property
    def problem_hour_count(self) -> int:
        return len(self.problem_ranked)

    @property
    def hours_in_scope(self) -> int:
        return len(self.window)

    @property
    def truncated(self) -> bool:
        return self.remainder_count > 0


def _impact_key(c: HourChecks):
    """
    Sort key for impact ranking. Descending impact, then descending severity,
    then ascending hour.

    Q4 DECISION -- why breached_count is the impact metric.

    is_problem_hour = 1 on 57.7% of all store-hours, because it means "at least
    one order breached". STORE_003 has 16 problem hours out of 16. Emitting the
    full mandated markdown block for each is ~150 lines -- unreadable in chat
    and poor in a live demo.

    So we rank rather than filter: EVERY problem hour stays in the analysis, and
    only the presentation is truncated (with the remainder count disclosed and
    expandable). Nothing is hidden.

    Candidate metrics considered:
      breached_rate  -- a 4-order hour with 2 breaches (50%) would outrank an
                        80-order hour with 20 breaches (25%). Ranks trivia above
                        real damage.
      avg_or2a       -- dominated by the 2,488-minute outliers, which may be
                        data artifacts we cannot verify without the raw dump.
      breached_count -- CHOSEN. A raw column meaning "how many customers were
                        affected". No derivation, no invented constant, and it
                        maps to a sentence an ops manager already thinks in.

    The tie-break chain makes ordering fully deterministic, so a live demo
    produces identical output on every run.
    """
    metric = config.IMPACT_METRIC
    primary = getattr(c, metric, c.breached_count)
    # None avg_or2a sorts last rather than raising
    severity = c.avg_or2a if c.avg_or2a is not None else float("-inf")
    return (-primary, -severity, c.hour)


def rank_problem_hours(
    con: duckdb.DuckDBPyConnection,
    store: str,
    date: str | None = None,
    top_n: int | None = config.TOP_N_PROBLEM_HOURS,
    hours: list[int] | None = None,
) -> RankedHours:
    """
    Run all four checks on every hour of a store-day and rank the problem hours.

    top_n = None returns everything -- this backs the "show me all hours"
            follow-up intent.
    hours = [6,7,8,...] scopes to a window (e.g. "morning"), which filters the
            candidate set but does not change the ranking logic.
    """
    date = date or config.DATA_DATE
    all_rows = fetch_store_hours(con, store, date)
    if not all_rows:
        return RankedHours(store=store, city="", date=date, scope_hours=[],
                           hour_filter=hours)

    # The pileup run detection must see the WHOLE day even when we are only
    # reporting a window -- a run can start before the window opens.
    scoped = [r for r in all_rows if hours is None or int(r["hour"]) in hours]

    window = [run_all_checks(r, all_rows) for r in scoped]
    window.sort(key=lambda c: c.hour)

    problem = [c for c in window if c.is_problem_hour]
    problem.sort(key=_impact_key)

    top = problem if top_n is None else problem[:top_n]
    remainder = len(problem) - len(top)

    return RankedHours(
        store=store,
        city=all_rows[0]["city"],
        date=date,
        scope_hours=[int(r["hour"]) for r in scoped],
        window=window,
        problem_ranked=problem,
        top=top,
        remainder_count=remainder,
        hour_filter=hours,
        available_hours=sorted(int(r["hour"]) for r in all_rows),
    )
