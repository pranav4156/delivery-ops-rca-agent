"""
Markdown rendering — the playbook's mandated output format.

WHY RENDERING IS DETERMINISTIC CODE, NOT THE LLM
------------------------------------------------
The instinct is "it's text, let the LLM write it". But look at the mandated
format from quick_commerce_rca_logic.md:

    ### [Store Name] - Hour [X] - avg OR2A: [Y] min (threshold: [Z] min)
    1. Demand Spike: [YES/NO] - [total_orders] orders vs [order_projection] ...
    2. Pileup: [YES/NO] - [pileup_count] orders carried ...
    3. Supply:
       a. Booking: [booked_size] of [current_size] slots booked ([X]%)
          (we additionally print the YES/NO verdict and, when non-zero, the
           cancelled slot count -- see the C3/C4 note in render_hour_block)
       b. Utilization: man_hour ratio [X] ([noshow_count] no-shows)
    **Summary**: ...

Every bracket is a value Python already computed. This is a TEMPLATE, not
writing. Rendering it here guarantees the structure is identical every time.

It also enforces a rule an LLM will quietly break for brevity:

    "Always answer all three checks, even when the answer is NO."

Exactly ONE thing is left to the model: the **Summary** sentence, which must
name every contributing factor in fluent English. That is genuine language
generation. Everything else is templating.

And even that has a deterministic fallback (build_summary below), so a failed or
slow LLM call degrades to a plainer sentence rather than a broken block. The
demo never shows a hole.
"""

from __future__ import annotations

import config
from app.rca import HourChecks, RankedHours
from app.rollups import CityDay, StoreDay


# ═══════════════════════════════════════════════════════════════════════
# NUMBER FORMATTING
# ═══════════════════════════════════════════════════════════════════════

def _min(v: float | None, dp: int = 1) -> str:
    return "n/a" if v is None else f"{v:,.{dp}f}"


def _pct(v: float | None, dp: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{dp}f}%"


def _signed_pct(v: float, dp: int = 1) -> str:
    return f"{v:+.{dp}f}%"


def _supply_phrase(booked: int, current: int, cancelled: int,
                   suffix: str = "") -> str:
    """
    B1 — "316 of 311 rider slots booked" reads as an error. It is not.

    Riders BOOK a slot and some later CANCEL, and `current_size` is the
    post-cancellation capacity — so `booked` legitimately exceeds `current` on
    24.4% of slots, and `booked - cancelled` tracks `current` closely. Nothing
    was wrong except the sentence, which showed two of the three numbers and let
    the reader conclude the agent could not count.

    So when booked exceeds current, the NET is what leads: it is the figure that
    reconciles, and the gross is kept beside it because a large cancellation
    count is itself an operational signal.
    """
    if cancelled and booked > current:
        net = booked - cancelled
        return (f"{net:,} of {current:,} rider slots filled{suffix} "
                f"({booked:,} booked, {cancelled:,} cancelled)")
    if cancelled:
        return (f"{booked:,} of {current:,} rider slots booked{suffix}, "
                f"{cancelled:,} cancelled")
    return f"{booked:,} of {current:,} rider slots booked{suffix}"


def _hour_span(hours) -> str:
    """
    B9 — "hours 13–15" was printed for a store with no hour 14.

    A dash promises everything in between. When the run is not contiguous the
    hours are listed instead, because a reader who acts on "13–15" will look for
    an hour that does not exist and conclude data is missing.

    Same rule as tools._hour_span; both exist because the two modules render
    hours independently and neither should have to import the other's internals.
    """
    hours = sorted(set(int(h) for h in hours))
    if not hours:
        return ""
    if len(hours) == 1:
        return str(hours[0])
    if hours[-1] - hours[0] + 1 == len(hours):
        return f"{hours[0]}–{hours[-1]}"
    return ", ".join(str(h) for h in hours)


def _join_phrases(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c' — reads as English, not as a list dump."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY SENTENCE  (deterministic fallback)
# ═══════════════════════════════════════════════════════════════════════

def _cause_phrases(c: HourChecks) -> list[str]:
    """
    Triggered flags as natural-language noun phrases.

    The pileup phrase names the ORIGINATING hour, because
    order_ready_to_assignment.md states:

        "Pileup inflates OR2A for the receiving hour, not the hour that caused
         the backlog."

    So this hour is the victim; the failure happened in the previous hour. A
    naive RCA blames the wrong hour. Naming the source makes the causal
    direction explicit.
    """
    out: list[str] = []

    if c.demand.flag:
        out.append(
            f"a demand spike ({c.demand.actual} orders vs "
            f"{c.demand.projected:,.0f} projected, {_signed_pct(c.demand.pct_delta)})"
        )

    if c.pileup.flag:
        origin = c.hour - 1
        if c.pileup.sustained:
            out.append(
                f"sustained pileup across {c.pileup.run_length} consecutive hours "
                f"(hours {_hour_span(c.pileup.run_hours)}), with "
                f"{c.pileup.count} orders carried in from hour {origin}"
            )
        else:
            out.append(
                f"{c.pileup.count} orders carried in unassigned from hour {origin}"
            )

    if c.booking.flag:
        # C3 -- same reason as the Booking line above: without the cancelled
        # count the two numbers do not produce the percentage.
        cancel_note = (f", {c.booking.cancelled} cancelled"
                       if c.booking.cancelled else "")
        out.append(
            f"a booking gap (only {c.booking.booked_size} of "
            f"{c.booking.current_size} rider slots filled{cancel_note}, "
            f"{_pct(c.booking.pct, 0)})"
        )

    if c.utilization.no_supply:
        out.append(
            f"no rider supply at all — {c.booking.current_size} slots were offered "
            f"and none were booked"
        )
    elif c.utilization.flag:
        noshow = (f", {c.utilization.noshows} no-shows"
                  if c.utilization.noshows else "")
        out.append(
            f"a utilization gap (man_hour {c.utilization.man_hour:.2f} vs "
            f"{config.UTILIZATION_GAP_THRESHOLD} threshold{noshow})"
        )

    return out


def build_summary(c: HourChecks) -> str:
    """
    Deterministic summary sentence.

    Used as the fallback when the LLM is unavailable, and as the baseline the
    LLM's version is spliced over in M3.

    Q6 DECISION -- the zero-flag branch.

    217 problem hours (9.9%) breach SLA while failing all four checks. The
    playbook has no branch for this, yet mandates that the summary "name every
    contributing factor in one sentence". With zero factors, a naive
    implementation emits a sentence ending in "due to ." -- which reads as a
    bug. This template states the finding honestly instead.
    """
    over = 0.0 if c.avg_or2a is None else c.avg_or2a - config.OR2A_THRESHOLD_MIN
    causes = _cause_phrases(c)

    if not causes:
        return (
            f"OR2A was {_min(over)} min over threshold, but none of the three "
            f"standard root causes were present — no demand spike, no pileup, "
            f"and rider supply was within thresholds. This points to a "
            f"dispatch-side or unmodelled factor. Recommend manual review."
        )

    return f"OR2A was {_min(over)} min over threshold due to {_join_phrases(causes)}."


# ═══════════════════════════════════════════════════════════════════════
# HOUR BLOCK  — the mandated format
# ═══════════════════════════════════════════════════════════════════════

def render_hour_block(c: HourChecks, summary: str | None = None) -> str:
    """
    One hour, in the playbook's mandated markdown.

    `summary` overrides the deterministic sentence -- this is the single seam
    where M3's LLM output is spliced in. If it is None or the LLM call failed,
    build_summary() is used and the block is still complete and correct.

    ALL THREE checks always render, including the NOs. That is an explicit
    playbook rule and the most likely thing an LLM would drop for brevity.
    """
    d, p, b, u = c.demand, c.pileup, c.booking, c.utilization

    # 2. Pileup — append SUSTAINED escalation when the run is long enough
    pileup_line = f"{p.count} orders carried from previous hour"
    if p.flag and p.sustained:
        pileup_line += (
            f" (SUSTAINED — {p.run_length} consecutive hours: "
            f"{_hour_span(p.run_hours)})"
        )
    elif not p.flag:
        pileup_line = "0 orders carried from previous hour"

    # 3b. Utilization — man_hour is NULL in the no-supply case, so state the
    #     cause rather than printing "n/a" and looking like missing data
    if u.no_supply:
        util_line = (
            f"no riders booked, so no man_hour ratio "
            f"({u.noshows} no-shows)"
        )
    elif u.man_hour is None:
        util_line = f"man_hour unavailable ({u.noshows} no-shows)"
    else:
        util_line = f"man_hour ratio {u.man_hour:.2f} ({u.noshows} no-shows)"

    body = summary if summary is not None else build_summary(c)

    # C3 + C4 -- THE BOOKING LINE.
    #
    # C4: 3a was the only check printed without a YES/NO. Checks 1, 2 and 3b
    # all carry a verdict, and the Summary below asserts "due to a booking
    # gap" -- so the one line the reader needs in order to see WHICH check
    # fired was the one line that did not say. BookingCheck.label already
    # existed; it was simply never rendered.
    #
    # C3: the cancelled slots are named only when there are any, so the common
    # case stays as short as the playbook's mandated format.
    cancel_note = (f", {b.cancelled} cancelled" if b.cancelled else "")
    booking_line = (f"   a. Booking: {b.label} — {b.booked_size} of "
                    f"{b.current_size} slots booked{cancel_note} "
                    f"({_pct(b.pct, 0)})\n")

    return (
        f"### {c.store} — Hour {c.hour} — avg OR2A: {_min(c.avg_or2a)} min "
        f"(threshold: {config.OR2A_THRESHOLD_MIN} min)\n"
        f"\n"
        f"1. Demand Spike: {d.label} — {d.actual} orders vs "
        f"{d.projected:,.0f} projected ({_signed_pct(d.pct_delta)})\n"
        f"2. Pileup: {p.label} — {pileup_line}\n"
        f"3. Supply:\n"
        f"{booking_line}"
        f"   b. Utilization: {u.label} — {util_line}\n"
        f"\n"
        f"**Summary**: {body}"
    )


# ═══════════════════════════════════════════════════════════════════════
# STORE DAY
# ═══════════════════════════════════════════════════════════════════════

def render_store_day(
    day: StoreDay,
    ranked: RankedHours,
    summaries: dict[int, str] | None = None,
) -> str:
    """
    Full store-day RCA: header KPIs, then the top-N hour blocks by impact,
    then an honest disclosure of what was not detailed.

    `summaries` maps hour -> LLM-written summary. Missing hours fall back to
    the deterministic sentence, so a partial LLM failure degrades gracefully.
    """
    summaries = summaries or {}
    parts: list[str] = []

    parts.append(f"## {day.store} — {day.city} — {day.date}\n")

    parts.append(
        f"**{day.total_orders:,} orders · {day.breached_count:,} breached "
        f"({_pct(day.w_breached_rate * 100)}) · weighted avg OR2A "
        f"{_min(day.w_avg_or2a)} min**\n"
    )

    detail = [
        f"{day.problem_hours} of {day.hours_present} hours flagged as problem hours",
        f"{day.pileup_hours} hours with pileup ({day.pileup_orders} orders carried)",
        _supply_phrase(day.booked_size, day.current_size,
                       getattr(day, "rider_cancelled", 0),
                       f" across {day.slot_count} slots"),
        f"demand ran {_signed_pct(day.demand_vs_forecast_pct)} vs forecast",
    ]
    if day.no_supply_hours:
        detail.append(f"**{day.no_supply_hours} hours with zero riders booked**")
    parts.append(" · ".join(detail) + "\n")

    if not ranked.problem_ranked:
        parts.append("\nNo problem hours were flagged for this store on this date.")
        return "\n".join(parts)

    scope = ""
    if ranked.hour_filter:
        scope = f" in hours {_hour_span(ranked.hour_filter)}"

    heading = (
        f"### All {len(ranked.top)} problem hours{scope}"
        if not ranked.truncated
        else f"### Top {len(ranked.top)} problem hours{scope}, "
             f"ranked by customers affected"
    )
    parts.append(f"\n{heading}\n")

    for c in ranked.top:
        parts.append(render_hour_block(c, summaries.get(c.hour)) + "\n")

    if ranked.truncated:
        # B5 -- name the two options precisely.
        #
        # This used to say only 'Ask "show me all hours" to expand', which
        # then returned just the PROBLEM hours. The wording promised the full
        # day and delivered a subset, so the omission was invisible.
        undetailed = ", ".join(str(c.hour)
                               for c in ranked.problem_ranked[len(ranked.top):])
        healthy = day.hours_present - day.problem_hours
        parts.append(
            f"_{ranked.remainder_count} further problem hours were analysed but "
            f"not detailed here, all with lower impact (hours {undetailed}). "
            f'Ask **"show me all problem hours"** for those'
            + (f', or **"show me all hours"** to include the '
               f'{healthy} hours with no breaches._' if healthy else "._")
        )

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# HOUR WALKTHROUGH  (narrative order, not impact order)
# ═══════════════════════════════════════════════════════════════════════

def render_hour_walkthrough(
    ranked: RankedHours,
    summaries: dict[int, str] | None = None,
) -> str:
    """
    "Walk me through the morning hours."

    Uses `window` (hour order) rather than `problem_ranked` (impact order),
    because a walkthrough is a narrative -- the user wants the day in sequence,
    including the hours that were fine.
    """
    summaries = summaries or {}
    if not ranked.window:
        # Say WHICH hours exist, not just that the window is empty.
        #
        # STORE_054 has hours [7, 10, 13, 15, 21, 22] -- no hour 12 -- so
        # "noon hours" is legitimately empty. But "No data in that time window"
        # leaves the user guessing whether the store is missing, the window is
        # wrong, or the agent failed. Listing the real hours turns a dead end
        # into an obvious next question.
        available = ", ".join(str(h) for h in ranked.available_hours) or "none"
        window = (f"hours {_hour_span(ranked.hour_filter)}"
                  if ranked.hour_filter else "that window")
        return (
            f"**{ranked.store}** has no data for {window} on {ranked.date}.\n\n"
            f"Hours recorded for this store: {available}."
        )

    # B9 -- the heading names the hours ACTUALLY in scope. min-max printed
    # "hours 13–15" for a store with no hour 14, sending the reader looking for
    # data that was never missing.
    parts = [
        f"## {ranked.store} — {ranked.city} — "
        f"hours {_hour_span(ranked.scope_hours)} — {ranked.date}\n",
        f"{len(ranked.window)} hours in scope, "
        f"{len(ranked.problem_ranked)} flagged as problem hours.\n",
    ]

    for c in ranked.window:
        if c.is_problem_hour:
            parts.append(render_hour_block(c, summaries.get(c.hour)) + "\n")
        else:
            parts.append(
                f"### {c.store} — Hour {c.hour} — avg OR2A: {_min(c.avg_or2a)} min "
                f"(threshold: {config.OR2A_THRESHOLD_MIN} min)\n\n"
                f"No breaches this hour — {c.total_orders} orders, all assigned "
                f"within SLA.\n"
            )

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# CITY DAY
# ═══════════════════════════════════════════════════════════════════════

def render_city_day(day: CityDay) -> str:
    """
    City summary: weighted KPIs, worst stores, drill-down invitation.

    Bangalore has 97 stores, so listing them all is unusable. The top N by
    severity plus a prompt sets up the natural next question -- which is also
    the multi-turn drill-down the brief asks us to demonstrate.
    """
    parts = [f"## {day.city} — {day.date}\n"]

    parts.append(
        f"**{day.total_orders:,} orders across {day.store_count} stores · "
        f"{day.breached_count:,} breached ({_pct(day.w_breached_rate * 100)}) · "
        f"weighted avg OR2A {_min(day.w_avg_or2a)} min**\n"
    )

    detail = [
        f"{day.problem_hours:,} problem hours of {day.store_hours:,} store-hours",
        f"{day.pileup_hours:,} hours with pileup",
        _supply_phrase(day.booked_size, day.current_size,
                       getattr(day, "rider_cancelled", 0)),
        f"demand ran {_signed_pct(day.demand_vs_forecast_pct)} vs forecast",
    ]
    if day.no_supply_hours:
        detail.append(f"**{day.no_supply_hours} hours with zero riders booked**")
    parts.append(" · ".join(detail) + "\n")

    if day.top_stores:
        label = {
            "w_avg_or2a": "weighted avg OR2A",
            "w_breached_rate": "breach rate",
            "breached_count": "orders breached",
            "total_orders": "order volume",
        }[day.rank_metric]

        parts.append(f"\n### Worst {len(day.top_stores)} stores by {label}\n")
        parts.append("| Store | Avg OR2A | Breach rate | Orders | Breached | Problem hours |")
        parts.append("|---|---:|---:|---:|---:|---:|")
        for s in day.top_stores:
            parts.append(
                f"| {s.store} | {_min(s.w_avg_or2a)} min | "
                f"{_pct(s.w_breached_rate * 100)} | {s.total_orders:,} | "
                f"{s.breached_count:,} | {s.problem_hours}/{s.hours_present} |"
            )
        parts.append(
            f"\nAsk about any of these stores for a full root-cause breakdown."
        )

    return "\n".join(parts)
