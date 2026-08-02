"""
Rendering tests — the playbook's mandated output format.

The format is prescribed, so these tests assert on exact strings. The most
important group verifies the rules an LLM would quietly break: printing all
three checks even when the answer is NO, and never emitting a summary sentence
with an empty cause list.
"""

import pytest

import config
from app import rca, render
from app.rollups import city_day, store_day


@pytest.fixture(scope="module")
def ranked(con):
    return rca.rank_problem_hours(con, "STORE_003")


@pytest.fixture(scope="module")
def hour8(ranked):
    return ranked.top[0]


# ═══════════════════════════════════════════════════════════════════════
# MANDATED FORMAT
# ═══════════════════════════════════════════════════════════════════════

def test_hour_block_matches_mandated_format(hour8):
    b = render.render_hour_block(hour8)
    assert "### STORE_003 — Hour 8 — avg OR2A: 562.0 min (threshold: 0 min)" in b
    assert "1. Demand Spike: NO — 76 orders vs 103 projected (-26.2%)" in b
    assert "2. Pileup: YES — 4 orders carried from previous hour" in b
    assert "3. Supply:" in b
    # C3/C4: the verdict is printed, and the cancelled slots are named because
    # the stored ratio is (booked - cancelled)/current -- 32/59 is 54%, not 49%.
    assert "a. Booking: YES — 32 of 59 slots booked, 3 cancelled (49%)" in b
    assert "b. Utilization: NO — man_hour ratio 0.88 (1 no-shows)" in b
    assert "**Summary**:" in b


def test_sustained_escalation_rendered(hour8):
    b = render.render_hour_block(hour8)
    # B9: hour spans use an en-dash everywhere now, and only when the run
    # is contiguous — hours 7,8,9,10 are, so a range is honest here.
    assert "SUSTAINED — 4 consecutive hours: 7–10" in b


def test_all_three_checks_render_even_when_no(ranked):
    """
    Explicit playbook rule: "Always answer all three checks, even when the
    answer is NO." An LLM would drop the NOs for brevity.
    """
    h18 = next(c for c in ranked.problem_ranked if c.hour == 18)
    assert h18.triggered == ["booking gap"]

    b = render.render_hour_block(h18)
    for required in ["1. Demand Spike:", "2. Pileup:", "3. Supply:",
                     "a. Booking:", "b. Utilization:"]:
        assert required in b


def test_summary_names_originating_hour(hour8):
    """
    "Pileup inflates OR2A for the receiving hour, not the hour that caused the
     backlog." Hour 8 is the victim; the failure began in hour 7.
    """
    assert "from hour 7" in render.render_hour_block(hour8)


# ═══════════════════════════════════════════════════════════════════════
# Q6 — ZERO-FLAG TEMPLATE
# ═══════════════════════════════════════════════════════════════════════

def test_zero_flag_hour_does_not_emit_empty_cause_list(con):
    """
    217 problem hours (9.9%) trigger nothing. The playbook has no branch for
    this but still mandates a summary naming every factor -- a naive
    implementation emits "due to ." and looks broken.
    """
    hours = rca.fetch_store_hours(con, "STORE_001")
    c = rca.run_all_checks({h["hour"]: h for h in hours}[22], hours)
    b = render.render_hour_block(c)

    assert c.triggered == []
    assert "due to ." not in b
    assert "none of the three standard root causes" in b
    assert "Recommend manual review" in b


# ═══════════════════════════════════════════════════════════════════════
# NO SUPPLY
# ═══════════════════════════════════════════════════════════════════════

def test_no_supply_renders_cause_not_na(con):
    """
    man_hour is NULL here. Printing "n/a" would read as missing data rather
    than total supply failure.
    """
    r = rca.rank_problem_hours(con, "STORE_128")
    c = next(x for x in r.problem_ranked if x.hour == 8)
    b = render.render_hour_block(c)

    # C3: no cancellations here, so no cancelled clause -- the common case stays
    # as short as the playbook's mandated format.
    assert "a. Booking: YES — 0 of 22 slots booked (0%)" in b
    assert "Utilization: YES (NO SUPPLY)" in b
    assert "no riders booked, so no man_hour ratio" in b
    assert "man_hour ratio n/a" not in b
    assert "no rider supply at all" in b


# ═══════════════════════════════════════════════════════════════════════
# STORE DAY
# ═══════════════════════════════════════════════════════════════════════

def test_store_day_header_and_truncation(con, ranked):
    out = render.render_store_day(store_day(con, "STORE_003"), ranked)

    assert "991 orders · 760 breached (76.7%)" in out
    assert "weighted avg OR2A 317.7 min" in out
    assert "Top 3 problem hours, ranked by customers affected" in out
    assert out.count("### STORE_003 — Hour") == 3
    assert "13 further problem hours were analysed" in out

    # B5 -- the notice must name BOTH options distinctly. It previously said
    # only 'show me all hours', which then returned just the problem hours:
    # the wording promised the full day and delivered a subset.
    assert '"show me all problem hours"' in out


def test_store_day_discloses_undetailed_hours(con, ranked):
    """Truncation must name what was left out -- nothing is silently dropped."""
    out = render.render_store_day(store_day(con, "STORE_003"), ranked)
    for hour in [c.hour for c in ranked.problem_ranked[3:]]:
        assert str(hour) in out


def test_store_day_is_readable_length(con, ranked):
    """~40 lines vs ~168 if every problem hour were rendered."""
    out = render.render_store_day(store_day(con, "STORE_003"), ranked)
    assert len(out.splitlines()) < 60


def test_show_all_hours_renders_everything(con):
    r = rca.rank_problem_hours(con, "STORE_003", top_n=None)
    out = render.render_store_day(store_day(con, "STORE_003"), r)
    assert out.count("### STORE_003 — Hour") == 16
    assert "further problem hours" not in out
    assert "### All 16 problem hours" in out


# ═══════════════════════════════════════════════════════════════════════
# CITY DAY
# ═══════════════════════════════════════════════════════════════════════

def test_city_day_render(con):
    out = render.render_city_day(city_day(con, "Bangalore"))
    assert "88,340 orders across 97 stores" in out
    assert "weighted avg OR2A 33.4 min" in out
    assert "| STORE_091 | 873.2 min |" in out
    assert out.count("| STORE_") == 5
    assert "Ask about any of these stores" in out


# ═══════════════════════════════════════════════════════════════════════
# WALKTHROUGH
# ═══════════════════════════════════════════════════════════════════════

def test_walkthrough_is_hour_ordered_not_impact_ordered(con):
    r = rca.rank_problem_hours(con, "STORE_003", hours=config.HOUR_LABELS["morning"])
    out = render.render_hour_walkthrough(r)
    hours = [int(l.split("Hour ")[1].split(" ")[0])
             for l in out.splitlines() if "— Hour " in l]
    assert hours == sorted(hours)
    assert hours == [7, 8, 9, 10, 11]


# ═══════════════════════════════════════════════════════════════════════
# LLM SEAM
# ═══════════════════════════════════════════════════════════════════════

def test_llm_summary_replaces_only_the_summary(hour8):
    """
    The single seam where M3 splices in model output. Every computed number
    must survive untouched.
    """
    b = render.render_hour_block(hour8, summary="Custom LLM sentence.")
    assert "**Summary**: Custom LLM sentence." in b
    assert "1. Demand Spike: NO — 76 orders vs 103 projected (-26.2%)" in b
    # C3/C4: the verdict is printed, and the cancelled slots are named because
    # the stored ratio is (booked - cancelled)/current -- 32/59 is 54%, not 49%.
    assert "a. Booking: YES — 32 of 59 slots booked, 3 cancelled (49%)" in b


def test_deterministic_fallback_when_no_summary(hour8):
    """No API key, or a failed call, must still produce a complete block."""
    b = render.render_hour_block(hour8, summary=None)
    assert "OR2A was 562.0 min over threshold" in b
    assert "sustained pileup" in b
    assert "booking gap" in b


def test_partial_llm_failure_degrades_per_hour(con, ranked):
    """One hour narrated by the LLM, the rest falling back, must both render."""
    out = render.render_store_day(
        store_day(con, "STORE_003"), ranked, summaries={8: "LLM wrote this one."})
    assert "LLM wrote this one." in out
    assert "OR2A was 492.9 min over threshold" in out


# ═══════════════════════════════════════════════════════════════════════
# CONFIG-DRIVEN
# ═══════════════════════════════════════════════════════════════════════

def test_threshold_comes_from_config(hour8, monkeypatch):
    """
    The OR2A threshold is contested (the metric doc says both "<= 0" and
    "typically 5-8 min"), so it must be changeable in one place.
    """
    monkeypatch.setattr(config, "OR2A_THRESHOLD_MIN", 8)
    b = render.render_hour_block(hour8)
    assert "(threshold: 8 min)" in b
    assert "OR2A was 554.0 min over threshold" in b


# ═══════════════════════════════════════════════════════════════════════
# REGRESSION
# ═══════════════════════════════════════════════════════════════════════

def test_all_233_stores_render(con):
    stores = [r[0] for r in con.execute("SELECT store FROM dim_store").fetchall()]
    for s in stores:
        render.render_store_day(store_day(con, s), rca.rank_problem_hours(con, s))


# ═══════════════════════════════════════════════════════════════════════
# C3 / C4 — THE BOOKING LINE
#
# Two separate defects in one line, both found by reading output that had
# been pasted as "working":
#
#   C3  "14 of 16 slots booked (81%)"   -- 14/16 is 87.5%. The percentage was
#       correct (the stored ratio is (booked-cancelled)/current, a deliberate
#       Q3 decision) but the cancelled slot was invisible, so the line read as
#       an arithmetic error.
#   C4  3a was the only check printed with no YES/NO, while the Summary went on
#       to assert "due to a booking gap".
# ═══════════════════════════════════════════════════════════════════════

def test_booking_percentage_is_reconcilable_from_the_printed_numbers(con):
    """
    The regression that matters: a reader doing the arithmetic in their head
    must reach the printed percentage. Asserted over EVERY store-hour with a
    cancellation, not one example.
    """
    import re
    from app import rca
    checked = 0
    for store in ("STORE_046", "STORE_003", "STORE_080"):
        hours = rca.fetch_store_hours(con, store)
        by_hour = {h["hour"]: h for h in hours}
        for hour, row in by_hour.items():
            c = rca.run_all_checks(row, hours)
            if c.booking.pct is None or not c.is_problem_hour:
                continue
            block = render.render_hour_block(c)
            m = re.search(r"a\. Booking: (YES|NO) — (\d+) of (\d+) slots booked"
                          r"(?:, (\d+) cancelled)? \((\d+)%\)", block)
            assert m, f"booking line did not parse for {store} h{hour}:\n{block}"
            _, booked, current, cancelled, pct = m.groups()
            booked, current = int(booked), int(current)
            cancelled = int(cancelled or 0)
            # ±1 for display rounding. STORE_003 h12 is the exact-half case:
            # (26-3)/40 is 57.5%, which the stored column renders as 58% while
            # the float evaluates to 57.4999... A tolerance of one point is
            # what "a reader doing this in their head" actually needs; anything
            # tighter tests IEEE-754, not the thing that was broken.
            assert abs((booked - cancelled) / current * 100 - int(pct)) <= 1, (
                f"{store} h{hour}: ({booked}-{cancelled})/{current} does not "
                f"give {pct}%")
            checked += 1
    assert checked > 20, f"only {checked} hours exercised — too weak a check"


def test_cancelled_clause_is_omitted_when_there_are_none(con):
    """The mandated format stays intact in the common case."""
    from app import rca
    hours = rca.fetch_store_hours(con, "STORE_128")
    c = rca.run_all_checks({h["hour"]: h for h in hours}[8], hours)
    assert c.booking.cancelled == 0
    assert "cancelled" not in render.render_hour_block(c).split("**Summary**")[0]


def test_every_check_carries_a_verdict(con):
    """C4 — 3a used to be the one line with no YES/NO."""
    from app import rca
    hours = rca.fetch_store_hours(con, "STORE_046")
    c = rca.run_all_checks({h["hour"]: h for h in hours}[7], hours)
    block = render.render_hour_block(c)
    for line in ("1. Demand Spike:", "2. Pileup:",
                 "a. Booking:", "b. Utilization:"):
        printed = [l for l in block.splitlines() if line in l][0]
        assert "YES" in printed or "NO" in printed, f"no verdict on: {printed}"


def test_booking_verdict_agrees_with_the_summary(con):
    """
    The specific confusion: 'Booking: 14 of 16 (81%)' with no verdict, followed
    by a Summary asserting a booking gap. Verdict and prose must never diverge.
    """
    from app import rca
    for store in ("STORE_046", "STORE_003", "STORE_080"):
        hours = rca.fetch_store_hours(con, store)
        for row in hours:
            c = rca.run_all_checks(row, hours)
            if not c.is_problem_hour:
                continue
            block = render.render_hour_block(c)
            head, _, summary = block.partition("**Summary**")
            booking_line = [l for l in head.splitlines() if "a. Booking:" in l][0]
            says_yes = "Booking: YES" in booking_line
            assert says_yes == ("booking gap" in summary), (
                f"{store} h{row['hour']}: verdict/summary disagree\n{block}")


# ═══════════════════════════════════════════════════════════════════════
# B1 — OVER-BOOKED SLOTS
#
# "316 of 311 rider slots booked" reads as an arithmetic error. It is not:
# riders BOOK a slot and some later CANCEL, and current_size is the
# post-cancellation capacity. 24.4% of slots are over-booked this way.
#
# Nothing was wrong except the sentence, which showed two of the three numbers
# and let the reader conclude the agent could not count.
# ═══════════════════════════════════════════════════════════════════════

def test_over_booking_really_happens_in_this_data(con):
    """The premise. If it stops being true, the special-case can go."""
    over, total = con.execute(
        "SELECT COUNT(*) FILTER (WHERE booked_size > current_size), COUNT(*) "
        "FROM slot_supply").fetchone()
    assert over > 0, "no slot is over-booked — B1's special case is now dead code"
    assert over / total > 0.1


def test_an_over_booked_day_reconciles_on_screen(con, ranked):
    """
    A reader must be able to make the printed numbers add up. STORE_080 books
    316 against 311 offered, with 22 cancellations.
    """
    import re
    from app import rca

    day = store_day(con, "STORE_080")
    assert day.booked_size > day.current_size, "STORE_080 is no longer over-booked"

    out = render.render_store_day(
        day, rca.rank_problem_hours(con, "STORE_080"))
    line = [l for l in out.splitlines() if "rider slots" in l][0]

    m = re.search(r"([\d,]+) of ([\d,]+) rider slots filled.*?"
                  r"\(([\d,]+) booked, ([\d,]+) cancelled\)", line)
    assert m, f"over-booked day did not render the reconciling form:\n{line}"

    net, current, booked, cancelled = (int(g.replace(",", "")) for g in m.groups())
    assert net == booked - cancelled, "the printed numbers do not reconcile"
    assert current == day.current_size
    assert booked == day.booked_size


def test_a_normal_day_keeps_the_short_phrasing(con):
    """The extra clause appears only where it is needed."""
    from app import rca

    day = store_day(con, "STORE_003")
    out = render.render_store_day(day, rca.rank_problem_hours(con, "STORE_003"))
    line = [l for l in out.splitlines() if "rider slots" in l][0]
    if day.booked_size <= day.current_size:
        assert "booked," not in line or "cancelled)" not in line


# ═══════════════════════════════════════════════════════════════════════
# B9 — A DASH PROMISES EVERYTHING IN BETWEEN
#
# "hours 13–15" was printed for a store with no hour 14. A reader who acts on
# that looks for data that was never missing.
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("hours,expected", [
    ([6, 7, 8], "6–8"),
    ([14], "14"),
    ([15, 19], "15, 19"),
    ([7, 10, 13, 15, 21, 22], "7, 10, 13, 15, 21, 22"),
    ([21, 22, 0, 1], "0, 1, 21, 22"),        # the wrapping night window
    ([], ""),
])
def test_hour_span_only_collapses_a_contiguous_run(hours, expected):
    assert render._hour_span(hours) == expected


def test_a_gappy_store_lists_its_hours_instead_of_claiming_a_range(con):
    """
    STORE_054 records hours 7, 10, 13, 15, 21, 22. Rendering that as "hours
    7–22" claims sixteen hours that do not exist.
    """
    from app import rca

    r = rca.rank_problem_hours(con, "STORE_054")
    heading = render.render_hour_walkthrough(r).splitlines()[0]

    assert "7–22" not in heading, "still printed a range over missing hours"
    assert "7, 10, 13, 15, 21, 22" in heading


def test_a_contiguous_store_still_gets_a_range(con):
    """The dash is correct when it is true — this must not over-correct."""
    from app import rca

    r = rca.rank_problem_hours(con, "STORE_128")
    assert "hours 6–22" in render.render_hour_walkthrough(r).splitlines()[0]
