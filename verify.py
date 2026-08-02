#!/usr/bin/env python3
"""
Manual verification script — see what the engine actually produces.

Run:  python verify.py            full walkthrough
      python verify.py traps      just the data traps
      python verify.py demo       just the demo answers
      python verify.py tools      just the Tier-2 tools

Nothing here uses an LLM. Every number printed comes from deterministic Python.
"""

import sys
import time

sys.path.insert(0, ".")

import config
from app import db, rca, render, tables, tools
from app.entities import EntityIndex, resolve_city, resolve_hours, resolve_store
from app.rollups import city_day, store_day
from app.validation import validate

W = 78


def h1(t):
    print("\n" + "=" * W)
    print(f"  {t}")
    print("=" * W)


def h2(t):
    print(f"\n{'-' * W}\n  {t}\n{'-' * W}")


def boot():
    t0 = time.time()
    con = db.load()
    results = validate(con, raise_on_fail=False)
    tables.build_all(con)
    ms = (time.time() - t0) * 1000
    return con, results, ms


# ═══════════════════════════════════════════════════════════════════════

def section_load(con, results, ms):
    h1("1. LOAD + VALIDATION")
    print(f"\n  Loaded and validated in {ms:.0f} ms\n")
    q = lambda s: con.execute(s).fetchone()[0]
    print(f"  rows              {q('SELECT COUNT(*) FROM fact_store_hour'):,}")
    print(f"  stores            {q('SELECT COUNT(DISTINCT store) FROM fact_store_hour')}")
    print(f"  cities            {q('SELECT COUNT(DISTINCT city) FROM fact_store_hour')}")
    print(f"  date              {config.DATA_DATE}")
    print(f"\n  NULL avg_or2a     {q('SELECT COUNT(*) FROM fact_store_hour WHERE avg_or2a IS NULL')}"
          f"      (preserved, not imputed)")
    print(f"  NULL man_hour     {q('SELECT COUNT(*) FROM fact_store_hour WHERE man_hour IS NULL')}"
          f"    (preserved, not imputed)")
    print(f"  MAX avg_or2a      {q('SELECT MAX(avg_or2a) FROM fact_store_hour'):,.1f}"
          f"  (not capped)")

    ok = sum(r.passed for r in results)
    print(f"\n  Invariants: {ok}/{len(results)} passed")
    for r in results:
        if not r.passed:
            print(f"    FAIL  {r.name}: {r.detail}")
    print("\n  Derived tables:")
    for t in ["dim_store", "slot_supply", "agg_store_day", "agg_city_day"]:
        print(f"    {t:<16} {q(f'SELECT COUNT(*) FROM {t}'):>5} rows")


def section_traps(con):
    h1("2. THE FIVE DATA TRAPS  (what a naive implementation gets wrong)")

    h2("TRAP 1 — weighted vs naive average")
    naive = con.execute(
        "SELECT AVG(avg_or2a) FROM fact_store_hour WHERE city='Bangalore'").fetchone()[0]
    ours = city_day(con, "Bangalore").w_avg_or2a
    print(f"  naive   AVG(avg_or2a)   {naive:8.4f} min   <- what LLM-written SQL produces")
    print(f"  ours    weighted        {ours:8.4f} min")
    print(f"  error                   {abs(naive-ours)/ours*100:.1f}%")

    h2("TRAP 2 — dirty city names")
    idx = EntityIndex(con)
    n = con.execute("SELECT COUNT(*) FROM fact_store_hour WHERE city='Mumbai'").fetchone()[0]
    print(f"  WHERE city = 'Mumbai'          -> {n} rows   <- stored value has TWO spaces")
    print(f"  resolve_city('mumbai')         -> {resolve_city('mumbai', idx).value!r}")
    print(f"  resolve_city('bengaluru')      -> {resolve_city('bengaluru', idx).value!r}")
    print(f"  resolve_city('Bangalor')       -> {resolve_city('Bangalor', idx).value!r}  (fuzzy)")

    h2("TRAP 3 — slot values replicated across hours")
    naive_b = con.execute(
        "SELECT SUM(booked_size) FROM fact_store_hour WHERE store='STORE_003'").fetchone()[0]
    ded = con.execute(
        "SELECT SUM(booked_size) FROM slot_supply WHERE store='STORE_003'").fetchone()[0]
    print(f"  SUM(booked_size) from fact     {naive_b}   <- double-counts 2-hour slots")
    print(f"  from slot_supply (deduped)     {ded}")
    print(f"  overstatement                  {naive_b/ded*100-100:.0f}%")

    h2("TRAP 4 — NULL in the weighted denominator")
    ours8 = store_day(con, "STORE_008").w_avg_or2a
    unsafe = con.execute("""SELECT SUM(avg_or2a*total_orders)/SUM(total_orders)
                            FROM fact_store_hour WHERE store='STORE_008'""").fetchone()[0]
    print(f"  FILTER on numerator only       {unsafe:.4f}   <- silently low, no error")
    print(f"  FILTER on both sides (ours)    {ours8:.4f}")

    h2("TRAP 5 — NULL comparison hides the worst rows")
    row = {h["hour"]: h for h in rca.fetch_store_hours(con, "STORE_128")}[8]
    naive_u = row["man_hour"] is not None and row["man_hour"] < 0.85
    u = rca.check_utilization(row)
    print(f"  STORE_128 h8: booked_size={row['booked_size']}, man_hour={row['man_hour']}")
    print(f"  naive `man_hour < 0.85`        {naive_u}    <- 'no supply problem' on a")
    print(f"                                            zero-rider hour (139 such rows)")
    print(f"  our check                      flag={u.flag}, reason={u.reason}")

    h2("BONUS — the TBC8 safety net")
    print("  The brief's sample questions name TBC8/TBF1/TBC9. They do not exist.")
    for code in ["TBC8", "TBF1", "TBC9"]:
        print(f"  resolve_store({code!r})           -> {resolve_store(code, idx).value}"
              f"    (declines; does NOT guess)")
    print(f"  resolve_store('8')               -> {resolve_store('8', idx).value}"
          f"   (bare number still works)")
    print("\n  A digit-extracting resolver would map TBC8 -> STORE_008 and deliver a")
    print("  confident, correct-looking RCA about a store nobody asked about.")


def section_demo(con):
    h1("3. THE DEMO ANSWERS")

    h2("Q1  'How did Bangalore do on 2026-04-22?'")
    print(render.render_city_day(city_day(con, "Bangalore")))

    h2("Q2  'Why did STORE_003 underperform that day?'")
    out = render.render_store_day(
        store_day(con, "STORE_003"), rca.rank_problem_hours(con, "STORE_003"))
    print(out)
    print(f"\n  [{len(out.splitlines())} lines — vs 168 if all 16 problem hours were rendered]")

    h2("Q3  'Walk me through the morning hours at STORE_003.'")
    print(render.render_hour_walkthrough(
        rca.rank_problem_hours(con, "STORE_003", hours=config.HOUR_LABELS["morning"])))

    h2("Q4/Q5  Other demo stores")
    for s in ["STORE_091", "STORE_210"]:
        d = store_day(con, s)
        r = rca.rank_problem_hours(con, s)
        print(f"  {s} ({d.city}): {d.total_orders:,} orders, "
              f"weighted OR2A {d.w_avg_or2a:,.1f} min, "
              f"{d.problem_hours}/{d.hours_present} problem hours")
        print(f"      top hour {r.top[0].hour}: {', '.join(r.top[0].triggered)}")

    h2("EDGE — a problem hour where NO root cause applies (9.9% of them)")
    hrs = rca.fetch_store_hours(con, "STORE_001")
    print(render.render_hour_block(rca.run_all_checks({h["hour"]: h for h in hrs}[22], hrs)))

    h2("EDGE — zero riders booked")
    r = rca.rank_problem_hours(con, "STORE_128")
    print(render.render_hour_block(next(c for c in r.problem_ranked if c.hour == 8)))


def section_tools(con):
    h1("4. TIER-2 PARAMETERIZED TOOLS  (ad-hoc questions)")
    R = tools.render_tool_result
    cases = [
        ("Which city had the most no-shows?",
         lambda: tools.rank_entities(con, level="city", metric="noshow_count", n=5)),
        ("Worst 5 stores in Mumbai",
         lambda: tools.rank_entities(con, level="store", scope="Mumbai  north", n=5)),
        ("Compare STORE_003 and STORE_091",
         lambda: tools.compare_entities(con, ["STORE_003", "STORE_091"])),
        ("What happened at 7pm across Bangalore?",
         lambda: tools.hour_profile(con, hours=[19], city="Bangalore")),
        ("How many stores had sustained pileup?",
         lambda: tools.count_matching(con, "sustained_pileup", level="store")),
        ("What was STORE_003's breach rate?",
         lambda: tools.metric_lookup(con, "STORE_003",
                                     metrics=["w_breached_rate", "w_avg_or2a", "noshow_count"])),
    ]
    for q, fn in cases:
        h2(q)
        res = fn()
        txt = R(res)
        print("\n".join(txt.splitlines()[:9]))
        if len(txt.splitlines()) > 9:
            print(f"  ... ({len(res.rows)} rows total)")
        print(f"\n  provenance: {res.provenance}")

    h2("SAFETY — malicious tool arguments")
    for label, fn in [
        ("metric='1; DROP TABLE dim_store'",
         lambda: tools.rank_entities(con, metric="1; DROP TABLE dim_store")),
        ("condition=\"'; DELETE FROM fact_store_hour--\"",
         lambda: tools.count_matching(con, "'; DELETE FROM fact_store_hour--")),
    ]:
        try:
            fn()
            print(f"  NOT REJECTED  {label}")
        except Exception as e:
            print(f"  rejected      {label}\n                -> {type(e).__name__}")
    print(f"\n  tables intact: dim_store="
          f"{con.execute('SELECT COUNT(*) FROM dim_store').fetchone()[0]}, "
          f"fact_store_hour="
          f"{con.execute('SELECT COUNT(*) FROM fact_store_hour').fetchone()[0]}")


# ═══════════════════════════════════════════════════════════════════════

def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    con, results, ms = boot()

    if what in ("all", "load"):
        section_load(con, results, ms)
    if what in ("all", "traps"):
        section_traps(con)
    if what in ("all", "demo"):
        section_demo(con)
    if what in ("all", "tools"):
        section_tools(con)

    h1("DONE")
    print("\n  Every number above was computed by deterministic Python.")
    print("  No LLM is involved anywhere in Modules 1 and 2.")
    print("\n  Run the test suite with:  python -m pytest -v\n")


if __name__ == "__main__":
    main()
