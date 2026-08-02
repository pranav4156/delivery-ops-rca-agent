#!/usr/bin/env python3
"""
End-to-end agent check — see the full rendered answers.

    python check_agent.py            scripted walkthrough, full output
    python check_agent.py --chat     interactive REPL, type your own questions
    python check_agent.py --quick    assertions only, no rendered output

Exercises the whole stack: DuckDB -> RCA engine -> LangGraph -> LLM narration.
Nothing here is mocked. Every number printed is computed by the same code the
frontend will call in M4.

The REPL exists because the frontend does not yet. It is also the fastest way
to probe unscripted questions, which is what the walkthrough will actually
consist of.
"""

import os
import sys
import time

sys.path.insert(0, ".")

from app.env import load_env
load_env()

import config
from app import db, rca, tables
from app.graph import build_agent
from app.llm import TELEMETRY
from app.rollups import city_day, store_day
from app.validation import validate

W = 88
MODE = ("--chat" if "--chat" in sys.argv else
        "--quick" if "--quick" in sys.argv else "--full")

passed = failed = 0


def check(ok, label, detail=""):
    global passed, failed
    ok = bool(ok)
    passed += ok
    failed += not ok
    print(f"    {'PASS' if ok else 'FAIL'}  {label}")
    if detail and not ok:
        print(f"          {detail}")


def rule(title=""):
    print("\n" + "=" * W)
    if title:
        print(f"  {title}")
        print("=" * W)


# ═══════════════════════════════════════════════════════════════════════
# BOOT
# ═══════════════════════════════════════════════════════════════════════

t0 = time.time()
con = db.load()
validate(con)
tables.build_all(con)
agent = build_agent(con)
boot_ms = (time.time() - t0) * 1000

model = getattr(agent.provider, "model", "-")
rule("QC RCA AGENT — END TO END")
print(f"\n  provider   {agent.provider.name} / {model}")
print(f"  data       {con.execute('SELECT COUNT(*) FROM fact_store_hour').fetchone()[0]:,} "
      f"rows · {con.execute('SELECT COUNT(*) FROM dim_store').fetchone()[0]} stores · "
      f"{config.DATA_DATE}")
print(f"  boot       {boot_ms:.0f} ms")
if agent.provider.name == "rule-based":
    print("\n  NOTE: no API key found. Running the deterministic classifier.")
    print("  Every number is still correct -- only the prose is plainer.")


def ask(message: str, session: str = "demo", show: bool = True) -> dict:
    t = time.time()
    r = agent.run_turn(session, message)
    ms = (time.time() - t) * 1000
    c = r["context"]

    if show:
        print("\n" + "─" * W)
        print(f"  > {message}")
        print("─" * W)
        print(f"  intent={r['intent']}   provenance={r['provenance']}   {ms:.0f}ms")
        print(f"  context: store={c['store']}  city={c['city']}  date={c['date']}"
              + (f"  hours={c['hours']}" if c.get("hours") else ""))
        if c.get("inherited"):
            print(f"  inherited: {c['inherited']}")
        if c.get("resolution_methods"):
            print(f"  resolved via: {c['resolution_methods']}")
        print()
        for line in r["reply"].splitlines():
            print(f"  {line}")
    r["_ms"] = ms
    return r


# ═══════════════════════════════════════════════════════════════════════
# INTERACTIVE
# ═══════════════════════════════════════════════════════════════════════

if MODE == "--chat":
    rule("INTERACTIVE — type questions, /help for commands")
    print("""
  Try:
    How did Bangalore do on 2026-04-22?
    Why did STORE_003 underperform that day?
    Walk me through the morning hours.
    What about STORE_091?
    How did STORE_210 do?
    show me all hours
    Which city had the most no-shows?
    Compare STORE_003 and STORE_091
    How many stores had sustained pileup?
    Why did TBC8 underperform?          <- should decline, not guess

  Commands:  /new   start a fresh session (clears context)
             /ctx   show the current context
             /stats LLM call telemetry
             /quit
""")
    sid = "chat"
    while True:
        try:
            msg = input("  you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  bye\n")
            break
        if not msg:
            continue
        if msg in ("/quit", "/exit", "/q"):
            print("  bye\n")
            break
        if msg == "/new":
            from app.state import SESSIONS
            SESSIONS.reset(sid)
            print("  -- new session, context cleared\n")
            continue
        if msg == "/ctx":
            from app.state import SESSIONS
            s = SESSIONS.get(sid)
            print(f"  store={s.current_store} city={s.current_city} "
                  f"date={s.current_date} hours={s.current_hours} "
                  f"last_intent={s.last_intent} turns={s.turn_count}\n")
            continue
        if msg == "/stats":
            print(f"  {TELEMETRY.summary()}\n")
            continue
        if msg == "/help":
            print("  /new  /ctx  /stats  /quit\n")
            continue
        ask(msg, sid)
        print()
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════
# SCRIPTED
# ═══════════════════════════════════════════════════════════════════════

SHOW = MODE == "--full"

rule("PART 1 — THE FIVE DEMO QUESTIONS")

r1 = ask("How did Bangalore do on 2026-04-22?", show=SHOW)
r2 = ask("Why did STORE_003 underperform that day?", show=SHOW)
r3 = ask("Walk me through the morning hours.", show=SHOW)
r4 = ask("What about STORE_091?", show=SHOW)
r5 = ask("How did STORE_210 do?", show=SHOW)

rule("PART 2 — AD-HOC QUESTIONS (Tier 2 tools)")

t1 = ask("Which city had the most no-shows?", "adhoc", show=SHOW)
t2 = ask("Compare STORE_003 and STORE_091", "adhoc", show=SHOW)
t3 = ask("How many stores had sustained pileup?", "adhoc", show=SHOW)
t4 = ask("What happened at 7pm across Bangalore?", "adhoc", show=SHOW)
t5 = ask("What's the pileup threshold?", "adhoc", show=SHOW)

rule("PART 3 — EDGE CASES")

e1 = ask("Why did TBC8 underperform?", "edge", show=SHOW)
e2 = ask("How did Kolkata do?", "edge", show=SHOW)
e3 = ask("Why did STORE_003 underperform on 2026-04-23?", "edge", show=SHOW)
e4 = ask("What is the capital of France?", "edge", show=SHOW)
# Uses its own session: the "demo" session ended on STORE_210, so expanding
# there would correctly show STORE_210's hours -- not what this assertion is
# about. The first version got that wrong and compared against STORE_003.
ask("Why did STORE_003 underperform on 2026-04-22?", "expand", show=False)
e5 = ask("Show me all hours", "expand", show=SHOW)

# ═══════════════════════════════════════════════════════════════════════
# ASSERTIONS
# ═══════════════════════════════════════════════════════════════════════

rule("ASSERTIONS")

print("\n  Multi-turn context")
check(r1["intent"] == "city_summary", "T1 classified as city_summary")
check(r2["intent"] == "store_rca", "T2 classified as store_rca")
check("date" in r2["context"]["inherited"], "T2 inherited the date from T1")
check(r2["context"]["date"] == config.DATA_DATE, "T2 date is 2026-04-22")
check(r3["context"]["store"] == "STORE_003", "T3 inherited the store from T2")
check(r3["context"]["hours"] == config.HOUR_LABELS["morning"], "T3 'morning' -> 6-11")
check(r4["context"]["store"] == "STORE_091", "T4 bare follow-up swapped the store")
check("intent" in r4["context"]["inherited"], "T4 inherited the intent")
check(r5["context"]["city"] == "Noida", "T5 cross-city switch to Noida")

print("\n  Numbers match the engine exactly")
bl = city_day(con, "Bangalore")
check(f"{bl.total_orders:,} orders across {bl.store_count} stores" in r1["reply"],
      f"Bangalore: {bl.total_orders:,} orders / {bl.store_count} stores")
check(f"{bl.w_avg_or2a:,.1f} min" in r1["reply"],
      f"Bangalore weighted OR2A = {bl.w_avg_or2a:.4f}")

s3 = store_day(con, "STORE_003")
check(f"{s3.total_orders:,} orders" in r2["reply"], f"STORE_003 orders = {s3.total_orders}")
check(f"{s3.breached_count:,} breached" in r2["reply"], f"breached = {s3.breached_count}")
check(f"{s3.w_avg_or2a:,.1f} min" in r2["reply"], f"weighted OR2A = {s3.w_avg_or2a:.4f}")
check(f"{s3.w_breached_rate * 100:.1f}%" in r2["reply"],
      f"breach rate = {s3.w_breached_rate:.6f}")

s210 = store_day(con, "STORE_210")
check(f"{s210.w_avg_or2a:,.1f} min" in r5["reply"],
      f"STORE_210 weighted OR2A = {s210.w_avg_or2a:.4f}")

print("\n  Impact ranking (Option 2 — rank, never filter)")
ranked = rca.rank_problem_hours(con, "STORE_003")
check(r2["reply"].count("### STORE_003 — Hour") == config.TOP_N_PROBLEM_HOURS,
      f"detailed exactly {config.TOP_N_PROBLEM_HOURS} hours")
check(f"{ranked.remainder_count} further problem hours" in r2["reply"],
      f"disclosed the other {ranked.remainder_count}")
check("Top 3 problem hours" in r2["reply"], "labelled as impact-ranked")
for h in [c.hour for c in ranked.top]:
    check(f"Hour {h} " in r2["reply"], f"hour {h} present (top by breached_count)")

print("\n  Playbook output format")
check("1. Demand Spike:" in r2["reply"], "check 1 rendered")
check("2. Pileup:" in r2["reply"], "check 2 rendered")
check("3. Supply:" in r2["reply"], "check 3 rendered")
check("a. Booking:" in r2["reply"], "check 3a rendered")
check("b. Utilization:" in r2["reply"], "check 3b rendered")
check("**Summary**:" in r2["reply"], "summary sentence present")
check("SUSTAINED" in r2["reply"], "sustained pileup escalation shown")
check(r2["reply"].count("Demand Spike: NO") >= 1,
      "NO answers still rendered (explicit playbook rule)")

print("\n  Show-all expansion")
check(e5["context"]["store"] == "STORE_003", "expands the store in context")
check(e5["reply"].count("### STORE_003 — Hour") == ranked.problem_hour_count,
      f"renders all {ranked.problem_hour_count} problem hours, not the top 3",
      f"got {e5['reply'].count('### STORE_003 — Hour')} blocks")
check("further problem hours" not in e5["reply"],
      "no remainder line -- nothing left undisclosed")

print("\n  Tier-2 tools")
check(t1["intent"] == "rank" and t1["provenance"] == config.PROVENANCE_VALIDATED,
      "ranking answered and marked validated")
check(t2["intent"] == "compare", "comparison routed")
check("STORE_003 vs STORE_003" not in t2["reply"], "comparison not duplicated")
check(t3["intent"] == "count", "count routed")
check(t4["intent"] == "hour_profile" and "Hour 19" in t4["reply"], "7pm -> hour 19")
check(t5["intent"] == "methodology", "methodology routed")

print("\n  Edge cases decline rather than guess")
check(e1["provenance"] == config.PROVENANCE_UNAVAILABLE and "TBC8" in e1["reply"],
      "TBC8 declined with a suggestion list")
check("STORE_008" not in e1["reply"].split("include:")[0],
      "TBC8 did NOT silently become STORE_008")
# Asserts the BEHAVIOUR, not one provider's phrasing. A real LLM copies the
# user's text so resolve_city can name it ("I don't have data for 'Kolkata'");
# the rule-based fallback cannot extract a city it does not recognise and asks
# generically. Both decline and list the valid cities, which is what matters.
check(all(c in e2["reply"] for c in ("Bangalore", "Noida", "Chennai")),
      "unknown city declined, valid cities listed")
check("2026-04-22" in e3["reply"], "wrong date explained")
check(e4["provenance"] == config.PROVENANCE_UNAVAILABLE, "off-topic declined")
check(all("Something went wrong" not in r["reply"]
          for r in (e1, e2, e3, e4, e5)), "no crashes on any edge case")

print("\n  Provenance")
check(r1["provenance"] == config.PROVENANCE_VALIDATED, "city summary = validated")
check(r2["provenance"] == config.PROVENANCE_VALIDATED, "store RCA = validated")
check(not any(r["provenance"] == config.PROVENANCE_EXPLORATORY
              for r in (r1, r2, r3, r4, r5, t1, t2, t3, t4, t5)),
      "nothing marked exploratory (Tier 3 not built)")

# ═══════════════════════════════════════════════════════════════════════

rule()
turns = [r1, r2, r3, r4, r5, t1, t2, t3, t4, t5, e1, e2, e3, e4, e5]
avg = sum(r["_ms"] for r in turns) / len(turns)
print(f"  {passed} passed, {failed} failed")
print(f"  {len(turns)} turns, {avg:.0f}ms average")
if agent.provider.name != "rule-based":
    print(f"  {TELEMETRY.summary()}")
print("=" * W)
if failed == 0:
    print("\n  RCA engine and agent are working end to end.")
    print("  Try it yourself:  python check_agent.py --chat\n")
else:
    print("\n  See the failures above.\n")
    sys.exit(1)
