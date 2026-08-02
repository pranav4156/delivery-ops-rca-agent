#!/usr/bin/env python3
"""
Multi-turn conversation check — shows context inheritance actually happening.

Run:  python check_turns.py

WHY THIS EXISTS SEPARATELY FROM check_llm.py
---------------------------------------------
check_llm.py tests the CLASSIFIER IN ISOLATION. Each question is a fresh call
with no conversation history, so "Why did STORE_003 underperform that day?"
correctly returns date=null -- null is the classifier's way of saying "the user
did not state this; inherit it".

That output looks like a failure if you read it as the final answer. It is not.
Inheritance happens one layer later, in resolve_entities, which is what this
script exercises: the same five questions, in sequence, in one session, showing
each null being filled from the previous turn.

Two layers, two jobs:

    classify_intent   "was a date stated?"        -> null means no
    resolve_entities  "then what date applies?"   -> fills from session state

Keeping them separate is what makes the inheritance inspectable. A single
prompt carrying the whole history would produce the same answer with no way to
show WHY.
"""

import os
import sys
import time

sys.path.insert(0, ".")

from app.env import load_env
load_env()

import config
from app import db, tables
from app.entities import EntityIndex
from app.llm import TELEMETRY, get_provider
from app.nodes import ResolutionError, classify_intent, commit_context, resolve_entities
from app.state import Session
from app.validation import validate

W = 78
passed = failed = 0


def check(ok: bool, label: str, detail: str = ""):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"    {'PASS' if ok else 'FAIL'}  {label}")
    if detail:
        print(f"          {detail}")


con = db.load()
validate(con)
tables.build_all(con)
index = EntityIndex(con)
provider = get_provider(index.cities)

print("=" * W)
print(f"  MULTI-TURN CONVERSATION  [provider: {provider.name}"
      f"{'/' + provider.model if hasattr(provider, 'model') else ''}]")
print("=" * W)

if provider.name == "rule-based":
    print("\n  No API key found -- running against the rule-based classifier.")
    print("  Inheritance is deterministic Python, so it works either way; this")
    print("  just does not exercise the real model.\n")

session = Session()


def turn(n: int, message: str, note: str = ""):
    """Run one turn and print what was stated vs what was inherited."""
    t0 = time.time()
    extraction = classify_intent(message, provider, index)
    try:
        ctx = resolve_entities(extraction, session, index)
        commit_context(session, ctx)
        err = None
    except ResolutionError as e:
        ctx, err = None, e
    ms = (time.time() - t0) * 1000

    print(f"\n  {'─' * (W - 4)}")
    print(f"  TURN {n}   \"{message}\"")
    print(f"  {'─' * (W - 4)}")

    # What the model extracted -- null means "not stated in this message"
    stated = {k: v for k, v in (
        ("intent", extraction.intent), ("city", extraction.city),
        ("store", extraction.store), ("date", extraction.date),
        ("hour_label", extraction.hour_label),
        ("metric", extraction.metric), ("condition", extraction.condition),
    ) if v is not None}
    print(f"    classifier extracted : {stated}")
    nulls = [k for k in ("city", "store", "date")
             if getattr(extraction, k) is None]
    if nulls:
        print(f"    returned null for    : {nulls}   <- 'inherit these'")

    if err:
        print(f"    DECLINED             : {err.message[:100]}")
        return None

    print(f"    resolved context     : intent={ctx.intent}  store={ctx.store}  "
          f"city={ctx.city}  date={ctx.date}")
    if ctx.hours:
        print(f"                           hours={ctx.hours}")
    if ctx.inherited:
        print(f"    INHERITED from state : {ctx.inherited}")
    if ctx.resolution_methods:
        print(f"    resolved via         : {ctx.resolution_methods}")
    if note:
        print(f"    {note}")
    print(f"    ({ms:.0f}ms)")
    return ctx


# ═══════════════════════════════════════════════════════════════════════
# The five demo questions, in sequence, in ONE session
# ═══════════════════════════════════════════════════════════════════════

c1 = turn(1, "How did Bangalore do on 2026-04-22?",
          "Everything stated explicitly. Nothing to inherit.")
c2 = turn(2, "Why did STORE_003 underperform that day?",
          "<< 'that day' -> classifier said null -> resolver filled 2026-04-22")
c3 = turn(3, "Walk me through the morning hours.",
          "<< no store named -> inherited STORE_003 from turn 2")
c4 = turn(4, "What about STORE_091?",
          "<< no verb at all -> inherited the INTENT from turn 3")
c5 = turn(5, "How did STORE_210 do?",
          "<< different city -> context switches Bangalore -> Noida")

print("\n" + "=" * W)
print("  ASSERTIONS")
print("=" * W)

print("\n  Turn 1 — explicit")
if c1:
    check(c1.intent == "city_summary", "intent = city_summary")
    check(c1.city == "Bangalore", "city = Bangalore")
    check(c1.date == "2026-04-22", "date taken from the message")

print("\n  Turn 2 — THE DATE QUESTION")
if c2:
    check(c2.store == "STORE_003", "store = STORE_003 (stated)")
    check("date" in c2.inherited, "date was INHERITED, not stated")
    check(c2.date == "2026-04-22", "inherited value is correct",
          "the classifier returned null; the resolver supplied the date")

print("\n  Turn 3 — store carries forward")
if c3:
    check(c3.intent == "hour_detail", "intent = hour_detail")
    check("store" in c3.inherited, "store was INHERITED")
    check(c3.store == "STORE_003", "still STORE_003")
    check(c3.hours == config.HOUR_LABELS["morning"], "'morning' -> hours 6-11")

print("\n  Turn 4 — bare follow-up")
if c4:
    check(c4.store == "STORE_091", "store swapped to STORE_091")
    check("intent" in c4.inherited, "intent was INHERITED from turn 3")
    check(c4.date == "2026-04-22", "date still carried")

print("\n  Turn 5 — cross-city switch")
if c5:
    check(c5.store == "STORE_210", "store = STORE_210")
    check(c5.city == "Noida", "city switched to Noida",
          "a store implies its city, so context stays coherent")

print("\n  Failure isolation")
before = session.current_store
turn(6, "What about TBC8?", "<< does not exist -- must decline, not guess")
check(session.current_store == before,
      "a declined turn did not poison the context",
      f"current_store still {session.current_store}")
c7 = turn(7, "Walk me through the evening hours.",
          "<< still works after the failed turn")
check(c7 is not None and c7.store == before, "conversation still usable")

print("\n" + "=" * W)
print(f"  {passed} passed, {failed} failed")
if provider.name != "rule-based":
    print(f"  {TELEMETRY.summary()}")
print(f"  final session state: store={session.current_store} "
      f"city={session.current_city} date={session.current_date}")
print("=" * W)
print()
