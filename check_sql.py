#!/usr/bin/env python3
"""
Tier 3 diagnostic — see exactly what the model wrote and why it was rejected.

Run:  python check_sql.py
      python check_sql.py "your own question here"

WHY THIS EXISTS
---------------
A Tier-3 failure inside the agent is one line of prose. This prints the whole
transaction: the question, every SQL attempt, the screen's verdict on each, the
retry feedback, and the final rows.

It also proves the two guardrails work against a REAL model rather than a test
double. The last two questions below are deliberately baited:

    "average OR2A across all stores"    tempts AVG(w_avg_or2a)
    "total rider slots offered per store"  tempts SUM over replicated columns

If the model takes the bait, you should see the rejection AND the corrected
query that follows it. That retry is the system working, not failing.
"""

import sys

sys.path.insert(0, ".")

from app.env import load_env
load_env()

import config
from app import db, sql_path, tables
from app.llm import TELEMETRY, get_provider
from app.entities import EntityIndex
from app.sql_path import ScreenRejection, screen
from app.validation import validate

W = 78

QUESTIONS = [
    "what percentage of total orders came from the 10 busiest stores",
    "what share of stores never had a problem hour",
    # ── baited: these tempt the two known traps ──
    "average OR2A across all stores",
    "total rider slots offered per store",
]


def main() -> int:
    con = db.load()
    validate(con)
    tables.build_all(con)
    provider = get_provider(EntityIndex(con).cities)

    print("=" * W)
    print(f"  TIER 3 — GENERATED SQL   [provider: {provider.name}"
          f"{'/' + provider.model if hasattr(provider, 'model') else ''}]")
    print("=" * W)

    if provider.name == "rule-based":
        print("\n  No API key found. Tier 3 needs a real model -- the")
        print("  rule-based provider cannot write SQL, and the agent")
        print("  deliberately refuses to try.\n")
        print("  Put GROQ_API_KEY in .env and re-run.\n")
        return 1

    if not config.ENABLE_SQL_FALLBACK:
        print("\n  ENABLE_SQL_FALLBACK is False in config.py -- Tier 3 is off.\n")
        return 1

    # ── RAW PROBE ──────────────────────────────────────────────────────
    #
    # One minimal call before anything wraps it. When every question fails
    # identically the cause is upstream of the SQL logic, and the agent's own
    # error message cannot show you the provider's reason -- this can.
    #
    # Two calls, deliberately: a trivial JSON request first, then the real SQL
    # prompt. If the first succeeds and the second fails, the problem is the
    # prompt (size, or the JSON-mode requirement), not the key or the quota.
    print("\n  RAW PROBE")
    print("  " + "-" * (W - 4))

    tiny = provider.complete(
        'Reply with JSON only: {"ok": true}', "say ok",
        json_mode=True, max_retries=0, timeout=15)
    print(f"    minimal JSON call : ok={tiny.ok}")
    if not tiny.ok:
        print(f"        error   : {tiny.error}")
    else:
        print(f"        returned: {(tiny.text or '')[:120]!r}")

    schema = sql_path._schema_text(con)
    full = prompts_sql = None
    from app import prompts
    prompts_sql = prompts.build_sql_prompt(schema, {})
    print(f"    sql prompt size   : {len(prompts_sql):,} chars")

    full = provider.complete(prompts_sql, "how many stores are there",
                             json_mode=True, max_retries=0, timeout=20)
    print(f"    full SQL call     : ok={full.ok}")
    if not full.ok:
        print(f"        error   : {full.error}")
    else:
        print(f"        returned: {(full.text or '')[:200]!r}")

    if not tiny.ok and not full.ok:
        print("\n    Both failed -> the key, the quota or the model name is the")
        print("    problem, not this prompt.")
    elif tiny.ok and not full.ok:
        print("\n    Small call works, large one does not -> the SQL prompt is")
        print("    the problem (length, or JSON-mode handling).")

    questions = sys.argv[1:] or QUESTIONS
    failures = 0

    for q in questions:
        print(f"\n{'─' * W}")
        print(f"  Q: {q}")
        print(f"{'─' * W}")

        res = sql_path.answer(con, q, provider)

        for i, attempt in enumerate(res.attempts, 1):
            print(f"\n  attempt {i}:")
            for line in (attempt or "(empty)").strip().splitlines():
                print(f"      {line}")
            try:
                screen(attempt)
                print("      >> screen: ACCEPTED")
            except ScreenRejection as e:
                print(f"      >> screen: REJECTED — {e.reason}")

        if res.error:
            failures += 1
            print(f"\n  RESULT: no answer — {res.error}")
            continue

        print(f"\n  RESULT: {res.total_rows} row(s)"
              f"{' (capped)' if res.truncated else ''}")
        if res.rows:
            print(f"      {' | '.join(res.columns)}")
            for row in res.rows[:5]:
                print(f"      {' | '.join(str(v) for v in row)}")
            if len(res.rows) > 5:
                print(f"      ... {len(res.rows) - 5} more")

        # THE TWO CHECKS THAT MATTER. Not "did it run" -- a wrong query runs.
        sql_low = res.sql.lower()
        if "or2a" in sql_low:
            weighted = "or2a_order_minutes" in sql_low
            print(f"      >> OR2A weighting: "
                  f"{'OK (order-weighted)' if weighted else 'CHECK THIS'}")
            if not weighted:
                failures += 1
        if any(c in sql_low for c in ("current_size", "booked_size")):
            ok = "safe_slot_supply" in sql_low
            print(f"      >> supply grain: "
                  f"{'OK (slot-level table)' if ok else 'CHECK THIS'}")
            if not ok:
                failures += 1

    print(f"\n{'=' * W}")
    print(f"  {len(questions)} question(s), {failures} problem(s)")
    print(f"  {TELEMETRY.summary()}")
    print("=" * W)
    print()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
