#!/usr/bin/env python3
"""
Live LLM connectivity check.

Run this once after putting GOOGLE_API_KEY in .env:

    python check_llm.py

It answers four questions, in order, and stops at the first real problem:

  1. Is the key present and readable?
  2. Which provider does the factory actually pick?
  3. Which models does the account have access to?
  4. Does the classifier prompt produce correct JSON on the demo questions?

Nothing here touches the RCA engine. If this fails, the agent still works --
it falls back to the rule-based classifier and every number stays correct.
This only tells you whether the *prose* path is live.
"""

import os
import sys
import time

sys.path.insert(0, ".")

from app.env import load_env
load_env()

import config
from app import db, tables, tools
from app.entities import EntityIndex
from app.llm import TELEMETRY, GoogleProvider, get_provider, parse_json
from app.prompts import build_classify_prompt
from app.validation import validate

W = 74
ok_count = 0
fail_count = 0


def result(passed: bool, label: str, detail: str = ""):
    global ok_count, fail_count
    if passed:
        ok_count += 1
    else:
        fail_count += 1
    print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    if detail:
        for line in str(detail).splitlines():
            print(f"        {line}")


print("=" * W)
print("  LIVE LLM CHECK")
print("=" * W)

# ── 1. key ──────────────────────────────────────────────────────────────
print("\n[1] API key")

# Provider-agnostic: find whichever key is present, honouring LLM_PROVIDER.
_KEYS = [("groq", "GROQ_API_KEY", "https://console.groq.com/keys"),
         ("google", "GOOGLE_API_KEY", "https://aistudio.google.com/apikey"),
         ("anthropic", "ANTHROPIC_API_KEY", "https://console.anthropic.com"),
         ("openai", "OPENAI_API_KEY", "https://platform.openai.com/api-keys")]

want = (os.getenv("LLM_PROVIDER") or "").strip().lower()
candidates = [k for k in _KEYS if not want or want in ("auto",) or k[0] == want]
found = next(((n, e, u) for n, e, u in candidates if os.getenv(e)), None)

if not found:
    expected = candidates[0] if candidates else _KEYS[0]
    result(False, f"{expected[1]} not found",
           f"Create .env from .env.example and paste your key:\n"
           f"    cp .env.example .env\n"
           f"    open -e .env\n\n"
           f"Get a free key at {expected[2]}\n"
           f"Then set, with no quotes:\n"
           f"    LLM_PROVIDER={expected[0]}\n"
           f"    {expected[1]}=...")
    sys.exit(1)

provider_name, key_var, _ = found
key = os.getenv(key_var, "")
result(True, f"{key_var} present ({key[:6]}...{key[-4:]}, {len(key)} chars)")

# Catch the copy-paste mistakes that produce a baffling 401 rather than an
# obvious "your key is malformed".
_problems = []
if key != key.strip():
    _problems.append("leading or trailing whitespace")
if key[:1] in "\"'" or key[-1:] in "\"'":
    _problems.append("surrounding quotes — remove them, .env needs none")
if " " in key:
    _problems.append("a space inside the key")
if _problems:
    result(False, "key looks malformed",
           "\n".join(f"- {p}" for p in _problems) +
           "\n\nThe line in .env should read exactly:\n"
           "    GOOGLE_API_KEY=AIzaSy...\n"
           "with no quotes, no spaces around '=', and nothing after the key.")

# Google issues API keys in two formats. Both are legitimate:
#
#   AIzaSy...   legacy format, 39 chars
#   AQ....      current format, rolled out during 2026
#
# The format itself tells you nothing about whether the key works. What DOES
# matter is a known platform issue: many accounts restricted to AQ. keys have
# their free-tier quota provisioned at ZERO, producing
#   "generativelanguage.googleapis.com/generate_content_free_tier_requests
#    limit: 0"
# on every request. That is an account-level restriction, not a configuration
# problem -- new keys and new projects do not resolve it. We surface the
# diagnosis in step 4 where the actual error appears, rather than guessing from
# the prefix here.
if key.startswith("AQ."):
    result(True, "key format: Google AQ. (current)",
           "Note: many accounts issued AQ. keys have free-tier quota set to 0.\n"
           "If step 4 shows 'limit: 0', that is the known Google restriction.")

result(True, f"LLM_PROVIDER = {os.getenv('LLM_PROVIDER') or '(auto)'}")

# ── 2. provider ─────────────────────────────────────────────────────────
print("\n[2] Provider selection")
con = db.load()
validate(con)
tables.build_all(con)
index = EntityIndex(con)

provider = get_provider(index.cities)
if provider.name == "rule-based":
    pkg = {"groq": "openai", "google": "google-genai",
           "anthropic": "anthropic", "openai": "openai"}.get(provider_name, "?")
    result(False, f"factory fell back to rule-based, expected {provider_name!r}",
           f"The SDK is probably not installed:\n"
           f"    pip install {pkg}\n"
           f"(The agent still runs -- it just uses the rule-based classifier,\n"
           f" and every number stays correct.)")
    sys.exit(1)
result(True, f"provider = {provider.name}")
result(True, f"model = {provider.model}")

# ── 3. models ───────────────────────────────────────────────────────────
print("\n[3] Models available on this account")
models = provider.list_models() if hasattr(provider, "list_models") else []
if models and not str(models[0]).startswith("<"):
    for m in models[:12]:
        marker = "  <-- configured" if provider.model in str(m) else ""
        print(f"        {m}{marker}")
    if len(models) > 12:
        print(f"        ... and {len(models) - 12} more")
    hit = any(provider.model in str(m) for m in models)
    result(hit, f"configured model {provider.model!r} is available",
           "" if hit else
           f"Set LLM_MODEL in .env to one of the names above, e.g.\n"
           f"    LLM_MODEL={str(models[0]).split('/')[-1]}")
else:
    result(True, "model listing unavailable (not fatal)",
           str(models[:1]) if models else "provider does not expose a model list")

# ── 4. classifier ───────────────────────────────────────────────────────
print("\n[4] Classifier prompt against the demo questions")
system = build_classify_prompt(index.cities, sorted(tools.METRICS),
                               sorted(tools._CONDITIONS))

CASES = [
    # (question,                                  expected intent, expected store, date must be)
    ("How did Bangalore do on 2026-04-22?",       "city_summary",  None,        "2026-04-22"),
    ("Why did STORE_003 underperform that day?",  "store_rca",     "STORE_003", None),
    ("Walk me through the morning hours.",        "hour_detail",   None,        None),
    ("What about STORE_091?",                     "follow_up",     "STORE_091", None),
    ("How did STORE_210 do?",                     "store_rca",     "STORE_210", None),
    ("show me all hours",                         "show_all",      None,        None),
    ("Which city had the most no-shows?",         "rank",          None,        None),
    ("How many stores had sustained pileup?",     "count",         None,        None),
    ("What's the pileup threshold?",              "methodology",   None,        None),
    ("What is the capital of France?",            "out_of_scope",  None,        None),
]

print(f"\n        {'question':<42}{'intent':<15}{'store':<12}date")
print("        " + "-" * 76)

t0 = time.time()
misses = []
api_errors = []
for question, want_intent, want_store, want_date in CASES:
    resp = provider.complete(system, question, json_mode=True)
    if not resp.ok:
        print(f"        ERR   {question[:40]:<42}{(resp.error or '')[:34]}")
        api_errors.append(resp.error or "")
        misses.append((question, "API error", (resp.error or "")[:60]))
        continue

    data = parse_json(resp.text) or {}
    got_intent = data.get("intent")
    got_store = data.get("store")
    got_date = data.get("date")

    good = (got_intent == want_intent
            and (want_store is None or got_store == want_store)
            and got_date == want_date)
    if not good:
        misses.append((question, f"{want_intent}/{want_store}/{want_date}",
                       f"{got_intent}/{got_store}/{got_date}"))

    mark = "  " if good else "XX"
    print(f"      {mark}{question[:40]:<42}{str(got_intent):<15}"
          f"{str(got_store):<12}{got_date}")

elapsed = time.time() - t0

print()
result(not misses, f"{len(CASES) - len(misses)}/{len(CASES)} classified correctly")

# If every call failed the same way, one full error is far more useful than ten
# truncated ones -- the quota TYPE (per-minute vs per-day) is the actionable
# detail and it lives at the end of the message.
if api_errors:
    from app.llm import is_transient, parse_retry_delay
    print("\n        --- FULL ERROR (first occurrence) ---")
    for line in str(api_errors[0]).replace(", '", ",\n        '").splitlines():
        print(f"        {line[:110]}")
    print(f"\n        classified as: "
          f"{'TRANSIENT (will retry)' if is_transient(api_errors[0]) else 'PERMANENT (fails fast)'}")
    delay = parse_retry_delay(api_errors[0])
    if delay:
        print(f"        server suggests retrying after {delay}s")
    low = api_errors[0].lower()
    if "limit: 0" in low or "limit:0" in low:
        print("\n        -> QUOTA ALLOCATED AS ZERO, not exhausted.")
        print("           This is a known Google account-level restriction, most")
        print("           common on accounts issued 'AQ.' prefix keys. Creating a")
        print("           new key or a new project does NOT fix it.")
        print("\n           Options, fastest first:")
        print("             1. Try a different Google account")
        print("             2. Use another free provider (Groq, OpenRouter) --")
        print("                app/llm.py takes a new provider in ~30 lines")
        print("             3. Run without a key: the rule-based classifier and")
        print("                deterministic summaries keep every number correct")
    elif "per day" in low or "daily" in low:
        print("\n        -> DAILY quota exhausted. Retrying will not help today.")
        print("           Wait for the reset, or set a smaller model in .env,")
        print("           e.g. LLM_MODEL=llama-3.1-8b-instant (Groq) or")
        print("           LLM_MODEL=gemini-2.0-flash-lite (Google).")
    elif "per minute" in low or "perminute" in low:
        print("\n        -> PER-MINUTE limit. Retry with backoff handles this.")
else:
    for q, want, got in misses:
        print(f"        {q}\n          expected {want}\n          got      {got}")

# ── 5. the critical rule ────────────────────────────────────────────────
#
# BUG THIS FIXES: the first version asserted `data.get("date") is None` without
# checking that the call succeeded. When every request 429'd, parse_json
# returned None, data became {}, and .get("date") returned None -- so a total
# API failure looked identical to correct behaviour and the check PASSED.
#
# A test that passes when the thing under test never ran is worse than no test.
print("\n[5] The null-means-inherit rule (multi-turn depends on this)")


def _classified(question: str):
    """Return parsed JSON, or None with a reason if the call did not succeed."""
    r = provider.complete(system, question, json_mode=True)
    if not r.ok:
        return None, f"API call failed: {(r.error or '')[:90]}"
    d = parse_json(r.text)
    if d is None:
        return None, f"response was not JSON: {r.text[:90]!r}"
    return d, None


data, why = _classified("Why did STORE_003 underperform that day?")
if data is None:
    result(False, "could not verify the null-inherit rule", why)
else:
    got = data.get("date")
    result(got is None,
           "'that day' returns date=null instead of inventing one",
           "" if got is None else
           f"Got {got!r}. The model is filling in a date instead of deferring\n"
           f"to conversation context. Multi-turn will break -- the prompt needs\n"
           f"hardening before the graph is built on top of it.")

data, why = _classified("What about STORE_091?")
if data is None:
    result(False, "could not verify bare follow-up handling", why)
else:
    result(data.get("city") is None and data.get("date") is None,
           "bare follow-up leaves city and date null",
           f"city={data.get('city')!r} date={data.get('date')!r}")

# ── 5b. Tier-2 parameters ───────────────────────────────────────────────
#
# Step 4 only compared intent, store and date. A "rank" intent with a null
# metric, or a "count" with a null condition, would have passed there and then
# failed at execution time -- the intent is right but the tool cannot be called.
print("\n[5b] Tier-2 tool parameters")

PARAM_CASES = [
    ("Which city had the most no-shows?",      "metric",    sorted(tools.METRICS)),
    ("Worst 5 stores in Mumbai",               "metric",    sorted(tools.METRICS)),
    ("How many stores had sustained pileup?",  "condition", sorted(tools._CONDITIONS)),
    ("How many hours had a demand spike?",     "condition", sorted(tools._CONDITIONS)),
]

for question, field, allowed in PARAM_CASES:
    data, why = _classified(question)
    if data is None:
        result(False, f"{field} for {question[:34]!r}", why)
        continue
    value = data.get(field)
    ok = value in allowed
    result(ok, f"{question[:40]:<42} {field}={value!r}",
           "" if ok else
           f"Must be one of the registry values, or the tool cannot be called.\n"
           f"Allowed: {', '.join(allowed[:6])}...")

# ── 6. narration ────────────────────────────────────────────────────────
print("\n[6] Narration")
from app import rca
from app.nodes import facts_for_narration, narrate
from app.render import build_summary

ranked = rca.rank_problem_hours(con, "STORE_003")
checks = ranked.top[0]
deterministic = build_summary(checks)
generated = narrate(facts_for_narration(checks), provider, fallback=deterministic)

print(f"\n        deterministic fallback:\n          {deterministic}\n")
print(f"        {provider.name} ({provider.model}):\n          {generated}\n")
result(generated != deterministic, "model produced its own sentence",
       "" if generated != deterministic else
       "Fell back to the deterministic summary -- check the errors above.")
result("562" in generated or "561" in generated or "STORE_003" in generated,
       "output references the computed facts")

# ── summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * W)
print(f"  {ok_count} passed, {fail_count} failed")
print(f"  {TELEMETRY.summary()}")
print(f"  classifier pass took {elapsed:.1f}s for {len(CASES)} calls "
      f"({elapsed / len(CASES):.1f}s each)")
print("=" * W)

if fail_count == 0:
    print(f"\n  {provider.name} is wired up correctly. Ready for the graph.")
    # Two LLM calls per turn, so per-call latency doubles into what the user
    # feels. Above ~1.5s/call the demo starts to drag noticeably.
    per_call = elapsed / len(CASES)
    if per_call > 1.5:
        print(f"\n  NOTE: {per_call:.1f}s per call means ~{per_call * 2:.1f}s per turn.")
        print( "  For a live demo, try a smaller model in .env:")
        print( "      LLM_MODEL=llama-3.1-8b-instant")
        print( "  Intent classification is a structured task -- a small model")
        print( "  handles it fine, and narration quality is explicitly not graded.")
    print()
else:
    print("\n  See the failures above. The agent still runs regardless --")
    print("  it falls back to the rule-based classifier and all numbers")
    print("  stay correct. Only the prose quality is affected.\n")
