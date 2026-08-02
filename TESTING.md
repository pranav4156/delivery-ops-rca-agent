# Backend Testing Guide

Seven steps, in order. Each one builds on the last, so a failure tells you
exactly which layer broke.

**Total time: ~10 minutes.**

Steps 1–3 need no API key and no internet. Steps 4–5 need a key. Step 6 needs
Node. Step 7 needs nothing extra.

---

## Step 0 — Setup (once)

```bash
cd "/Users/pranavpujara/Desktop/QC RCA Agent - Test Data/qc-rca-agent"

python3 -m venv .venv
source .venv/bin/activate

which python          # must show .../qc-rca-agent/.venv/bin/python

pip install -r requirements.txt
```

If you open a new terminal tab later, run `source .venv/bin/activate` again.

**Expected:** installs duckdb, pandas, pytest, langgraph, openai, mcp, fastapi,
uvicorn. Two to three minutes on a first run.

---

## Step 1 — Unit tests

**Proves:** the data layer, RCA engine and query tools are correct.

```bash
python -m pytest -q
```

**Expected:**

```
203 passed in 1.4s
```

Want to see what they cover?

```bash
python -m pytest -v | head -40
python -m pytest -k "trap or supply or tbc8 or weighted" -v
```

### Prove the tests can actually fail

A test that can't fail is worse than no test. Break something and watch:

```bash
# open app/rca.py, find:  if booked == 0 and config.ZERO_BOOKING_AS_NO_SUPPLY:
# change it to:           if False:
python -m pytest -k "no_supply" -v      # should FAIL — 2 tests
# undo the change
python -m pytest -q                      # back to 203 passed
```

**⚠️ Known gap:** these 203 tests cover Modules 1–2 only. The agent layer
(`graph.py`, `nodes.py`, `llm.py`, `state.py` — about 2,200 lines) is covered by
the check scripts in steps 3–7, **not** by pytest. Do not read "203 passed" as
full coverage.

---

## Step 2 — The engine and the data traps

**Proves:** the numbers are right, and shows the five mistakes a naive
implementation makes on this dataset.

```bash
python verify.py traps
```

**Expected — each trap with the wrong answer beside the right one:**

```
TRAP 1  naive AVG(avg_or2a) 43.1354  vs  weighted 33.3561    29.3% error
TRAP 2  WHERE city = 'Mumbai'  ->  0 rows
TRAP 3  SUM(booked_size) 426   vs  deduped 249               71% over
TRAP 4  numerator-only FILTER 76.3029  vs  correct 79.9034
TRAP 5  naive man_hour<0.85 = False on a zero-rider hour
```

Then see the actual answers the agent produces:

```bash
python verify.py demo
```

**Look for:** the `STORE_003` RCA is ~40 lines with 3 hour blocks and a
"13 further problem hours" notice. That is the impact-ranking decision working.

Other sections: `python verify.py load` · `python verify.py tools`

---

## Step 3 — Multi-turn context (no API key needed)

**Proves:** the agent carries context across turns and knows when *not* to.

```bash
python check_turns.py
```

**Expected: `17 passed, 0 failed`**

**The thing to actually read** is turn 2:

```
TURN 2   "Why did STORE_003 underperform that day?"
  classifier extracted : {'intent': 'store_rca', 'store': 'STORE_003'}
  returned null for    : ['city', 'date']   <- 'inherit these'
  resolved context     : store=STORE_003  city=Bangalore  date=2026-04-22
  INHERITED from state : ['date', 'city']
```

`null` from the classifier means *"the user did not state this"*. The resolver
then fills it from the conversation. That two-layer split is what makes
inheritance inspectable.

Also note **turn 6** — `TBC8` is declined, and **turn 7 still works**, proving a
failed turn does not poison the context.

---

## Step 4 — LLM connectivity

**Needs an API key.** Free from https://console.groq.com/keys

```bash
cp .env.example .env
open -e .env
```

Set these two lines. **No quotes, no spaces around `=`:**

```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
```

Then:

```bash
python check_llm.py
```

**Expected: `10 passed, 0 failed`** plus a table of the demo questions and how
the model classified each.

**The section that matters is [5]:**

```
PASS  'that day' returns date=null instead of inventing one
```

If the model invents a date there instead of returning null, multi-turn breaks
and the prompt needs hardening. Tell me if it fails.

**If everything 429s:** you have hit a rate limit. The script prints the full
error and says whether it is per-minute (retry handles it) or per-day (switch
model). Try `LLM_MODEL=llama-3.1-8b-instant` in `.env`.

**If the key is missing or wrong:** the agent still works. It falls back to a
deterministic classifier and every number stays correct — only the prose gets
plainer.

---

## Step 5 — Full agent, end to end

**Proves:** the whole stack works — DuckDB → RCA engine → LangGraph → LLM.

```bash
python check_agent.py --quick
```

**Expected: `48 passed, 0 failed`**

To see every rendered answer in full:

```bash
python check_agent.py
```

**Then test it yourself — this is the important one:**

```bash
python check_agent.py --chat
```

Try the scripted questions, then go off-script. Commands: `/new` `/ctx`
`/stats` `/quit`.

**Worth trying to break it:**

```
Compare bangalore and noida, which did better and why?
does bangalore come in the worst 5 cities?
give me all cities from worst to best
how many stores had sustained pileup?
Why did TBC8 underperform?
what about the morning?                 (with no store in context)
Why did STORE_003 underperform on 2026-04-23?
```

**What good behaviour looks like:** when the agent cannot answer, it should
**ask or decline**, never guess. `TBC8` must never silently become `STORE_008`.

---

## Step 6 — MCP integration

**Needs Node.** Check with `node --version`.

```bash
# Pre-install so npx does not download during the demo
npm install -g @modelcontextprotocol/server-filesystem

python check_mcp.py
```

**Expected:** connects, lists the server's tools, reads all three reference
docs, and answers *"What's the pileup threshold?"* **from the playbook** rather
than from config.

**Two things to look for:**

```
[8]  PASS  answered from the DOCUMENTS, not from config
[9]  PASS  zero MCP calls during an RCA
```

The second one proves MCP is genuinely optional — **no number depends on it**.

Step 10 prints the tool-call log. **Screenshot that** — it is the evidence that
the MCP integration is real rather than decorative.

**If it fails:** the agent still runs. Methodology questions fall back to
config-derived values, clearly labelled. Nothing else is affected.

---

## Step 7 — The HTTP API

**Proves:** the endpoint your UI team will build against.

```bash
python -m uvicorn app.main:app --reload --port 8000
```

> **Use `python -m uvicorn`, not bare `uvicorn`.** If uvicorn is also installed
> globally, the bare command may resolve to that copy and run under system
> Python — which has none of this project's packages, giving a confusing
> `ModuleNotFoundError: No module named 'duckdb'` even though the venv is
> active. `python -m` uses the venv's interpreter explicitly and cannot pick
> the wrong one.
>
> Check with: `which uvicorn` vs `which python`

**Expected on startup:**

```
agent ready in 445 ms (llm: groq, mcp: connecting)
```

### Browser

Open **http://localhost:8000/docs** — interactive OpenAPI. You can run every
endpoint from there without writing a client.

### Terminal (in a second tab)

```bash
curl -s localhost:8000/health | jq
```

```bash
# Turn 1 — no session_id, server generates one
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"How did Bangalore do on 2026-04-22?"}' | jq -r '.reply'
```

```bash
# Turn 2 — reuse the session, "that day" is inherited
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"Why did STORE_003 underperform that day?","session_id":"demo"}' \
  | jq '{intent, provenance, context}'
```

```bash
# A decline — HTTP 200, with suggestions
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"Why did TBC8 underperform?","session_id":"demo"}' \
  | jq '{reply, provenance, suggestions, context}'
```

**Look for:** `context.store` is still `STORE_003` on that last one. The
context survives a decline, which is why the UI panel will not blank out.

```bash
curl -s localhost:8000/entities | jq '.cities'
curl -s localhost:8000/debug/stats | jq
```

---

## Summary — what each step proves

| Step | Command | Proves | Needs |
|---|---|---|---|
| 1 | `pytest -q` | Engine correctness (203 tests) | — |
| 2 | `verify.py traps` | The 5 data traps are handled | — |
| 3 | `check_turns.py` | Multi-turn context inheritance | — |
| 4 | `check_llm.py` | LLM connectivity + prompt behaviour | API key |
| 5 | `check_agent.py --chat` | Whole stack, interactively | API key |
| 6 | `check_mcp.py` | MCP integration is real | Node |
| 7 | `uvicorn app.main:app` | HTTP API for the UI team | — |

**Everything except step 7 runs offline if you skip steps 4–6.** The agent is
designed so that no external dependency can affect a number.

---

## If something fails

Paste the output. Include which step and the full error — every script is
written to explain what went wrong and what to do about it, so the message
itself is usually the answer.
