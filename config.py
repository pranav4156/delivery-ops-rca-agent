"""
Central configuration — every threshold and judgment call in one place.

DESIGN NOTE
-----------
The RCA playbook specifies four numeric thresholds. Several other decisions were
forced by contradictions or gaps between the reference docs and the actual data
(see 00_PRE_DESIGN_ANALYSIS.md). Rather than scatter those through the code, all
of them live here.

This means any threshold or design decision can be changed and the system re-run
without touching logic — useful for the walkthrough, where the panel may want to
see how results shift under a different threshold.

Each constant cites the source that justifies it.
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"

CSV_PATH = DATA_DIR / "quick_commerce_orders_gold_20260422.csv"

# The single date present in the dataset. Verified: charge_date has exactly
# one distinct value. Used to reject out-of-range date requests politely.
DATA_DATE = "2026-04-22"


# ═══════════════════════════════════════════════════════════════════════
# PLAYBOOK THRESHOLDS  — from quick_commerce_rca_logic.md
# These four are prescribed by the playbook. Do not change without reason.
# ═══════════════════════════════════════════════════════════════════════

# Check 1 — Demand Spike
# "If total_orders > order_projection x 1.10 -> flag DEMAND SPIKE"
DEMAND_SPIKE_MULTIPLIER = 1.10

# Check 2 — Pileup escalation
# "Check if pileup is continuous for 3+ consecutive hours -> SUSTAINED PILEUP"
SUSTAINED_PILEUP_HOURS = 3

# Check 3 L1 — Booking Gap
# "If current_capacity_booked < 0.90 -> flag BOOKING GAP"
BOOKING_GAP_THRESHOLD = 0.90

# Check 3 L2 — Utilization Gap
# "If man_hour < 0.85 -> flag UTILIZATION GAP"
UTILIZATION_GAP_THRESHOLD = 0.85


# ═══════════════════════════════════════════════════════════════════════
# RESOLVED AMBIGUITIES  — decisions forced by doc/data contradictions
# ═══════════════════════════════════════════════════════════════════════

# ── Q2: OR2A threshold ────────────────────────────────────────────────
# CONTRADICTION: order_ready_to_assignment.md states BOTH
#   "Healthy: avg OR2A <= 0"  AND  "(client-specific, typically 5-8 min)"
# on consecutive lines. The mandated output format requires printing a
# threshold on every RCA block, so a value must be chosen.
#
# DECISION: default 0, because it makes our verdicts agree exactly with the
# is_problem_hour column in the data (verified: is_problem_hour = 1 is
# equivalent to breached_rate > 0, with zero exceptions across 3,813 rows).
# Configurable because the doc says "client-specific".
OR2A_THRESHOLD_MIN = 0

# ── Q3: capacity ratio source ─────────────────────────────────────────
# CONTRADICTION: schema doc says current_capacity_booked = booked_size /
# current_size. That formula reproduces only 32.7% of rows. The formula that
# reproduces 100.0% is (booked_size - cancelled_count) / current_size.
#
# DECISION: read the stored column, never recompute. The stored value is what
# the ops team's own SQL returns, so our numbers always agree with theirs.
USE_STORED_CAPACITY_RATIO = True

# ── Q4: problem-hour presentation ─────────────────────────────────────
# ISSUE: is_problem_hour = 1 on 2,202 of 3,813 rows (57.7%) because it means
# "at least one order breached". STORE_003 has 16 problem hours out of 16 --
# emitting a full markdown block for each is unreadable.
#
# DECISION: keep ALL problem hours in the analysis; rank by impact and detail
# only the top N in the response, disclosing the remainder count and
# supporting a "show me all hours" follow-up. Nothing is filtered out.
TOP_N_PROBLEM_HOURS = 3

# Impact metric for that ranking. breached_count chosen over:
#   - breached_rate: a 4-order hour with 2 breaches (50%) would outrank an
#     80-order hour with 20 breaches (25%) -- ranks trivia above real damage
#   - avg_or2a: dominated by the 2,488-minute outliers, which may be artifacts
# breached_count is a raw column meaning "how many customers were affected".
IMPACT_METRIC = "breached_count"

# Deterministic tie-break so the live demo is reproducible run to run.
IMPACT_TIEBREAK = ("avg_or2a", "hour")

# ── Q5: zero-booking rows ─────────────────────────────────────────────
# ISSUE: 139 rows have booked_size = 0 -> booked_hours = 0 -> man_hour NULL.
# These are the most severe supply failures in the dataset (capacity offered,
# nobody booked). But NaN < 0.85 evaluates to False, so a naive implementation
# reports "no utilization problem" on exactly the worst rows.
#
# DECISION: emit a distinct NO SUPPLY flag rather than folding into
# UTILIZATION GAP, which would conflate "riders booked but underused" with
# "no riders at all".
ZERO_BOOKING_AS_NO_SUPPLY = True

# ── Q8: consecutive-hour definition ───────────────────────────────────
# ISSUE: 25 stores have non-contiguous hour coverage, and hours 2-5 are absent
# dataset-wide. Does pileup at hours 10, 12, 13 (11 missing) count as 3
# consecutive hours?
#
# DECISION: strict adjacency -- hour values must differ by exactly 1. This is
# the plain reading of "consecutive hours", and gaps are most likely genuine
# store closures rather than missing data.
STRICT_HOUR_ADJACENCY = True

# ── Q10: city-level response scope ────────────────────────────────────
# Bangalore has 97 stores. A city answer lists weighted KPIs plus the worst N
# stores, then invites a drill-down.
CITY_TOP_N_STORES = 5

# ── Grouped ranking: "top N stores of each city" ──────────────────────
#
# Deliberately smaller than CITY_TOP_N_STORES. A per-group ranking multiplies
# by the number of cities: at 5 x 9 cities the answer is 45 rows, which stops
# being readable in a chat window. 3 x 9 = 27 fits, and the user can always
# state a number -- an explicit n always wins (see grain_facts.explicit_n).
PER_GROUP_DEFAULT_N = 3

# ── Multi-question messages ───────────────────────────────────────────
#
# "How did Bangalore do, and which store was worst?" is two questions. The
# splitter cuts the message up and each piece runs through the ordinary graph.
#
# CAPPED AT 3, for the same reason the narration budget exists: N questions is
# N full runs, each with its own narration. Four questions is four RCAs and a
# request the frontend will abandon before it returns. Three is enough for every
# realistic ops question and keeps the worst case bounded.
MAX_SUBQUESTIONS = 3

# Whole-turn budget shared across all sub-questions. Once spent, the remaining
# questions are not run -- and are LISTED, so the user can see what was dropped
# rather than silently receiving a partial answer.
SUBQUESTION_BUDGET_SECONDS = 25.0

# The splitter runs on every turn, so it must be cheap and must never be the
# reason a single-question message fails. On timeout or error the message is
# treated as ONE question, which is the pre-existing behaviour.
SPLIT_TIMEOUT_SECONDS = 6.0
SPLIT_MAX_RETRIES = 0

# ── Tier 3: generated SQL for questions no tool covers ────────────────
#
# ON by default, with a switch so it can be killed without a code change.
# Everything above this line returns `validated` provenance -- a tested tool
# computed it. This path returns `screened`: we generated the query, screened it
# against the traps we know about, and did not verify the answer.
#
# The two labels exist because they mean different things. Collapsing them would
# either overstate this path or insult the rest.
ENABLE_SQL_FALLBACK = True

# Rows returned to the user. A generated query can select 3,813 rows; nobody
# reads that in a chat window, and the count is disclosed either way.
SQL_ROW_LIMIT = 50

# Retries after the SCREEN rejects a query -- not after a network error. The
# rejection reason is fed back, and most rejections are the model reaching for
# AVG(avg_or2a), which it corrects when told. Two attempts, then decline.
SQL_MAX_RETRIES = 2
SQL_TIMEOUT_SECONDS = 20.0

# Hard cap on execution. A generated query can cross-join by accident.
SQL_EXECUTION_TIMEOUT_SECONDS = 5.0

# ── Documentation Q&A ─────────────────────────────────────────────────
#
# One call over the WHOLE corpus (~1,800 tokens -- smaller than the classifier
# prompt), so there is no retrieval step and no RAG. See
# MCPDocServer.read_corpus for why that is the right call at this size, and
# where the trade-off would flip.
#
# Retries allowed, unlike narration: the fallback here is a raw excerpt dump,
# which is materially worse than a written answer -- so one retry is worth it.
# Narration's fallback is a complete tested sentence, so it retries zero times.
DOC_MAX_RETRIES = 1
DOC_TIMEOUT_SECONDS = 20.0

# Which metric orders "worst stores" in a city summary.
#
# NOTE this deliberately differs from IMPACT_METRIC above, and the distinction
# is worth being able to defend:
#
#   Within a store, ranking HOURS by breached_count answers "which hour hurt the
#   most customers" -- the right triage question when every hour belongs to the
#   same store and shares its volume profile.
#
#   Across stores, ranking by breached_count would just rank by store size --
#   the biggest store almost always has the most breaches. "Which store
#   performed worst" is a question about severity, not volume, so w_avg_or2a is
#   the better ordering. breached_count is still shown on every row so volume
#   context is never hidden.
#
# Switchable to "breached_count" if a reviewer prefers volume ordering.
CITY_RANK_METRIC = "w_avg_or2a"


# ═══════════════════════════════════════════════════════════════════════
# HOUR LABELS
# Not defined in any source document -- our judgment call, pinned here so it
# is consistent and defensible rather than re-invented per query.
# Sample question 3 ("walk me through the morning hours") requires this.
# Note: hours 2-5 are absent from the dataset entirely.
# ═══════════════════════════════════════════════════════════════════════

HOUR_LABELS = {
    "morning":   [6, 7, 8, 9, 10, 11],
    "noon":      [12],
    "afternoon": [12, 13, 14, 15, 16],
    "evening":   [17, 18, 19, 20],
    "night":     [21, 22, 0, 1],
}

# Synonyms, resolved to the windows above. "in noon hours" previously matched
# nothing, so hour_label stayed None and the constraint was DROPPED SILENTLY --
# the user asked about one hour and got the whole day, with no indication the
# window had been ignored.
HOUR_LABEL_ALIASES = {
    "midday": "noon",
    "mid-day": "noon",
    "lunch": "noon",
    "lunchtime": "noon",
    "am": "morning",
    "early": "morning",
    "pm": "afternoon",
    "late": "night",
    "overnight": "night",
}


# ═══════════════════════════════════════════════════════════════════════
# ENTITY RESOLUTION
# ═══════════════════════════════════════════════════════════════════════

# Minimum difflib similarity for a fuzzy entity match. Below this we decline
# with suggestions rather than guessing.
FUZZY_MATCH_CUTOFF = 0.75

# How many alternatives to offer when an entity cannot be resolved.
MAX_SUGGESTIONS = 5

# ── City aliases ──────────────────────────────────────────────────────
# The stored city values are dirty and a user will never type them exactly:
#
#     'Mumbai  north'    <- DOUBLE space, lowercase 'north'
#     'New delhi'        <- lowercase 'd'
#     'Pune city east'   <- 'city' embedded mid-name
#     'Hyderabad city'   <- 'city' suffix
#
# Without aliasing, "How did Mumbai do?" matches zero rows and the agent
# reports no data -- on the very first question of the demo. Keys are in
# normalized form (lowercase, whitespace-collapsed); values are the exact
# stored strings.
CITY_ALIASES = {
    "bangalore": "Bangalore",
    "bengaluru": "Bangalore",
    "blr": "Bangalore",

    "mumbai": "Mumbai  north",
    "mumbai north": "Mumbai  north",
    "north mumbai": "Mumbai  north",
    "bombay": "Mumbai  north",

    "pune": "Pune city east",
    "pune east": "Pune city east",
    "east pune": "Pune city east",

    "delhi": "New delhi",
    "new delhi": "New delhi",
    "ncr": "New delhi",

    "noida": "Noida",

    "gurgaon": "Gurgaon",
    "gurugram": "Gurgaon",

    "hyderabad": "Hyderabad city",
    "hyd": "Hyderabad city",

    "chennai": "Chennai",
    "madras": "Chennai",

    "faridabad": "Faridabad",
}


# ═══════════════════════════════════════════════════════════════════════
# RESPONSE PROVENANCE
# Every response declares where its numbers came from. "exploratory" is
# reserved for the optional Tier 3 text-to-SQL path (Task 5.5, stretch) and is
# never emitted by Tiers 1 and 2. Building the field now keeps that addition
# purely additive instead of a late refactor.
# ═══════════════════════════════════════════════════════════════════════

PROVENANCE_VALIDATED = "validated"      # deterministic engine + parameterized tools
PROVENANCE_EXPLORATORY = "exploratory"  # reserved: unscreened generated SQL
PROVENANCE_UNAVAILABLE = "unavailable"  # graceful decline

# Tier 3. A model wrote the query; it passed the mechanical screen in
# sql_path.screen() and ran against read-only views shaped to make this
# dataset's two known traps unwritable. The ANSWER is still unverified.
#
# Deliberately NOT folded into "exploratory": that value means "generated, with
# no safety net", and conflating the two would either overstate this path or
# understate the difference the screen makes. `exploratory` is kept for a raw
# path we do not currently expose.
PROVENANCE_SCREENED = "screened"

EXPLORATORY_CAUTION = (
    "Exploratory result - this query was generated by the model, not from the "
    "validated RCA path. Verify before acting on it."
)


# ═══════════════════════════════════════════════════════════════════════
# LLM
#
# The model does exactly two things: classify intent, and write one summary
# sentence from facts that are already computed. It never calculates, compares
# or decides a flag -- all of that happened in tested Python before it is
# called. So the binding constraint on model choice is LATENCY, not reasoning
# depth: two round-trips per turn is what the user feels in a live demo.
#
# Provider is chosen from the environment (see app/llm.get_provider):
#   LLM_PROVIDER=google|anthropic|openai|rule-based
# and falls back to a deterministic rule-based classifier if no key is present,
# so a dead API degrades the wording rather than breaking the agent.
# ═══════════════════════════════════════════════════════════════════════

LLM_TEMPERATURE = 0          # deterministic intent classification

# ── Narration budget ──────────────────────────────────────────────────
# Each detailed hour block costs ONE LLM call. "Show me all hours" expands to
# 16+ blocks, which at ~500ms per call is 8+ seconds of sequential work -- past
# the client timeout, surfacing to the user as "Can't reach the agent".
#
# So narration is CAPPED. Beyond this many blocks the response uses the
# deterministic summaries throughout, which are complete and correct; only the
# phrasing is plainer. Consistency matters here: a response with three
# LLM-written summaries and thirteen deterministic ones reads as inconsistent,
# so it is all-or-nothing per response.
#
# 6 covers store_rca (3 blocks) and a typical hour_detail window (5), while
# excluding the show_all case that caused the failure.
MAX_NARRATED_HOURS = 6

# Narration calls are independent, so they run concurrently. Kept small to
# avoid tripping free-tier rate limits, which would cost more in retries than
# the parallelism saves.
NARRATION_CONCURRENCY = 4

# HARD CEILING on total narration time per turn.
#
# Without this the turn had no upper bound. A single narration call can take
# LLM_TIMEOUT_SECONDS x (1 + LLM_MAX_RETRIES) plus backoff -- about 93 seconds
# -- and as_completed() waited indefinitely. One slow or rate-limited call
# therefore stalled the whole response past the client's 45s timeout, which
# surfaced as "Can't reach the agent" with an empty body.
#
# Past this budget the remaining blocks keep their deterministic summaries.
# Nothing is lost but phrasing: every number is rendered before the model is
# called, so a skipped narration cannot change a figure.
#
# 10s, not 8s: measured cold-start latency is ~3.7s per call, and 6 blocks at
# concurrency 4 is two waves -- ~7.4s worst case for calls that are slow but
# working. The budget should only fire on genuine trouble, never on those.
NARRATION_BUDGET_SECONDS = 10.0

# ── Narration retries and timeout: deliberately different from the defaults ──
#
# Retries exist so a transient 429 does not DEGRADE an answer. That reasoning
# holds for CLASSIFICATION -- if it fails we drop to a regex classifier, which
# is genuinely worse.
#
# It does not hold for NARRATION. Every block already carries a complete,
# correct, deterministic summary. Retrying a narration call spends up to 93
# seconds and burns free-tier quota to avoid a purely cosmetic downgrade -- and
# the abandoned call keeps retrying in the background after the budget fires,
# making the NEXT turn more likely to be rate-limited too.
#
# So narration fails fast to its fallback. Classification keeps its retries.
NARRATION_MAX_RETRIES = 0

# A two-sentence summary should never take 8 seconds. The 30s default exists
# for long generations; applied here it just delays the inevitable fallback.
# 8s leaves ~2x headroom over the 3.7s cold-start worst case.
NARRATION_TIMEOUT_SECONDS = 8.0

# Retries apply only to TRANSIENT failures (429 rate limit, 5xx, timeout).
# Permanent failures (bad key, unknown model) fail fast -- retrying them just
# delays the same error.
#
# This matters here because the Gemini free tier allows ~10-15 requests/minute
# and the agent makes 2 calls per turn. A five-turn demo plus a few unscripted
# questions crosses that limit, and a 429 means "wait a second", not "give up".
LLM_MAX_RETRIES = 2
LLM_RETRY_BACKOFF_SECONDS = 1.0     # doubles each attempt: 1s, 2s

# If the server supplies its own retryDelay we honour it -- but only up to this
# cap. Google returns delays of 30-60s when a DAILY quota is exhausted, which no
# amount of backoff will outrun. In a live demo, falling back to the
# deterministic path instantly beats a minute of silence.
LLM_MAX_RETRY_WAIT_SECONDS = 5.0

# Applies to every provider. Without it a hung request never returns and never
# errors, so the turn stalls with no path to the fallback.
LLM_TIMEOUT_SECONDS = 30

# ── Groq (configured provider) ────────────────────────────────────────
# Chosen after Gemini proved unusable: this account was issued an 'AQ.' prefix
# key, and those are affected by a known Google restriction that provisions
# free-tier quota at ZERO -- every call returns
#   429 ... generate_content_free_tier_requests limit: 0
# which no amount of new keys or new projects resolves.
#
# Groq is free with no card, wire-compatible with the OpenAI API (so it reuses
# that code path), and fast -- which matters more than raw model quality here,
# since the model only classifies intent and writes one sentence.
#
# MODEL CHOICE — WHY NOT llama-3.3-70b-versatile
#
# That was the original choice and it worked. Groq DEPRECATED it on 2026-06-17,
# free and developer tier alike. A deprecated model still answers right up until
# it does not, and "it stopped mid-demo" is a failure with no recovery and no
# warning. Groq's own migration note points at openai/gpt-oss-120b as the
# replacement.
#
# Swapping is a one-line change precisely because everything model-specific
# lives behind the provider abstraction in app/llm.py -- Groq is wire-compatible
# with the OpenAI API, so this reuses that code path unchanged.
#
# Override with LLM_MODEL if this name is retired too; check_llm.py lists what
# the account can actually reach.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"

# The previous model, kept as a documented fallback rather than deleted: if the
# replacement misbehaves mid-demo, LLM_MODEL=<this> is a one-env-var revert.
GROQ_PREVIOUS_MODEL = "llama-3.3-70b-versatile"    # deprecated 2026-06-17

# ── Google (kept as an alternative) ───────────────────────────────────
GOOGLE_DEFAULT_MODEL = "gemini-2.0-flash"


# ═══════════════════════════════════════════════════════════════════════
# MCP  — reference documents over the Model Context Protocol
#
# We consume the official open-source filesystem server, pointed at docs/, so
# the agent answers "what is the pileup threshold?" from the RCA playbook
# itself rather than from the constants above.
#
# PLACEMENT RATIONALE: MCP sits on DOCUMENTS, not on the database. A SQL server
# would let the model write its own queries -- reintroducing exactly the
# non-determinism this build exists to remove, on a dataset with five measured
# traps that all fail silently (see app/mcp_client.py). Reading a document
# cannot produce a wrong number.
#
# The server runs under Node via npx. To avoid a download at demo time:
#     npm install -g @modelcontextprotocol/server-filesystem
# Swap MCP_SERVER_COMMAND for any other stdio MCP server; nothing else changes.
# ═══════════════════════════════════════════════════════════════════════

MCP_SERVER_COMMAND = ["npx", "-y", "@modelcontextprotocol/server-filesystem"]

# The docs directory is appended to the command at connect time.
MCP_DOC_FILES = [
    "quick_commerce_rca_logic.md",       # the playbook -- source of truth
    "quick_commerce_orders_gold.md",     # schema and data-quality notes
    "order_ready_to_assignment.md",      # OR2A metric definition
]

MCP_CONNECT_TIMEOUT_SECONDS = 20.0   # npx may fetch the package on first run
MCP_CALL_TIMEOUT_SECONDS = 10.0

# Attempt a connection at startup rather than on the first methodology
# question. A cold npx fetch mid-demo is a visible stall; failing early also
# surfaces a missing Node before it matters.
MCP_CONNECT_ON_STARTUP = True


# ═══════════════════════════════════════════════════════════════════════
# DATASET INVARIANTS  — verified true at time of analysis.
# db.validate() asserts these on load so that a future data swap fails loudly
# instead of producing silently wrong answers.
# ═══════════════════════════════════════════════════════════════════════

EXPECTED_ROW_COUNT = 3813
EXPECTED_STORE_COUNT = 233
EXPECTED_CITY_COUNT = 9
EXPECTED_NULL_OR2A_ROWS = 2       # STORE_008 hours 8, 9 -- 100% cancelled
EXPECTED_NULL_MANHOUR_ROWS = 139  # booked_size = 0
EXPECTED_PROBLEM_HOURS = 2202     # 57.7%

# Fail the load if an invariant breaks. Set False to warn instead.
STRICT_VALIDATION = True
