"""
LLM provider abstraction.

WHY AN ABSTRACTION RATHER THAN A DIRECT SDK CALL
-------------------------------------------------
Three reasons, in order of importance:

1. GRACEFUL DEGRADATION. The walkthrough is live. If the API key is wrong, the
   network is down, or the provider rate-limits mid-demo, the agent must still
   answer -- because every number is already computed deterministically and only
   the *prose* depends on the model. RuleBasedProvider below is that fallback:
   it handles the common phrasings with regex, so a dead API degrades the
   wording rather than breaking the demo.

2. TESTABILITY. The whole LangGraph flow can be unit-tested with no API key, no
   network, and no non-determinism. `test_turns.py` runs the full five-turn
   conversation in CI.

3. PROVIDER CHOICE. Anthropic vs OpenAI is one environment variable.

WHAT THE LLM IS AND IS NOT ALLOWED TO DO
-----------------------------------------
It does exactly two things:

  classify_intent  -- turn a sentence into {intent, entities} JSON
  narrate          -- turn already-computed facts into one summary sentence

It never computes, compares, aggregates, or decides a flag. Those all happened
in Modules 1 and 2, in tested Python. This is the reason the numbers cannot be
wrong because the model had a bad day -- and it is the single most important
architectural claim in the project.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Protocol

import config


@dataclass
class LLMResponse:
    text: str
    provider: str
    ok: bool = True
    error: str | None = None
    attempts: int = 1
    latency_ms: float = 0.0


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str, json_mode: bool = False,
                 max_retries: int | None = None,
                 timeout: float | None = None) -> LLMResponse:
        """
        `max_retries` and `timeout` are per-call overrides.

        They exist because the two callers have opposite needs. Classification
        should retry -- its fallback is a regex classifier, which is worse.
        Narration should not -- its fallback is a complete, tested sentence, so
        retrying spends up to 93 seconds and burns quota to avoid a cosmetic
        downgrade.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════
# RETRY
#
# Why this exists: the Gemini free tier allows roughly 10-15 requests per
# minute. The agent makes TWO calls per turn, so a five-turn walkthrough plus a
# few unscripted questions is ~16 calls in a few minutes -- comfortably over the
# limit. A 429 is not a failure, it is "slow down"; waiting one second and
# retrying clears it.
#
# Without this, one 429 drops the turn to the rule-based fallback. The numbers
# stay correct (they were computed before the model was called) but the prose
# visibly flattens mid-demo, for a condition that would have resolved in a
# second.
# ═══════════════════════════════════════════════════════════════════════

# ── Status codes are AUTHORITATIVE ──────────────────────────────────────
#
# BUG THIS FIXES: the first version matched keywords only, and listed "billing"
# as a permanent signal. Google's rate-limit response reads:
#
#   429 RESOURCE_EXHAUSTED ... "You exceeded your current quota, please check
#   your plan and BILLING details"
#
# so a textbook retryable 429 was classified permanent and failed instantly --
# 13 calls, 0 retries, in a live test. Prose-only keyword matching is not
# reliable enough for this; the HTTP status is.
_TRANSIENT_CODES = ("429", "500", "502", "503", "504", "408")
_PERMANENT_CODES = ("400", "401", "403", "404", "405", "422")

# Keywords are a FALLBACK for SDK errors that carry no status code at all
# (socket resets, DNS failures, client-side timeouts).
_TRANSIENT_WORDS = (
    "rate limit", "rate_limit", "resource_exhausted", "overloaded",
    "unavailable", "internal error", "timeout", "timed out", "deadline",
    "connection reset", "connection aborted", "connection error",
    "temporarily", "try again", "retry",
)
_PERMANENT_WORDS = (
    "invalid api key", "api key not valid", "api_key_invalid",
    "unauthorized", "permission denied", "permission_denied",
    "not found", "invalid_argument", "unsupported", "unregistered callers",
)


def _has_code(error: str, codes: tuple) -> bool:
    """Match a status code as a standalone token, not as a substring of a number."""
    return any(re.search(rf"(?<!\d){c}(?!\d)", error) for c in codes)


def is_transient(error: str) -> bool:
    """
    Should this failure be retried?

    Order matters and is deliberate:
      1. explicit status code  -- authoritative, checked first
      2. keyword heuristics    -- only for errors with no status code

    Getting this wrong in either direction is costly: treating a 429 as
    permanent loses the retry that would have saved the turn, while treating a
    bad API key as transient wastes 3 seconds before the identical failure.
    """
    if not error:
        return False
    e = error.lower()

    # 1. status code wins
    if _has_code(e, _TRANSIENT_CODES):
        return True
    if _has_code(e, _PERMANENT_CODES):
        return False

    # 2. keyword fallback
    if any(p in e for p in _PERMANENT_WORDS):
        return False
    return any(t in e for t in _TRANSIENT_WORDS)


_RETRY_DELAY_PATTERNS = (
    # Google:  retryDelay: '31s'
    re.compile(r"retry[_-]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", re.I),
    # OpenAI:  Please retry after 20 seconds
    re.compile(r"retry\s+after\s+(\d+(?:\.\d+)?)\s*s", re.I),
)

# Groq:  "Please try again in 8m24.575999999s"  /  "in 1h2m3s"  /  "in 30s"
_GROQ_DELAY = re.compile(
    r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s", re.I)


def parse_retry_delay(error: str) -> float | None:
    """
    How long the server says to wait, in seconds.

    WHY THE FORMATS MATTER MORE THAN THEY LOOK
    ------------------------------------------
    This is what decides "retry" vs "give up now". Below the threshold we wait;
    above it we fall back immediately, because a quota that resets in eight
    minutes is not something exponential backoff can outrun.

    Only Google's format was understood. Groq writes

        Please try again in 8m24.575999999s

    which parsed as None, so the "not worth waiting" branch never ran and an
    exhausted DAILY quota was retried twice with backoff -- three failed calls
    and three seconds of sleeping for a wall that would not move for eight
    minutes. The guard was correct; it was simply never reached.

    A compound duration must be parsed as a whole: taking the trailing "24.5s"
    from "8m24.5s" yields 24 seconds, which is under the threshold and would
    have produced exactly the pointless retry this exists to prevent.
    """
    m = _GROQ_DELAY.search(error)
    if m:
        hours, minutes, seconds = m.group(1), m.group(2), m.group(3)
        return (float(hours or 0) * 3600
                + float(minutes or 0) * 60
                + float(seconds))

    for pattern in _RETRY_DELAY_PATTERNS:
        m = pattern.search(error)
        if m:
            return float(m.group(1))
    return None


def _with_retry(call, provider_name: str, max_retries: int | None = None) -> LLMResponse:
    """
    Run `call`, retrying transient failures with exponential backoff.

    Backoff doubles (1s, 2s) rather than retrying instantly: hammering a server
    that has just asked you to slow down keeps you rate-limited. Worst case this
    costs 3 seconds; the common case costs nothing.

    Never raises. A provider that throws is converted into a failed
    LLMResponse, because the graph must never crash on model behaviour.
    """
    retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries
    started = time.time()
    last: LLMResponse | None = None

    for attempt in range(1, retries + 2):        # 1 initial try + N retries
        try:
            resp = call()
        except Exception as e:                   # provider raised instead of returning
            resp = LLMResponse(text="", provider=provider_name, ok=False, error=str(e))

        resp.attempts = attempt
        resp.latency_ms = (time.time() - started) * 1000

        if resp.ok:
            return resp

        last = resp
        err = resp.error or ""
        if not is_transient(err):
            return resp                          # permanent -- fail fast
        if attempt > retries:
            return resp                          # out of attempts

        # Prefer the server's own advice over our schedule, but cap it: a
        # 60-second retryDelay is correct for a daily quota and useless in a
        # live demo, where falling back immediately is the better outcome.
        suggested = parse_retry_delay(err)
        wait = config.LLM_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
        if suggested is not None:
            if suggested > config.LLM_MAX_RETRY_WAIT_SECONDS:
                return resp                      # not worth waiting -- fall back now
            wait = suggested

        time.sleep(wait)

    return last or LLMResponse(text="", provider=provider_name, ok=False,
                               error="exhausted retries")


@dataclass
class Telemetry:
    """
    Per-provider call counters.

    Cheap to keep and useful twice: it turns "the demo felt slow" into a number,
    and it lets the walkthrough state plainly how much work the model actually
    did -- e.g. "11 calls, 2 retried, 780ms average, and none of them produced a
    number".
    """
    calls: int = 0
    failures: int = 0
    retries: int = 0
    fallbacks: int = 0
    total_ms: float = 0.0

    def record(self, resp: LLMResponse) -> None:
        self.calls += 1
        self.total_ms += resp.latency_ms
        self.retries += max(0, resp.attempts - 1)
        if not resp.ok:
            self.failures += 1

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0

    def summary(self) -> str:
        return (f"{self.calls} calls, {self.retries} retried, "
                f"{self.failures} failed, {self.avg_ms:.0f}ms avg")


TELEMETRY = Telemetry()


# ═══════════════════════════════════════════════════════════════════════
# ANTHROPIC
# ═══════════════════════════════════════════════════════════════════════

class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str | None = None):
        import anthropic  # imported lazily so the package is optional
        self._client = anthropic.Anthropic(api_key=api_key,
                                           timeout=config.LLM_TIMEOUT_SECONDS)
        self.model = model or "claude-sonnet-4-20250514"

    def complete(self, system: str, user: str, json_mode: bool = False,
                 max_retries: int | None = None,
                 timeout: float | None = None) -> LLMResponse:
        def _call() -> LLMResponse:
            # Prefilling '{' is the reliable way to force JSON-only output from
            # a chat model -- more dependable than asking politely in the prompt.
            messages = [{"role": "user", "content": user}]
            if json_mode:
                messages.append({"role": "assistant", "content": "{"})

            kwargs = {"timeout": timeout} if timeout is not None else {}
            r = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=config.LLM_TEMPERATURE,
                system=system,
                messages=messages,
                **kwargs,
            )
            text = r.content[0].text
            if json_mode:
                text = "{" + text
            return LLMResponse(text=text, provider=self.name)

        resp = _with_retry(_call, self.name, max_retries=max_retries)
        TELEMETRY.record(resp)
        return resp


# ═══════════════════════════════════════════════════════════════════════
# OPENAI
# ═══════════════════════════════════════════════════════════════════════

class OpenAIProvider:
    """
    OpenAI, and anything speaking its API.

    `base_url` exists because several providers are wire-compatible with the
    OpenAI chat-completions API -- Groq among them. Sharing this implementation
    means one code path, one set of retry semantics, and one place for a bug to
    be fixed, rather than a near-duplicate class per vendor.
    """
    name = "openai"

    def __init__(self, api_key: str, model: str | None = None,
                 base_url: str | None = None):
        from openai import OpenAI  # lazy
        kwargs = {"api_key": api_key, "timeout": config.LLM_TIMEOUT_SECONDS}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = model or "gpt-4o-mini"

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self._client.models.list())
        except Exception as e:
            return [f"<could not list: {e}>"]

    def complete(self, system: str, user: str, json_mode: bool = False,
                 max_retries: int | None = None,
                 timeout: float | None = None) -> LLMResponse:
        def _call() -> LLMResponse:
            kwargs = {}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if timeout is not None:
                # Per-request override. The client-level timeout is set at
                # construction and applies to everything; narration needs a
                # tighter one so it fails fast to its deterministic fallback.
                kwargs["timeout"] = timeout
            r = self._client.chat.completions.create(
                model=self.model,
                temperature=config.LLM_TEMPERATURE,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                **kwargs,
            )
            return LLMResponse(text=r.choices[0].message.content, provider=self.name)

        resp = _with_retry(_call, self.name, max_retries=max_retries)
        TELEMETRY.record(resp)
        return resp


# ═══════════════════════════════════════════════════════════════════════
# GROQ
# ═══════════════════════════════════════════════════════════════════════

class GroqProvider(OpenAIProvider):
    """
    Groq — the configured provider for this build.

    WHY GROQ RATHER THAN GEMINI
    ---------------------------
    Gemini was the original choice, but this account hit a known Google
    platform restriction: keys issued with the 'AQ.' prefix are provisioned
    with free-tier quota set to ZERO, so every generate_content call returns

        429 RESOURCE_EXHAUSTED ... generate_content_free_tier_requests limit: 0

    That is account-level and not fixable by creating new keys or projects.

    Groq is a good replacement for this specific workload:
      - free tier, no credit card
      - wire-compatible with the OpenAI API, so it reuses the code path above
      - very fast inference, which matters here because latency (two calls per
        turn) is what the demo audience actually feels

    The swap cost one config entry and this subclass. Everything else -- the
    graph, the prompts, the retry layer, the entire RCA engine -- is untouched,
    which is the point of having the abstraction.
    """
    name = "groq"

    def __init__(self, api_key: str, model: str | None = None,
                 base_url: str | None = None):
        super().__init__(
            api_key,
            model or config.GROQ_DEFAULT_MODEL,
            base_url or config.GROQ_BASE_URL,
        )


# ═══════════════════════════════════════════════════════════════════════
# GOOGLE GEMINI
# ═══════════════════════════════════════════════════════════════════════

class GoogleProvider:
    """
    Google Gemini.

    Chosen for this build: the free tier is generous enough for a take-home,
    and the Flash models are fast -- which matters more than raw quality here,
    because the model only classifies intent and writes one summary sentence.
    Every number is computed before it is ever called, so latency is the
    binding constraint on demo feel, not reasoning depth.

    Supports both the current `google-genai` SDK and the older
    `google-generativeai` package, since which one is installed varies.
    """
    name = "google"

    def __init__(self, api_key: str, model: str | None = None):
        self.model = model or config.GOOGLE_DEFAULT_MODEL
        self._new_sdk = True
        try:
            from google import genai                       # google-genai
            self._genai = genai
            # Timeout is in MILLISECONDS in this SDK. Without it a hung request
            # never returns and never errors -- the turn stalls forever and the
            # fallback never fires, because nothing ever reports a failure. That
            # is the one failure mode with no graceful exit, so it must be set.
            self._client = genai.Client(
                api_key=api_key,
                http_options={"timeout": int(config.LLM_TIMEOUT_SECONDS * 1000)},
            )
        except ImportError:
            import google.generativeai as genai            # google-generativeai
            genai.configure(api_key=api_key)
            self._genai = genai
            self._client = None
            self._new_sdk = False

    def complete(self, system: str, user: str, json_mode: bool = False,
                 max_retries: int | None = None,
                 timeout: float | None = None) -> LLMResponse:
        def _call() -> LLMResponse:
            if self._new_sdk:
                cfg = self._genai.types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=config.LLM_TEMPERATURE,
                    # Native JSON mode -- more reliable than asking the model to
                    # "respond with JSON only", which it will sometimes wrap in
                    # prose or a code fence anyway.
                    response_mime_type="application/json" if json_mode else "text/plain",
                )
                r = self._client.models.generate_content(
                    model=self.model, contents=user, config=cfg)
                return LLMResponse(text=r.text or "", provider=self.name)

            gm = self._genai.GenerativeModel(self.model, system_instruction=system)
            r = gm.generate_content(
                user,
                generation_config={
                    "temperature": config.LLM_TEMPERATURE,
                    "response_mime_type": "application/json" if json_mode else "text/plain",
                },
                request_options={"timeout": config.LLM_TIMEOUT_SECONDS},
            )
            return LLMResponse(text=r.text or "", provider=self.name)

        resp = _with_retry(_call, self.name, max_retries=max_retries)
        TELEMETRY.record(resp)
        return resp

    def list_models(self) -> list[str]:
        """Available models, for verifying the configured name works."""
        try:
            if self._new_sdk:
                return [m.name for m in self._client.models.list()]
            return [m.name for m in self._genai.list_models()]
        except Exception as e:
            return [f"<could not list: {e}>"]


# ═══════════════════════════════════════════════════════════════════════
# RULE-BASED  — fallback and test double
# ═══════════════════════════════════════════════════════════════════════

# Ordered most-specific first: "show me all hours" must beat the generic
# store_rca pattern, and "how many" must beat "how did".
_INTENT_PATTERNS: list[tuple[str, str]] = [
    # "What about X?" / "And X?" -- no verb, no metric. Means "same question,
    # different entity". Must be matched FIRST, before any keyword pattern.
    ("follow_up",    r"^\s*(what|how)\s+about\b|^\s*and\s+\w+\s*\?*\s*$"),
    # "show me the morning hours at X" is an hour WALKTHROUGH, not an expansion.
    # Checked before show_all so a named window wins over the word "show".
    ("hour_detail",  r"\b(show|give|list|see|display)\b.*\b(morning|afternoon|"
                     r"evening|night|\d{1,2}\s*(am|pm)|hour\s*\d{1,2})\b"),
    # show_all means "expand the previous answer to every HOUR".
    #
    # The negative lookahead is load-bearing: without it, "give me all CITIES
    # from worst to best" matched here and was routed as an hour expansion,
    # which then demanded a store. "all" alone is not the signal -- "all
    # hours" is. A following level noun means the user is ranking, not
    # expanding.
    ("show_all",     r"\b(show|list|see|give)\s+(me\s+)?(all|every|the full|"
                     r"the rest|the remaining|remaining)\b"
                     r"(?!\s+(?:\w+\s+)?(cit(y|ies)|stores?))"),
    ("show_all",     r"\b(all|every)\s+(the\s+)?hours?\b|\bexpand\b|"
                     r"\bthe rest of (the\s+)?(them|hours)\b"),
    # Doc questions: "what's the pileup threshold", "how is OR2A defined",
    # "how do you calculate the breach rate". The verb and the subject can be
    # separated by several words, so these are not adjacency patterns.
    ("methodology",  r"\bthreshold\b|\bplaybook\b|\bmethodology\b|\bformula\b|"
                     r"\bhow (is|are|do you|does)\b.*\b(defined?|calculated?|computed?|"
                     r"measured?|derived?|work(s)?)\b|"
                     r"\bwhat (is|are|does)\b.*\b(mean|defined?|formula|definition)\b|"
                     r"\bwhere does .* come from\b|"
                     # A BARE "what is X" IS A DEFINITION QUESTION.
                     #
                     # "what is or2a?" fell through to the `lookup` pattern
                     # below -- which matches "what is" plus a metric word --
                     # and was answered as a data lookup with no entity:
                     #
                     #     **No data for None**
                     #
                     # A definition question and a value question differ by
                     # whether an ENTITY is named. "What is OR2A" asks what the
                     # metric means; "what was STORE_003's OR2A" asks for a
                     # number. The negative lookahead is that distinction: no
                     # store, no city, no possessive, no date -> a definition.
                     r"\bwhat(?:'?s| is| are)\s+(?:the\s+|a\s+|an\s+)?"
                     r"(?:or2a|pileup|breach(?:ed)?\s*rate|man.?hour|"
                     r"utilization|utilisation|no.?supply|booking\s*gap|"
                     r"demand\s*spike|slot|rca)\b"
                     r"(?!.*\b(?:store|city|cities|at|for|in|on)\b)"),
    ("count",        r"\bhow many\b"),
    # "which one did better" is the natural second half of a comparison and
    # carries no entity names -- the graph reuses the previous comparison.
    ("compare",      r"\b(compare|versus|vs\.?|against|difference between)\b|"
                     r"\bwhich (one|store|city|of them)\b.*\b(better|worse|won)\b|"
                     r"\bwho (did|performed)\s+(better|worse)\b|"
                     r"\bwhich .*(did|performed)\s+better\b"),
    ("rank",         r"\b(worst|best|top|bottom|which (city|store).*(most|least|highest|lowest)|"
                     r"most|least|highest|lowest|rank)\b"),
    # Bare listing requests: "list all cities", "show me all stores in Mumbai".
    # No superlative, so the patterns above miss them, and they used to fall
    # through to city_summary or out_of_scope. A listing IS a ranking with no
    # stated direction.
    ("rank",         r"\b(show|list|give|display|name)\s+(me\s+)?(all|every|the)?\s*"
                     r"(the\s+)?(cit(y|ies)|stores?)\b"),
    ("hour_profile", r"\bwhat happened (at|during)\b|\bacross (all|the)?\s*stores\b"),
    ("hour_detail",  r"\b(walk me through|break ?down|hour by hour|during the)\b|"
                     r"\b(morning|afternoon|evening|night)\s+hours?\b"),
    ("lookup",       r"\bwhat (was|were|is)\b.*\b(rate|or2a|orders|no-?shows?|count)\b"),
    ("store_rca",    r"\b(why|what (went )?wrong|underperform|root cause|rca|diagnos)\b"),
    ("city_summary", r"\bhow did\b|\bhow was\b|\bperformance of\b|\bsummar"),
]

_HOUR_WORDS = r"(morning|noon|midday|afternoon|evening|night)"
_STORE_RE = re.compile(r"\b(store[\s_\-]*\d{1,4}|[A-Z]{2,4}\d{1,3})\b", re.I)
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# B7 -- BARE STORE NUMBERS.
#
# "How did 054 do?" named a store; the classifier extracted nothing and the
# answer came back as a whole-city summary. entities.resolve_store("054")
# already returns STORE_054 -- the classifier simply never handed it over.
#
# But a number in a sentence can mean three different things, and treating
# every number as a store is unusable:
#
#     "worst 5 stores"    5  is a COUNT      -> would become STORE_005
#     "hours 8-11"        8,11 are HOURS     -> would become STORE_008/011
#     "at 7pm"            7  is a CLOCK TIME -> would become STORE_007
#
# So a bare number is read as a store ONLY when it is not sitting in one of
# those positions. The patterns below mark the positions to skip.
_NUMBER_IS_NOT_A_STORE = re.compile(
    r"\b(?:worst|best|top|bottom|last|first|show|give|list)\s+\d{1,4}\b"   # a count
    r"|\b\d{1,4}\s+(?:stores?|cit(?:y|ies)|hours?|orders?|min)\b"          # a count/unit
    r"|\bhours?\s+\d{1,2}\b"                                              # "hour 19"
    r"|\b\d{1,2}\s*[-–]\s*\d{1,2}\b"                                      # "8-11"
    r"|\b\d{1,2}\s*(?:am|pm)\b"                                           # "7pm"
    r"|\b\d{4}-\d{2}-\d{2}\b"                                             # a date
    # A THRESHOLD VALUE IS NOT A STORE CODE.
    #
    # "stores with breach rate over 50%" resolved 50 to STORE_050 and pinned it
    # as the conversation's store. Nothing said so. Three turns later "evening
    # hours" failed, because the session held a Bangalore store while the user
    # had moved to Mumbai -- and the real cause was four turns back, invisible.
    #
    # A number governed by a comparator, or carrying a unit, is a QUANTITY. It
    # was never a name.
    r"|\b(?:over|under|above|below|more\s+than|less\s+than|fewer\s+than|"
    r"greater\s+than|higher\s+than|lower\s+than|at\s+least|at\s+most|"
    r"exceed(?:ing|s)?|within|between)\s+\d{1,4}\b"
    r"|\b\d{1,4}\s*(?:%|percent|pct)\b"                                   # "50%"
    r"|\b\d+(?:\.\d+)?\s*(?:min|mins|minutes?|hrs?|hours?)\b",            # "30 min"
    re.I)
_BARE_NUMBER_RE = re.compile(r"\b(\d{1,4})\b")


# ═══════════════════════════════════════════════════════════════════════
# GRAIN FACTS — computed for EVERY provider, never asked of the model
# ═══════════════════════════════════════════════════════════════════════

# "of each city" and "per city" are the two commonest phrasings and were both
# missing, so "top 5 stores of each city" never set the scope veto -- it hit the
# compound path and returned a city table AND a store table.
#
# "per" needs no article ("per city"), the rest usually take one, so the article
# group stays optional for all of them.
_CITY_AS_SCOPE_RE = re.compile(
    r"\b(in|across|over|for|of|per|throughout|among|within|by)\s+"
    r"(all|every|each|the)?\s*cit(y|ies)\b"
    r"|\b(everywhere|nationwide|network[\s-]?wide)\b")

# GROUPED RANKING: "top 5 stores PER city" asks for five per city, not five
# overall. Reported as a fact; Agent._rank decides it means a window function.
_PER_GROUP_RE = re.compile(
    r"\b(per|each|every|within each|for each|in each|by)\s+"
    r"(cit(y|ies)|stores?)\b"
    r"|\bbreak(?:down|\s+down)\s+by\b|\bgrouped?\s+by\b")

_WANT_ALL_RE = re.compile(r"\b(all|every|complete|full list|each)\b")

# ── THRESHOLD QUESTIONS ───────────────────────────────────────────────
#
# "stores with breach rate over 50%", "cities under 30 min".
#
# Parsed deterministically for the same reason as the grain facts: this is
# reading a number and a comparator out of a string. Python does it exactly;
# a model does it slowly and occasionally drops the "not" in "not more than".
_METRIC_PHRASES = [
    (r"breach(?:ed)?\s*rate|breach\s*%|%\s*breach",      "w_breached_rate"),
    (r"or2a|assignment\s*time|ready.to.assign",          "w_avg_or2a"),
    (r"problem\s*hours?",                                 "problem_hours"),
    (r"pileup\s*hours?",                                  "pileup_hours"),
    (r"no.?supply\s*hours?|zero.rider\s*hours?",          "no_supply_hours"),
    (r"no.?shows?",                                       "noshow_count"),
    (r"breached\s*orders?|orders?\s*breached",            "breached_count"),
    (r"orders?",                                          "total_orders"),
]
_ABOVE = r"(?:above|over|more than|greater than|higher than|exceed(?:ing|s)?|>=?|at least)"
_BELOW = r"(?:below|under|less than|lower than|fewer than|<=?|at most|within)"
# NEGATION MUST BE CAPTURED, NOT IGNORED.
#
# "stores with breach rate NOT MORE THAN 50%" parsed as ">" -- the exact
# opposite of the question -- because `_ABOVE` matched "more than" and the "not"
# sat harmlessly in front of it. The agent then returned the 14 WORST stores to
# someone asking for the compliant ones, with no indication anything was lost.
#
# A silent inversion is the worst failure available here: every row is real,
# the table looks right, and the answer is backwards.
_NEGATION = r"(?:not|no|isn'?t|aren'?t)\s+"
_THRESHOLD_RE = re.compile(
    rf"\b(?P<neg>{_NEGATION})?(?P<cmp>{_ABOVE}|{_BELOW})\s*"
    rf"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>%|min|minutes?|hours?)?",
    re.I)

# "morning vs evening", "by time of day", "across the day" -- a comparison of
# named hour windows, which is neither an hour walkthrough (one window, hour by
# hour) nor a ranking.
_WINDOW_COMPARE_RE = re.compile(
    r"\bby time of day\b|\bacross the day\b|\btime[- ]of[- ]day\b"
    r"|\b(morning|noon|afternoon|evening|night)\b[^.?!]*\b(vs\.?|versus|compared to|"
    r"against|or)\b[^.?!]*\b(morning|noon|afternoon|evening|night)\b"
    r"|\bwhich (?:part|time) of the day\b|\bbusiest time\b|\bworst time of day\b",
    re.I)

# The dataset is a SINGLE DAY. Anything asking about change over time cannot be
# answered, and must say so rather than describe one day as if it were a trend.
_TREND_RE = re.compile(
    r"\b(trend|over time|getting (?:better|worse)|improv(?:ing|ed|ement)|"
    r"declin(?:ing|ed)|week over week|month over month|wow|mom|"
    r"compared to (?:last|yesterday|previous)|last week|yesterday|"
    r"since|history|historical|forecast next|predict)\b", re.I)

# AN UNKNOWN PLACE MUST BE DECLINED, NOT DROPPED.
#
# "worst 5 stores in Atlantis" ranked the ENTIRE NETWORK. The classifier only
# proposes a city when it recognises the name, so "Atlantis" was extracted by
# nothing, the scope stayed empty, and the answer came back confident and
# global -- to a question about a place that does not exist.
#
# This is the TBC8 failure exactly, on the city path: a named entity that
# cannot be resolved must produce a decline, never a silently widened answer.
# The store path already gets this right.
#
# Words that legitimately follow "in"/"at"/"for" and are NOT places. Kept
# explicit rather than inferred, because a false positive here declines a valid
# question -- the one outcome worse than the bug being fixed.
_NOT_A_PLACE = {
    "all", "every", "each", "the", "a", "an", "any", "both", "this", "that",
    "these", "those", "my", "our", "their", "its",
    "city", "cities", "store", "stores", "hour", "hours", "slot", "slots",
    "morning", "noon", "midday", "afternoon", "evening", "night", "midnight",
    "am", "pm", "today", "yesterday", "tomorrow", "general", "total", "detail",
    "which", "what", "who", "them", "it", "one", "order", "orders", "riders",
    "rider", "minutes", "min", "mins", "terms", "comparison", "short", "fact",
    "scope", "place", "question", "example", "particular", "line", "time",
    "times", "day", "days", "week", "month", "breach", "breaches", "sla",
}
_PLACE_AFTER_PREP = re.compile(
    r"\b(?:in|at|for|across|within|around)\s+([a-z][a-z .\-]{2,30}?)"
    r"(?=\s*(?:$|[,.?!;]|\bwith\b|\bwhich\b|\bthat\b|\band\b|\bon\b|\bby\b|"
    r"\bover\b|\bunder\b|\bhad\b|\bhas\b|\bfrom\b))",
    re.I)

# A SHARE-OF-A-WHOLE QUESTION IS NOT A RANKING.
#
# "what percentage of total orders came from the 10 busiest stores" was
# classified as `rank` -- reasonably, it does contain "10 busiest stores" -- and
# answered with a table of ten stores and NO PERCENTAGE ANYWHERE. The question
# was never addressed, and nothing said so.
#
# Ranking answers "which are the extremes". This asks what fraction of a total
# a subset accounts for: a concentration measure. No tool computes it, so it
# belongs in Tier 3, where a query can be generated and shown.
#
# Deliberately narrow. It requires BOTH a ratio word AND a whole-vs-subset
# phrase, so "what percentage of orders breached at STORE_003" -- which the
# breach-rate metric answers exactly, with `validated` provenance -- is left
# alone. Routing a question to Tier 3 that a tested tool could answer is a
# downgrade, not a fix.
_SHARE_RE = re.compile(
    # '%' is not a word character, so \b%\b can never match after a space.
    # Alternated separately rather than folded into the word group.
    r"(?:\b(?:percentage|percent|share|proportion|fraction)\b|%)"
    r"(?=.*\b(of\s+(?:the\s+)?(?:total|all|overall|network)"
    r"|came\s+from|come\s+from|contributed?\s+by|account(?:ed|s)?\s+for"
    r"|concentrat|cumulative)\b)"
    r"|\b(?:what|how much)\s+(?:%|percent(?:age)?|share|proportion)\b"
    r"(?=.*\b(?:top|busiest|biggest|largest|worst|best)\s+\d+\b)",
    re.I)

_HEALTHY_RE = re.compile(
    r"\b(health(?:y|iest)|doing well|performing well|no (?:issues|problems)|"
    r"in good shape|fine|okay|ok)\b", re.I)


def threshold_facts(low: str) -> dict | None:
    """
    Pull `metric comparator value` out of a threshold question, or None.

    Percentages are normalised to the fraction the column actually stores:
    w_breached_rate is 0.0-1.0, so "over 50%" must become 0.5, not 50. Getting
    this wrong returns zero rows and looks like "nothing matched" rather than
    like a bug -- a silent-wrong-answer of exactly the kind this project keeps
    hunting, so it is asserted in the tests.
    """
    m = _THRESHOLD_RE.search(low)
    if not m:
        return None

    raw = m.group("cmp").lower()
    if re.fullmatch(_ABOVE, raw, re.I):
        comparator = ">=" if raw in (">=", "at least") else ">"
    else:
        comparator = "<=" if raw in ("<=", "at most", "within") else "<"

    # "not more than 50" means "<= 50", not "> 50". Negating flips the direction
    # AND the strictness: the excluded boundary becomes included, because "not
    # more than 50" includes exactly 50 while "more than 50" excludes it.
    if m.group("neg"):
        comparator = {">": "<=", ">=": "<", "<": ">=", "<=": ">"}[comparator]

    value = float(m.group("val"))
    unit = (m.group("unit") or "").lower()

    metric = None
    for pattern, name in _METRIC_PHRASES:
        if re.search(pattern, low, re.I):
            metric = name
            break

    # The unit disambiguates when no metric phrase was named: "% " means a rate,
    # "min" means OR2A. Without this, "stores over 50%" has no metric at all.
    if metric is None:
        metric = "w_breached_rate" if unit == "%" else \
                 "w_avg_or2a" if unit.startswith("min") else None
    if metric is None:
        return None

    if metric == "w_breached_rate" and value > 1:
        value /= 100.0

    return {"metric": metric, "comparator": comparator, "value": value}

# An explicitly stated number. "top 5 stores of each city" contains BOTH a
# number and the word "each" -- and "each" used to win, so the 5 was discarded
# and all 97 rows came back. A number the user typed is the least ambiguous
# signal in the message; nothing infers over it.
_EXPLICIT_N_RE = re.compile(
    r"\b(?:top|bottom|worst|best|first|last|show|give|list)\s+(\d{1,3})\b"
    r"|\b(\d{1,3})\s+(?:worst|best|top|bottom)\b")


def _unknown_place(low: str, cities: list[str]) -> str | None:
    """
    A place name after "in"/"at"/"for" that resolves to nothing. None if fine.

    Reported as a FACT. `resolve_entities` decides it means "decline", the same
    way an unresolvable store does -- this function never decides anything.

    Deliberately conservative: it fires only on a word that survives every
    filter below. A false positive rejects a valid question, which is worse
    than the bug it prevents, so anything ambiguous is left alone.
    """
    known = {c.lower() for c in cities}
    known |= {a.lower() for a in config.CITY_ALIASES}
    known |= {v.lower() for v in config.CITY_ALIASES.values()}

    for raw in _PLACE_AFTER_PREP.findall(low):
        candidate = raw.strip(" .-")
        if not candidate or candidate in _NOT_A_PLACE:
            continue
        if any(w in _NOT_A_PLACE for w in candidate.split()):
            continue        # "all cities", "the morning" etc.
        if candidate in known or any(candidate in k or k in candidate
                                     for k in known):
            continue        # a real city, however partially written
        if re.search(r"\d", candidate) or "store" in candidate:
            continue        # a store code or an hour, handled elsewhere
        return candidate
    return None


def grain_facts(low: str, cities: list[str]) -> dict:
    """
    Which grain nouns appeared in the message, and how.

    WHY THIS IS A FUNCTION AND NOT A PROMPT FIELD
    ---------------------------------------------
    It used to live inside RuleBasedProvider._classify, which meant it ran ONLY
    when the model failed. `mentions_store` and `mentions_city` are not in the
    classify prompt at all, so on the LLM path -- the path that actually serves
    users -- `data.get("mentions_store")` was always None, and every handler
    that consults it saw False.

    So the A2 fix, and the follow_up re-route built on top of it, worked in the
    test suite (rule-based) and silently did nothing in production (Groq). Two
    different causes, one identical symptom -- which is why "what about all
    stores" still repeated the previous answer live.

    These are LEXICAL FACTS about a string: did the word "stores" appear, was a
    city noun governed by a preposition. Python answers them exactly, for free,
    every time. Asking a model is slower, costs a token budget, varies between
    runs, and can be wrong. There is no version of this where the model is the
    better tool -- so it is computed here and OVERLAID on whatever any provider
    returned, rather than merged or trusted.

    Recognised city names are stripped first: "Hyderabad city" and "Pune city
    east" literally contain the word "city", so without this, naming one of
    those two cities reads as "asking about cities in general".
    """
    without_city_names = low
    for city_name in cities:
        without_city_names = without_city_names.replace(city_name.lower(), " ")
    for alias in config.CITY_ALIASES:
        if " city" in alias or " east" in alias or " north" in alias:
            without_city_names = without_city_names.replace(alias, " ")

    mentions_city = bool(re.search(r"\bcit(y|ies)\b", without_city_names))
    mentions_store = bool(re.search(r"\bstores?\b", without_city_names))

    # C2 -- SCOPE MODIFIER vs SECOND SUBJECT.
    #
    #     "all stores IN all cities"       -> scope   -> one store table
    #     "worst 5 stores ACROSS cities"   -> scope   -> one store table
    #     "the best city AND store"        -> subject -> two tables
    #
    # Both mention stores and cities; only the preposition separates them.
    city_as_scope = bool(_CITY_AS_SCOPE_RE.search(without_city_names))

    # "top 5 stores PER city" is neither one ranking nor two -- it is one
    # ranking run inside each group. Only meaningful when both nouns appear.
    per_group = bool(_PER_GROUP_RE.search(without_city_names)) and mentions_store

    m = _EXPLICIT_N_RE.search(low)
    explicit_n = int(m.group(1) or m.group(2)) if m else None

    # "best 0 stores" is not a request for zero rows -- it is a typo or a probe.
    # Treated as unstated so the default applies, rather than rendering an empty
    # table that looks like "no stores qualify".
    if explicit_n is not None and explicit_n < 1:
        explicit_n = None

    return {
        "mentions_city": mentions_city,
        "mentions_store": mentions_store,
        "city_as_scope": city_as_scope,
        "per_group": per_group,
        "explicit_n": explicit_n,
        # "top 5 stores" is genuinely ambiguous -- top of the leaderboard, or
        # top of the problem list? We read it as WORST, because that is what it
        # means in an operations review. But an assumption the user cannot see
        # is the same as a silent one, so the handler discloses it and names
        # the unambiguous phrasing. Reported as a fact; the handler decides.
        "said_top": bool(re.search(r"\btop\b", low)),
        "threshold": threshold_facts(low),
        "wants_healthy": bool(_HEALTHY_RE.search(low)),
        "wants_windows": bool(_WINDOW_COMPARE_RE.search(low)),
        "wants_trend": bool(_TREND_RE.search(low)),
        "unknown_place": _unknown_place(low, cities),
        "wants_share": bool(_SHARE_RE.search(low)),
        # A scope modifier is not a second subject, so it must not reach the
        # compound-question path. Nor is a per-group request.
        "both_levels": (mentions_city and mentions_store
                        and not city_as_scope and not per_group),
        # An explicit number outranks a quantifier: "top 5 ... of each city"
        # states both, and the 5 is what the user actually asked for.
        "want_all": bool(_WANT_ALL_RE.search(low)) and explicit_n is None,
    }


# ═══════════════════════════════════════════════════════════════════════
# DETERMINISTIC QUESTION SPLITTING
# ═══════════════════════════════════════════════════════════════════════

# Split points, in the order they are tried. Each is a separator that a second
# QUESTION starts after -- not merely a conjunction.
#
# The distinction that matters: "and STORE_203" continues a request, "and which
# store" starts one. So every alternative below requires an interrogative or an
# imperative verb AFTER the connector. That single rule is what keeps "compare
# STORE_003 and STORE_203" intact.
_SPLIT_RE = re.compile(
    r"\s*(?:[;]|\?\s+|,?\s*(?:and|also|then|plus|as well as|after that)\s+)"
    r"(?=(?:also\s+)?(?:what|which|who|when|where|why|how|show|give|tell|list|"
    r"compare|walk|is|are|do|does|did|can)\b)",
    re.I)

# A fragment that is only a connective or too short to carry a question.
_TRIVIAL = re.compile(r"^\W*(?:and|also|then|plus|ok|okay)?\W*$", re.I)


def rule_split(text: str, max_questions: int | None = None) -> list[str]:
    """
    Split a message into questions without a model.

    Used by RuleBasedProvider so the feature behaves identically with no API key
    -- and so the test suite exercises the real code path rather than a stub.

    Conservative by construction: it only splits where a connector is followed
    by a word that can BEGIN a question. Anything else stays whole, which is the
    safe failure. Over-splitting destroys the user's question; under-splitting
    merely leaves the previous behaviour in place.
    """
    max_questions = max_questions or config.MAX_SUBQUESTIONS
    text = (text or "").strip()
    if not text:
        return [text]

    parts = [p.strip(" ,;") for p in _SPLIT_RE.split(text)]
    parts = [p for p in parts if p and not _TRIVIAL.match(p)]

    if len(parts) < 2:
        return [text]

    # Restore the question mark the split consumed, so each fragment still reads
    # as a question to the classifier.
    out = []
    for p in parts[:max_questions]:
        out.append(p if p.endswith(("?", ".")) else p + "?")
    return out


class RuleBasedProvider:
    """
    Deterministic classifier. No network, no key, no variance.

    This is NOT the intended production path -- a real LLM handles phrasings
    this cannot. It exists so that (a) the graph is testable without a key and
    (b) a mid-demo API failure degrades the wording rather than breaking the
    agent. Every number is already computed deterministically, so the loss is
    fluency, not correctness.
    """
    name = "rule-based"

    def __init__(self, cities: list[str] | None = None):
        self.cities = cities or []

    def complete(self, system: str, user: str, json_mode: bool = False,
                 max_retries: int | None = None,
                 timeout: float | None = None) -> LLMResponse:
        if not json_mode:
            return LLMResponse(text="", provider=self.name, ok=False,
                               error="rule-based provider does not generate prose")

        # The splitter and the classifier share this entry point, so the prompt
        # decides which job this is. Without this branch the fallback returns a
        # classification for a split request, `questions` is absent, and
        # splitting silently never happens on the rule-based path -- exactly the
        # provider-divergence bug that made mentions_store dead in production.
        # A feature that works on only one provider is a feature that does not
        # work.
        from app.prompts import SPLIT_MARKER
        if SPLIT_MARKER in system:
            return LLMResponse(text=json.dumps({"questions": rule_split(user)}),
                               provider=self.name)

        return LLMResponse(text=json.dumps(self._classify(user)), provider=self.name)

    def _classify(self, text: str) -> dict:
        low = text.lower()
        out: dict = {"intent": "out_of_scope", "city": None, "store": None,
                     "date": None, "hour_label": None, "entities": []}

        for intent, pattern in _INTENT_PATTERNS:
            if re.search(pattern, low):
                out["intent"] = intent
                break

        # Collect ALL entities, not just the first.
        #
        # The original version used .search() and stopped at the first store and
        # the first city. "Compare STORE_003 and STORE_091" therefore yielded a
        # single name, and the compare handler -- which needs two -- rejected
        # every comparison. A real LLM populates `entities`; the fallback has to
        # as well, or the offline path silently loses a whole intent.
        stores = [m for m in _STORE_RE.findall(text)
                  if not re.fullmatch(r"(am|pm)\d*", m, re.I)]
        stores = list(dict.fromkeys(stores))

        # B7 -- fall back to a bare number only when nothing explicit matched,
        # and only outside count/time positions. Explicit "store 003" always
        # wins, so "hours 8-11 at store 003" cannot be misread as STORE_011.
        if not stores:
            masked = _NUMBER_IS_NOT_A_STORE.sub(" ", text)
            stores = [n for n in _BARE_NUMBER_RE.findall(masked)]
            stores = list(dict.fromkeys(stores))

        if stores:
            out["store"] = stores[0]

        cities: list[str] = []
        for city in self.cities:
            tokens = [t for t in city.lower().split() if len(t) > 3]
            if tokens and re.search(rf"\b{re.escape(tokens[0])}\b", low):
                cities.append(city)
        for alias in config.CITY_ALIASES:
            if re.search(rf"\b{re.escape(alias)}\b", low):
                target = config.CITY_ALIASES[alias]
                if target not in cities:
                    cities.append(target)
        cities = list(dict.fromkeys(cities))
        if cities:
            out["city"] = cities[0]

        if out["intent"] == "compare":
            out["entities"] = stores if len(stores) >= 2 else cities if len(cities) >= 2 else (stores + cities)

        # ── GRAIN: does the message mention stores, cities, or both? ──────
        #
        # A2 -- EXTRACTION IS UNCONDITIONAL.
        #
        # This block used to be gated on `intent in (rank, count, lookup)`, so
        # a follow_up like "what about all stores" had the word "stores"
        # matched and then thrown away -- and the handler, seeing nothing,
        # guessed cities. That same coupling produced three separate bugs
        # (hour_label, level, then this one).
        #
        # Extraction now reports FACTS ONLY: which nouns appeared. No intent
        # check, and no decision about what they imply -- deciding whether a
        # named city is a subject or a scope is the handler's job, and the
        # handler has more to go on.
        #
        # Recognised city names are stripped before the noun check, because
        # 'Hyderabad city' and 'Pune city east' literally contain the word
        # "city". Without this, naming one of those two cities reads as
        # "asking about cities in general".
        out.update(grain_facts(low, self.cities))

        # CONDITION for count questions.
        #
        # The fallback never extracted this, so every offline "how many stores
        # had X" answered with "which condition should I count?" -- the LLM
        # path worked and the fallback silently did not. Ordered
        # most-specific-first so "sustained pileup" is not swallowed by
        # "pileup".
        for condition, pattern in (
            ("sustained_pileup", r"sustained\s+pile\s?-?up|sustained"),
            ("no_supply",        r"no\s+(rider\s+)?supply|zero\s+riders?|"
                                 r"nobody\s+booked|no\s+riders?\s+booked"),
            ("utilization_gap",  r"utili[sz]ation|man[\s_]?hour|no[\s-]?shows?"),
            ("booking_gap",      r"booking\s+gap|slots?\s+(un)?filled|"
                                 r"under[\s-]?booked"),
            ("demand_spike",     r"demand\s+spike|over\s+forecast|"
                                 r"exceeded\s+(the\s+)?forecast"),
            ("pileup",           r"pile\s?-?up|carried\s+over|backlog"),
            ("problem_hour",     r"problem\s+hours?|breach(ed|es)?|missed\s+sla"),
        ):
            if re.search(pattern, low):
                out["condition"] = condition
                break

        # B5 -- "all hours" and "all PROBLEM hours" are different requests.
        # They were conflated: "show me all hours" returned only the problem
        # hours, silently omitting the healthy ones from a request for ALL.
        out["problem_only"] = bool(re.search(
            r"\b(problem|problematic|breach(ed|ing)?|flagged|bad|failing)\b", low))
        m = re.search(r"\b(?:worst|best|top|bottom)\s+(\d+)\b", low)
        if m:
            out["n"] = int(m.group(1))
        # DIRECTION -- ordering phrases are checked FIRST.
        #
        # "from worst to best" contains the word "best", so a naive
        # best-before-worst check flipped a worst-first ranking into a
        # best-first one. The phrase describes the ORDER, not the preference,
        # and the first-named end is the top of the list.
        if re.search(r"\bworst\s+to\s+best\b|\bbest\s+last\b", low):
            out["direction"] = "worst"
        elif re.search(r"\bbest\s+to\s+worst\b|\bworst\s+last\b", low):
            out["direction"] = "best"
        elif re.search(r"\bworst\b|\bslowest\b|\bhighest\b|\bneeds? improvement\b", low):
            out["direction"] = "worst"
        elif re.search(r"\bbest\b|\bfastest\b|\blowest\b|\btop performing\b", low):
            out["direction"] = "best"

        m = _DATE_RE.search(text)
        if m:
            out["date"] = m.group(1)
        elif re.search(r"\bthat day\b|\bsame day\b|\bthen\b", low):
            out["date"] = None            # explicitly inherit

        m = re.search(_HOUR_WORDS, low)
        if m:
            out["hour_label"] = m.group(1)
        else:
            m = re.search(r"\b(\d{1,2}\s*(?:am|pm))\b", low)
            if m:
                out["hour_label"] = m.group(1)
            else:
                # B8 -- capture ANY word used as a time window, even one we do
                # not recognise. "in noon hours" used to match nothing, so the
                # constraint vanished and the user got a whole-day answer with
                # no hint that their question had been narrowed away.
                #
                # Extracting it unconditionally means resolve_hours() gets to
                # decide, and an unknown window DECLINES with the valid options
                # instead of being silently dropped.
                m = re.search(r"\b(\w+)[\s-]+hours?\b", low)
                if m and m.group(1) not in (
                        "all", "the", "these", "those", "many", "problem",
                        "problematic", "breach", "breached", "further",
                        "remaining", "other", "some", "few", "in", "of", "and"):
                    out["hour_label"] = m.group(1)

        # "How did STORE_210 do?" matches the city_summary phrasing ("how did")
        # but names a STORE. The named entity wins: a store-scoped question
        # cannot be a city summary.
        if out["store"] and out["intent"] == "city_summary":
            out["intent"] = "store_rca"

        # A NAMED TIME WINDOW IMPLIES AN HOUR WALKTHROUGH.
        #
        # "show me morning hours at STORE_202" fell through to out_of_scope,
        # and "how did store_054 do in noon hours?" fell through to store_rca
        # -- in both cases the extracted hour_label was then ignored for
        # routing, and the user got a whole-day answer to a question they had
        # explicitly narrowed.
        #
        # If a window was named, that is the scope of the answer. The only
        # intents that legitimately mention hours WITHOUT meaning a
        # walkthrough are the aggregate ones -- count, rank, compare,
        # hour_profile -- which are excluded here.
        if out.get("hour_label") and out["intent"] in (
                "out_of_scope", "show_all", "store_rca", "city_summary"):
            out["intent"] = "hour_detail"

        # Conversely, "How did Bangalore do?" with no store is a city summary
        # even if some other keyword matched first.
        if out["city"] and not out["store"] and out["intent"] == "out_of_scope":
            out["intent"] = "city_summary"

        return out


# ═══════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════

def get_provider(cities: list[str] | None = None) -> LLMProvider:
    """
    Choose a provider from the environment.

        LLM_PROVIDER=groq|google|anthropic|openai|rule-based
        GROQ_API_KEY / GOOGLE_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY

    Groq is checked first in auto mode -- it is the configured provider for this
    build. See GroqProvider for why Gemini was dropped.

    Falls back to RuleBasedProvider whenever a real provider cannot be
    constructed: missing key, missing package, bad config. The agent always
    starts; it never refuses to boot because of an LLM problem.
    """
    want = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    model = os.getenv("LLM_MODEL") or None

    # (env var, class) — order defines auto-detection precedence
    registry = [
        ("groq",      "GROQ_API_KEY",      GroqProvider),
        ("google",    "GOOGLE_API_KEY",    GoogleProvider),
        ("anthropic", "ANTHROPIC_API_KEY", AnthropicProvider),
        ("openai",    "OPENAI_API_KEY",    OpenAIProvider),
    ]

    if want in ("", "auto"):
        want = next((name for name, env, _ in registry if os.getenv(env)),
                    "rule-based")

    for name, env_var, cls in registry:
        if name != want:
            continue
        key = os.getenv(env_var)
        if key:
            try:
                return cls(key, model)
            except Exception:
                break                     # package missing -> fall through
        break

    return RuleBasedProvider(cities)


def parse_json(text: str) -> dict | None:
    """
    Best-effort JSON extraction from a model response.

    Models wrap JSON in prose or fences despite instructions, so we try the
    whole string, then a fenced block, then the first balanced object. Returns
    None on failure -- callers must treat that as out_of_scope rather than
    crashing.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    break
    return None
