"""
LangGraph nodes.

Each node is a small function taking the graph state and returning updates. Two
of them call the LLM; the rest are deterministic Python that delegates to the
Module 1/2 engine.

    classify_intent    LLM #1 — sentence -> {intent, entities}
    resolve_entities   Python — inherit context, resolve names, or decline
    handlers           Python — thin wrappers over the tested RCA engine
    narrate            LLM #2 — computed facts -> one summary sentence

Exactly two LLM calls per turn. That is a latency budget as much as an
architectural choice: every extra call is another 1-3 seconds the panel watches
a spinner.
"""

from __future__ import annotations

import re
from dataclasses import asdict

import config
from app import llm as llm_mod
from app import prompts, tools
from app.entities import EntityIndex, resolve_city, resolve_hours, resolve_store
from app.state import FOLLOW_UP_DEFAULT, Extraction, ResolvedContext, Session


# ═══════════════════════════════════════════════════════════════════════
# NODE 0 — split_questions
# ═══════════════════════════════════════════════════════════════════════

# Conjunctions that almost always join ONE request rather than two. If the
# message contains no separator at all, there is nothing to split and the LLM
# call is skipped entirely -- which is the overwhelming majority of turns.
_MULTI_HINT = re.compile(
    r"[;?]|\band\s+(?:also|then|what|which|who|how|why|show|give|tell|list)\b"
    r"|\balso\b|\bthen\b|\bplus\b|\bas well as\b|\bafter that\b", re.I)


def split_questions(message: str, provider) -> list[str]:
    """
    Cut a message into at most MAX_SUBQUESTIONS self-contained questions.

    WHY A SEPARATE STEP AND NOT A RICHER CLASSIFIER
    ----------------------------------------------
    The alternative was to make the classifier return a LIST of extractions.
    That fails as soon as the sub-questions have different intents: a city
    summary and an hour walkthrough need different fields, so the schema becomes
    a list of full Extractions -- which is this design, with more surgery and a
    much larger prompt doing two jobs at once.

    Splitting first keeps every downstream fix applying to every sub-question
    for free, because each piece runs through the ordinary graph untouched.

    THE FAILURE MODE IS OVER-SPLITTING. "Compare STORE_003 and STORE_203" reads
    like two questions and is one; splitting it destroys the comparison and
    answers something nobody asked. So this is biased hard toward returning one:

      * no separator in the message      -> skip the call entirely
      * model errors, times out, or
        returns junk                     -> treat as one question
      * model returns one element        -> one question

    Every one of those degrades to exactly the pre-existing behaviour. The
    feature can fail completely without making anything worse than it was.
    """
    text = (message or "").strip()
    if not text or not _MULTI_HINT.search(text):
        return [text]

    try:
        resp = provider.complete(
            prompts.build_split_prompt(config.MAX_SUBQUESTIONS), text,
            json_mode=True,
            max_retries=config.SPLIT_MAX_RETRIES,
            timeout=config.SPLIT_TIMEOUT_SECONDS)
        data = llm_mod.parse_json(resp.text) if resp.ok else None
    except Exception:
        data = None

    if not data:
        return [text]

    parts = [str(q).strip() for q in (data.get("questions") or [])
             if str(q).strip()]

    # A splitter that drops content is worse than no splitter. If it returned
    # nothing usable, or so little text that it clearly summarised rather than
    # split, fall back to the original message.
    if not parts or sum(len(p) for p in parts) < len(text) * 0.5:
        return [text]

    return parts[: config.MAX_SUBQUESTIONS]


# ═══════════════════════════════════════════════════════════════════════
# NODE 1 — classify_intent   (LLM call #1)
# ═══════════════════════════════════════════════════════════════════════

def classify_intent(
    message: str,
    provider,
    index: EntityIndex,
) -> Extraction:
    """
    Turn a user message into a structured Extraction.

    Failure handling is deliberately total: a bad response, malformed JSON, a
    dead API or an unrecognised intent all resolve to a usable Extraction rather
    than an exception. The graph must never crash on model output -- the model
    is the least trustworthy component in the system and is treated as such.
    """
    if not message or not message.strip():
        return Extraction(intent="out_of_scope", raw={"reason": "empty message"})

    system = prompts.build_classify_prompt(
        cities=index.cities,
        metrics=sorted(tools.METRICS),
        conditions=sorted(tools._CONDITIONS),
    )

    resp = provider.complete(system, message, json_mode=True)
    data = llm_mod.parse_json(resp.text) if resp.ok else None

    if data is None:
        # Rule-based fallback: if the real provider failed or returned garbage,
        # classify locally rather than giving up. Wording degrades; the agent
        # keeps working.
        fallback = llm_mod.RuleBasedProvider(index.cities)
        data = llm_mod.parse_json(fallback.complete(system, message, json_mode=True).text)

    if data is None:
        return Extraction(intent="out_of_scope",
                          raw={"reason": "unparseable", "provider": resp.provider})

    # ── GRAIN FACTS ARE COMPUTED, NEVER ASKED ───────────────────────────
    #
    # `mentions_store` / `mentions_city` are not in the classify prompt, so the
    # model never emitted them: on the LLM path every handler that consults
    # them saw False. The A2 fix and the follow_up re-route therefore passed in
    # the test suite (rule-based) and did nothing in production (Groq).
    #
    # These are lexical facts about the message string. Python answers them
    # exactly; a model answers them slowly, variably, and sometimes wrongly.
    # Overlaid here so the guarantee holds for EVERY provider, present or
    # future, including one that returns valid JSON with these keys absent.
    facts = llm_mod.grain_facts((message or "").lower(), index.cities)
    data["mentions_city"] = facts["mentions_city"]
    data["mentions_store"] = facts["mentions_store"]
    data["city_as_scope"] = facts["city_as_scope"]

    # want_all and both_levels are OR-ed rather than overwritten: the model can
    # legitimately spot phrasings the regex does not ("give me the lot",
    # "rank every single one of them"). The regex is a floor, not a ceiling.
    #
    # The one thing the model may NOT do is call a prepositional scope a second
    # subject -- "all stores in all cities" asks for one list -- so city_as_scope
    # vetoes both_levels regardless of who set it.
    data["per_group"] = facts["per_group"]
    data["said_top"] = facts["said_top"]
    data["threshold"] = facts["threshold"]
    data["wants_healthy"] = facts["wants_healthy"]
    data["wants_windows"] = facts["wants_windows"]
    data["wants_trend"] = facts["wants_trend"]
    data["wants_share"] = facts["wants_share"]

    # AN UNRESOLVABLE PLACE IS A NAMED ENTITY, SO IT GOES IN `city`.
    #
    # "worst 5 stores in Atlantis" ranked the entire network: nothing extracted
    # "Atlantis", the scope stayed empty, and a confident global answer came
    # back for a place that does not exist.
    #
    # Putting it in `city` routes it into resolve_city, which already knows how
    # to decline an unknown name with suggestions -- the same path that handles
    # an unknown STORE. Only set when the model did not name a city itself; a
    # real extraction always wins.
    if facts["unknown_place"] and not data.get("city"):
        data["city"] = facts["unknown_place"]

    # A threshold or health question is a RANKING-FAMILY question. Providers
    # routinely classify "stores with breach rate over 50%" as out_of_scope or
    # count, because neither prompt example covers it -- and the request then
    # dies at a handler that has no idea a threshold was parsed. Recognising the
    # shape here keeps the two in step without another prompt round-trip.
    if (facts["threshold"] or facts["wants_healthy"]) and \
            data.get("intent") in (None, "out_of_scope", "lookup", "count"):
        data["intent"] = "rank"

    # A window comparison outranks whatever else matched: "morning vs evening"
    # contains hour words, so hour_detail claims it and walks ONE window hour by
    # hour, silently dropping the comparison the user asked for.
    if facts["wants_windows"]:
        data["intent"] = "window_rollup"

    # A trend question outranks everything. Answering it from one day is the
    # worst failure available here -- the numbers would all be real and the
    # conclusion would be fabricated.
    if facts["wants_trend"]:
        data["intent"] = "no_trend_data"

    # A share-of-a-whole question is not a ranking, however many superlatives it
    # contains. Sent to out_of_scope, which is where Tier 3 lives -- generating
    # a query and showing it beats a table that silently omits the number asked
    # for. Trend still outranks it: a share question about last week is still
    # unanswerable.
    if facts["wants_share"] and not facts["wants_trend"]:
        data["intent"] = "out_of_scope"

    # An explicitly typed number is the least ambiguous signal in a message and
    # nothing infers over it -- not the model, not a quantifier. "top 5 stores
    # of each city" contains both a 5 and the word "each"; "each" used to win,
    # so the 5 was discarded and all 97 rows came back.
    if facts["explicit_n"] is not None:
        data["n"] = facts["explicit_n"]
        data["want_all"] = False
    else:
        data["want_all"] = bool(data.get("want_all")) or facts["want_all"]

    data["both_levels"] = ((bool(data.get("both_levels")) or facts["both_levels"])
                           and not facts["city_as_scope"]
                           and not facts["per_group"])

    intent = data.get("intent")
    if intent not in (
        "follow_up", "city_summary", "store_rca", "hour_detail", "show_all",
        "rank", "compare", "hour_profile", "count", "lookup", "methodology",
        "window_rollup", "no_trend_data", "out_of_scope",
    ):
        intent = "out_of_scope"

    def _clean(v):
        """Models sometimes emit the strings "null"/"none" instead of JSON null."""
        if isinstance(v, str) and v.strip().lower() in ("", "null", "none", "n/a"):
            return None
        return v

    n = _clean(data.get("n"))
    try:
        n = int(n) if n is not None else None
    except (TypeError, ValueError):
        n = None

    level = _clean(data.get("level"))
    if level not in ("store", "city"):
        level = None

    return Extraction(
        intent=intent,
        city=_clean(data.get("city")),
        store=_clean(data.get("store")),
        date=_clean(data.get("date")),
        hour_label=_clean(data.get("hour_label")),
        metric=_clean(data.get("metric")),
        condition=_clean(data.get("condition")),
        direction=_clean(data.get("direction")),
        n=n,
        level=level,
        want_all=bool(data.get("want_all")),
        both_levels=bool(data.get("both_levels")),
        mentions_city=bool(data.get("mentions_city")),
        mentions_store=bool(data.get("mentions_store")),
        city_as_scope=bool(data.get("city_as_scope")),
        per_group=bool(data.get("per_group")),
        said_top=bool(data.get("said_top")),
        threshold=data.get("threshold") or None,
        wants_healthy=bool(data.get("wants_healthy")),
        wants_windows=bool(data.get("wants_windows")),
        wants_trend=bool(data.get("wants_trend")),
        wants_share=bool(data.get("wants_share")),
        problem_only=bool(data.get("problem_only")),
        entities=[e for e in (data.get("entities") or []) if e],
        raw={"provider": resp.provider, "data": data},
    )


# ═══════════════════════════════════════════════════════════════════════
# NODE 2 — resolve_entities   (deterministic)
# ═══════════════════════════════════════════════════════════════════════

class ResolutionError(Exception):
    """Carries a user-facing message plus suggestions."""

    def __init__(self, message: str, suggestions: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.suggestions = suggestions or []


def resolve_entities(
    ext: Extraction,
    session: Session,
    index: EntityIndex,
) -> ResolvedContext:
    """
    Apply inheritance, then resolve names to values that exist in the data.

    INHERITANCE is the multi-turn mechanism: any field the classifier returned
    as None is filled from the session. `inherited` records which ones, so the
    UI can show that "that day" became 2026-04-22 by carrying context rather
    than by guessing.

    Raises ResolutionError for anything unresolvable -- this is the TBC8 path.
    A wrong entity match produces a confident answer about the wrong store,
    which is far worse than declining.
    """
    ctx = ResolvedContext(intent=ext.intent)

    # ── follow_up: "What about STORE_091?" means "repeat the last question" ──
    if ext.intent == "follow_up":
        inherited_intent = session.last_intent or FOLLOW_UP_DEFAULT
        if inherited_intent in ("follow_up", "out_of_scope", "methodology"):
            inherited_intent = FOLLOW_UP_DEFAULT
        # C1-a -- A FOLLOW-UP THAT CHANGES THE GRAIN IS A NEW QUESTION.
        #
        # "what about all stores" returned the PREVIOUS answer verbatim:
        #
        #     extraction  intent=follow_up  mentions_store=True  want_all=True
        #     resolve     inherits intent=hour_detail
        #     answer      STORE_080 hour 12 -- byte-identical to the turn before
        #
        # Both flags were extracted correctly and then never read. This is the
        # FOURTH instance of one failure mode: something the user said is
        # extracted, discarded because no downstream condition consumes it, and
        # the answer is delivered as if nothing was lost (previously hour_label,
        # level, and mentions_store). A2 made extraction unconditional, which
        # fixed the recording half; this is the reading half.
        #
        # A bare grain noun with no store named is not "ask the last question
        # about a new entity" -- it is "ask about that whole population", which
        # is a ranking. `_decide_level` then picks store vs city from the same
        # flags, so the decision still lives in exactly one place.
        #
        # Guarded on `not ext.store` so "what about STORE_091?" is untouched --
        # and note "store_091" does not match \bstores?\b anyway, because the
        # underscore is a word character, so there is no boundary after "store".
        if not ext.store and (ext.mentions_store or ext.mentions_city):
            inherited_intent = "rank"
        # A bare city reference should become a city summary, not a store RCA.
        # Checked second: "what about stores in Mumbai" names a city AND a
        # grain, and the grain is what changed.
        elif ext.city and not ext.store:
            inherited_intent = "city_summary"
        ctx.intent = inherited_intent
        ctx.inherited.append("intent")

    # ── date ────────────────────────────────────────────────────────────
    if ext.date:
        ctx.date = ext.date
    else:
        ctx.date = session.current_date or config.DATA_DATE
        if session.current_date:
            ctx.inherited.append("date")

    if ctx.date != config.DATA_DATE:
        raise ResolutionError(
            f"I only have data for {config.DATA_DATE}. "
            f"There is no data for {ctx.date}.")

    # ── city ────────────────────────────────────────────────────────────
    #
    # PRECEDENCE, highest first:
    #   1. STATED     the user named a city in this message
    #   2. DERIVED    a named store implies its city (resolved below)
    #   3. INHERITED  carried from the previous turn
    #
    if ext.city:
        r = resolve_city(ext.city, index)
        if not r.matched:
            raise ResolutionError(r.message, r.suggestions)
        ctx.city = r.value
        ctx.resolution_methods["city"] = r.method
    elif session.current_city:
        ctx.city = session.current_city
        ctx.inherited.append("city")

    # ── store ───────────────────────────────────────────────────────────
    if ext.store:
        r = resolve_store(ext.store, index, city_hint=ctx.city)
        if not r.matched:
            raise ResolutionError(r.message, r.suggestions)
        ctx.store = r.value
        ctx.resolution_methods["store"] = r.method

        # A named store OUTRANKS an inherited city. "How did Bangalore do?"
        # followed by "why did STORE_210 underperform?" must switch to Noida --
        # STORE_210 is unambiguously a Noida store, so there is nothing to
        # inherit. Record this as DERIVED, not inherited: the previous label
        # said "city: inherited" while displaying the derived value, which
        # misrepresented what actually happened.
        derived_city = index.store_city[r.value]
        if derived_city != ctx.city:
            if "city" in ctx.inherited:
                ctx.inherited.remove("city")
            ctx.resolution_methods["city"] = "derived from store"
        ctx.city = derived_city

    elif ctx.intent in ("store_rca", "hour_detail", "show_all", "window_rollup"):
        carried = session.current_store

        # THE PIVOT CASE.
        #
        #   T1  "Why did STORE_003 underperform?"    store = STORE_003 (Bangalore)
        #   T2  "How did Mumbai do?"                  city  = Mumbai, store cached
        #   T3  "Walk me through the morning hours"   <-- which store?
        #
        # Blindly inheriting answers about a Bangalore store while the user is
        # plainly discussing Mumbai. No error, no warning, wrong answer.
        #
        # The condition is NOT "did the user state a city this turn" -- the
        # first version checked that and never fired, because by T3 the city
        # was itself inherited and the guard was skipped. The pivot happened a
        # turn earlier.
        #
        # The right condition is a CONSISTENCY CHECK on the context itself:
        # whenever a store is about to be inherited, does it belong to the city
        # currently in scope? If the two disagree, the session is internally
        # inconsistent and guessing which one the user meant is exactly the
        # wrong move. Ask instead.
        #
        # This cannot false-positive on the normal path: a store-scoped turn
        # sets current_city FROM the store, so the two always agree unless a
        # city-only turn has since moved the context elsewhere.
        if carried and ctx.city and index.store_city[carried] != ctx.city:
            options = index.stores_in(ctx.city)[: config.MAX_SUGGESTIONS]
            raise ResolutionError(
                f"You were looking at {carried} ({index.store_city[carried]}), "
                f"but now asking about {ctx.city}. Which store in {ctx.city}? "
                f"For example: {', '.join(options)}.",
                options)

        if carried:
            ctx.store = carried
            ctx.city = index.store_city[carried]
            ctx.inherited.append("store")
        elif ctx.intent == "window_rollup":
            # A time-of-day breakdown is meaningful at EITHER grain, so it
            # inherits a store when there is one and falls back to the city
            # rather than demanding a store the user never mentioned.
            #
            # The other three intents genuinely need a store: an RCA or an hour
            # walkthrough has no city-level form. This one does, so refusing to
            # answer "by time of day" during a city conversation would be the
            # agent inventing a requirement.
            pass
        else:
            examples = index.stores_in(ctx.city)[:3] if ctx.city else [
                "STORE_003", "STORE_091", "STORE_210"]
            where = f" in {ctx.city}" if ctx.city else ""
            raise ResolutionError(
                f"Which store{where} would you like me to look at? "
                f"For example: {', '.join(examples)}.")

    # ── hours ───────────────────────────────────────────────────────────
    if ext.hour_label:
        r = resolve_hours(ext.hour_label)
        if not r.matched:
            raise ResolutionError(r.message, r.suggestions)
        ctx.hours = r.value
        ctx.hour_label = ext.hour_label
        ctx.resolution_methods["hours"] = r.method
    elif ctx.intent == "hour_detail" and session.current_hours:
        ctx.hours = session.current_hours
        ctx.inherited.append("hours")

    # ── city_summary needs a city ───────────────────────────────────────
    if ctx.intent == "city_summary" and not ctx.city:
        raise ResolutionError(
            "Which city would you like me to look at? Available: "
            + ", ".join(index.cities) + ".")

    return ctx


def commit_context(session: Session, ctx: ResolvedContext) -> None:
    """
    Write the resolved slots back so the NEXT turn can inherit them.

    Called only after a turn succeeds -- a failed turn must not poison the
    context. If "TBC8" overwrote current_store with None, the following
    "walk me through the morning" would lose the store it should have kept.
    """
    if ctx.city:
        session.current_city = ctx.city
    if ctx.store:
        session.current_store = ctx.store
        session.last_store_rca = ctx.store
    if ctx.date:
        session.current_date = ctx.date
    if ctx.hours:
        session.current_hours = ctx.hours
    if ctx.intent and ctx.intent not in ("out_of_scope", "follow_up"):
        session.last_intent = ctx.intent


# ═══════════════════════════════════════════════════════════════════════
# NODE 3 — narrate   (LLM call #2)
# ═══════════════════════════════════════════════════════════════════════

def narrate(facts: dict, provider, fallback: str) -> str:
    """
    Turn computed facts into one summary sentence. Falls back on any failure.

    Prefer `narrate_or_none` in new code -- it distinguishes "the model wrote
    this" from "the model failed and you are looking at the template", which
    this signature cannot. Kept because the distinction is irrelevant to callers
    that narrate exactly one block.
    """
    return narrate_or_none(facts, provider) or fallback


def narrate_or_none(facts: dict, provider) -> str | None:
    """
    One summary sentence, or None if the model did not produce a usable one.

    B3 -- WHY THE CALLER MUST BE ABLE TO TELL.

    `narrate()` substitutes the deterministic sentence internally, so a caller
    narrating several blocks cannot see which ones fell back. Three blocks went
    out, one came back as model prose and two as templates, and the answer was
    rendered with all three -- so a single response spoke in two voices:

        Hour 21  "OR2A was 755.0 min over threshold due to..."          template
        Hour 14  "The SLA was missed with an average OR2A of 846.6..."  prose

    Returning None makes the failure visible one level up, which is what lets
    _narrate_blocks enforce all-or-nothing.
    """
    # Fail fast. Narration has a complete deterministic fallback, so retrying
    # spends up to 93 seconds -- and burns free-tier quota the NEXT turn will
    # want -- to avoid a purely cosmetic downgrade. Classification keeps its
    # retries because its fallback is genuinely worse.
    resp = provider.complete(prompts.NARRATE_SYSTEM,
                             prompts.build_narrate_user(facts),
                             json_mode=False,
                             max_retries=config.NARRATION_MAX_RETRIES,
                             timeout=config.NARRATION_TIMEOUT_SECONDS)

    if not resp.ok or not resp.text or not resp.text.strip():
        return None

    text = resp.text.strip()

    # Reject obviously malformed output rather than rendering it. A model that
    # returns a JSON blob or a wall of markdown has misunderstood the task.
    if text.startswith("{") or text.startswith("#") or len(text) > 1200:
        return None

    return " ".join(text.split())


def narrate_comparison(raw: dict, verdict: str, provider) -> str:
    """
    Rephrase a deterministic comparison verdict.

    Uses COMPARE_SYSTEM, not NARRATE_SYSTEM -- the latter is about store-hours
    and root-cause flags, and handing it an aggregate comparison produced
    confident nonsense about untriggered flags and manual review.

    The output is VALIDATED before use. The verdict was computed in Python; the
    model is only rephrasing it, so if the result does not name the winner, or
    smuggles in hour-level vocabulary that was never computed, the
    deterministic sentence is kept instead. A rephrasing that changes the
    meaning is not a rephrasing.
    """
    resp = provider.complete(prompts.COMPARE_SYSTEM,
                             prompts.build_compare_user(raw, verdict),
                             json_mode=False,
                             max_retries=config.NARRATION_MAX_RETRIES,
                             timeout=config.NARRATION_TIMEOUT_SECONDS)

    if not resp.ok or not resp.text or not resp.text.strip():
        return verdict

    text = " ".join(resp.text.split())

    # The winner's name must survive. Its absence means the model wrote about
    # something else.
    winner = verdict.split("**")[1] if "**" in verdict else None
    if winner and winner.lower() not in text.lower():
        return verdict

    # Vocabulary that belongs to hour-level RCA and was never computed here.
    forbidden = ("store-hour", "manual review", "demand spike", "pileup",
                 "booking gap", "utilization gap", "no rider supply",
                 "root cause")
    if any(f in text.lower() for f in forbidden):
        return verdict

    if text.startswith("{") or text.startswith("#") or len(text) > 900:
        return verdict

    return text


def facts_for_narration(checks) -> dict:
    """
    Serialise one hour's checks for the narrator.

    Only computed values are included, and every flag is stated explicitly --
    the model is assembling a sentence from finished facts, not deciding
    anything.
    """
    return {
        "store": checks.store,
        "hour": checks.hour,
        "avg_or2a_min": checks.avg_or2a,
        "threshold_min": config.OR2A_THRESHOLD_MIN,
        "total_orders": checks.total_orders,
        "breached_orders": checks.breached_count,
        "demand_spike": asdict(checks.demand),
        "pileup": asdict(checks.pileup),
        "booking_gap": asdict(checks.booking),
        "utilization": asdict(checks.utilization),
        "triggered_flags": checks.triggered,
        "previous_hour": checks.hour - 1,
    }
