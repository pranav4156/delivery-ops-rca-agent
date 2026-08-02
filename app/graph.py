"""
LangGraph orchestration.

WHY LANGGRAPH RATHER THAN A PLAIN AGENT LOOP
---------------------------------------------
A LangChain-style agent is a black box: LLM -> tool -> LLM -> tool -> answer,
with the model deciding the path. Simple to write, impossible to reason about.

LangGraph makes the control flow an explicit state machine that I draw, not the
model. The practical consequences:

  - every node's input and output can be inspected and asserted
  - the deterministic nodes are ordinary Python and CANNOT be wrong
  - the LLM chooses which BRANCH to take, never what the answer is
  - the graph is a diagram, and a diagram can be defended in a design review

That last point is the real reason. In a walkthrough, pointing at a node and
saying "this one is deterministic, so it cannot produce a wrong number" is worth
more than any amount of prompt engineering.

THE SHAPE
---------
                        START
                          |
                   classify_intent          LLM #1 -- the only place a
                          |                          sentence becomes structure
                   resolve_entities         Python -- inheritance + resolution
                       /      \\
                 error?        ok
                   |             \\
                  END        route_by_intent
                                   |
        +---------+---------+------+------+---------+---------+
        |         |         |             |         |         |
   city_summary  store_rca  hour_detail  show_all  tier2   methodology
        |         |         |             |         |         |
        +---------+---------+------+------+---------+---------+
                                   |
                                narrate       LLM #2 -- prose only, over
                                   |                    finished facts
                                  END

Exactly two LLM calls per turn. Everything between them is the tested engine
from Modules 1 and 2.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, TypedDict

import duckdb
from langgraph.graph import END, START, StateGraph

import config
from app import mcp_client, nodes, prompts, rca, render, sql_path, tools
from app.entities import EntityIndex, resolve_city, resolve_store
from app.llm import get_provider
from app.nodes import (
    ResolutionError,
    classify_intent,
    commit_context,
    facts_for_narration,
    narrate,
    narrate_comparison,
    resolve_entities,
)
from app.rollups import NoDataError, city_day, store_day
from app.state import SESSIONS, ResolvedContext, Session


# ═══════════════════════════════════════════════════════════════════════
# GRAPH STATE
# ═══════════════════════════════════════════════════════════════════════

class GraphState(TypedDict, total=False):
    """
    Passed between nodes. Each node returns a partial dict that LangGraph
    merges in.

    Deliberately flat and JSON-serialisable: `context` and `provenance` are
    returned straight to the frontend, and being able to print the whole state
    at any node is what makes the graph debuggable.
    """
    message: str
    session_id: str

    extraction: Any                 # Extraction from classify_intent
    context: dict                   # ResolvedContext.as_dict()
    intent: str

    reply: str                      # rendered markdown
    provenance: str
    error: str | None
    suggestions: list[str]

    facts: list[dict]               # per-hour facts handed to the narrator
    summaries: dict                 # hour -> LLM sentence

    # Rich objects passed from a handler to the narrator.
    #
    # These MUST be declared here. LangGraph filters each node's return value
    # against the declared state keys and silently drops anything unknown --
    # an undeclared key arrives as None in the next node, surfacing as a
    # confusing "'NoneType' object has no attribute ..." rather than a schema
    # error. The TypedDict is the schema, not documentation.
    ranked: Any                     # rca.RankedHours
    day: Any                        # rollups.StoreDay
    tool_result: Any                # tools.ToolResult, for Tier-2 narration
    compare_level: str              # "store" | "city" — changes which metrics
                                    # are comparable in the verdict


# ═══════════════════════════════════════════════════════════════════════
# AGENT
# ═══════════════════════════════════════════════════════════════════════

class Agent:
    """
    Owns the connection, the entity index, the LLM provider and the compiled
    graph. Built once at startup; `run_turn` is called per message.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, provider=None,
                 mcp: "mcp_client.MCPDocServer | None" = None,
                 connect_mcp: bool | None = None):
        self.con = con
        self.index = EntityIndex(con)
        self.provider = provider or get_provider(self.index.cities)

        # MCP is optional by construction, and warmed up in the BACKGROUND.
        #
        # Connecting synchronously here stalled boot: npx may fetch the server
        # package on first run, and with no network it hangs rather than
        # failing, so startup blocked for the full timeout. Nothing depends on
        # MCP, so waiting for it is pure cost.
        self.mcp = mcp if mcp is not None else mcp_client.MCPDocServer()
        should = (config.MCP_CONNECT_ON_STARTUP if connect_mcp is None
                  else connect_mcp)
        if should and self.mcp is not None:
            self.mcp.connect_background()

        self.graph = self._build()

    # ── nodes ──────────────────────────────────────────────────────────

    def _classify(self, state: GraphState) -> dict:
        ext = classify_intent(state["message"], self.provider, self.index)
        return {"extraction": ext, "intent": ext.intent}

    def _resolve(self, state: GraphState) -> dict:
        session = SESSIONS.get(state["session_id"])
        try:
            ctx = resolve_entities(state["extraction"], session, self.index)
        except ResolutionError as e:
            # Not an exception path -- a legitimate outcome. The agent asks a
            # question instead of guessing, which is the correct behaviour for
            # an unknown store or an inconsistent context.
            #
            # Return the SESSION'S CURRENT context, not an empty one. The
            # session deliberately keeps its slots through a failed turn, so
            # returning blanks would make a client's context panel clear itself
            # on every declined question -- showing "no store" while the
            # conversation is still very much about STORE_003.
            return {
                "error": e.message,
                "suggestions": e.suggestions,
                "reply": e.message,
                "provenance": config.PROVENANCE_UNAVAILABLE,
                "context": self._session_context(session),
            }

        commit_context(session, ctx)
        return {"context": ctx.as_dict(), "intent": ctx.intent, "error": None}

    @staticmethod
    def _session_context(session: Session) -> dict:
        """
        The session's persisted slots, for turns that produced no new context.

        Used on every failure path. The session deliberately keeps its slots
        through a declined turn, so returning an empty context would make a
        client's context panel blank itself on every "I don't have that store" --
        showing "no store selected" while the conversation is still about
        STORE_003.
        """
        return ResolvedContext(
            city=session.current_city,
            store=session.current_store,
            date=session.current_date,
            hours=session.current_hours,
            intent=session.last_intent,
        ).as_dict()

    # ── handlers: thin wrappers over the tested engine ─────────────────

    def _city_summary(self, state: GraphState) -> dict:
        ctx = state["context"]
        day = city_day(self.con, ctx["city"], ctx["date"])
        return {"reply": render.render_city_day(day),
                "provenance": config.PROVENANCE_VALIDATED}

    def _store_rca(self, state: GraphState, expand: bool = False) -> dict:
        """
        Store-day RCA.

        `expand` rather than `top_n=None`: rank_problem_hours() uses None to
        mean "return everything", while a default argument uses None to mean
        "not specified". Those two collided -- _show_all passed top_n=None
        intending "all hours" and got the default of 3 back, silently
        truncating the expansion the user explicitly asked for.
        A boolean cannot be ambiguous.
        """
        ctx = state["context"]
        top_n = None if expand else config.TOP_N_PROBLEM_HOURS
        day = store_day(self.con, ctx["store"], ctx["date"])
        ranked = rca.rank_problem_hours(self.con, ctx["store"], ctx["date"],
                                        top_n=top_n)
        return {
            "reply": render.render_store_day(day, ranked),
            "provenance": config.PROVENANCE_VALIDATED,
            "facts": [facts_for_narration(c) for c in ranked.top],
            "ranked": ranked,
            "day": day,
        }

    def _show_all(self, state: GraphState) -> dict:
        """
        Expand a truncated answer.

        B5 -- "all hours" means ALL hours.

        This used to return only the PROBLEM hours regardless of phrasing, so
        "show me all hours" silently dropped the healthy ones -- 6 of 17 for
        STORE_080. The heading said "All 11 problem hours", which is honest,
        but it answers a narrower question than the one asked and the word
        "all" makes that easy to miss.

        Now the two requests are distinct:
            "show me all hours"          -> every hour, healthy included
            "show me all problem hours"  -> problem hours only
        """
        if state["extraction"].problem_only:
            return self._store_rca(state, expand=True)

        ctx = state["context"]
        ranked = rca.rank_problem_hours(self.con, ctx["store"], ctx["date"],
                                        top_n=None)
        return {
            "reply": render.render_hour_walkthrough(ranked),
            "provenance": config.PROVENANCE_VALIDATED,
            "facts": [facts_for_narration(c) for c in ranked.window
                      if c.is_problem_hour],
            "ranked": ranked,
        }

    def _hour_detail(self, state: GraphState) -> dict:
        ctx = state["context"]
        ranked = rca.rank_problem_hours(self.con, ctx["store"], ctx["date"],
                                        top_n=None, hours=ctx.get("hours"))
        return {
            "reply": render.render_hour_walkthrough(ranked),
            "provenance": config.PROVENANCE_VALIDATED,
            "facts": [facts_for_narration(c) for c in ranked.window
                      if c.is_problem_hour],
            "ranked": ranked,
        }

    # ── ranking ────────────────────────────────────────────────────────

    def _rank(self, state: GraphState) -> dict:
        """
        Rank stores or cities -- or BOTH, when the user asked for both.

        TWO FIXES LIVE HERE:

        1. LEVEL COMES FROM THE CLASSIFIER, NOT FROM CONTEXT. The first version
           inferred it -- level = "city" only when no city was in context -- so
           asking "is Bangalore in the worst 5 CITIES?" while Bangalore was in
           context ranked the STORES INSIDE Bangalore. A confident answer to a
           different question. An inherited city is context, not subject; only
           an explicitly stated city scopes a store ranking.

        2. COMPOUND QUESTIONS. "Tell me the best performing city and store" is
           two questions. A single `level` could only answer one, so the user
           had to ask again for the other. Now both rankings run and are
           returned together.
        """
        ctx, ext = state["context"], state["extraction"]
        metric = ext.metric or "w_avg_or2a"
        direction = ext.direction or "worst"

        # THRESHOLD: "stores with breach rate over 50%". Checked before every
        # ranking branch because a threshold that also mentions a grain noun
        # would otherwise be answered as a ranking -- which always returns rows
        # and so implies they crossed a line they may not have crossed.
        if ext.threshold:
            level = self._decide_level(ext, ctx)
            t = ext.threshold
            scope, scope_note = self._scope_for(ext, ctx, level)
            try:
                res = tools.filter_by_threshold(
                    self.con, level=level, metric=t["metric"],
                    comparator=t["comparator"], value=t["value"],
                    scope=scope, date=ctx["date"])
            except tools.UnknownMetricError:
                # The metric exists but not at this grain (e.g. store_count is
                # city-only). Retry at the grain where it does exist rather than
                # failing the turn.
                other = "city" if level == "store" else "store"
                scope_note = ""
                res = tools.filter_by_threshold(
                    self.con, level=other, metric=t["metric"],
                    comparator=t["comparator"], value=t["value"],
                    date=ctx["date"])
            return {"reply": tools.render_tool_result(res) + scope_note,
                    "provenance": res.provenance}

        # HEALTHY: an inverted ranking that states the bar. Never a pass list.
        if ext.wants_healthy:
            level = self._decide_level(ext, ctx)
            scope, scope_note = self._scope_for(ext, ctx, level)
            res = tools.healthiest(
                self.con, level=level, n=ext.n or config.CITY_TOP_N_STORES,
                scope=scope, date=ctx["date"])
            return {"reply": tools.render_tool_result(res) + scope_note,
                    "provenance": res.provenance}

        # PER-GROUP: "top 5 stores of each city". Checked first because it is
        # the most specific shape -- it mentions both nouns, so every branch
        # below would also match it, and each would answer a different
        # question. Specificity, not order of discovery, decides precedence.
        if ext.per_group:
            res = tools.rank_within_groups(
                self.con, metric=metric, n=ext.n or config.PER_GROUP_DEFAULT_N,
                direction=direction, date=ctx["date"])
            return {"reply": self._with_top_note(tools.render_tool_result(res), ext),
                    "provenance": res.provenance}

        if ext.both_levels:
            parts = []
            for level in ("city", "store"):
                res = self._rank_one(ext, ctx, level, metric, direction)
                parts.append(tools.render_tool_result(res))
            return {"reply": self._with_top_note("\n\n".join(parts), ext),
                    "provenance": config.PROVENANCE_VALIDATED}

        level = self._decide_level(ext, ctx)
        res = self._rank_one(ext, ctx, level, metric, direction)
        return {"reply": self._with_top_note(tools.render_tool_result(res), ext),
                "provenance": res.provenance}

    def _window_rollup(self, state: GraphState) -> dict:
        """
        "Morning vs evening across Bangalore" — KPIs per named hour window.

        Routed ahead of hour_detail because the phrase contains hour words, so
        hour_detail claims it and walks ONE window hour by hour — answering half
        the question and never mentioning the other half.
        """
        ctx = state["context"]
        res = tools.window_rollup(
            self.con, scope=ctx.get("city"), store=ctx.get("store"),
            date=ctx["date"])
        return {"reply": tools.render_tool_result(res),
                "provenance": res.provenance,
                "context": ctx}

    def _no_trend_data(self, state: GraphState) -> dict:
        """
        Decline every question about change over time.

        THE DATASET IS ONE DAY. "Is it getting worse?", "how does this compare
        to last week", "what's the trend" cannot be answered from it.

        This is the most dangerous question class in the project, because a
        plausible answer is trivially available: describe 2026-04-22 in the
        language of a trend and every number quoted is real. The reasoning would
        be fabricated and nothing in the output would reveal it.

        So the decline is explicit, names the single date, and redirects to a
        question the data CAN answer. Declining well is a feature -- an agent
        that never says "I can't" cannot be trusted when it says "I can".
        """
        ctx = state["context"]
        return {
            "reply": (
                f"I can't answer that — the dataset is a **single day**, "
                f"{config.DATA_DATE}. There is no earlier data to compare "
                f"against, so any trend, improvement or week-on-week figure "
                f"would be invented.\n\n"
                f"What I can do for {config.DATA_DATE}:\n"
                f"- **\"morning vs evening at STORE_080\"** — how it moved "
                f"across that day\n"
                f"- **\"worst 5 stores in Bangalore\"** — where it stands "
                f"against its peers\n"
                f"- **\"why did STORE_080 underperform?\"** — the root cause "
                f"behind the day's numbers"),
            "provenance": config.PROVENANCE_UNAVAILABLE,
            "context": ctx,
        }

    @staticmethod
    def _scope_for(ext, ctx: dict, level: str) -> tuple[str | None, str]:
        """
        Resolve a store-level scope and the disclosure that goes with it.

        Shared by the threshold and healthy handlers so they inherit a city on
        the same terms as a ranking does -- and, more importantly, DISCLOSE it
        on the same terms. A new tool that silently narrows is the same bug as
        the old one; it just has not been reported yet.
        """
        if level != "store":
            return None, ""
        scope = ctx.get("city")
        if not scope or ext.city or ext.store:
            return scope, ""
        return scope, (f"\n\n_Scoped to {scope}, carried over from earlier in "
                       f"this conversation._")

    @staticmethod
    def _with_top_note(reply: str, ext) -> str:
        """
        Disclose that "top" was read as "worst".

        "Top 5 stores" is genuinely ambiguous -- top of the leaderboard, or top
        of the problem list? In an operations review it means the latter, and
        that is how we read it. But a reasonable assumption the user cannot see
        is indistinguishable from a wrong one: they would have to notice the
        title said "Worst" and work backwards.

        Only fires when the user actually typed "top" AND did not state a
        direction. Saying "best 5 stores" or "worst 5 stores" is unambiguous,
        so there is nothing to disclose and no note appears.
        """
        if not (ext.said_top and ext.direction is None):
            return reply
        return (f"{reply}\n\n_Read \"top\" as **worst-performing** — the top of "
                f"the problem list. For the best performers, ask "
                f"**\"best 5 stores\"**._")

    @staticmethod
    def _decide_level(ext, ctx: dict) -> str:
        """
        Decide whether a ranking is over STORES or CITIES.

        A2 -- THIS DECISION LIVES HERE, NOT IN EXTRACTION.

        The classifier reports facts: which grain nouns appeared, and which
        entities were named. It used to also make this decision, gated on
        intent -- so a follow_up never got a level at all and the fallback
        guessed. Moving the decision here fixes that, and gives it more to work
        with: the handler can see whether a named entity actually RESOLVED,
        which a bare word-match cannot.

        Precedence, most explicit first:
          1. the model stated a level outright
          2. the message said "stores"          -> rank stores
          3. the message said "cities"          -> rank cities
          4. a city was NAMED                   -> it scopes a store ranking
          5. a store is in context              -> stores
          6. nothing to go on                   -> cities (the coarser view)
        """
        if ext.level in ("store", "city"):
            return ext.level
        if ext.mentions_store:
            return "store"
        if ext.mentions_city:
            return "city"
        if ext.city or ctx.get("store"):
            return "store"
        return "city"

    def _rank_one(self, ext, ctx: dict, level: str, metric: str, direction: str):
        # C2 -- AN INHERITED CITY IS KEPT, AND SAID OUT LOUD.
        #
        # The first cut of this fix had `want_all` DISCARD an inherited scope,
        # on the reasoning that `_decide_level` calls an inherited city "context,
        # not subject". That was a bad transplant. `_decide_level`'s rule was
        # written for a LEVEL bug -- "is Bangalore in the worst 5 cities?"
        # returned stores inside Bangalore, a confident answer to a DIFFERENT
        # question. Scope is not grain:
        #
        #     wrong grain   -> the wrong question, answered confidently  (a bug)
        #     narrow scope  -> the right question, over a smaller set    (a default)
        #
        # Only the first is a correctness failure. And discarding the scope is
        # the more surprising behaviour, on three counts:
        #
        #   * every other intent inherits -- "noon hours" inherits the store,
        #     "why did it underperform" inherits the date. Breaking inheritance
        #     for exactly one intent is the inconsistency.
        #   * "all" contrasts with "top 5", not with "in Bangalore". The
        #     quantifier scopes the COUNT, not the geography.
        #   * 233 rows spanning 9 cities is a worse answer to an ops manager
        #     three turns into Bangalore than 97 coherent ones.
        #
        # So the scope is kept and DISCLOSED, with the widening phrase named.
        # `city_as_scope` -- "in all cities", "everywhere" -- is the explicit
        # opt-out, and is the only thing that drops it.
        scope = ctx.get("city") if level == "store" else None

        # `not ext.store` matters as much as `not ext.city`: naming a store
        # DERIVES its city (see resolve_entities), and a derived city is as
        # explicit as a stated one -- the user did name the thing it came from.
        # Without this, "what about STORE_091?" was told its own store's city
        # had been "carried over from earlier", which is simply untrue.
        scope_inherited = bool(scope) and not ext.city and not ext.store

        if level == "store" and ext.city_as_scope and not ext.city:
            scope = None
            scope_inherited = False
        total = (len(self.index.cities) if level == "city"
                 else len(self.index.stores_in(scope)))
        n = total if ext.want_all else (ext.n or config.CITY_TOP_N_STORES)
        return tools.rank_entities(
            self.con, level=level, metric=metric, scope=scope,
            n=max(1, min(n, total)), direction=direction, date=ctx["date"],
            scope_inherited=scope_inherited)

    # ── comparison ─────────────────────────────────────────────────────

    def _compare(self, state: GraphState) -> dict:
        """
        Compare two stores or two cities.

        THREE BUGS THIS FIXES, all found by unscripted testing:

        1. LEVEL WAS NEVER DETECTED. compare_entities defaults to level="store",
           so "compare Bangalore and Noida" looked up cities in agg_store_day
           and returned "No data" -- city comparison was entirely broken.

        2. NAMES WERE PASSED RAW. Without resolution, "mumbai" never matches the
           stored 'Mumbai  north', and "store 3" never matches STORE_003.

        3. NO MEMORY. "which one did better?" as a follow-up had nothing to
           compare and asked for two entities again, when the user had just
           named them.
        """
        ctx, ext = state["context"], state["extraction"]
        session = SESSIONS.get(state["session_id"])

        raw_names = list(dict.fromkeys(
            ext.entities or [x for x in (ext.store, ext.city) if x]))

        # Bare follow-up -- "which did better?" -- reuses the last comparison.
        if len(raw_names) < 2 and session.last_compare:
            names, level = session.last_compare, session.last_compare_level
        else:
            names, level, unresolved = self._resolve_comparands(raw_names)
            if unresolved:
                return {
                    "reply": (f"I don't recognise {', '.join(repr(u) for u in unresolved)}. "
                              f"Compare two stores (e.g. STORE_003 and STORE_091) "
                              f"or two cities (e.g. Bangalore and Noida)."),
                    "provenance": config.PROVENANCE_UNAVAILABLE,
                }
            if level == "mixed":
                return {
                    "reply": ("I can compare two stores or two cities, but not a "
                              "store against a city — the metrics are not "
                              "comparable at different grains."),
                    "provenance": config.PROVENANCE_UNAVAILABLE,
                }
            if len(names) < 2:
                return self._compare_needs_two(state, names)

        res = tools.compare_entities(self.con, names, level=level, date=ctx["date"])
        session.last_compare, session.last_compare_level = names, level

        return {
            "reply": tools.render_tool_result(res),
            "provenance": res.provenance,
            "tool_result": res,
            "compare_level": level,
        }

    # "compare all stores" / "compare every city" -- a bulk request, not a pair.
    _COMPARE_ALL = re.compile(
        r"\b(all|every|each|the)\s+(\w+\s+)?(store|stores|cit(y|ies))\b", re.I)

    def _compare_needs_two(self, state: GraphState, names: list[str]) -> dict:
        """
        A comparison was asked for but fewer than two entities were named.

        "Compare all stores" is not really a comparison of a pair -- it is a
        ranking, and it has only one sensible reading. But routing it silently
        to `rank` would GUESS at intent, which is exactly what this system
        refuses to do everywhere else (TBC8 never becomes STORE_008; an
        inconsistent context asks rather than picking a side).

        So: name the limitation, offer the likely reading, and make taking it
        one tap. The suggestion strings are sent verbatim as the next message
        when a client renders them as chips.
        """
        message = (state.get("message") or "")
        ctx = state["context"]
        m = self._COMPARE_ALL.search(message)

        if not m:
            return {
                "reply": ("I need two entities to compare. Try "
                          "_\"compare STORE_003 and STORE_091\"_ or "
                          "_\"Bangalore vs Noida\"_."),
                "provenance": config.PROVENANCE_UNAVAILABLE,
            }

        noun = m.group(3).lower()
        is_city = noun.startswith("cit")
        scope = "" if is_city else (f" in {ctx['city']}" if ctx.get("city") else "")
        plural = "cities" if is_city else "stores"
        named = f" ({names[0]})" if names else ""

        options = [
            f"Worst 5 {plural}{scope}",
            f"Best 5 {plural}{scope}",
            f"All {plural}{scope} from worst to best",
        ]

        return {
            "reply": (
                f"I compare two entities at a time{named}, so I can't compare "
                f"all {plural} against each other. Did you want a **ranking** "
                f"of all {plural}{scope} instead?\n\n"
                + "\n".join(f"- _{o}_" for o in options)
            ),
            "provenance": config.PROVENANCE_UNAVAILABLE,
            "suggestions": options,
        }

    def _resolve_comparands(self, names: list[str]) -> tuple[list[str], str, list[str]]:
        """
        Resolve each name to a store or a city, and infer the grain.

        Store is tried first: STORE_003 is unambiguous, whereas a fuzzy city
        match on a store code would be noise.
        """
        stores, cities, unresolved = [], [], []
        for n in names:
            rs = resolve_store(n, self.index)
            if rs.matched:
                stores.append(rs.value)
                continue
            rc = resolve_city(n, self.index)
            if rc.matched:
                cities.append(rc.value)
                continue
            unresolved.append(n)

        if unresolved:
            return [], "", unresolved
        if stores and cities:
            return [], "mixed", []
        return (stores, "store", []) if stores else (cities, "city", [])

    # ── Tier 2 ─────────────────────────────────────────────────────────

    def _tier2(self, state: GraphState) -> dict:
        """
        Parameterized query tools.

        The LLM chose the tool and filled parameters; the SQL is hand-written
        and tested. An unrecognised metric or condition raises rather than
        being interpolated -- that whitelist is the injection boundary.
        """
        ctx, ext = state["context"], state["extraction"]
        intent = ctx["intent"]

        try:
            if intent == "rank":
                return self._rank(state)
            elif intent == "compare":
                return self._compare(state)
            elif intent == "hour_profile":
                res = tools.hour_profile(self.con, hours=ctx.get("hours"),
                                         city=ctx.get("city"), date=ctx["date"])
            elif intent == "count":
                # No silent default. Defaulting to "problem_hour" answered
                # "how many stores had sustained pileup?" with a breach count --
                # a confidently wrong answer to a different question. Ask
                # instead.
                if not ext.condition:
                    return {
                        "reply": (
                            "Which condition should I count? One of: "
                            + ", ".join(f"`{c}`" for c in sorted(tools._CONDITIONS))
                            + "."),
                        "provenance": config.PROVENANCE_UNAVAILABLE,
                    }
                res = tools.count_matching(
                    self.con, ext.condition,
                    level="store", city=ctx.get("city"), date=ctx["date"])
            else:  # lookup
                entity = ctx.get("store") or ctx.get("city")
                res = tools.metric_lookup(
                    self.con, entity,
                    level="store" if ctx.get("store") else "city",
                    metrics=[ext.metric] if ext.metric else None,
                    date=ctx["date"])

            return {"reply": tools.render_tool_result(res),
                    "provenance": res.provenance}

        except (ValueError, NoDataError) as e:
            # Includes the whitelist rejections. Surfaced as a normal reply so
            # the graph never crashes on an LLM-supplied parameter.
            return {"reply": f"I couldn't run that: {e}",
                    "provenance": config.PROVENANCE_UNAVAILABLE}

    # ── methodology + fallback ─────────────────────────────────────────

    # Which reference document answers which kind of question. Checked in
    # order, so a more specific topic wins over a general one.
    _DOC_TOPICS = [
        (("or2a", "order ready", "assignment", "sla", "metric", "threshold"),
         "order_ready_to_assignment.md"),
        (("pileup", "demand", "spike", "booking", "utilization", "man_hour",
          "sustained", "playbook", "root cause", "rca", "check"),
         "quick_commerce_rca_logic.md"),
        (("column", "schema", "field", "table", "capacity", "slot", "grain"),
         "quick_commerce_orders_gold.md"),
    ]

    def _methodology(self, state: GraphState) -> dict:
        """
        Answer definition questions FROM THE SOURCE DOCUMENTS, via MCP.

        This is the MCP integration point. "What is the pileup threshold?" is
        answered by reading the RCA playbook -- the declared source of truth --
        rather than by echoing a constant we typed into config.py. When the two
        ever disagree, the document wins, and the user can see the quoted text.

        Degrades to config-derived values if the server is unavailable, clearly
        labelled as such. The RCA path never touches MCP, so an outage here
        cannot affect a single number.
        """
        question = state.get("message") or ""

        if self.mcp is not None:
            # WHOLE CORPUS, NOT A RETRIEVED SUBSET.
            #
            # ~1,800 tokens -- less than the classifier prompt. Selecting part
            # of that to send part of that buys nothing and adds a way to miss.
            # See MCPDocServer.read_corpus for the full argument, including why
            # RAG would be strictly worse here.
            corpus = self.mcp.read_corpus()

            if corpus:
                answer = self._synthesise_docs(question, corpus)
                if answer:
                    return {"reply": answer,
                            "provenance": config.PROVENANCE_VALIDATED}
                # No model, or the call failed. The excerpts are still real
                # source text -- degrade to showing them rather than to nothing.
                return {"reply": self._doc_excerpts(question, corpus),
                        "provenance": config.PROVENANCE_VALIDATED}

        return {"reply": mcp_client.config_thresholds(),
                "provenance": config.PROVENANCE_VALIDATED}

    def _synthesise_docs(self, question: str,
                         corpus: list[tuple[str, str]]) -> str | None:
        """
        One written answer over the whole corpus. None if unavailable.

        The model receives every document and writes prose with inline
        citations. It is not asked for a single number the engine already
        knows -- it is asked to explain what the documents SAY, which is
        genuine language work and the one thing a chunk dump cannot do.

        Returns None rather than raising, so the caller can fall back.
        """
        if self.provider.name == "rule-based":
            return None
        try:
            resp = self.provider.complete(
                prompts.build_doc_prompt(corpus), question,
                max_retries=config.DOC_MAX_RETRIES,
                timeout=config.DOC_TIMEOUT_SECONDS)
        except Exception:
            return None

        if not resp.ok or not (resp.text or "").strip():
            return None

        return (f"{resp.text.strip()}\n\n"
                f"_Answered from {len(corpus)} reference document"
                f"{'s' if len(corpus) != 1 else ''} read via MCP from "
                f"`{self.mcp.docs_dir.name}/`._")

    def _doc_excerpts(self, question: str,
                      corpus: list[tuple[str, str]]) -> str:
        """
        Deterministic fallback: the relevant sections, ONE BLOCK PER FILE.

        The bug this replaces: three excerpts were concatenated with a header
        each, and for "what is OR2A?" all three came from the SAME file --
        so the answer repeated `order_ready_to_assignment.md` three times and
        read like a search-results page.

        Merging per file fixes that without a model, which matters because this
        path runs whenever there is no key.
        """
        wanted = set(self._doc_query(question.lower()).split())
        scored = []
        for name, text in corpus:
            sections = [s for s in self.mcp.search_text(text, wanted)] \
                if hasattr(self.mcp, "search_text") else []
            body = "\n\n".join(sections) if sections else text
            hits = sum(body.lower().count(w) for w in wanted if len(w) > 3)
            scored.append((hits, name, body))

        scored.sort(key=lambda r: -r[0])
        best = [s for s in scored if s[0] > 0] or scored[:1]

        parts = ["**From the reference documentation**\n"]
        for _, name, body in best[:2]:
            body = body.strip()
            if len(body) > 1400:
                body = body[:1400].rsplit("\n", 1)[0] + "\n…"
            parts.append(f"_{name}_\n\n{body}\n")
        parts.append(f"_Read via MCP from `{self.mcp.docs_dir.name}/`._")
        return "\n".join(parts)

    @staticmethod
    def _doc_query(message: str) -> str:
        """
        Pick the search term from the question.

        Searching the whole sentence would match nothing -- the documents do
        not contain "what is the pileup threshold". A single domain term is
        what actually appears in them.
        """
        for term in ("sustained pileup", "pileup", "demand spike", "booking",
                     "utilization", "man_hour", "or2a", "breach", "threshold",
                     "capacity", "slot", "projection", "no-show", "grain"):
            if term in message:
                return term
        return "threshold"

    def _out_of_scope(self, state: GraphState) -> dict:
        """
        Never a dead end.

        Two ways out, in order:

          1. TIER 3 -- no tool covers this, so generate SQL, screen it, run it,
             and label the answer `screened` rather than `validated`.
          2. A decline that lists capabilities, converting a miss into a
             redirect.

        Tier 3 is tried FIRST because "I can't" is only the right answer once
        every honest option is exhausted. It is tried LAST in the graph because
        a tested tool always beats a generated query when both could answer.

        Off-topic questions still land here, and the model returns {"sql": ""}
        for them -- which falls through to the decline below. So the capability
        list is never replaced, only preceded.
        """
        if config.ENABLE_SQL_FALLBACK and self.provider.name != "rule-based":
            question = state.get("message") or ""
            res = sql_path.answer(self.con, question, self.provider,
                                  state.get("context") or {})

            if res.sql and not res.error:
                return {"reply": sql_path.render(res, question),
                        "provenance": config.PROVENANCE_SCREENED}

            # A FAILED TIER-3 ATTEMPT MUST NOT LOOK LIKE AN OUT-OF-SCOPE
            # QUESTION.
            #
            # Falling straight through to the capability list said "I don't
            # handle this kind of question" when the truth was "I tried, wrote a
            # query, and it was rejected". Those are different facts and the
            # second one is actionable -- the user can rephrase, or tell me the
            # screen is too strict.
            #
            # `attempts` is non-empty only when the model actually produced SQL,
            # so a genuinely off-topic question (model returns {"sql": ""}) still
            # gets the plain capability list below, which is correct for it.
            if res.attempts and any(a.strip() for a in res.attempts):
                detail = f"\n\n```sql\n{res.attempts[-1]}\n```" if res.attempts else ""
                return {
                    "reply": (
                        f"I tried to answer that with a generated query and "
                        f"couldn't: **{res.error}**{detail}\n\n"
                        f"Rephrasing usually fixes it. Or ask something the "
                        f"tested tools cover — rankings, thresholds, "
                        f"comparisons, hour walkthroughs, time-of-day splits."),
                    "provenance": config.PROVENANCE_UNAVAILABLE,
                }

        return {
            "reply": (
                "I can only answer questions about quick-commerce delivery "
                f"performance for {config.DATA_DATE}. I can:\n\n"
                "- summarise a city — _\"How did Bangalore do?\"_\n"
                "- run a root-cause analysis on a store — "
                "_\"Why did STORE_003 underperform?\"_\n"
                "- walk through specific hours — "
                "_\"Walk me through the morning hours\"_\n"
                "- rank or compare stores and cities — "
                "_\"Worst 5 stores in Mumbai\"_\n"
                "- count stores or hours matching a condition — "
                "_\"How many stores had sustained pileup?\"_\n"
                "- explain the methodology — _\"What's the pileup threshold?\"_"
            ),
            "provenance": config.PROVENANCE_UNAVAILABLE,
        }

    # ── narration: LLM call #2 ─────────────────────────────────────────

    def _narrate(self, state: GraphState) -> dict:
        """
        Replace ONLY the **Summary** line of each hour block with model prose.

        Every number is already rendered. The model receives finished facts and
        cannot alter them -- if it fails, render.build_summary()'s deterministic
        sentence stays and the answer is still complete and correct.
        """
        # Tier-2 results are narrated differently: a table, then a verdict.
        if state.get("tool_result") is not None:
            return self._narrate_tool(state)

        facts = state.get("facts") or []
        ranked = state.get("ranked")
        if not facts or ranked is None:
            return {}                      # nothing to narrate; keep the render

        is_walkthrough = state.get("intent") == "hour_detail"
        blocks = ([c for c in ranked.window if c.is_problem_hour]
                  if is_walkthrough else ranked.top)

        # BUDGET CHECK -- the fix for "Can't reach the agent".
        #
        # Every block is one LLM call. "Show me all hours" produces 16+ blocks,
        # so narrating them sequentially took 8+ seconds and exceeded the
        # client timeout. Past the cap we skip narration entirely and keep the
        # deterministic summaries, which are already complete and correct.
        # All-or-nothing, because a response mixing model prose and template
        # prose reads as inconsistent.
        if len(blocks) > config.MAX_NARRATED_HOURS:
            return {}

        session = SESSIONS.get(state["session_id"])
        summaries = self._narrate_blocks(facts, blocks, session, ranked.store)
        if summaries is None:
            # B3 -- ALL OR NOTHING.
            #
            # At least one block did not come back from the model. Rendering the
            # rest anyway produced a single answer in two voices:
            #
            #   Hour 21  "OR2A was 755.0 min over threshold due to..."   template
            #   Hour 14  "The SLA was missed with an average OR2A..."    prose
            #
            # Returning {} keeps the deterministic summaries the handler already
            # rendered. Consistency beats having one nicer sentence, and the
            # template carries MORE figures than the prose does anyway.
            return {}

        if is_walkthrough:
            reply = render.render_hour_walkthrough(ranked, summaries)
        else:
            day = state.get("day")
            if day is None:
                return {}                  # cannot re-render without the rollup
            reply = render.render_store_day(day, ranked, summaries)

        return {"reply": reply,
                "summaries": {str(k): v for k, v in summaries.items()}}

    def _narrate_blocks(self, facts: list[dict], blocks: list,
                        session=None, store: str | None = None
                        ) -> dict[int, str] | None:
        """
        Narrate several hour blocks concurrently. None if ANY block failed.

        B3 -- returning None rather than a partial dict is what makes the answer
        speak in one voice. A caller handed a partial dict cannot tell which
        entries are model prose and which are templates.

        B4 -- SUMMARIES ARE CACHED PER SESSION, keyed by (store, hour).

        The same hour asked twice produced two different sentences:

            Q1  "The SLA was missed with an average OR2A of 846.6 minutes..."
            Q2  "OR2A was 846.6 min over threshold due to a demand spike..."

        Identical facts, identical numbers, different words -- because one turn
        narrated successfully and the other fell back. Nothing was wrong, but it
        reads as though the agent changed its mind, which costs exactly the
        trust the numbers are meant to earn.

        The facts for a given store-hour are fixed (one day of data, computed
        deterministically), so a summary of them is safe to reuse for the life
        of the conversation. Caching also makes re-asking free, which matters on
        a token budget.
        """
        pairs = list(zip(facts, blocks))
        if not pairs:
            return {}

        cache = getattr(session, "hour_summaries", None) if session else None

        def cached(hour: int) -> str | None:
            return cache.get((store, hour)) if cache is not None else None

        def remember(hour: int, text: str) -> None:
            if cache is not None:
                cache[(store, hour)] = text

        # Everything already seen this conversation is reused verbatim; only the
        # rest is sent to the model.
        summaries: dict[int, str] = {}
        todo = []
        for f, c in pairs:
            hit = cached(c.hour)
            if hit is not None:
                summaries[c.hour] = hit
            else:
                todo.append((f, c))

        if not todo:
            return summaries

        if len(todo) == 1:
            f, c = todo[0]
            text = nodes.narrate_or_none(f, self.provider)
            if text is None:
                return None
            remember(c.hour, text)
            summaries[c.hour] = text
            return summaries

        pairs = todo

        # NOT a `with` block. ThreadPoolExecutor.__exit__ calls
        # shutdown(wait=True), which blocks until every worker finishes -- that
        # would wait out the very calls the deadline exists to abandon, making
        # the budget meaningless. The pool is shut down explicitly with
        # wait=False instead.
        pool = ThreadPoolExecutor(max_workers=config.NARRATION_CONCURRENCY)
        done: dict[int, str] = {}
        try:
            futures = {
                pool.submit(nodes.narrate_or_none, f, self.provider): c
                for f, c in pairs
            }

            deadline = time.monotonic() + config.NARRATION_BUDGET_SECONDS
            try:
                for future in as_completed(
                        futures, timeout=max(0.1, deadline - time.monotonic())):
                    checks = futures[future]
                    try:
                        text = future.result()
                    except Exception:
                        text = None
                    if text is not None:
                        done[checks.hour] = text
            except FuturesTimeout:
                pass    # budget spent

            # ALL OR NOTHING. Anything that failed or ran out of budget means
            # the whole answer goes deterministic -- see the note in _narrate.
            # Previously the survivors were rendered alongside templates, which
            # is how one response ended up in two voices.
            if len(done) != len(pairs):
                return None
        finally:
            # Do not wait. An in-flight call may run on for a while; it writes
            # to a future nobody reads.
            pool.shutdown(wait=False, cancel_futures=True)

        for hour, text in done.items():
            remember(hour, text)
        summaries.update(done)
        return summaries

    def _narrate_tool(self, state: GraphState) -> dict:
        """
        Append an interpretation to a Tier-2 table.

        A comparison table does not answer "which one did better" -- it hands
        the user the raw numbers and leaves them to do the comparison
        themselves, which is exactly the work they asked the agent to do.

        The verdict is computed deterministically from ToolResult.raw first;
        the model only rephrases it. So the ANSWER cannot be wrong even if the
        prose is, and with no model available the deterministic sentence stands
        on its own.
        """
        res = state["tool_result"]
        table = tools.render_tool_result(res)

        verdict = tools.summarize_comparison(res, state.get("compare_level", "store"))
        if not verdict:
            return {}

        prose = narrate_comparison(res.raw, verdict, self.provider)
        return {"reply": f"{table}\n\n{prose}"}

    # ── routing ────────────────────────────────────────────────────────

    @staticmethod
    def _route(state: GraphState) -> str:
        intent = state.get("intent", "out_of_scope")
        if intent in ("rank", "compare", "hour_profile", "count", "lookup"):
            return "tier2"
        if intent in ("city_summary", "store_rca", "hour_detail",
                      "show_all", "methodology", "window_rollup",
                      "no_trend_data"):
            return intent
        return "out_of_scope"

    @staticmethod
    def _needs_narration(state: GraphState) -> str:
        return "narrate" if (state.get("facts")
                             or state.get("tool_result") is not None) else "done"

    # ── assembly ───────────────────────────────────────────────────────

    def _build(self):
        g = StateGraph(GraphState)

        g.add_node("classify_intent", self._classify)
        g.add_node("resolve_entities", self._resolve)
        g.add_node("city_summary", self._city_summary)
        g.add_node("store_rca", self._store_rca)
        g.add_node("hour_detail", self._hour_detail)
        g.add_node("show_all", self._show_all)
        g.add_node("tier2", self._tier2)
        g.add_node("methodology", self._methodology)
        g.add_node("window_rollup", self._window_rollup)
        g.add_node("no_trend_data", self._no_trend_data)
        g.add_node("out_of_scope", self._out_of_scope)
        g.add_node("narrate", self._narrate)

        g.add_edge(START, "classify_intent")
        g.add_edge("classify_intent", "resolve_entities")

        # Resolution either asks a question (-> END) or dispatches by intent.
        # Both branches live in one conditional edge rather than two chained
        # ones: an unresolvable entity is a normal outcome, not an error path,
        # and modelling it as a route keeps the graph flat and readable.
        g.add_conditional_edges(
            "resolve_entities",
            lambda s: "error" if s.get("error") else self._route(s),
            {
                "error": END,
                "city_summary": "city_summary",
                "store_rca": "store_rca",
                "hour_detail": "hour_detail",
                "show_all": "show_all",
                "tier2": "tier2",
                "methodology": "methodology",
                "window_rollup": "window_rollup",
                "no_trend_data": "no_trend_data",
                "out_of_scope": "out_of_scope",
            },
        )

        # Only the RCA handlers produce per-hour facts worth narrating.
        for node in ("store_rca", "hour_detail", "show_all"):
            g.add_conditional_edges(node, self._needs_narration,
                                    {"narrate": "narrate", "done": END})

        # tier2 joins the narration branch too: a comparison table needs a
        # verdict appended, not just the numbers.
        g.add_conditional_edges("tier2", self._needs_narration,
                                {"narrate": "narrate", "done": END})

        for node in ("city_summary", "methodology", "window_rollup",
                     "no_trend_data", "out_of_scope"):
            g.add_edge(node, END)
        g.add_edge("narrate", END)

        return g.compile()

    # ── public API ─────────────────────────────────────────────────────

    def run_turn(self, session_id: str, message: str) -> dict:
        """
        One conversational turn, which may contain several questions.

        Returns the contract M4's API and frontend depend on:
            reply, context, intent, provenance, error, suggestions
        """
        parts = nodes.split_questions(message, self.provider)
        if len(parts) > 1:
            return self._run_multi(session_id, message, parts)
        return self._run_one(session_id, parts[0] if parts else message)

    def _run_multi(self, session_id: str, message: str,
                   parts: list[str]) -> dict:
        """
        Several questions in one message, answered in order.

        WHY IN ORDER, SHARING SESSION STATE
        -----------------------------------
        "How did Bangalore do, and which store was worst?" -- question 2 has no
        subject. Running the pieces sequentially through the ordinary turn
        machinery means Bangalore is already in context by the time it runs, so
        the existing inheritance resolves it with no special handling at all.
        Running them in parallel would be faster and would break exactly this.

        THE BUDGET IS SHARED, NOT PER-QUESTION
        --------------------------------------
        Three questions each taking a comfortable 8s is 24 seconds -- long past
        the point the frontend gives up, which is the "can't reach the agent"
        failure already seen once. A whole-turn budget bounds the worst case.

        Questions that do not get run are LISTED. A partial answer that looks
        complete is the failure this project keeps finding; a partial answer
        that says so is just an answer.
        """
        deadline = time.monotonic() + config.SUBQUESTION_BUDGET_SECONDS
        blocks, answered, last = [], 0, None

        for i, part in enumerate(parts, 1):
            if i > 1 and time.monotonic() > deadline:
                break
            last = self._run_one(session_id, part)
            answered += 1
            blocks.append(f"### {i}. {part.rstrip('?')}?\n\n{last['reply']}")

        if answered < len(parts):
            skipped = "\n".join(f"- {p}" for p in parts[answered:])
            blocks.append(
                f"_I ran out of time before these. Ask them on their own:_\n"
                f"{skipped}")

        # Provenance for the whole turn is the WEAKEST of its parts: one
        # unavailable answer makes the combined reply not fully validated, and
        # claiming otherwise would overstate the strongest piece.
        session = SESSIONS.get(session_id)
        return {
            "session_id": session.session_id,
            "reply": "\n\n---\n\n".join(blocks),
            "context": (last or {}).get("context") or self._session_context(session),
            "intent": "multi_question",
            "provenance": (last or {}).get("provenance",
                                           config.PROVENANCE_UNAVAILABLE),
            "error": None,
            "suggestions": [],
        }

    def _run_one(self, session_id: str, message: str) -> dict:
        session = SESSIONS.get(session_id)
        try:
            out = self.graph.invoke({"message": message,
                                     "session_id": session.session_id})
        except NoDataError as e:
            out = {"reply": str(e), "provenance": config.PROVENANCE_UNAVAILABLE,
                   "error": str(e), "context": self._session_context(session)}
        except Exception as e:                      # never crash a live demo
            out = {"reply": f"Something went wrong handling that: {e}",
                   "provenance": config.PROVENANCE_UNAVAILABLE,
                   "error": str(e), "context": self._session_context(session)}

        result = {
            "session_id": session.session_id,
            "reply": out.get("reply", ""),
            "context": out.get("context") or self._session_context(session),
            "intent": out.get("intent"),
            "provenance": out.get("provenance", config.PROVENANCE_UNAVAILABLE),
            "error": out.get("error"),
            "suggestions": out.get("suggestions", []),
        }
        session.record(message, result["reply"], ResolvedContext(**{
            k: v for k, v in result["context"].items()
            if k in ResolvedContext.__dataclass_fields__
        }))
        return result

    def diagram(self) -> str:
        """Mermaid source for the README and the walkthrough."""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            return f"<diagram unavailable: {e}>"


def build_agent(con: duckdb.DuckDBPyConnection, provider=None,
                mcp=None, connect_mcp: bool | None = None) -> Agent:
    return Agent(con, provider, mcp=mcp, connect_mcp=connect_mcp)
