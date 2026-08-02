"""
Conversation state — the mechanism behind multi-turn.

WHY EXPLICIT SLOTS RATHER THAN MESSAGE HISTORY
-----------------------------------------------
The brief requires 5+ turn conversations with drill-downs, follow-ups and
context switches. Sample question 4 is literally "What about TBF1?" -- no verb,
no metric, no date. To answer it the agent must remember that the previous turn
was an RCA, carry the date forward, and swap only the store.

Two ways to do that:

  A. Dump the whole message history at the LLM every turn and hope it works out.
     Trivial to write. Unpredictable, grows unboundedly, and fails silently --
     you cannot tell WHY it resolved something a particular way.

  B. Keep explicit slots (current_store, current_city, current_date) and let the
     LLM's only job be "which slots does this turn change?". More code, but the
     resolved context is INSPECTABLE -- it can be printed, asserted in a test,
     and rendered in the UI beside the chat.

We use B. During the walkthrough the frontend shows the resolved context panel
updating live, which is the most direct way to prove multi-turn is real rather
than an illusion produced by a long prompt.

NO PERSISTENCE
--------------
The brief says explicitly: "Session persistence is not required -- refresh can
start a new session." So sessions live in a plain dict. No database, no Redis,
no serialisation. That is a real scope saving, taken deliberately.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import config

Intent = Literal[
    "follow_up",        # "What about STORE_091?" -- inherit the previous intent
    "city_summary",     # "How did Bangalore do?"
    "store_rca",        # "Why did STORE_003 underperform?"
    "hour_detail",      # "Walk me through the morning hours"
    "show_all",         # "Show me all hours"
    "rank",             # "Worst 5 stores in Mumbai"      -> Tier 2
    "compare",          # "Compare STORE_003 and STORE_091" -> Tier 2
    "hour_profile",     # "What happened at 7pm?"          -> Tier 2
    "count",            # "How many stores had pileup?"    -> Tier 2
    "lookup",           # "What was STORE_003's breach rate?" -> Tier 2
    "methodology",      # "What's the pileup threshold?"   -> MCP docs
    "out_of_scope",     # anything else
]

# Intents that operate on a store and can therefore inherit one from context.
STORE_INTENTS = {"store_rca", "hour_detail", "show_all"}

# "What about STORE_091?" carries no verb and no metric -- it means "do the same
# thing as last turn, but with this entity". The classifier flags it and
# resolve_entities substitutes the previous intent. Handling it as an explicit
# intent rather than letting the model guess keeps the inheritance inspectable.
FOLLOW_UP_DEFAULT = "store_rca"

# Intents that need no entity at all -- they must not trigger a "which store?"
# clarification when context is empty.
ENTITY_FREE_INTENTS = {"methodology", "out_of_scope", "count", "rank"}


@dataclass
class Extraction:
    """
    What the LLM pulled out of a single user message.

    A None field means "not mentioned in this turn" -- NOT "unknown". That
    distinction is the whole mechanism: None is the signal to inherit from
    state. If the model guessed a value instead of returning None, inheritance
    would break and "that day" would silently become today's date.
    """
    intent: Intent = "out_of_scope"
    city: str | None = None
    store: str | None = None
    date: str | None = None
    hour_label: str | None = None
    # Tier-2 parameters
    metric: str | None = None
    condition: str | None = None
    direction: str | None = None
    n: int | None = None

    # "store" | "city" -- WHAT the user is asking about, which is not the same
    # as what city is in context. "Which city is worst?" while Bangalore is in
    # context must rank CITIES, not stores within Bangalore. Inferring this
    # from the presence of an inherited city produced exactly that wrong
    # answer, so the classifier states it explicitly.
    level: str | None = None
    want_all: bool = False          # "list ALL cities" -> do not truncate to N

    # FACTS, not decisions. Which grain nouns appeared in the message, with
    # recognised city names stripped first so "Hyderabad city" does not read
    # as "cities". The handler decides what they imply -- see Agent._rank.
    mentions_city: bool = False
    mentions_store: bool = False

    # "show me all hours" vs "show me all PROBLEM hours" are different
    # requests. True means restrict to problem hours.
    problem_only: bool = False

    # "best performing city AND store" asks two questions at once. A single
    # `level` can only answer one, so the user had to ask again for the other.
    # This flag makes the handler run both rankings and return them together.
    both_levels: bool = False

    # C2 -- A CITY NOUN GOVERNED BY A PREPOSITION IS A SCOPE, NOT A SUBJECT.
    #
    #   "the best city and store"        two SUBJECTS   -> both_levels, two tables
    #   "all stores in all cities"       subject+SCOPE  -> ONE table, unscoped
    #
    # Both mention stores and cities, so `both_levels` alone could not tell
    # them apart and sent the second down the compound-question path -- which
    # returned a city ranking AND a store ranking for a question that asked for
    # one list. That mattered because "list all stores in all cities" is the
    # phrase the agent itself tells users to type to widen an inherited scope:
    # the advertised escape hatch returned the wrong shape.
    #
    # Reported as a FACT, per A2: extraction says "a city noun appeared under a
    # preposition". `_rank_one` decides that this means "drop the inherited
    # city"; `both_levels` decides it means "not a compound question".
    city_as_scope: bool = False

    # "top 5 stores PER city" -- one ranking run inside each group, which is
    # neither `level="store"` (five overall) nor `both_levels` (two tables).
    # A third shape the system had no way to express, so it silently became one
    # of the other two.
    per_group: bool = False

    # The message contained the word "top", which we read as "worst". Carried
    # so the handler can disclose that reading rather than assume silently.
    said_top: bool = False

    # Threshold questions: {"metric", "comparator", "value"} or None.
    # "stores with breach rate over 50%" -- a different question from a ranking,
    # because the answer may legitimately be NONE and a ranking can never say so.
    threshold: dict | None = None

    # "which cities are healthy" -- an inverted ranking, NOT a pass/fail verdict.
    # With OR2A_THRESHOLD_MIN = 0 nothing passes, so the handler must say what
    # the bar is rather than invent a softer one (D4.2).
    wants_healthy: bool = False

    # "morning vs evening", "by time of day" -- KPIs per named hour window.
    wants_windows: bool = False

    # "is it getting worse?", "compared to last week". The dataset is ONE DAY,
    # so this cannot be answered. Carried so the agent can say that explicitly
    # instead of describing a single day as though it were a trend.
    wants_trend: bool = False

    # "what percentage of total orders came from the 10 busiest stores" -- a
    # concentration measure, not a ranking. No tool computes it, so it goes to
    # Tier 3 rather than being answered with a table that omits the percentage.
    wants_share: bool = False
    entities: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedContext:
    """
    The slots after inheritance and entity resolution.

    This is what the frontend renders beside the chat. Being able to point at it
    mid-demo and say "the agent understood 'that day' as 2026-04-22 and carried
    STORE_003 forward" is worth more than any amount of prose about multi-turn.
    """
    city: str | None = None
    store: str | None = None
    date: str | None = None
    hours: list[int] | None = None
    hour_label: str | None = None
    intent: str | None = None
    inherited: list[str] = field(default_factory=list)   # which slots came from history
    resolution_methods: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "city": self.city,
            "store": self.store,
            "date": self.date,
            "hours": self.hours,
            "hour_label": self.hour_label,
            "intent": self.intent,
            "inherited": self.inherited,
            "resolution_methods": self.resolution_methods,
        }


@dataclass
class Session:
    """One conversation. Held in memory only."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # carried slots
    current_city: str | None = None
    current_store: str | None = None
    current_date: str | None = config.DATA_DATE
    current_hours: list[int] | None = None
    last_intent: str | None = None

    # last successful RCA, so "show me all hours" knows what to expand
    last_store_rca: str | None = None

    # Entities from the last successful comparison, so a bare follow-up like
    # "which one did better?" reuses them instead of asking again. Without
    # this, the natural second half of a comparison -- the interpretation --
    # is unanswerable.
    last_compare: list[str] = field(default_factory=list)
    last_compare_level: str | None = None

    # B4 -- (store, hour) -> the narrated summary sentence.
    #
    # The same hour asked twice produced two different sentences: one turn
    # narrated successfully, the other fell back to the template. Same facts,
    # same numbers, different words -- which reads as the agent changing its
    # mind, and costs exactly the trust the numbers are meant to earn.
    #
    # Safe to cache because the underlying facts CANNOT change: one day of data,
    # computed deterministically. It also makes re-asking free, which matters on
    # a token budget.
    #
    # Per-session, not global: a cache that outlived the conversation would make
    # a fix to the narration prompt invisible until the process restarted.
    hour_summaries: dict[tuple[str | None, int], str] = field(default_factory=dict)

    turns: list[dict] = field(default_factory=list)

    def record(self, user: str, reply: str, ctx: ResolvedContext) -> None:
        self.turns.append({
            "turn": len(self.turns) + 1,
            "user": user,
            "reply": reply,
            "context": ctx.as_dict(),
        })

    @property
    def turn_count(self) -> int:
        return len(self.turns)


class SessionStore:
    """
    In-memory session registry.

    A plain dict is the correct choice here, not a shortcut: the brief lists
    session persistence as out of scope. Sessions are isolated by id so two
    browser tabs cannot contaminate each other.
    """

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get(self, session_id: str | None) -> Session:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        s = Session(session_id=session_id or uuid.uuid4().hex[:12])
        self._sessions[s.session_id] = s
        return s

    def reset(self, session_id: str) -> Session:
        self._sessions.pop(session_id, None)
        return self.get(session_id)

    def __len__(self) -> int:
        return len(self._sessions)


SESSIONS = SessionStore()
