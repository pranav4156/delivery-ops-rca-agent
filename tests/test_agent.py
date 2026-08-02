"""
Agent-layer tests — multi-turn routing, over the real graph.

WHY THIS FILE EXISTS
--------------------
Everything below the graph (db, tables, entities, rca, render, tools) had
pytest coverage from the start. The conversational layer had none — it was
exercised only by the check_* scripts, which are demonstrations: they print,
a human reads, and a regression is caught only if that human happens to be
looking at the right line.

Every bug in this file was found by a human reading output. That is the
argument for the file.

Each test names the failure it locks down, in the sequence that produced it.
A single-turn assertion cannot catch any of them: they are all bugs about what
the agent carried, dropped, or silently reused BETWEEN turns.

MCP is disabled (`connect_mcp=False`) so the suite never touches the network.
The provider is whatever `get_provider` finds; the routing asserted here is
deterministic Python downstream of extraction, so it holds either way.
"""

import itertools

import pytest

import config

from app.graph import build_agent


_ids = itertools.count()


@pytest.fixture(scope="module")
def agent(con):
    return build_agent(con, connect_mcp=False)


@pytest.fixture
def convo(agent):
    """
    A fresh session per test. Session state is global and keyed by id, so tests
    sharing one would leak context into each other — which is the exact class
    of bug this file is about.
    """
    sid = f"pytest-{next(_ids)}"

    def ask(message: str) -> dict:
        return agent.run_turn(sid, message)

    return ask


# ═══════════════════════════════════════════════════════════════════════
# C1 — A FOLLOW-UP THAT CHANGES THE GRAIN IS A NEW QUESTION
#
# The reported failure, exactly as it happened:
#
#   "How did STORE_080 do?"    -> store_rca
#   "noon hours"               -> hour_detail, STORE_080 hour 12
#   "what about all stores"    -> hour_detail, STORE_080 hour 12  (IDENTICAL)
#
# `mentions_store` and `want_all` were both extracted correctly and then never
# read by anything. Fourth instance of one failure mode: extracted, discarded
# because no downstream condition consumed it, answered as if nothing was lost.
# ═══════════════════════════════════════════════════════════════════════

def test_grain_followup_does_not_repeat_the_previous_answer(convo):
    convo("How did STORE_080 do?")
    previous = convo("noon hours")
    after = convo("what about all stores")

    assert after["reply"] != previous["reply"], (
        "the agent replied with the previous answer verbatim")
    assert after["intent"] == "rank"
    assert "hours 12–12" not in after["reply"]


def test_grain_followup_works_for_cities_too(convo):
    convo("How did STORE_080 do?")
    convo("noon hours")
    out = convo("what about all cities")

    assert out["intent"] == "rank"
    assert "| City |" in out["reply"]


def test_named_store_followup_is_not_hijacked_by_the_grain_rule(convo):
    """
    The guard. "what about STORE_091?" must still mean "ask the last question
    about this store" — it names an entity, so it is not a question about the
    store population.

    Note "STORE_091" cannot match \\bstores?\\b anyway: the underscore is a word
    character, so there is no boundary after "store". This test pins that,
    because a later tweak to the grain regex could quietly break it.
    """
    convo("How did STORE_080 do?")
    out = convo("what about STORE_091?")

    assert out["intent"] == "store_rca"
    assert "STORE_091" in out["reply"]
    assert "| Store |" not in out["reply"], "became a ranking table"


def test_bare_city_followup_still_becomes_a_city_summary(convo):
    """The pre-existing rule, unchanged: a bare city reference is a summary."""
    convo("How did STORE_080 do?")
    out = convo("what about Mumbai?")

    assert out["intent"] == "city_summary"


# ═══════════════════════════════════════════════════════════════════════
# C2 — AN INHERITED CITY IS KEPT, AND SAID OUT LOUD
#
# The first cut of this fix had `want_all` DISCARD an inherited scope, so
# "list all stores" mid-Bangalore returned all 233 stores network-wide. That
# was wrong, and the reasoning behind it was a bad transplant: `_decide_level`
# calls an inherited city "context, not subject", but that rule was written for
# a LEVEL bug. Wrong grain answers a different question; a narrow scope answers
# the right question over a smaller set. Only the first is a correctness bug.
#
# Scope is now inherited like everything else, disclosed, and dropped only by
# an explicit "in all cities".
# ═══════════════════════════════════════════════════════════════════════

def test_inherited_scope_is_kept_and_disclosed(convo):
    convo("How did STORE_080 do?")          # Bangalore enters context
    out = convo("list all stores")

    assert "in Bangalore" in out["reply"]
    assert out["reply"].count("| STORE_") == 97, "Bangalore has 97 stores"
    assert "carried over from earlier" in out["reply"]


def test_the_disclosure_names_the_exact_widening_phrase(convo):
    """
    The phrase is load-bearing, not decoration: "in all cities" is what sets
    city_as_scope, which is what drops the scope. If the note and the parser
    ever drift apart, the agent sends the user round the same loop.
    """
    convo("How did STORE_080 do?")
    note = convo("list all stores")["reply"]

    assert 'Ask "worst stores in all cities" to widen it.' in note

    widened = convo("worst stores in all cities")
    assert "in Bangalore" not in widened["reply"]
    assert widened["reply"].count("| STORE_") == 233


def test_in_all_cities_returns_one_table_not_two(convo):
    """
    "all stores in all cities" mentions BOTH grain nouns, so it used to trip
    `both_levels` — the compound-question path built for "the best city and
    store" — and return a city ranking AND a store ranking. The agent was
    advertising a phrase that gave the wrong shape.

    A city noun under a preposition is a SCOPE; joined by "and" it is a SECOND
    SUBJECT. That grammatical difference is the whole fix.
    """
    out = convo("list all stores in all cities")

    assert "| City |" not in out["reply"], "returned a city ranking too"
    assert out["reply"].count("Avg OR2A**") == 1, "returned two tables"
    assert out["reply"].count("| STORE_") == 233


def test_compound_questions_still_return_two_tables(convo):
    """The other half of the same distinction must not regress."""
    out = convo("tell me the best performing city and store")

    assert "| City |" in out["reply"]
    assert "| Store |" in out["reply"]


def test_a_stated_city_still_scopes_the_ranking(convo):
    convo("How did STORE_080 do?")
    out = convo("list all stores in Mumbai")

    assert "Mumbai" in out["reply"]
    assert out["reply"].count("| STORE_") < 97


def test_no_scope_note_when_there_was_nothing_to_inherit(convo):
    """A first turn has no context, so there is nothing to disclose."""
    out = convo("list all stores")

    assert "carried over from earlier" not in out["reply"]
    assert out["reply"].count("| STORE_") == 233


def test_a_city_derived_from_a_named_store_is_not_called_inherited(convo):
    """
    Defect found in the C2 fix itself: naming a store DERIVES its city, and a
    derived city is as explicit as a stated one. The first cut told the user
    that STORE_091's own city had been "carried over from earlier", which is
    untrue and undermines the disclosure everywhere else it appears.
    """
    convo("worst 5 cities")
    out = convo("what about STORE_091?")

    assert "carried over from earlier" not in out["reply"]


# ═══════════════════════════════════════════════════════════════════════
# NO TURN MAY CRASH
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("message", [
    "what about all stores",
    "list all stores",
    "show me all stores",
    "what about all cities",
    "list all cities",
    "all stores",
    "and the stores?",
])
def test_grain_phrasings_never_error(convo, message):
    """
    These all reach the same routing decision by different wordings. A live
    demo cannot afford any of them to raise.
    """
    out = convo(message)
    assert out["error"] is None, out["error"]
    assert out["reply"].strip()


# ═══════════════════════════════════════════════════════════════════════
# GRAIN FACTS ARE COMPUTED, NEVER ASKED
#
# The bug that would have made every fix above a no-op in production.
#
# `mentions_store` / `mentions_city` are not in the classify prompt, so a real
# model never emits them. `bool(data.get("mentions_store"))` was therefore
# always False on the LLM path -- the path that serves users. The A2 fix and
# the follow_up re-route both passed against the rule-based provider and did
# nothing against Groq.
#
# Two different causes, one identical symptom, which is exactly why it survived
# a green test suite AND a live bug report.
# ═══════════════════════════════════════════════════════════════════════

class _SilentProvider:
    """
    A model that returns perfectly valid JSON with the grain keys absent —
    i.e. every real model, since the prompt never asks for them.
    """
    name = "silent-test-double"

    def complete(self, system, user, **kw):
        from app.llm import LLMResponse
        return LLMResponse(
            text='{"intent":"follow_up","city":null,"store":null,"date":null,'
                 '"hour_label":null,"metric":null,"condition":null,'
                 '"direction":null,"n":null,"level":null,'
                 '"want_all":false,"both_levels":false}',
            provider=self.name)


@pytest.mark.parametrize("message,expected", [
    ("what about all stores",       {"mentions_store": True,  "want_all": True}),
    ("what about all cities",       {"mentions_city":  True,  "want_all": True}),
    ("list all stores in all cities",
     {"mentions_store": True, "city_as_scope": True, "both_levels": False}),
    ("the best city and store",     {"both_levels": True, "city_as_scope": False}),
    ("how did STORE_080 do",        {"mentions_store": False, "mentions_city": False}),
])
def test_grain_facts_survive_a_provider_that_omits_them(index, message, expected):
    """
    The contract: whatever the provider returns, these facts are correct.
    A model cannot suppress them by staying silent, because it is never asked.
    """
    from app.nodes import classify_intent

    ext = classify_intent(message, _SilentProvider(), index)
    for field, want in expected.items():
        assert getattr(ext, field) is want, (
            f"{field} was {getattr(ext, field)}, expected {want} — "
            f"the provider omitted it and nothing recomputed it")


def test_hyderabad_city_is_not_read_as_asking_about_cities(index):
    """
    Two real city names contain the word "city" ("Pune city east"). Stripping
    recognised names before the noun check is what stops naming one of them
    from reading as "asking about cities in general".
    """
    from app.nodes import classify_intent

    ext = classify_intent("how did Pune city east do", _SilentProvider(), index)
    assert ext.mentions_city is False


def test_model_may_widen_want_all_but_not_veto_it(index):
    """
    want_all and both_levels are OR-ed, not overwritten: the regex is a floor,
    so a model spotting a phrasing it misses still counts. The one thing the
    model may not do is call a prepositional scope a second subject.
    """
    from app.nodes import classify_intent

    class _Eager(_SilentProvider):
        def complete(self, system, user, **kw):
            from app.llm import LLMResponse
            return LLMResponse(
                text='{"intent":"rank","want_all":true,"both_levels":true}',
                provider=self.name)

    ext = classify_intent("list all stores in all cities", _Eager(), index)
    assert ext.want_all is True
    assert ext.both_levels is False, "city_as_scope must veto the model here"


# ═══════════════════════════════════════════════════════════════════════
# D4.1 — THE EXPANDED MENU
#
# Every shape here was previously answered as something else, silently.
# ═══════════════════════════════════════════════════════════════════════

def test_top_n_per_city_is_a_grouped_ranking(convo):
    """
    "top 5 stores of each city" used to return a CITY table plus all 97
    Bangalore stores — three bugs compounding: `each` overrode the explicit 5,
    `of` was missing from the scope check, so both grain nouns tripped the
    compound path.
    """
    out = convo("get me top 5 stores of each city")

    assert "| City |" in out["reply"] and "| Store |" in out["reply"]
    assert out["reply"].count("Avg OR2A**") == 1, "returned two tables"
    # 9 cities, at most 5 each, and not the 97 of a single city
    assert 20 < out["reply"].count("| STORE_") <= 45


def test_an_explicit_number_beats_a_quantifier(convo):
    """
    "top 5 ... of each city" contains both a 5 and the word "each". A number the
    user typed is the least ambiguous signal in the message; nothing infers
    over it.
    """
    assert convo("top 5 stores of all cities")["reply"].count("| STORE_") == 5
    assert convo("worst 3 stores per city")["reply"].count("| STORE_") <= 27


def test_top_is_disclosed_as_worst(convo):
    """An assumption the user cannot see is the same as a silent one."""
    assert 'Read "top"' in convo("top 5 stores")["reply"]
    assert 'Read "top"' not in convo("best 5 stores")["reply"]
    assert 'Read "top"' not in convo("worst 5 stores")["reply"]


def test_threshold_questions_can_answer_none(convo):
    """
    A ranking sorted by the same metric ALWAYS returns rows, so the reader
    infers those rows crossed the line. A threshold must be able to say nothing
    did.
    """
    out = convo("cities with or2a above 200 min")
    assert "0 match" in out["reply"]
    assert "Nothing crossed that line" in out["reply"]
    assert "| STORE_" not in out["reply"] and "| Noida |" not in out["reply"]


def test_threshold_percentages_are_normalised(convo):
    """
    w_breached_rate is stored 0.0-1.0. "over 50%" must become 0.5, not 50 —
    which would match nothing and read as "no stores qualify" rather than as
    a bug.
    """
    out = convo("stores with breach rate over 50%")
    assert "0 match" not in out["reply"]
    assert "| STORE_" in out["reply"]


def test_healthy_states_the_bar_instead_of_inventing_one(convo):
    """
    D4.2. With OR2A_THRESHOLD_MIN = 0 nothing is healthy. The agent must say so
    and name the bar — never quietly apply a softer threshold and present the
    survivors as a pass list.
    """
    out = convo("which cities are healthy")

    assert "Healthiest" in out["reply"]
    assert "not a pass list" in out["reply"]
    assert f"{config.OR2A_THRESHOLD_MIN} min" in out["reply"]
    assert out["reply"].count("| ") > 5, "gave an explanation but no list"


def test_time_of_day_beats_hour_detail(convo):
    """
    "morning vs evening" contains hour words, so hour_detail claims it and walks
    ONE window — answering half the question without mentioning the other half.
    """
    convo("How did STORE_080 do?")
    out = convo("morning vs evening")

    assert out["intent"] == "window_rollup"
    assert "STORE_080 by time of day" in out["reply"]
    assert "| Morning |" in out["reply"] and "| Evening |" in out["reply"]


def test_window_rollup_works_at_city_grain_too(convo):
    """It must not demand a store the user never mentioned."""
    convo("How did Bangalore do?")
    out = convo("by time of day")
    assert "Bangalore by time of day" in out["reply"]
    assert out["error"] is None


def test_a_wrapping_window_is_not_printed_as_a_range(convo):
    """
    `night` is [21, 22, 0, 1]. min–max renders "0–22", claiming the window
    covers the whole day. A dash promises everything in between.
    """
    out = convo("by time of day")
    assert "| 0–22 |" not in out["reply"]
    assert "21, 22, 0, 1" in out["reply"]


def test_trend_questions_are_declined_not_answered(convo):
    """
    The most dangerous question class here: a plausible answer is trivially
    available — describe one day in the language of a trend and every number
    quoted is real while the conclusion is fabricated.
    """
    for q in ("is it getting worse?",
              "how does bangalore compare to last week",
              "what's the trend for STORE_080"):
        out = convo(q)
        assert out["intent"] == "no_trend_data", q
        assert "single day" in out["reply"]
        assert config.DATA_DATE in out["reply"]
        assert "|" not in out["reply"], f"{q!r} still produced a table"


# ═══════════════════════════════════════════════════════════════════════
# Q2 — MULTI-QUESTION MESSAGES
# ═══════════════════════════════════════════════════════════════════════

def test_two_questions_get_two_answers(convo):
    out = convo("How did Bangalore do, and which store was worst?")

    assert out["intent"] == "multi_question"
    assert "### 1." in out["reply"] and "### 2." in out["reply"]
    assert "88,340 orders" in out["reply"]        # the city summary
    assert "Worst 5 stores" in out["reply"]       # the ranking


def test_the_second_question_inherits_from_the_first(convo):
    """
    Sequential, not parallel. "which store was worst" has no subject; running
    the pieces in order through the ordinary turn machinery means Bangalore is
    already in context by the time it runs.
    """
    out = convo("How did Bangalore do, and which store was worst?")
    assert "in Bangalore" in out["reply"]


@pytest.mark.parametrize("message", [
    "Compare STORE_003 and STORE_203",
    "Tell me the best performing city and store",
    "Show me morning and evening hours at STORE_046",
    "What were the orders and breaches for STORE_080?",
    "How did STORE_080 do?",
])
def test_single_questions_are_never_split(convo, message):
    """
    OVER-SPLITTING is the failure mode. "Compare STORE_003 and STORE_203" reads
    like two questions and is one; splitting it destroys the comparison and
    answers something nobody asked.
    """
    out = convo(message)
    assert out["intent"] != "multi_question", f"over-split: {message!r}"
    assert "### 1." not in out["reply"]


def test_splitting_degrades_to_one_question_when_the_model_fails(con):
    """
    Every failure path — no separator, an error, junk JSON, a truncated split —
    returns the ORIGINAL message. The feature can fail completely without making
    anything worse than it was before it existed.
    """
    from app.nodes import split_questions

    class _Broken:
        name = "broken"

        def complete(self, *a, **kw):
            raise RuntimeError("boom")

    class _Junk:
        name = "junk"

        def complete(self, *a, **kw):
            from app.llm import LLMResponse
            return LLMResponse(text="not json at all", provider="junk")

    class _Lossy:
        name = "lossy"

        def complete(self, *a, **kw):
            import json
            from app.llm import LLMResponse
            return LLMResponse(text=json.dumps({"questions": ["how"]}),
                               provider="lossy")

    msg = "How did Bangalore do, and which store was worst?"
    for provider in (_Broken(), _Junk(), _Lossy()):
        assert split_questions(msg, provider) == [msg], type(provider).__name__


def test_the_split_marker_is_a_constant_not_a_phrase():
    """
    RuleBasedProvider routes on this to decide split-vs-classify. Matching on
    wording meant an innocuous edit to the prompt's first sentence silently
    disabled splitting for every user without an API key — the same
    provider-divergence class as the dead mentions_store flag.
    """
    from app import prompts
    assert prompts.SPLIT_MARKER in prompts.build_split_prompt(3)


# ═══════════════════════════════════════════════════════════════════════
# DOCUMENTATION Q&A — WHOLE-CORPUS SYNTHESIS
#
# "what is or2a?" returned THREE excerpts from the SAME file, each with its own
# header. That is a search-results page, not an answer: the reader was handed
# the retrieval output and left to synthesise it themselves.
#
# The fix is synthesis, not better retrieval. The whole corpus is ~1,800 tokens
# — smaller than the classifier prompt — so the model receives all of it and a
# retrieval miss becomes structurally impossible. Which is also why there is no
# RAG here: it would select a subset of 1,800 tokens in order to send a subset
# of 1,800 tokens, buying nothing and adding a way to be wrong.
# ═══════════════════════════════════════════════════════════════════════

from pathlib import Path                                        # noqa: E402


class _LocalMCP:
    """Reads the real docs/ from disk. No npx, no network, same interface."""
    docs_dir = Path(__file__).resolve().parents[1] / "docs"

    def __init__(self):
        self.corpus_reads = 0

    def list_documents(self):
        return [p.name for p in sorted(self.docs_dir.glob("*.md"))]

    def read_document(self, name):
        return (self.docs_dir / name).read_text()

    def read_corpus(self):
        self.corpus_reads += 1
        return [(p.name, p.read_text().strip())
                for p in sorted(self.docs_dir.glob("*.md"))]

    def search(self, q):
        return []

    def connect_background(self):
        pass


class _CapturingProvider:
    """Records the prompt so the corpus contents can be asserted."""
    name = "capturing"

    def __init__(self, reply="OR2A is a metric [order_ready_to_assignment.md]."):
        self.reply = reply
        self.systems = []

    def complete(self, system, user, **kw):
        from app.llm import LLMResponse
        self.systems.append(system)
        if "TASK::ANSWER_FROM_DOCS" in system:
            return LLMResponse(text=self.reply, provider=self.name)
        import json
        return LLMResponse(text=json.dumps({"intent": "methodology"}),
                           provider=self.name)


def test_the_whole_corpus_reaches_the_model(con):
    """
    Not a retrieved subset. Every document, every time — that is what makes a
    retrieval miss impossible rather than merely unlikely.
    """
    from app.graph import build_agent

    p = _CapturingProvider()
    mcp = _LocalMCP()
    agent = build_agent(con, provider=p, mcp=mcp, connect_mcp=False)
    agent.run_turn("doc-1", "what is or2a?")

    doc_prompts = [s for s in p.systems if "TASK::ANSWER_FROM_DOCS" in s]
    assert len(doc_prompts) == 1, "the doc prompt was not used"

    prompt = doc_prompts[0]
    assert prompt.count("### FILE:") == len(mcp.list_documents()) == 3
    for name in mcp.list_documents():
        assert name in prompt, f"{name} was not sent to the model"


def test_the_corpus_still_fits_comfortably_in_one_prompt(con):
    """
    The measurement the no-RAG decision rests on. If the docs ever grow past
    roughly 50k tokens this stops being true and retrieval starts to earn its
    keep — so the assumption is asserted, not assumed.
    """
    mcp = _LocalMCP()
    total = sum(len(text) for _, text in mcp.read_corpus())
    assert total // 4 < 8_000, (
        f"corpus is now ~{total // 4:,} tokens — revisit the no-RAG decision")


def test_a_doc_answer_is_synthesised_not_dumped(con):
    from app.graph import build_agent

    agent = build_agent(con, provider=_CapturingProvider(),
                        mcp=_LocalMCP(), connect_mcp=False)
    reply = agent.run_turn("doc-2", "what is or2a?")["reply"]

    assert "OR2A is a metric" in reply
    assert "## Source Table" not in reply, "raw document section was pasted"
    assert "read via MCP" in reply, "provenance must stay visible"


def test_the_fallback_merges_sections_per_file(con):
    """
    The specific bug: three excerpts from ONE file, three identical headers.
    This path runs whenever there is no API key, so it must be fixed WITHOUT
    a model — not merely papered over by synthesis.
    """
    from app.graph import build_agent
    from app.llm import RuleBasedProvider
    from app.entities import EntityIndex

    agent = build_agent(con, provider=RuleBasedProvider(EntityIndex(con).cities),
                        mcp=_LocalMCP(), connect_mcp=False)
    reply = agent.run_turn("doc-3", "what is or2a?")["reply"]

    assert reply.count("_order_ready_to_assignment.md_") <= 1, (
        "the same file is still announced more than once")
    assert "Read via MCP" in reply


def test_a_failed_model_call_degrades_to_excerpts_not_to_nothing(con):
    """Real source text beats an apology."""
    from app.graph import build_agent
    from app.llm import LLMResponse

    class _Broken(_CapturingProvider):
        def complete(self, system, user, **kw):
            if "TASK::ANSWER_FROM_DOCS" in system:
                return LLMResponse(text="", provider="broken", ok=False,
                                   error="429")
            import json
            return LLMResponse(text=json.dumps({"intent": "methodology"}),
                               provider="broken")

    agent = build_agent(con, provider=_Broken(), mcp=_LocalMCP(),
                        connect_mcp=False)
    reply = agent.run_turn("doc-4", "what is or2a?")["reply"]

    assert "reference documentation" in reply
    assert "Read via MCP" in reply
    assert "DATEDIFF" in reply, "the real source text should still be shown"


@pytest.mark.parametrize("question,intent", [
    # definitions — no entity named
    ("what is or2a?", "methodology"),
    ("whats or2a", "methodology"),
    ("what is a booking gap", "methodology"),
    ("what is the pileup threshold", "methodology"),
    ("what is demand spike", "methodology"),
    # values — an entity IS named, so these stay data lookups
    ("what was STORE_003 or2a", "lookup"),
    ("what is the breach rate for STORE_003", "lookup"),
    ("what is the or2a in Bangalore", "lookup"),
])
def test_a_definition_question_is_not_a_data_lookup(con, question, intent):
    """
    "what is or2a?" matched the `lookup` pattern — "what is" plus a metric word
    — and was answered as a lookup with no entity, producing:

        **No data for None**

    The two differ by whether an ENTITY is named: "what is OR2A" asks what the
    metric means, "what was STORE_003's OR2A" asks for a number.
    """
    from app.llm import RuleBasedProvider
    from app.entities import EntityIndex

    got = RuleBasedProvider(EntityIndex(con).cities)._classify(question)["intent"]
    assert got == intent, f"{question!r} -> {got}, expected {intent}"


# ═══════════════════════════════════════════════════════════════════════
# ADVERSARIAL EDGE CASES
#
# Every test below is a bug that was live until adversarial probing found it.
# All three are the same shape: something the user said was misread, and the
# answer came back confident with no sign anything had gone wrong.
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("question,comparator", [
    ("stores with breach rate not more than 50%", "<="),
    ("stores with breach rate no more than 50%", "<="),
    ("stores with breach rate not less than 50%", ">="),
    ("stores with breach rate not below 50%", ">="),
    # un-negated forms must be untouched
    ("stores with breach rate more than 50%", ">"),
    ("stores with breach rate less than 50%", "<"),
    ("stores with breach rate at least 50%", ">="),
    ("stores with breach rate at most 50%", "<="),
])
def test_negated_thresholds_are_not_silently_inverted(question, comparator):
    """
    "not more than 50%" parsed as ">" — the exact opposite question. The agent
    returned the 14 WORST stores to someone asking for the compliant ones, and
    every row in that table was real.

    A silent inversion is the worst failure available here: nothing about the
    output looks wrong.
    """
    from app.llm import threshold_facts
    got = threshold_facts(question.lower())
    assert got is not None, f"{question!r} did not parse at all"
    assert got["comparator"] == comparator, (
        f"{question!r} -> {got['comparator']}, expected {comparator}")


def test_a_threshold_value_is_not_read_as_a_store_code(con):
    """
    "stores with breach rate over 50%" resolved 50 to STORE_050 and pinned it
    as the conversation's store. Three turns later an unrelated question failed
    because the session held a Bangalore store while the user had moved on —
    and the real cause was four turns back, invisible.

    A number governed by a comparator, or carrying a unit, is a QUANTITY.
    """
    from app.graph import build_agent
    from app.llm import RuleBasedProvider
    from app.entities import EntityIndex

    agent = build_agent(con, provider=RuleBasedProvider(EntityIndex(con).cities),
                        connect_mcp=False)
    out = agent.run_turn("thresh-ctx", "stores with breach rate over 50%")

    assert out["context"].get("store") is None, (
        f"threshold value became {out['context']['store']}")
    assert out["error"] is None


@pytest.mark.parametrize("question", [
    "stores with breach rate over 50%",
    "cities under 30 min or2a",
    "stores with more than 5 problem hours",
    "stores with or2a above 200 min",
])
def test_no_threshold_phrasing_pins_a_store(con, question):
    from app.graph import build_agent
    from app.llm import RuleBasedProvider
    from app.entities import EntityIndex

    agent = build_agent(con, provider=RuleBasedProvider(EntityIndex(con).cities),
                        connect_mcp=False)
    out = agent.run_turn(f"t-{abs(hash(question))}", question)
    assert out["context"].get("store") is None, question


def test_an_unknown_city_is_declined_not_silently_dropped(con):
    """
    "worst 5 stores in Atlantis" ranked the ENTIRE NETWORK. Nothing extracted
    "Atlantis", so the scope stayed empty and a confident global answer came
    back for a place that does not exist.

    This is the TBC8 failure on the city path. The store path already declined
    correctly; the city path did not.
    """
    from app.graph import build_agent
    from app.llm import RuleBasedProvider
    from app.entities import EntityIndex

    agent = build_agent(con, provider=RuleBasedProvider(EntityIndex(con).cities),
                        connect_mcp=False)
    out = agent.run_turn("atlantis", "worst 5 stores in Atlantis")

    assert out["error"] is not None, "answered a question about a fake city"
    assert "atlantis" in out["reply"].lower()
    assert "| STORE_" not in out["reply"], "returned a ranking anyway"


@pytest.mark.parametrize("question", [
    "worst 5 stores in Mumbai",
    "list all stores in all cities",
    "top 5 stores of each city",
    "morning hours at STORE_046",
    "walk me through the morning hours",
    "what happened at 7pm in Bangalore",
    "how did Pune city east do",
    "which cities are healthy",
    "stores with breach rate over 50%",
    "morning vs evening",
    "worst stores in all cities",
    "what about all stores",
])
def test_the_unknown_place_guard_has_no_false_positives(con, question):
    """
    A false positive here declines a VALID question — worse than the bug it
    prevents. So the guard is deliberately conservative and this list is the
    proof, not the intention.
    """
    from app.llm import grain_facts
    from app.entities import EntityIndex

    flagged = grain_facts(question.lower(), EntityIndex(con).cities)["unknown_place"]
    assert flagged is None, f"{question!r} wrongly flagged as {flagged!r}"


def test_mixed_tiers_in_one_session_never_error(con):
    """
    Tier 1 (RCA), Tier 2 (tools), methodology (docs) and multi-question,
    interleaved in ONE session. The failure this guards against is not a crash
    — it is one tier POISONING the context for the next, which is exactly how
    the STORE_050 bug surfaced three turns after it was caused.
    """
    from app.graph import build_agent
    from app.llm import RuleBasedProvider
    from app.entities import EntityIndex

    agent = build_agent(con, provider=RuleBasedProvider(EntityIndex(con).cities),
                        connect_mcp=False)
    script = [
        "Why did STORE_003 underperform?",       # T1
        "what is the pileup threshold",          # methodology
        "walk me through the morning hours",     # T1, store must survive
        "worst 5 stores",                        # T2
        "what is or2a",                          # methodology
        "stores with breach rate over 50%",      # T2 threshold
        "morning vs evening",                    # T2 window
        "top 5 stores of each city",             # T2 grouped
        "which cities are healthy",              # T2 inverse
        "walk me through the evening hours",     # T1 again
    ]
    for q in script:
        out = agent.run_turn("mixed-tiers", q)
        assert out["error"] is None, f"{q!r} failed: {out['error']}"
        assert out["reply"].strip(), f"{q!r} returned nothing"


# ═══════════════════════════════════════════════════════════════════════
# THE RANKED METRIC IS ALREADY A COLUMN
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("metric", sorted(["total_orders", "breached_count",
                                           "w_avg_or2a", "problem_hours",
                                           "w_breached_rate", "noshow_count"]))
@pytest.mark.parametrize("level", ["store", "city"])
def test_a_ranking_never_prints_the_same_column_twice(con, metric, level):
    """
    "best 10 stores by Orders" rendered

        | Store | Orders | Orders | Breached |
        | STORE_036 | 2,673 | 2673 | 500 |

    The metric column and the fixed context column were the same field, once
    formatted and once raw. Two identical headers with different-looking numbers
    under them reads as a data error, and a reader cannot tell which to trust.
    """
    from app import tools

    res = tools.rank_entities(con, level=level, metric=metric, n=3,
                              direction="best")
    labels = [label for _, label in res.columns]
    assert len(labels) == len(set(labels)), f"duplicate columns: {labels}"

    keys = [key for key, _ in res.columns]
    assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"


# ═══════════════════════════════════════════════════════════════════════
# A SHARE-OF-A-WHOLE QUESTION IS NOT A RANKING
#
# "what percentage of total orders came from the 10 busiest stores" was
# classified as `rank` — reasonably, it does contain "10 busiest stores" — and
# answered with a table of ten stores and NO PERCENTAGE ANYWHERE. The question
# was never addressed and nothing said so.
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("question", [
    "what percentage of total orders came from the 10 busiest stores",
    "what share of total orders came from the top 5 stores",
    "what proportion of all orders came from Bangalore",
    "what % of total breaches came from the worst 10 stores",
    "what percent of overall orders do the top 3 stores account for",
])
def test_share_questions_are_recognised(con, question):
    from app.llm import grain_facts
    from app.entities import EntityIndex

    assert grain_facts(question.lower(), EntityIndex(con).cities)["wants_share"], \
        f"{question!r} would still be answered as a ranking"


@pytest.mark.parametrize("question", [
    # a breach RATE is a share, and a tested tool computes it — routing this to
    # Tier 3 would downgrade a `validated` answer to `screened`
    "what percentage of orders breached at STORE_003",
    "what is the breach rate for STORE_003",
    "stores with breach rate over 50%",
    "stores with breach rate not more than 50%",
    "worst 5 stores",
    "which cities are healthy",
    "how many stores had pileup",
    "top 5 stores of each city",
    "morning vs evening",
])
def test_share_detection_does_not_steal_tool_questions(con, question):
    from app.llm import grain_facts
    from app.entities import EntityIndex

    assert not grain_facts(question.lower(), EntityIndex(con).cities)["wants_share"], \
        f"{question!r} was diverted away from a tested tool"


def test_a_share_question_overrides_the_classifier(con):
    """
    The model called it `rank`, with n=10 and metric=total_orders — a defensible
    reading of "10 busiest stores". The deterministic fact must still win, or
    the fix only works when the model already agrees.
    """
    import json
    from app.graph import build_agent
    from app.llm import LLMResponse

    class _SaysRank:
        name = "says-rank"

        def complete(self, system, user, json_mode=False, **kw):
            if "TASK::GENERATE_SQL" in system:
                return LLMResponse(text=json.dumps({"sql":
                    "SELECT SUM(total_orders) AS t FROM safe_store_day"}),
                    provider=self.name)
            if "TASK::SPLIT_QUESTIONS" in system:
                return LLMResponse(text=json.dumps({"questions": [user]}),
                                   provider=self.name)
            return LLMResponse(text=json.dumps(
                {"intent": "rank", "n": 10, "direction": "best",
                 "metric": "total_orders"}), provider=self.name)

    agent = build_agent(con, provider=_SaysRank(), connect_mcp=False)
    out = agent.run_turn(
        "share-1", "what percentage of total orders came from the 10 busiest stores")

    assert out["provenance"] == config.PROVENANCE_SCREENED
    assert "```sql" in out["reply"]
    assert "| STORE_" not in out["reply"], "still answered with a ranking table"


# ═══════════════════════════════════════════════════════════════════════
# B3 / B4 — ONE VOICE PER ANSWER, ONE ANSWER PER HOUR
# ═══════════════════════════════════════════════════════════════════════

def _json(payload):
    import json
    from app.llm import LLMResponse
    return LLMResponse(text=json.dumps(payload), provider="scripted")


class _NarratorBase:
    """Classifies to a fixed intent; subclasses decide how narration behaves."""
    name = "scripted"

    def __init__(self, intent_payload):
        self.intent_payload = intent_payload
        self.narrations = 0

    def complete(self, system, user, json_mode=False, **kw):
        from app.llm import LLMResponse
        if "TASK::SPLIT_QUESTIONS" in system:
            return _json({"questions": [user]})
        if json_mode:
            return _json(self.intent_payload)
        self.narrations += 1
        return self._narrate()


def test_a_partial_narration_makes_the_whole_answer_deterministic(con):
    """
    B3. Three blocks went out, one came back as model prose and two as
    templates, and all three were rendered — so one response spoke in two
    voices:

        Hour 21  "OR2A was 755.0 min over threshold due to..."          template
        Hour 14  "The SLA was missed with an average OR2A of 846.6..."  prose

    All-or-nothing is the fix. The template carries MORE figures than the prose
    anyway, so consistency costs nothing that matters.
    """
    from app.graph import build_agent
    from app.llm import LLMResponse

    class _Flaky(_NarratorBase):
        def _narrate(self):
            if self.narrations == 1:
                return LLMResponse(
                    text="The SLA was missed with an average OR2A of 846.6 minutes.",
                    provider=self.name)
            return LLMResponse(text="", provider=self.name, ok=False, error="429")

    p = _Flaky({"intent": "store_rca", "store": "STORE_080"})
    reply = build_agent(con, provider=p, connect_mcp=False).run_turn(
        "b3-partial", "why did STORE_080 underperform")["reply"]

    assert "The SLA was missed" not in reply, "mixed prose and templates"
    assert reply.count("over threshold due to") >= 2, "lost the summaries entirely"


def test_full_narration_is_still_used_when_every_block_succeeds(con):
    """All-or-nothing must not become never."""
    from app.graph import build_agent
    from app.llm import LLMResponse

    class _Reliable(_NarratorBase):
        def _narrate(self):
            return LLMResponse(text="Narrated sentence for this hour.",
                               provider=self.name)

    p = _Reliable({"intent": "store_rca", "store": "STORE_080"})
    reply = build_agent(con, provider=p, connect_mcp=False).run_turn(
        "b3-full", "why did STORE_080 underperform")["reply"]

    assert reply.count("Narrated sentence for this hour.") >= 2
    assert "over threshold due to" not in reply


def test_the_same_hour_is_worded_identically_across_turns(con):
    """
    B4. Hour 14 of STORE_080, one conversation:

        Q1  "The SLA was missed with an average OR2A of 846.6 minutes..."
        Q2  "OR2A was 846.6 min over threshold due to a demand spike..."

    Same facts, same numbers, different words — because one turn narrated and
    the other fell back. Nothing was wrong, but it reads as the agent changing
    its mind.
    """
    from app.graph import build_agent
    from app.llm import LLMResponse

    class _Varying(_NarratorBase):
        def _narrate(self):
            return LLMResponse(text=f"Variant number {self.narrations}.",
                               provider=self.name)

    p = _Varying({"intent": "hour_detail", "store": "STORE_080",
                  "hour_label": "morning"})
    agent = build_agent(con, provider=p, connect_mcp=False)

    first = agent.run_turn("b4", "walk me through the morning hours")["reply"]
    calls_after_first = p.narrations
    second = agent.run_turn("b4", "walk me through the morning hours")["reply"]

    assert first == second, "the same hours were worded differently"
    assert p.narrations == calls_after_first, (
        "re-asking called the model again instead of using the cache")


def test_the_summary_cache_does_not_leak_between_sessions(con):
    """
    Per-session, not global. A global cache would make a change to the
    narration prompt invisible until the process restarted.
    """
    from app.graph import build_agent
    from app.llm import LLMResponse

    class _Counting(_NarratorBase):
        def _narrate(self):
            return LLMResponse(text=f"Sentence {self.narrations}.",
                               provider=self.name)

    p = _Counting({"intent": "hour_detail", "store": "STORE_080",
                   "hour_label": "morning"})
    agent = build_agent(con, provider=p, connect_mcp=False)

    agent.run_turn("sess-a", "walk me through the morning hours")
    after_a = p.narrations
    agent.run_turn("sess-b", "walk me through the morning hours")

    assert p.narrations > after_a, "a different session reused cached summaries"
