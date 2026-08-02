"""
Tier 3 — the generated-SQL path.

WHAT THESE TESTS ARE ACTUALLY PROTECTING
----------------------------------------
Not "does the SQL run". A wrong query runs perfectly.

The danger is a query that is valid, fast, and silently wrong — and this dataset
has two traps that produce exactly that:

  TRAP 1  AVG(w_avg_or2a) instead of the order-weighted rollup.
          Off by -40% to +35% at city level, and it REORDERS the cities.
  TRAP 2  Summing slot-level rider-supply columns across hours.
          Inflates by ~1.75x — 128,824 rider slots against a true 73,465.

Both are asserted numerically below, so the tests fail if anyone ever "tidies
up" the safe views and reintroduces the footgun.

Three layers are tested separately, because they fail independently:

  layer 2  SHAPED VIEWS   the wrong query cannot be WRITTEN
  layer 3  THE SCREEN     the wrong query cannot be EXECUTED
  the loop RETRY          a rejection is fed back with the reason
"""

import pytest

import config
from app import sql_path, tables
from app.sql_path import ScreenRejection, screen


# ═══════════════════════════════════════════════════════════════════════
# LAYER 2 — the traps, measured
# ═══════════════════════════════════════════════════════════════════════

def test_weighted_rollup_via_safe_view_matches_the_tested_aggregate(con):
    """
    `or2a_order_minutes` must reproduce agg_city_day exactly. If it drifts, the
    "correct query is the obvious query" claim quietly stops being true and
    every Tier-3 answer is wrong in a way nothing detects.
    """
    rows = con.execute("""
        SELECT s.city,
               SUM(s.or2a_order_minutes) / SUM(s.total_orders) AS via_view,
               ANY_VALUE(c.w_avg_or2a)                          AS tested
        FROM safe_store_day s
        JOIN safe_city_day c ON c.city = s.city
        GROUP BY s.city
    """).fetchall()

    assert len(rows) == 9
    for city, via_view, tested in rows:
        assert abs(via_view - tested) < 0.05, (
            f"{city}: safe view gives {via_view:.2f}, tested table {tested:.2f}")


def test_the_naive_average_really_is_wrong_and_reorders_cities(con):
    """
    The measurement behind the whole design. If this ever stops being true the
    screen's strictness is unjustified — and if it stays true, nobody may
    quietly relax it.
    """
    rows = con.execute("""
        SELECT city,
               SUM(or2a_order_minutes) / SUM(total_orders) AS correct,
               AVG(w_avg_or2a)                             AS naive
        FROM safe_store_day GROUP BY city
    """).fetchall()

    worst_err = max(abs(n - c) / c for _, c, n in rows)
    assert worst_err > 0.30, "the naive average is no longer badly wrong"

    correct_order = [c for c, _, _ in sorted(rows, key=lambda r: -r[1])]
    naive_order = [c for c, _, _ in sorted(rows, key=lambda r: -r[2])]
    assert correct_order != naive_order, (
        "the naive average no longer changes the ranking — the strongest "
        "argument for the screen has gone")


def test_replicated_supply_columns_are_absent_from_the_hourly_view(con):
    """
    Layer 2 in its purest form: a column that is not in the schema cannot be
    double-counted, no matter what the model writes.
    """
    cols = {c for (c,) in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'safe_store_hour'").fetchall()}

    for banned in ("current_size", "booked_size", "orginal_size",
                   "noshow_count", "booked_hours_per_hour"):
        assert banned not in cols, f"{banned} is reachable per-hour again"

    with pytest.raises(Exception):
        con.execute("SELECT SUM(current_size) FROM safe_store_hour").fetchall()


def test_slot_supply_is_deduplicated(con):
    """The 1.75x inflation, asserted rather than described."""
    naive = con.execute(
        "SELECT SUM(current_size) FROM fact_store_hour").fetchone()[0]
    true = con.execute(
        "SELECT SUM(current_size) FROM safe_slot_supply").fetchone()[0]
    assert naive > true * 1.5
    assert true == con.execute(
        "SELECT SUM(current_size) FROM slot_supply").fetchone()[0]


# ═══════════════════════════════════════════════════════════════════════
# LAYER 3 — the screen
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("sql,fragment", [
    # the two traps
    ("SELECT city, AVG(w_avg_or2a) FROM safe_store_day GROUP BY city", "AVG"),
    ("SELECT city, AVG(avg_or2a) FROM safe_store_hour GROUP BY city", "AVG"),
    ("SELECT MEDIAN(avg_or2a) FROM safe_store_hour", "MEDIAN"),
    ("SELECT AVG(DISTINCT w_avg_or2a) FROM safe_store_day", "AVG"),
    ("SELECT SUM(orginal_size) FROM safe_slot_supply", "slot-level"),
    # reaching outside the allowlist
    ("SELECT * FROM agg_store_day", "not available"),
    ("SELECT * FROM fact_store_hour", "not available"),
    ("SELECT * FROM safe_store_day JOIN dim_store USING (store)", "not available"),
    # not read-only / not a single SELECT
    ("DROP TABLE fact_store_hour", "SELECT or WITH"),
    ("UPDATE safe_store_day SET total_orders = 0", "SELECT or WITH"),
    ("SELECT 1; DROP TABLE x", "single statement"),
    ("SELECT * FROM read_csv('/etc/passwd')", "read-only"),
    ("SELECT * FROM safe_city_day; ATTACH 'x.db'", "single statement"),
    ("", "empty"),
])
def test_screen_rejects(sql, fragment):
    with pytest.raises(ScreenRejection) as e:
        screen(sql)
    assert fragment.lower() in e.value.reason.lower()


@pytest.mark.parametrize("sql", [
    "SELECT city, SUM(or2a_order_minutes)/SUM(total_orders) AS or2a "
    "FROM safe_store_day GROUP BY city",
    "WITH t AS (SELECT city, total_orders FROM safe_city_day) SELECT * FROM t",
    "SELECT store, hour, avg_or2a FROM safe_store_hour WHERE hour = 14",
    "SELECT SUM(current_size) FROM safe_slot_supply",
])
def test_screen_accepts_correct_queries(sql):
    assert screen(sql).lstrip().lower().startswith(("select", "with"))


def test_screen_always_enforces_a_row_cap():
    assert f"LIMIT {config.SQL_ROW_LIMIT}" in screen("SELECT store FROM safe_store_day")
    tightened = screen("SELECT store FROM safe_store_day LIMIT 5000")
    assert f"LIMIT {config.SQL_ROW_LIMIT}" in tightened
    assert "5000" not in tightened
    # a limit BELOW the cap is the user's business and is left alone
    assert "LIMIT 3" in screen("SELECT store FROM safe_store_day LIMIT 3")


def test_a_cte_name_is_not_mistaken_for_an_unknown_table():
    screen("WITH ranked AS (SELECT store FROM safe_store_day) "
           "SELECT * FROM ranked")


def test_every_safe_view_is_queryable(con):
    """The allowlist and the database must not drift apart."""
    for view in tables.SAFE_VIEWS:
        con.execute(f"SELECT * FROM {view} LIMIT 1").fetchall()


# ═══════════════════════════════════════════════════════════════════════
# THE RETRY LOOP
# ═══════════════════════════════════════════════════════════════════════

class _Scripted:
    """A provider returning a fixed sequence of queries."""
    name = "scripted"

    def __init__(self, *sqls):
        self.sqls = list(sqls)
        self.prompts = []

    def complete(self, system, user, **kw):
        import json
        from app.llm import LLMResponse
        self.prompts.append(user)
        sql = self.sqls.pop(0) if self.sqls else ""
        return LLMResponse(text=json.dumps({"sql": sql}), provider=self.name)


def test_a_rejection_is_fed_back_with_its_reason(con):
    """
    The retry is not hope. The screen already computed the exact correction, so
    it is handed over verbatim — most rejections are the model reaching for
    AVG(avg_or2a), which it fixes when told what to use instead.
    """
    p = _Scripted(
        "SELECT city, AVG(w_avg_or2a) AS or2a FROM safe_store_day GROUP BY city",
        "SELECT city, SUM(or2a_order_minutes)/SUM(total_orders) AS or2a "
        "FROM safe_store_day GROUP BY city",
    )
    res = sql_path.answer(con, "average or2a per city", p)

    assert res.error is None
    assert len(res.rows) == 9
    assert len(p.prompts) == 2
    assert "REJECTED" in p.prompts[1]
    assert "or2a_order_minutes" in p.prompts[1], "the fix was not passed back"


def test_an_execution_error_is_also_fed_back(con):
    p = _Scripted(
        "SELECT nonexistent_column FROM safe_store_day",
        "SELECT store FROM safe_store_day",
    )
    res = sql_path.answer(con, "list stores", p)
    assert res.error is None
    assert "FAILED to execute" in p.prompts[1]


def test_it_gives_up_after_the_configured_retries(con):
    bad = "SELECT AVG(w_avg_or2a) FROM safe_store_day"
    p = _Scripted(*([bad] * (config.SQL_MAX_RETRIES + 5)))
    res = sql_path.answer(con, "average or2a", p)

    assert res.error is not None
    assert "screen" in res.error.lower()
    assert len(p.prompts) == config.SQL_MAX_RETRIES + 1, "wrong number of attempts"


def test_an_empty_sql_response_is_a_clean_decline(con):
    res = sql_path.answer(con, "what's the weather", _Scripted(""))
    assert res.error is not None
    assert not res.rows


# ═══════════════════════════════════════════════════════════════════════
# THE LABEL
# ═══════════════════════════════════════════════════════════════════════

def test_the_answer_always_shows_the_sql_and_the_caveat(con):
    p = _Scripted("SELECT city, total_orders FROM safe_city_day")
    out = sql_path.render(sql_path.answer(con, "orders per city", p), "orders per city")

    assert "```sql" in out, "the query must always be visible — it IS the audit"
    assert "screened, not verified" in out.lower()
    assert "order-weighted" in out


def test_a_failed_answer_still_redirects(con):
    out = sql_path.render(sql_path.answer(con, "x", _Scripted("")), "x")
    assert "tested tools cover" in out


# ═══════════════════════════════════════════════════════════════════════
# RETRY-AFTER PARSING
#
# This decides "retry" vs "give up now", so a format it cannot read costs
# real quota. Groq's "try again in 8m24.575999999s" parsed as None, so the
# not-worth-waiting branch never ran and an EXHAUSTED DAILY quota was retried
# twice with backoff — three failed calls for a wall that would not move for
# eight minutes.
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("error,expected", [
    ("Please try again in 8m24.575999999s.", 504.576),   # Groq, compound
    ("Please try again in 30s", 30.0),                   # Groq, simple
    ("Please try again in 1h2m3s", 3723.0),              # Groq, hours
    ("retryDelay: '60s'", 60.0),                         # Google
    ("Please retry after 20 seconds", 20.0),             # OpenAI
    ("some unrelated failure", None),
])
def test_retry_delay_is_parsed_from_every_provider_format(error, expected):
    from app.llm import parse_retry_delay
    got = parse_retry_delay(error)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected, abs=0.01)


def test_a_compound_duration_is_not_truncated_to_its_seconds():
    """
    The specific trap: reading only the trailing "24.5s" from "8m24.5s" gives
    24 seconds, which is UNDER the wait threshold — producing exactly the
    pointless retry this parsing exists to prevent.
    """
    from app.llm import parse_retry_delay
    assert parse_retry_delay("try again in 8m24.5s") > config.LLM_MAX_RETRY_WAIT_SECONDS


def test_an_exhausted_daily_quota_is_not_retried():
    """
    A 429 is transient in general, so it retries — but not when the server has
    said the wall lasts minutes. Backoff cannot outrun a daily quota.
    """
    from app.llm import _with_retry, LLMResponse

    calls = []
    quota = ("Error code: 429 - Rate limit reached ... on tokens per day (TPD): "
             "Limit 100000, Used 99644. Please try again in 8m24.575999999s")

    def failing():
        calls.append(1)
        return LLMResponse(text="", provider="groq", ok=False, error=quota)

    resp = _with_retry(failing, "groq", max_retries=2)
    assert resp.ok is False
    assert len(calls) == 1, f"retried {len(calls) - 1}x against an 8-minute wall"


def test_a_short_rate_limit_is_still_retried():
    """The guard must not disable retries altogether."""
    from app.llm import _with_retry, LLMResponse

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            return LLMResponse(text="", provider="groq", ok=False,
                               error="Error code: 503 - service unavailable")
        return LLMResponse(text="{}", provider="groq")

    assert _with_retry(flaky, "groq", max_retries=2).ok is True
    assert len(calls) == 2
