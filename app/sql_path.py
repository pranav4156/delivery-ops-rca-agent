"""
Tier 3 — generated SQL for questions no tool covers.

THE HONEST FRAMING
------------------
Everything else in this codebase computes numbers with tested Python and tested
SQL, and returns `validated`. This module lets a language model write the query.
That is a different guarantee and it gets a different label: `screened`.

Not "unverified" as a disclaimer bolted on, but as an accurate description --
we generated the query, we screened it against the traps we know about, and we
did not check the answer.

WHY THIS IS SAFE ENOUGH TO SHIP, AND WHY IT IS STILL NOT `validated`
--------------------------------------------------------------------
Three layers, in increasing order of strength:

  1. CONTEXT (advice)      the prompt explains the grain and the pitfalls
  2. SHAPED VIEWS (structural)  the wrong query cannot be written
  3. THE SCREEN (mechanical)    the wrong query cannot be executed

Layer 1 alone is what most text-to-SQL systems ship, and it is the weakest: a
warning in a prompt is a suggestion. The model follows it or does not, and
nothing downstream can tell which happened. Worse, it lowers the error rate
without telling you WHICH answers are wrong -- so you start trusting it.

Layer 2 is the real work. See tables.py: `or2a_order_minutes` makes the correct
weighted rollup the OBVIOUS query, and the replicated supply columns are simply
absent from the hourly view. A column that is not in the schema cannot be
double-counted.

Layer 3 catches what is left, mechanically, before anything runs.

But "we removed the traps we know about" is not "this answer is correct", and
the label must not pretend otherwise. Hence `screened`.

THE CONCRETE DANGER, MEASURED
-----------------------------
    SELECT city, AVG(w_avg_or2a) FROM agg_store_day GROUP BY city

Valid SQL. Runs in 4ms. Returns clean numbers. Wrong by -40% to +35%, and it
REORDERS the cities -- Chennai is 3rd worst correctly and 6th worst naively.
Ask "which city needs attention" and an ops director is sent to the wrong city.

Nothing about that output looks wrong. That is the entire argument for layers
2 and 3.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import duckdb

import config
from app import llm as llm_mod
from app import prompts, tables


class ScreenRejection(Exception):
    """A generated query failed the screen. Carries a reason the model can act on."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class SqlAnswer:
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    total_rows: int = 0
    truncated: bool = False
    error: str | None = None
    attempts: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# THE SCREEN
# ═══════════════════════════════════════════════════════════════════════

# Anything that writes, changes shape, or reaches outside the database.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|truncate|replace|grant|revoke|"
    r"attach|detach|copy|export|import|install|load|pragma|set|call|"
    r"read_csv|read_parquet|read_json|glob)\b", re.I)

# TRAP 1, mechanically enforced. Any unweighted aggregate over an OR2A column
# is the exact mistake measured above. MEDIAN and MODE are included because they
# are equally unweighted and equally plausible to a model reaching for "average".
_UNWEIGHTED_OR2A = re.compile(
    r"\b(avg|mean|median|mode)\s*\(\s*(distinct\s+)?[\w.]*\b"
    r"(avg_or2a|w_avg_or2a)\b", re.I)

# TRAP 2, mechanically enforced. These columns exist only on the raw fact table
# and are slot-level values replicated across hours; the safe hourly view omits
# them entirely. Naming one means the model reached past the allowlist.
_REPLICATED = re.compile(
    r"\b(orginal_size|booked_hours_per_hour|rider_hours_per_hour)\b", re.I)

_IDENTIFIER_AFTER_FROM = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][\w.]*)", re.I)
_LIMIT_RE = re.compile(r"\blimit\s+(\d+)", re.I)
_CTE_NAME = re.compile(r"(?:with|,)\s+([a-zA-Z_]\w*)\s+as\s*\(", re.I)

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"--[^\n]*")
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")


def _normalise(sql: str) -> str:
    """
    A copy of the query with comments removed and string literals blanked.

    EVERY PATTERN CHECK RUNS AGAINST THIS; THE ORIGINAL IS WHAT EXECUTES.

    Three real cases, found by adversarial testing:

        SELECT 'AVG(avg_or2a) is wrong' AS note ...
            The trap word is DATA, not code. Rejecting it is a false positive.

        SELECT city FROM safe_city_day -- ; DROP TABLE x
            The ';' is inside a comment, so DuckDB sees one statement. Reading
            it as statement-stacking rejects a legitimate query -- and models do
            write comments.

        SELECT city FROM safe_city_day /* DROP TABLE x */
            Same, for block comments.

    Stripping comments is not a weakening: a comment cannot execute, so nothing
    can hide in one. Blanking literals rather than deleting them preserves
    offsets and keeps the surrounding syntax intact.

    Literals become '' rather than vanishing, so `WHERE city = 'x'` stays valid
    to a reader of the normalised text.
    """
    out = _BLOCK_COMMENT.sub(" ", sql)
    out = _LINE_COMMENT.sub(" ", out)
    return _STRING_LITERAL.sub("''", out)


def screen(sql: str) -> str:
    """
    Reject anything unsafe or known-wrong; return the query to actually run.

    Returns a possibly-modified query (a LIMIT is appended if absent).
    Raises ScreenRejection with a reason the model can act on.

    Every rule here is mechanical. None of them requires understanding the
    question -- which is the point: a screen that needed judgment would need the
    same judgment the model already failed to apply.
    """
    if not sql or not sql.strip():
        raise ScreenRejection("empty query")

    q = sql.strip().rstrip(";").strip()

    # Comments stripped and literals blanked before ANY pattern check, so a trap
    # word appearing as data or inside a comment is not mistaken for code.
    # `q` is what runs; `probe` is what is inspected.
    probe = _normalise(q).strip().rstrip(";").strip()

    if not probe:
        raise ScreenRejection("empty query")

    # ONE statement. Checked on the normalised text so a ';' inside a comment or
    # a string is not read as statement-stacking.
    if ";" in probe:
        raise ScreenRejection(
            "only a single statement is allowed — remove the ';'")

    if not re.match(r"^(select|with)\b", probe, re.I):
        raise ScreenRejection("the query must start with SELECT or WITH")

    m = _FORBIDDEN.search(probe)
    if m:
        raise ScreenRejection(
            f"'{m.group(1)}' is not allowed — this path is read-only")

    m = _UNWEIGHTED_OR2A.search(probe)
    if m:
        raise ScreenRejection(
            f"{m.group(1).upper()}() on an OR2A column is wrong: OR2A must be "
            f"weighted by order count. Use "
            f"SUM(or2a_order_minutes) / SUM(total_orders) instead.")

    m = _REPLICATED.search(probe)
    if m:
        raise ScreenRejection(
            f"'{m.group(1)}' is a slot-level column replicated across hours; "
            f"summing it double-counts. Use safe_slot_supply, which is already "
            f"one row per slot.")

    # TABLE ALLOWLIST. CTE names are collected first so a query's own WITH
    # clauses are not mistaken for unknown tables.
    cte_names = {n.lower() for n in _CTE_NAME.findall(probe)}
    for ident in _IDENTIFIER_AFTER_FROM.findall(probe):
        name = ident.lower().split(".")[-1]
        if name in cte_names or name in {"(", ""}:
            continue
        if name not in tables.SAFE_VIEWS:
            raise ScreenRejection(
                f"'{ident}' is not available. Query only: "
                f"{', '.join(tables.SAFE_VIEWS)}.")

    # ROW CAP. Tightened rather than trusted: a model-supplied LIMIT 5000 is
    # still a wall of rows.
    m = _LIMIT_RE.search(probe)
    if m and int(m.group(1)) > config.SQL_ROW_LIMIT:
        q = _LIMIT_RE.sub(f"LIMIT {config.SQL_ROW_LIMIT}", q, count=1)
    elif not m:
        q = f"{q}\nLIMIT {config.SQL_ROW_LIMIT}"

    return q


# ═══════════════════════════════════════════════════════════════════════
# GENERATE -> SCREEN -> RUN
# ═══════════════════════════════════════════════════════════════════════

def answer(con: duckdb.DuckDBPyConnection, question: str, provider,
           context: dict | None = None) -> SqlAnswer:
    """
    Generate SQL for a question, screen it, run it.

    On rejection the REASON is fed back and the model tries again, up to
    SQL_MAX_RETRIES. Most rejections are the model reaching for AVG(avg_or2a),
    and it corrects that when told exactly what to use instead -- so the retry
    is not hope, it is a specific instruction the screen already computed.
    """
    out = SqlAnswer()
    system = prompts.build_sql_prompt(_schema_text(con), context or {})
    message = question
    deadline = time.monotonic() + config.SQL_TIMEOUT_SECONDS

    for attempt in range(config.SQL_MAX_RETRIES + 1):
        if time.monotonic() > deadline:
            out.error = "timed out generating a query"
            return out

        try:
            resp = provider.complete(system, message, json_mode=True,
                                     timeout=config.SQL_TIMEOUT_SECONDS)
            data = llm_mod.parse_json(resp.text) if resp.ok else None
        except Exception as e:                      # never crash a live demo
            out.error = f"could not reach the model ({e})"
            return out

        # THREE DIFFERENT FAILURES, THREE DIFFERENT MESSAGES.
        #
        # These were all reported as "the model did not return a query", which
        # is a lie in two of the three cases and sent a real debugging session
        # down the wrong path: the API was failing outright and the message said
        # the model had declined.
        #
        #   resp.ok False    the HTTP call failed      -> the provider's reason
        #   data None        the reply was not JSON    -> show what came back
        #   sql empty        the model genuinely
        #                    cannot answer this        -> the honest decline
        if not resp.ok:
            out.error = f"the {resp.provider} call failed: {resp.error}"
            return out

        if data is None:
            snippet = (resp.text or "").strip().replace("\n", " ")[:160]
            out.error = (f"the model replied with something that is not JSON: "
                         f"{snippet!r}")
            return out

        if not data.get("sql"):
            out.error = "the model said it cannot answer this from the data"
            return out

        raw = str(data["sql"])
        out.attempts.append(raw)

        try:
            safe = screen(raw)
        except ScreenRejection as rej:
            if attempt == config.SQL_MAX_RETRIES:
                out.sql = raw
                out.error = f"query rejected by the safety screen: {rej.reason}"
                return out
            # Hand back the exact reason. Vague feedback produces a vague retry.
            message = (f"{question}\n\nYour previous query was REJECTED:\n"
                       f"{raw}\n\nReason: {rej.reason}\n"
                       f"Return a corrected query.")
            continue

        out.sql = safe
        try:
            cur = con.execute(safe)
            out.columns = [d[0] for d in cur.description]
            out.rows = cur.fetchall()
        except Exception as e:
            if attempt == config.SQL_MAX_RETRIES:
                out.error = f"the query failed to run: {e}"
                return out
            message = (f"{question}\n\nYour previous query FAILED to execute:\n"
                       f"{safe}\n\nDatabase error: {e}\nReturn a corrected query.")
            continue

        out.total_rows = len(out.rows)
        out.truncated = out.total_rows >= config.SQL_ROW_LIMIT
        return out

    return out


def _schema_text(con: duckdb.DuckDBPyConnection) -> str:
    """The safe views' real columns, read from the database rather than typed."""
    lines = []
    for view in tables.SAFE_VIEWS:
        cols = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position", [view]).fetchall()
        lines.append(f"{view}:")
        lines.extend(f"    {c:26} {t}" for c, t in cols)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════

def render(res: SqlAnswer, question: str) -> str:
    """
    Always shows the SQL.

    That is not decoration. It is the whole audit story: an analyst who can read
    the query can check the answer in five seconds, which is the only thing that
    makes `screened` an honest label rather than a hedge. Hiding it behind a
    toggle would mean most users never see it, and the label would be a promise
    nobody could cash.
    """
    if res.error:
        body = [f"I couldn't answer that with a generated query — {res.error}."]
        if res.sql:
            body.append(f"\nLast attempt:\n```sql\n{res.sql}\n```")
        body.append("\nTry rephrasing, or ask something the tested tools cover — "
                    "rankings, comparisons, thresholds, hour walkthroughs.")
        return "\n".join(body)

    parts = [f"```sql\n{res.sql}\n```"]

    if not res.rows:
        parts.append("\nThe query ran and returned **no rows**.")
    else:
        parts.append("")
        parts.append("| " + " | ".join(res.columns) + " |")
        parts.append("|" + "|".join(["---"] * len(res.columns)) + "|")
        for row in res.rows:
            parts.append("| " + " | ".join(_cell(v) for v in row) + " |")
        if res.truncated:
            parts.append(f"\n_Showing the first {config.SQL_ROW_LIMIT} rows._")

    parts.append(
        "\n> ⚠️ **Generated query — screened, not verified.** No tested tool "
        "covers this question, so the SQL above was written by the model. It "
        "was checked against this dataset's known pitfalls (OR2A must be "
        "order-weighted; rider-supply columns are slot-level and must not be "
        "summed across hours) and restricted to read-only views. The numbers "
        "have not been independently confirmed — read the query before acting "
        "on them.")
    return "\n".join(parts)


def _cell(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)
