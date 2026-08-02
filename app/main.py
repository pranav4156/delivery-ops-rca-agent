"""
FastAPI application.

A DELIBERATELY THIN LAYER.

The endpoint wraps Agent.run_turn() 1:1 and adds nothing: no business logic, no
reshaping, no computation. Everything that decides an answer already happened in
Modules 1-3, where it is tested. If this file grew logic, that logic would be
the only part of the system without coverage.

The response schema here is the contract handed to the UI team in
03_UI_HANDOFF.md. It must not change.

STARTUP COST
------------
Loading the CSV, validating it, building derived tables and compiling the graph
takes ~500ms. That happens ONCE at startup via the lifespan handler, not per
request. MCP warms up in a background thread and never blocks a request.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.env import describe as describe_env
from app.env import load_env

# MUST run before get_provider() reads the environment. Without it the API
# started with no key and silently used the rule-based classifier, while the
# check scripts -- which load .env themselves -- worked from the same file.
load_env()

import config                                       # noqa: E402
from app import db, tables                          # noqa: E402
from app.graph import build_agent                   # noqa: E402
from app.llm import TELEMETRY                       # noqa: E402
from app.state import SESSIONS                      # noqa: E402
from app.validation import validate                 # noqa: E402

STATIC_DIR = Path(__file__).parent.parent / "static"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build everything once. A cold request must never pay this cost."""
    t0 = time.time()
    con = db.load()
    validate(con)
    tables.build_all(con)
    agent = build_agent(con)

    _state["con"] = con
    _state["agent"] = agent
    _state["boot_ms"] = (time.time() - t0) * 1000
    _state["started_at"] = time.time()

    model = getattr(agent.provider, "model", "")
    print(f"  agent ready in {_state['boot_ms']:.0f} ms")
    print(f"    llm  : {agent.provider.name}{'/' + model if model else ''}")
    print(f"    env  : {describe_env()}")
    print(f"    mcp  : {'connecting in background' if config.MCP_CONNECT_ON_STARTUP else 'off'}")
    if agent.provider.name == "rule-based":
        print("    NOTE : no API key found — using the deterministic classifier.")
        print("           All numbers stay correct; only prose is plainer.")
        print("           Set GROQ_API_KEY in .env to enable the model.")
    yield

    if agent.mcp:
        agent.mcp.close()


app = FastAPI(
    title="Delivery Operations RCA Agent",
    description="Conversational root-cause analysis for quick-commerce "
                "delivery SLA breaches.",
    version="1.0.0",
    lifespan=lifespan,
)

# Wide open by design: this is a local demo tool with no auth and no user data.
# Anything narrower just breaks the UI team's dev server for no security gain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════
# SCHEMA  — mirrors 03_UI_HANDOFF.md §2
# ═══════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str = Field(..., description="Raw user text, unprocessed")
    session_id: str | None = Field(
        None, description="Omit on the first turn; send the returned value after")


class ContextModel(BaseModel):
    city: str | None = None
    store: str | None = None
    date: str | None = None
    hours: list[int] | None = None
    hour_label: str | None = None
    intent: str | None = None
    inherited: list[str] = []
    resolution_methods: dict[str, str] = {}


class ChatResponse(BaseModel):
    session_id: str
    reply: str = Field(..., description="Markdown. Render as-is; do not parse.")
    context: ContextModel
    intent: str | None = None
    provenance: str = Field(
        ..., description="validated | unavailable | exploratory")
    error: str | None = None
    suggestions: list[str] = []
    elapsed_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str
    rows: int
    stores: int
    cities: int
    date: str
    llm_provider: str
    mcp: str
    sessions: int
    uptime_seconds: float


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    One conversational turn.

    Returns HTTP 200 for every user-facing outcome, including declines and
    out-of-scope questions -- those are normal results carrying
    provenance="unavailable", not errors. A non-200 means the server itself is
    broken, which lets the client treat the two cases differently.
    """
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(503, "Agent is still starting up.")

    t0 = time.time()
    result = agent.run_turn(req.session_id, req.message)
    result["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
    return ChatResponse(**result)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    agent, con = _state.get("agent"), _state.get("con")
    if agent is None or con is None:
        raise HTTPException(503, "Not ready.")

    mcp_state = "off"
    if agent.mcp is not None:
        mcp_state = "connected" if agent.mcp.status.available else "unavailable"

    return HealthResponse(
        status="ok",
        rows=con.execute("SELECT COUNT(*) FROM fact_store_hour").fetchone()[0],
        stores=con.execute("SELECT COUNT(*) FROM dim_store").fetchone()[0],
        cities=len(agent.index.cities),
        date=config.DATA_DATE,
        llm_provider=agent.provider.name,
        mcp=mcp_state,
        sessions=len(SESSIONS),
        uptime_seconds=round(time.time() - _state["started_at"], 1),
    )


@app.get("/entities")
def entities() -> dict:
    """
    Valid cities and stores.

    Lets a client build autocomplete or starter chips without guessing, and
    without hardcoding a list that could drift from the data.
    """
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(503, "Not ready.")
    return {
        "date": config.DATA_DATE,
        "cities": agent.index.cities,
        "stores": agent.index.stores,
        "stores_by_city": agent.index.city_stores,
        "hour_labels": config.HOUR_LABELS,
    }


@app.get("/session/{session_id}")
def get_session(session_id: str) -> dict:
    """
    Full transcript for one conversation.

    Every turn is already recorded in Session.turns; this just exposes it. Two
    uses:

      - a client can restore the chat after a page refresh, without having to
        keep its own copy in sync with the server's context
      - it makes the inheritance auditable after the fact -- each turn carries
        the context that was resolved for it, so you can see exactly where
        "that day" became 2026-04-22

    IN-MEMORY ONLY. A server restart clears it. Persistence is out of scope,
    and pretending otherwise would invite a client to rely on it.
    """
    session = SESSIONS.get(session_id)
    return {
        "session_id": session.session_id,
        "turn_count": session.turn_count,
        "current_context": {
            "city": session.current_city,
            "store": session.current_store,
            "date": session.current_date,
            "hours": session.current_hours,
            "last_intent": session.last_intent,
        },
        "turns": session.turns,
    }


@app.post("/session/{session_id}/reset")
def reset_session(session_id: str) -> dict:
    """Clear one conversation's context. Backs a "New conversation" control."""
    SESSIONS.reset(session_id)
    return {"session_id": session_id, "status": "reset"}


@app.get("/sessions")
def list_sessions() -> dict:
    """Active sessions with turn counts. Debugging aid, not for the UI."""
    return {
        "count": len(SESSIONS),
        "sessions": [
            {"session_id": s.session_id, "turns": s.turn_count,
             "store": s.current_store, "city": s.current_city}
            for s in SESSIONS._sessions.values()
        ],
    }


@app.get("/debug/stats")
def stats() -> dict:
    """LLM and MCP telemetry. Useful during the walkthrough, not for the UI."""
    agent = _state.get("agent")
    return {
        "boot_ms": _state.get("boot_ms"),
        "llm": {
            "provider": agent.provider.name if agent else None,
            "model": getattr(agent.provider, "model", None) if agent else None,
            "calls": TELEMETRY.calls,
            "retries": TELEMETRY.retries,
            "failures": TELEMETRY.failures,
            "avg_ms": round(TELEMETRY.avg_ms, 1),
        },
        "mcp": {
            "available": agent.mcp.status.available if agent and agent.mcp else False,
            "server": agent.mcp.status.server if agent and agent.mcp else None,
            "tools": agent.mcp.status.tools if agent and agent.mcp else [],
            "calls": len(agent.mcp.status.calls) if agent and agent.mcp else 0,
        },
        "sessions": len(SESSIONS),
    }


@app.get("/debug/graph")
def graph_diagram() -> dict:
    """Mermaid source for the LangGraph state machine."""
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(503, "Not ready.")
    return {"mermaid": agent.diagram()}


# ═══════════════════════════════════════════════════════════════════════
# STATIC
# ═══════════════════════════════════════════════════════════════════════

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        page = STATIC_DIR / "index.html"
        if page.exists():
            return FileResponse(page)
        raise HTTPException(404, "No frontend built yet. The API is at /docs.")
