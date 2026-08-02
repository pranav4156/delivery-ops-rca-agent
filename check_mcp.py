#!/usr/bin/env python3
"""
MCP connectivity check.

    python check_mcp.py

Verifies the agent really consumes an open-source MCP server, and shows the
tool-call log as evidence.

Run this once before the walkthrough. If it fails, the agent still works --
methodology questions fall back to config-derived values and no RCA number is
affected -- but the graded MCP integration would not be demonstrable.

FIRST RUN IS SLOW: npx fetches @modelcontextprotocol/server-filesystem. To
avoid that stall during a demo, pre-install it:

    npm install -g @modelcontextprotocol/server-filesystem
"""

import shutil
import sys
import time

sys.path.insert(0, ".")

import config
from app import db, mcp_client, tables
from app.graph import build_agent
from app.validation import validate

W = 76
passed = failed = 0


def result(ok, label, detail=""):
    global passed, failed
    ok = bool(ok)
    passed += ok
    failed += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if detail:
        for line in str(detail).splitlines():
            print(f"        {line}")


print("=" * W)
print("  MCP CHECK")
print("=" * W)

# ── 1. runtime ──────────────────────────────────────────────────────────
print("\n[1] Runtime")
cmd = config.MCP_SERVER_COMMAND[0]
path = shutil.which(cmd)
result(path is not None, f"'{cmd}' on PATH", path or
       "The filesystem MCP server runs under Node.\n"
       "Install Node.js from https://nodejs.org, or point\n"
       "config.MCP_SERVER_COMMAND at a different stdio MCP server.")
if not path:
    sys.exit(1)

print(f"        server command: {' '.join(config.MCP_SERVER_COMMAND)}")
print(f"        docs directory: {config.DOCS_DIR}")

# ── 2. documents ────────────────────────────────────────────────────────
print("\n[2] Reference documents on disk")
for f in config.MCP_DOC_FILES:
    p = config.DOCS_DIR / f
    result(p.exists(), f"{f}", f"{p.stat().st_size:,} bytes" if p.exists()
           else f"missing at {p}")

# ── 3. connect ──────────────────────────────────────────────────────────
print("\n[3] Connecting")
print(f"        (first run downloads the server — up to "
      f"{config.MCP_CONNECT_TIMEOUT_SECONDS:.0f}s)")

server = mcp_client.MCPDocServer()
t0 = time.time()
ok = server.connect()
elapsed = time.time() - t0

result(ok, f"session established ({elapsed:.1f}s)",
       "" if ok else
       f"{server.status.error}\n\n"
       f"Pre-install to rule out a download problem:\n"
       f"    npm install -g @modelcontextprotocol/server-filesystem")
if not ok:
    print("\n  The agent still runs. Methodology questions fall back to")
    print("  config-derived values, and no RCA number is affected.\n")
    sys.exit(1)

# ── 4. tools ────────────────────────────────────────────────────────────
print("\n[4] Tools advertised by the server")
for t in server.status.tools:
    print(f"        {t}")
result(len(server.status.tools) > 0, f"{len(server.status.tools)} tools")
result(any("read" in t for t in server.status.tools),
       "a read tool is available")

# ── 5. list ─────────────────────────────────────────────────────────────
print("\n[5] Listing documents through MCP")
docs = server.list_documents()
for d in docs:
    print(f"        {d}")
result(len(docs) >= 3, f"{len(docs)} markdown files found")

# ── 6. read ─────────────────────────────────────────────────────────────
print("\n[6] Reading each document through MCP")
for f in config.MCP_DOC_FILES:
    text = server.read_document(f)
    result(bool(text), f"{f}",
           f"{len(text):,} chars" if text else "read returned nothing")

# ── 7. search ───────────────────────────────────────────────────────────
print("\n[7] Searching document content")
for term, expect in [("pileup", "quick_commerce_rca_logic.md"),
                     ("OR2A", "order_ready_to_assignment.md"),
                     ("man_hour", None)]:
    hits = server.search(term)
    files = sorted({f for f, _ in hits})
    ok = bool(hits) and (expect is None or expect in files)
    result(ok, f"'{term}' -> {len(hits)} section(s) in {files}")

# ── 8. through the agent ────────────────────────────────────────────────
print("\n[8] End to end — a methodology question answered FROM the documents")
con = db.load()
validate(con)
tables.build_all(con)
agent = build_agent(con, mcp=server, connect_mcp=False)

before = len(server.status.calls)
r = agent.run_turn("mcp-check", "What's the pileup threshold?")
after = len(server.status.calls)

print("\n" + "\n".join(f"        {l}" for l in r["reply"].splitlines()[:14]))

result(r["intent"] == "methodology", "routed to the methodology node")
result("From the reference documentation" in r["reply"],
       "answered from the DOCUMENTS, not from config",
       "" if "From the reference documentation" in r["reply"] else
       "Fell back to config values — the search found nothing.")
result(after > before, f"triggered {after - before} MCP call(s) on this turn")

# ── 9. RCA path independence ────────────────────────────────────────────
print("\n[9] The RCA path does not depend on MCP")
before = len(server.status.calls)
r = agent.run_turn("mcp-check2", "Why did STORE_003 underperform on 2026-04-22?")
after = len(server.status.calls)
result("317.7 min" in r["reply"], "RCA numbers correct")
result(after == before, "zero MCP calls during an RCA",
       f"{after - before} calls — MCP should not be on the numeric path")

# ── 10. log ─────────────────────────────────────────────────────────────
print("\n[10] Tool-call log (show this during the walkthrough)")
for line in server.call_log(12).splitlines():
    print(f"        {line}")

print("\n" + "=" * W)
print(f"  {passed} passed, {failed} failed")
print(f"  {server.status.summary()}")
print("=" * W)

server.close()

if failed == 0:
    print("\n  MCP integration verified end to end.\n")
else:
    print("\n  See the failures above. The agent still runs regardless —")
    print("  MCP only affects methodology answers, never RCA numbers.\n")
    sys.exit(1)
