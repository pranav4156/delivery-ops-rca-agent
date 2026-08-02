"""
Environment loading — one implementation, used by every entry point.

WHY THIS EXISTS
---------------
Each check script grew its own copy of a .env loader, and app/main.py grew
none -- so `python -m uvicorn app.main:app` started with LLM_PROVIDER unset and
silently fell back to the rule-based classifier, while the check scripts using
the same .env worked fine. The API looked broken for no visible reason.

Duplicated config loading also drifts: python-dotenv strips surrounding quotes
and a naive hand-rolled loader does not, so KEY="value" worked on one path and
produced a key with literal quote characters on another -- surfacing as a
confusing 401 rather than "your quotes are wrong".

One loader, imported everywhere, cannot drift.
"""

from __future__ import annotations

import os
from pathlib import Path

_loaded = False


def load_env(path: str | Path = ".env", force: bool = False) -> bool:
    """
    Load .env into os.environ. Idempotent; returns True if a file was read.

    Real environment variables always win over the file -- `GROQ_API_KEY=x
    python foo.py` must override .env, which is the standard expectation and
    how deployment overrides work.
    """
    global _loaded
    if _loaded and not force:
        return True

    # Resolve relative to the project root, not the current working directory,
    # so the loader works regardless of where the process was started from.
    p = Path(path)
    if not p.is_absolute():
        candidates = [Path.cwd() / p, Path(__file__).resolve().parent.parent / p]
        p = next((c for c in candidates if c.exists()), candidates[0])

    if not p.exists():
        _loaded = True
        return False

    try:
        from dotenv import load_dotenv
        load_dotenv(p)
        _loaded = True
        return True
    except ImportError:
        pass

    # Hand-rolled fallback, deliberately matching python-dotenv's behaviour:
    # strip surrounding quotes, ignore comments and blanks, do not overwrite
    # variables already set in the real environment.
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value.strip())

    _loaded = True
    return True


def describe() -> str:
    """One-line summary of what was picked up, for startup logs."""
    provider = os.getenv("LLM_PROVIDER") or "(auto)"
    keys = [name for name in ("GROQ_API_KEY", "GOOGLE_API_KEY",
                              "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
            if os.getenv(name)]
    return f"LLM_PROVIDER={provider} keys={keys or 'none'}"
