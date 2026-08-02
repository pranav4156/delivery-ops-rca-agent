"""
Shared pytest fixtures.

The database is session-scoped: loading the CSV, validating it and building the
derived tables takes ~200ms, and nothing in the suite mutates it. Tests that
need a mutable copy request `fresh_con` instead.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db, tables                       # noqa: E402
from app.entities import EntityIndex             # noqa: E402
from app.validation import validate              # noqa: E402


@pytest.fixture(scope="session")
def con():
    """Loaded, validated, fully built database. Read-only for tests."""
    c = db.load()
    validate(c)
    tables.build_all(c)
    return c


@pytest.fixture
def fresh_con():
    """A throwaway copy for tests that deliberately corrupt the data."""
    c = db.load()
    tables.build_all(c)
    return c


@pytest.fixture(scope="session")
def index(con):
    return EntityIndex(con)
