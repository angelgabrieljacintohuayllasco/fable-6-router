"""sqlite prompt cache — skip the network/CLI call if we've asked this exact
(provider, model, prompt) before."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "cache.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    text TEXT NOT NULL
)
"""


def _key(provider: str, model: str, prompt: str) -> str:
    return hashlib.sha256(f"{provider}:{model}:{prompt}".encode()).hexdigest()


def get(provider: str, model: str, prompt: str) -> str | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_SCHEMA)
        row = conn.execute(
            "SELECT text FROM cache WHERE key = ?", (_key(provider, model, prompt),)
        ).fetchone()
        return row[0] if row else None


def put(provider: str, model: str, prompt: str, text: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_SCHEMA)
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, text) VALUES (?, ?)",
            (_key(provider, model, prompt), text),
        )
