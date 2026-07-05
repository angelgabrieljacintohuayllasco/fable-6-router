"""Append-only log of every completion call: latency, tokens, cost, errors."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .adapters.base import Result

DB_PATH = Path(__file__).parent.parent.parent / "ledger.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    ok INTEGER NOT NULL,
    latency_s REAL NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL,
    error TEXT,
    mode TEXT
)
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def record(result: Result, *, mode: str = "router") -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO calls (ts, provider, model, ok, latency_s, input_tokens,"
            " output_tokens, cost_usd, error, mode) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                time.time(), result.provider, result.model, int(result.ok),
                result.latency_s, result.input_tokens, result.output_tokens,
                result.cost_usd, result.error, mode,
            ),
        )


def stats() -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT provider, model, COUNT(*) as calls, SUM(ok) as ok_calls,"
            " AVG(latency_s) as avg_latency, SUM(input_tokens) as total_input,"
            " SUM(output_tokens) as total_output, SUM(cost_usd) as total_cost"
            " FROM calls GROUP BY provider, model ORDER BY calls DESC"
        ).fetchall()
        return [dict(r) for r in rows]
