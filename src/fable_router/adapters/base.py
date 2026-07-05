"""Common interface every adapter implements, plus a shared NDJSON-subprocess helper."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Result:
    text: str
    model: str
    provider: str
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    raw: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class AdapterError(RuntimeError):
    pass


def run_ndjson_cli(
    cmd: list[str],
    *,
    timeout: float,
    line_handler: Callable[[dict], None],
) -> tuple[str, int]:
    """Run a CLI that streams newline-delimited JSON on stdout.

    Feeds each parsed JSON line to `line_handler`. Returns (stderr_text, returncode).
    stdin is explicitly closed so CLIs that peek at stdin (codex, opencode) never block.

    On Windows, `opencode`/`codex` are npm-installed `.cmd` shims, which
    CreateProcess cannot launch directly — shell=True is required. With a
    list argv, subprocess uses `list2cmdline` to quote each arg, so this is
    the safe, documented way to do it (not string-interpolation shell=True).
    """
    proc = subprocess.run(
        cmd,
        shell=(os.name == "nt"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        line_handler(event)
    return proc.stderr, proc.returncode
