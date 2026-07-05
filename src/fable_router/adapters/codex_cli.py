"""ChatGPT 5.5 via the Codex CLI (ChatGPT plan Go, OAuth login already done).

`codex exec --json <prompt>` runs one non-interactive turn and streams
newline-delimited JSON events. stdin is forced closed (see run_ndjson_cli) so
it never blocks on "Reading additional input from stdin...". Stray MCP
connector errors on stderr (e.g. an unrelated Figma MCP auth failure) are
harmless noise and ignored as long as the process exits 0.
"""
from __future__ import annotations

import subprocess
import time

from .base import Result, run_ndjson_cli

DEFAULT_MODEL = "gpt-5.5"


def complete(prompt: str, *, model: str = DEFAULT_MODEL, timeout: float = 180.0) -> Result:
    cmd = ["codex", "exec", "--json", "-m", model, "--skip-git-repo-check", prompt]

    last_text = ""
    input_tokens = 0
    output_tokens = 0

    def handle(event: dict) -> None:
        nonlocal last_text, input_tokens, output_tokens
        etype = event.get("type")
        if etype == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                last_text = item.get("text", "")
        elif etype == "turn.completed":
            usage = event.get("usage", {})
            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0

    start = time.monotonic()
    try:
        stderr, returncode = run_ndjson_cli(cmd, timeout=timeout, line_handler=handle)
    except subprocess.TimeoutExpired:
        return Result(
            text="", model=model, provider="codex",
            latency_s=time.monotonic() - start, error=f"timeout after {timeout}s",
        )

    latency = time.monotonic() - start
    if returncode != 0:
        return Result(
            text="", model=model, provider="codex", latency_s=latency,
            error=f"exit {returncode}: {stderr.strip()[-500:]}",
        )

    return Result(
        text=last_text, model=model, provider="codex", latency_s=latency,
        input_tokens=input_tokens, output_tokens=output_tokens,
    )
