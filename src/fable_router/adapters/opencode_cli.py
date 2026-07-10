"""GLM / Qwen / DeepSeek / Kimi / Minimax via the OpenCode CLI (Go plan).

`opencode run -m <provider/model> --format json` is a non-interactive
one-shot completion that streams newline-delimited JSON events. It already
uses the credentials in `~/.local/share/opencode/auth.json` (no key handling
needed here). Each call carries a large fixed system-prompt overhead
(~35k input tokens observed in smoke tests) — fine for real completions,
too expensive for a cheap classifier call.

Note: the "opencode" (Zen) provider ran out of credits when this was tested
(401 CreditsError) — only "opencode-go" models are used by default.
"""
from __future__ import annotations

import subprocess
import time

from .base import Result, run_ndjson_cli

MODELS = {
    "glm": "opencode-go/glm-5.1",
    "glm52": "opencode-go/glm-5.2",
    "qwen": "opencode-go/qwen3.6-plus",
    "deepseek": "opencode-go/deepseek-v4-pro",
    "deepseek-flash": "opencode-go/deepseek-v4-flash",
    "kimi": "opencode-go/kimi-k2.6",
    "minimax": "opencode-go/minimax-m2.7",
}


def complete(model_key: str, prompt: str, *, timeout: float = 120.0) -> Result:
    model_id = MODELS.get(model_key, model_key)
    cmd = ["opencode", "run", "-m", model_id, "--format", "json", prompt]

    text_by_part: dict[str, str] = {}
    part_order: list[str] = []
    input_tokens = 0
    output_tokens = 0
    cost = 0.0

    def handle(event: dict) -> None:
        nonlocal input_tokens, output_tokens, cost
        etype = event.get("type")
        part = event.get("part", {})
        if etype == "text" and part.get("id"):
            pid = part["id"]
            if pid not in text_by_part:
                part_order.append(pid)
            text_by_part[pid] = part.get("text", "")
        elif etype == "step_finish":
            tokens = part.get("tokens", {})
            input_tokens += tokens.get("input", 0) or 0
            output_tokens += tokens.get("output", 0) or 0
            cost += part.get("cost", 0.0) or 0.0
        elif etype == "error":
            raise RuntimeError(event.get("error", {}).get("data", {}).get("message", str(event)))

    start = time.monotonic()
    try:
        stderr, returncode = run_ndjson_cli(cmd, timeout=timeout, line_handler=handle)
    except subprocess.TimeoutExpired:
        return Result(
            text="", model=model_id, provider="opencode",
            latency_s=time.monotonic() - start, error=f"timeout after {timeout}s",
        )
    except RuntimeError as exc:
        return Result(
            text="", model=model_id, provider="opencode",
            latency_s=time.monotonic() - start, error=str(exc),
        )

    latency = time.monotonic() - start
    if returncode != 0:
        return Result(
            text="", model=model_id, provider="opencode", latency_s=latency,
            error=f"exit {returncode}: {stderr.strip()[-500:]}",
        )

    text = "".join(text_by_part[pid] for pid in part_order)
    return Result(
        text=text, model=model_id, provider="opencode", latency_s=latency,
        input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost or None,
    )
