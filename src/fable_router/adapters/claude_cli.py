"""Claude (Opus 4.8 / Fable 5) via el CLI de Claude Code (suscripción, sin API key).

`claude -p --output-format json` corre un turno headless bajo el plan del
usuario y devuelve UNA línea JSON con `result` + usage. El prompt viaja por
STDIN (no argv): argumentos no confiables por argv es lo que abrió la
inyección cmd.exe en los otros CLIs, y stdin lo evita de raíz.

Este es el agregador/candidato de mayor calidad del router — la tesis Sakana
fugu: el ensemble rinde ≥ que su mejor miembro solo si quien sintetiza/juzga
es el modelo más fuerte disponible.
"""
from __future__ import annotations

import os
import subprocess
import time

from .base import Result, run_ndjson_cli

DEFAULT_MODEL = os.environ.get("FABLE_ROUTER_CLAUDE_MODEL", "claude-opus-4-8")


def complete(prompt: str, *, model: str = DEFAULT_MODEL, timeout: float = 240.0) -> Result:
    # --strict-mcp-config sin --mcp-config: el hijo headless NO carga los MCP
    # servers del usuario (si no, cada llamada del ensemble arrancaría blender,
    # obs, chrome-devtools, y hasta OTRO fable-6-router recursivo).
    cmd = ["claude", "-p", "--output-format", "json", "--model", model, "--strict-mcp-config"]

    text = ""
    input_tokens = 0
    output_tokens = 0
    error: str | None = None

    def handle(event: dict) -> None:
        nonlocal text, input_tokens, output_tokens, error
        if event.get("type") != "result":
            return
        if event.get("is_error"):
            error = str(event.get("result", ""))[:300] or "claude CLI is_error"
            return
        text = event.get("result", "") or ""
        usage = event.get("usage", {})
        input_tokens = (usage.get("input_tokens", 0) or 0) + (
            usage.get("cache_creation_input_tokens", 0) or 0
        )
        output_tokens = usage.get("output_tokens", 0) or 0

    start = time.monotonic()
    try:
        stderr, returncode = run_ndjson_cli(
            cmd, timeout=timeout, line_handler=handle, stdin_data=prompt
        )
    except subprocess.TimeoutExpired:
        return Result(
            text="", model=model, provider="claude",
            latency_s=time.monotonic() - start, error=f"timeout after {timeout}s",
        )

    latency = time.monotonic() - start
    if returncode != 0:
        return Result(
            text="", model=model, provider="claude", latency_s=latency,
            error=f"exit {returncode}: {stderr.strip()[-300:]}",
        )
    if error:
        return Result(text="", model=model, provider="claude", latency_s=latency, error=error)

    return Result(
        text=text, model=model, provider="claude", latency_s=latency,
        input_tokens=input_tokens, output_tokens=output_tokens,
    )
