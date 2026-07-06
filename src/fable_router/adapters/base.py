"""Common interface every adapter implements, plus a shared NDJSON-subprocess helper."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
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


# Binarios nativos dentro del paquete npm, relativos a <npm>\node_modules.
# Invocarlos directo evita cmd.exe por completo: pasar argumentos no confiables
# (prompts con comillas, %, &) a un .cmd via shell es inyectable por diseño
# (BatBadBut; la doc de Python lo marca como inseguro sin fix posible).
_WINDOWS_NATIVE_BIN: dict[str, tuple[str, ...]] = {
    "opencode": (r"opencode-ai\node_modules\opencode-windows-*\bin\opencode.exe",),
    "codex": (r"@openai\codex\node_modules\@openai\codex-win32-*\vendor\*\bin\codex.exe",),
}

_resolved: dict[str, list[str]] = {}


def _resolve_cli(name: str) -> list[str]:
    """Resuelve un CLI npm a su exe nativo real (sin shim .cmd, sin cmd.exe)."""
    if name in _resolved:
        return _resolved[name]
    argv = [name]
    path = shutil.which(name)
    if path:
        argv = [path]
        if os.name == "nt" and path.lower().endswith((".cmd", ".bat")):
            modules = Path(path).parent / "node_modules"
            for pattern in _WINDOWS_NATIVE_BIN.get(name, ()):
                hits = glob.glob(str(modules / pattern))
                if hits:
                    argv = [hits[0]]
                    break
    _resolved[name] = argv
    return argv


def run_ndjson_cli(
    cmd: list[str],
    *,
    timeout: float,
    line_handler: Callable[[dict], None],
) -> tuple[str, int]:
    """Run a CLI that streams newline-delimited JSON on stdout.

    Feeds each parsed JSON line to `line_handler`. Returns (stderr_text, returncode).
    stdin is explicitly closed so CLIs that peek at stdin (codex, opencode) never block.

    cmd[0] se resuelve al binario nativo via _resolve_cli. shell=True queda solo
    como último recurso si no se encontró exe (shim .cmd) — en ese caso los
    args no confiables viajan por cmd.exe, evitarlo siempre que se pueda.

    Timeout uses Popen + taskkill /T: with shell=True, subprocess.run(timeout)
    only kills the intermediate cmd.exe — the node grandchild survives holding
    the stdout/stderr pipes and the follow-up communicate() blocks forever
    (deadlock observed live with `codex exec` > 180s inside the MCP server).
    taskkill /T /F also covers the native exes' own children.
    """
    argv = _resolve_cli(cmd[0]) + cmd[1:]
    needs_shell = os.name == "nt" and argv[0].lower().endswith((".cmd", ".bat"))
    proc = subprocess.Popen(
        argv,
        shell=needs_shell,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdin=subprocess.DEVNULL, capture_output=True,
            )
        else:
            proc.kill()
        proc.communicate()  # pipes now closed by the tree kill; reap and drop
        raise
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        line_handler(event)
    return stderr, proc.returncode
