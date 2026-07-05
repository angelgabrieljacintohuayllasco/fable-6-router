"""Chequeo de setup: qué credenciales faltan y el comando exacto para arreglarlas.

Nada de esto automatiza el login (opencode/codex abren su propio flujo
interactivo/OAuth) — solo detecta qué falta y dice qué correr.
"""
from __future__ import annotations

import os
import subprocess

from .adapters.dashscope import is_configured as dashscope_configured


def _run(cmd: list[str], timeout: float = 15.0) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd, shell=(os.name == "nt"), capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        return proc.returncode == 0, (proc.stdout + proc.stderr).strip()
    except FileNotFoundError:
        return False, "comando no encontrado"
    except subprocess.TimeoutExpired:
        return False, "timeout"


def check_vertex() -> tuple[bool, str]:
    ok, _ = _run(["gcloud", "auth", "application-default", "print-access-token"])
    fix = "gcloud auth application-default login"
    return ok, ("" if ok else fix)


def check_opencode() -> tuple[bool, str]:
    # `opencode auth list` imprime la etiqueta "OpenCode Go" (con espacio),
    # no la clave interna "opencode-go" de auth.json — no son el mismo string.
    ok, out = _run(["opencode", "auth", "list"])
    configured = ok and "opencode go" in out.lower()
    fix = (
        "opencode auth login -p opencode-go "
        "(si te muestra un selector de provider igual, elegi 'OpenCode Go', NO 'OpenCode Zen')"
    )
    return configured, ("" if configured else fix)


def check_codex() -> tuple[bool, str]:
    ok, out = _run(["codex", "login", "status"])
    logged_in = ok and "logged in" in out.lower()
    fix = "codex login"
    return logged_in, ("" if logged_in else fix)


def check_dashscope() -> tuple[bool, str]:
    ok = dashscope_configured()
    fix = (
        "Conseguí una API key gratis en https://bailian.console.aliyun.com/ "
        "(Model Studio) y agregá DASHSCOPE_API_KEY=... a tu .env"
    )
    return ok, ("" if ok else fix)


CHECKS: list[tuple[str, str]] = [
    ("Vertex AI (Gemini)", "check_vertex"),
    ("OpenCode Go (GLM/Qwen/DeepSeek/Kimi/Minimax)", "check_opencode"),
    ("Codex CLI (GPT-5.5)", "check_codex"),
    ("Qwen Model Studio (opcional, qwen3.7-max)", "check_dashscope"),
]

_FUNCS = {
    "check_vertex": check_vertex,
    "check_opencode": check_opencode,
    "check_codex": check_codex,
    "check_dashscope": check_dashscope,
}


def report() -> str:
    from concurrent.futures import ThreadPoolExecutor

    # Los checks son subprocesos CLI lentos (gcloud ~2s, opencode ~2s, codex ~1s)
    # — en serie suman 5-8s; en paralelo, el más lento.
    with ThreadPoolExecutor(max_workers=len(CHECKS)) as pool:
        futures = [pool.submit(_FUNCS[fn_name]) for _, fn_name in CHECKS]
        results = [f.result() for f in futures]

    lines = ["Estado de credenciales:\n"]
    any_missing = False
    for (label, _), (ok, fix) in zip(CHECKS, results):
        mark = "OK" if ok else "FALTA"
        lines.append(f"[{mark}] {label}")
        if not ok:
            any_missing = True
            lines.append(f"       -> corre: {fix}")
    if not any_missing:
        lines.append("\nTodo listo, podés usar ask/ask_ensemble/ask_deep.")
    else:
        lines.append(
            "\nQwen Model Studio es opcional (hay fallback a OpenCode Go). "
            "Vertex, OpenCode y Codex son necesarios para el router completo."
        )
    return "\n".join(lines)
