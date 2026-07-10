"""Modo "Fable 6": Generar-Verificar-Seleccionar entre modelos frontera.

La tesis (Sakana Fugu, roles Thinker/Worker/Verifier) aplicada con una
mejora clave para código: el Verifier no es un LLM opinando, es EJECUCIÓN.

  Workers   -> 5 candidatos de vendors distintos, en paralelo (errores no
               correlacionados): Sonnet 5, GPT-5.6 Terra, Gemini 3.1 Pro,
               GLM 5.2, Qwen 3.7 Max.
  Verifier  -> Gemini Flash genera asserts desde el enunciado (corre en
               paralelo con los workers, no después). Cada candidato se
               ejecuta contra los asserts; puntaje = asserts pasados.
  Selección -> gana el candidato con más asserts. Nunca se sintetiza código
               mezclando soluciones: la síntesis LLM puede fusionar dos
               respuestas medio-correctas en una rota.
  Thinker   -> Claude (Opus 4.8) solo entra si la verificación no decide
               (sin código detectable, asserts inservibles o empate en 0):
               sintetiza estilo MoA. Uso quirúrgico del modelo más caro.

Para prompts sin código la etapa de verificación no aplica y cae directo a
síntesis — este modo brilla en tareas verificables.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from .adapters import claude_cli, copilot, dashscope, opencode_cli, vertex
from .adapters.base import Result
from .ensemble import _SYNTH_PROMPT
from .ledger import record

Candidate = tuple[str, str | None]

WORKERS: list[Candidate] = [
    ("copilot", "sonnet"),      # Claude Sonnet 5
    ("copilot", "terra"),       # GPT-5.6 Terra
    ("vertex", "gemini-pro"),   # Gemini 3.1 Pro
    ("opencode", "glm52"),      # GLM 5.2
    ("dashscope", "qwen-max"),  # Qwen 3.7 Max
]

_TESTGEN_PROMPT = """A continuación hay una tarea de programación. Escribe entre 5 y 8 \
`assert` de Python que verifiquen una implementación correcta de la función pedida. \
Usa SOLO los ejemplos/propiedades que se deducen del enunciado. Responde únicamente \
con un bloque ```python que contenga los asserts (sin implementación, sin prints).

Tarea:
---
{prompt}
---"""


def _dispatch(provider: str, model_key: str | None, prompt: str) -> Result:
    if provider == "vertex":
        return vertex.complete(model_key or "gemini-flash", prompt)
    if provider == "opencode":
        return opencode_cli.complete(model_key or "glm52", prompt)
    if provider == "dashscope":
        return dashscope.complete(model_key or "qwen-max", prompt)
    if provider == "copilot":
        return copilot.complete(model_key or "sonnet", prompt)
    if provider == "claude":
        return claude_cli.complete(prompt, model=model_key or claude_cli.DEFAULT_MODEL)
    raise ValueError(f"unknown provider: {provider}")


def _extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else None


def _extract_asserts(text: str) -> list[str]:
    code = _extract_code(text) or text
    # Un assert puede ocupar varias líneas (paréntesis abiertos); statement por
    # statement vía compile es overkill — heurística: líneas que arrancan con
    # "assert" + continuaciones indentadas.
    asserts: list[str] = []
    current: list[str] = []
    for line in code.splitlines():
        if line.startswith("assert "):
            if current:
                asserts.append("\n".join(current))
            current = [line]
        elif current and (line.startswith((" ", "\t", ")"))):
            current.append(line)
        else:
            if current:
                asserts.append("\n".join(current))
                current = []
    if current:
        asserts.append("\n".join(current))
    return asserts


def _run_asserts(code: str, asserts: list[str], timeout: float = 15.0) -> int:
    """Cuántos asserts pasa el candidato. Una sola ejecución con contadores —
    lanzar un subproceso por assert sería 8x más lento."""
    numbered = "\n".join(
        f"try:\n"
        + "".join(f"    {ln}\n" for ln in a.splitlines())
        + f"    _passed += 1\nexcept Exception:\n    pass\n"
        for a in asserts
    )
    program = f"{code}\n\n_passed = 0\n{numbered}\nprint(_passed)\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return 0
    if proc.returncode != 0:
        return 0
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return 0


def ask_fable6(prompt: str) -> Result:
    start = time.monotonic()

    # Workers + Verifier (generación de tests) en paralelo: el testgen solo
    # depende del prompt, no de los candidatos.
    with ThreadPoolExecutor(max_workers=len(WORKERS) + 1) as pool:
        cand_futures = [pool.submit(_dispatch, p, m, prompt) for p, m in WORKERS]
        test_future = pool.submit(
            vertex.complete, "gemini-flash", _TESTGEN_PROMPT.format(prompt=prompt)
        )
        candidates = [f.result() for f in cand_futures]
        test_result = test_future.result()

    for r in candidates:
        record(r, mode="fable6_candidate")
    record(test_result, mode="fable6_testgen")

    ok = [r for r in candidates if r.ok]
    if not ok:
        return candidates[0]

    def _finish(text: str, source: str, extra: Result | None = None) -> Result:
        used = candidates + [test_result] + ([extra] if extra else [])
        return Result(
            text=text,
            model=f"fable6-{source}",
            provider="fable6",
            latency_s=time.monotonic() - start,
            input_tokens=sum(r.input_tokens for r in used),
            output_tokens=sum(r.output_tokens for r in used),
        )

    # Etapa de verificación: solo si hay código que ejecutar y asserts útiles.
    coded = [(r, _extract_code(r.text)) for r in ok]
    coded = [(r, c) for r, c in coded if c]
    asserts = _extract_asserts(test_result.text) if test_result.ok else []

    if coded and asserts:
        scored = [(r, _run_asserts(c, asserts)) for r, c in coded]
        best_r, best_score = max(scored, key=lambda rc: rc[1])
        if best_score > 0:
            return _finish(best_r.text, f"verified({best_r.model},{best_score}/{len(asserts)})")

    # Verificación no decidió: Thinker (Claude) sintetiza estilo MoA.
    block = "\n\n".join(f"[{r.provider}/{r.model}]\n{r.text}" for r in ok)
    judge = _dispatch("claude", None, _SYNTH_PROMPT.format(prompt=prompt, candidates_block=block))
    record(judge, mode="fable6_judge")
    if judge.ok:
        return _finish(judge.text, "judged", judge)
    return _finish(ok[0].text, f"fallback({ok[0].model})")
