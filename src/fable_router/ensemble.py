"""Capa B: Mixture-of-Agents. N modelos responden en paralelo (hilos, ya que
los adapters son bloqueantes: subprocess.run / cliente Vertex síncrono) y un
agregador de vendor distinto sintetiza la mejor respuesta.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .adapters import claude_cli, codex_cli, dashscope, opencode_cli, vertex
from .adapters.base import Result
from .ledger import record

Candidate = tuple[str, str | None]  # (provider, model_key)

DEFAULT_CANDIDATES: list[Candidate] = [
    ("claude", None),
    ("vertex", "gemini-pro"),
    ("opencode", "glm"),
    ("codex", None),
]
# Tesis Sakana fugu: quien sintetiza debe ser el modelo MÁS fuerte disponible,
# si no el ensemble queda capado a la calidad del agregador. Claude (Opus 4.8
# via suscripción) primero; Gemini 3.1 Pro de respaldo si el CLI falla.
AGGREGATOR_CHAIN: list[Candidate] = [("claude", None), ("vertex", "gemini-pro")]

_SYNTH_PROMPT = """Varios modelos de IA respondieron la misma pregunta. Sintetiza \
la MEJOR respuesta posible: toma lo correcto de cada una, corrige errores si un \
modelo se equivocó y los demás coinciden en algo distinto, y da una respuesta \
final única, completa y directa. No menciones que hubo varios modelos.

Pregunta original:
---
{prompt}
---

Respuestas de los modelos:
{candidates_block}

Respuesta final sintetizada:"""


def _dispatch(provider: str, model_key: str | None, prompt: str) -> Result:
    if provider == "vertex":
        return vertex.complete(model_key or "gemini-flash", prompt)
    if provider == "opencode":
        return opencode_cli.complete(model_key or "glm", prompt)
    if provider == "codex":
        return codex_cli.complete(prompt)
    if provider == "dashscope":
        return dashscope.complete(model_key or "qwen-max", prompt)
    if provider == "claude":
        return claude_cli.complete(prompt, model=model_key or claude_cli.DEFAULT_MODEL)
    raise ValueError(f"unknown provider: {provider}")


# Pa cuando el AGREGADOR es el modelo que llama la tool (Claude via MCP):
# sin claude en los candidatos — la calidad Claude la pone el caller al
# sintetizar, y meterlo también de candidato duplicaría consumo del plan.
CALLER_AGGREGATED_CANDIDATES: list[Candidate] = [
    ("vertex", "gemini-pro"),
    ("opencode", "glm"),
    ("dashscope", "qwen-max"),
    ("codex", None),
]


def gather_candidates(
    prompt: str, *, candidates: list[Candidate] | None = None
) -> list[Result]:
    """Fan-out paralelo sin agregación: devuelve las respuestas crudas de cada
    modelo. El caller (idealmente el modelo más fuerte disponible) sintetiza."""
    candidates = candidates or CALLER_AGGREGATED_CANDIDATES
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        futures = [pool.submit(_dispatch, p, m, prompt) for p, m in candidates]
        results = [f.result() for f in futures]
    for r in results:
        record(r, mode="ensemble_candidate")
    return results


def ask_ensemble(
    prompt: str,
    *,
    candidates: list[Candidate] | None = None,
    aggregators: list[Candidate] | None = None,
) -> Result:
    aggregators = aggregators or AGGREGATOR_CHAIN

    results = gather_candidates(prompt, candidates=candidates or DEFAULT_CANDIDATES)

    ok_results = [r for r in results if r.ok]
    if not ok_results:
        return results[0]
    if len(ok_results) == 1:
        return ok_results[0]

    candidates_block = "\n\n".join(
        f"[{r.provider}/{r.model}]\n{r.text}" for r in ok_results
    )
    synth_prompt = _SYNTH_PROMPT.format(prompt=prompt, candidates_block=candidates_block)

    synthesis: Result | None = None
    for agg_provider, agg_model in aggregators:
        synthesis = _dispatch(agg_provider, agg_model, synth_prompt)
        record(synthesis, mode="ensemble_aggregate")
        if synthesis.ok:
            break

    if synthesis is None or not synthesis.ok:
        # Every aggregator failed — fall back to the first successful candidate.
        return ok_results[0]

    total_latency = sum(r.latency_s for r in results) + synthesis.latency_s
    return Result(
        text=synthesis.text,
        model=f"moa({'+'.join(r.model for r in ok_results)})",
        provider="ensemble",
        latency_s=total_latency,
        input_tokens=sum(r.input_tokens for r in results) + synthesis.input_tokens,
        output_tokens=sum(r.output_tokens for r in results) + synthesis.output_tokens,
    )
