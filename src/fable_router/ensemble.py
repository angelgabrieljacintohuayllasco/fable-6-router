"""Capa B: Mixture-of-Agents. N modelos responden en paralelo (hilos, ya que
los adapters son bloqueantes: subprocess.run / cliente Vertex síncrono) y un
agregador de vendor distinto sintetiza la mejor respuesta.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .adapters import codex_cli, dashscope, opencode_cli, vertex
from .adapters.base import Result
from .ledger import record

Candidate = tuple[str, str | None]  # (provider, model_key)

DEFAULT_CANDIDATES: list[Candidate] = [
    ("vertex", "gemini-pro"),
    ("opencode", "glm"),
    ("codex", None),
]
DEFAULT_AGGREGATOR: Candidate = ("vertex", "gemini-pro")

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
    raise ValueError(f"unknown provider: {provider}")


def ask_ensemble(
    prompt: str,
    *,
    candidates: list[Candidate] | None = None,
    aggregator: Candidate = DEFAULT_AGGREGATOR,
) -> Result:
    candidates = candidates or DEFAULT_CANDIDATES

    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        futures = [pool.submit(_dispatch, p, m, prompt) for p, m in candidates]
        results = [f.result() for f in futures]

    for r in results:
        record(r, mode="ensemble_candidate")

    ok_results = [r for r in results if r.ok]
    if not ok_results:
        return results[0]
    if len(ok_results) == 1:
        return ok_results[0]

    candidates_block = "\n\n".join(
        f"[{r.provider}/{r.model}]\n{r.text}" for r in ok_results
    )
    synth_prompt = _SYNTH_PROMPT.format(prompt=prompt, candidates_block=candidates_block)

    agg_provider, agg_model = aggregator
    synthesis = _dispatch(agg_provider, agg_model, synth_prompt)
    record(synthesis, mode="ensemble_aggregate")

    if not synthesis.ok:
        # Aggregation failed — fall back to the first successful candidate.
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
