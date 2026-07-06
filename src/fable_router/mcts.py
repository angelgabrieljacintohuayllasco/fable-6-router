"""Capa C: AB-MCTS (TreeQuest, Sakana AI — Apache-2.0) para tareas difíciles.

En vez de pedir N respuestas independientes (como el ensemble), explora un
árbol: cada paso elige una rama prometedora y la refina, o abre una rama
nueva con otro modelo. Evaluación por defecto: LLM-judge (Gemini Flash,
barato) puntúa 0-10 cada candidato. Para código, pasa tu propio `evaluate`
que ejecute tests reales — un juez ejecutando tests es más confiable que uno
opinando.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import treequest as tq

from .adapters import codex_cli, dashscope, opencode_cli, vertex
from .adapters.base import Result
from .ledger import record

Candidate = tuple[str, str | None]

# Sin codex a propósito: `codex exec` con reasoning xhigh tarda 1-3 min POR
# LLAMADA y MCTS hace varias secuenciales — el tool se va a >10 min y el
# cliente MCP muere antes. dashscope entra solo si hay DASHSCOPE_API_KEY
# (falla rápido y el árbol sigue con las demás ramas si no).
DEFAULT_ACTIONS: list[Candidate] = [
    ("vertex", "gemini-pro"),
    ("opencode", "glm"),
    ("opencode", "qwen"),
    ("dashscope", "qwen-max"),
]


@dataclass
class NodeState:
    text: str
    provider: str
    model: str


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


_JUDGE_PROMPT = """Evalúa la siguiente respuesta a la pregunta dada. Responde SOLO \
con un número entero del 0 al 10 (10 = perfecta, correcta y completa; 0 = inútil \
o incorrecta). No expliques, solo el número.

Pregunta:
---
{prompt}
---
Respuesta a evaluar:
---
{answer}
---
Puntaje (0-10):"""


def _llm_judge(prompt: str, answer: str) -> float:
    judge_prompt = _JUDGE_PROMPT.format(prompt=prompt, answer=answer)
    # max_output_tokens bajo aquí volvía la respuesta VACÍA: el thinking de
    # gemini-3.x consume el mismo presupuesto (gotcha documentado en vault
    # "13 Referencia - Gemini Vertex"), aunque solo queramos un número corto.
    result = vertex.complete("gemini-flash", judge_prompt, max_output_tokens=8000)
    record(result, mode="mcts_judge")
    if not result.ok:
        return 0.0
    digits = "".join(c for c in result.text if c.isdigit())
    score = int(digits[:2]) if digits else 0
    return max(0.0, min(10.0, float(score))) / 10.0


def _refine_prompt(original_prompt: str, prev_text: str) -> str:
    return (
        f"Pregunta original:\n---\n{original_prompt}\n---\n\n"
        f"Tu respuesta anterior fue:\n---\n{prev_text}\n---\n\n"
        "Mejórala: corrige errores, completa lo que falte, hazla más precisa. "
        "Da la versión mejorada completa, no solo los cambios."
    )


def _make_generate_fn(
    provider: str,
    model_key: str | None,
    prompt: str,
    evaluate: Callable[[str], float],
    call_log: list[Result],
):
    def generate_fn(parent: NodeState | None) -> tuple[NodeState, float]:
        call_prompt = prompt if parent is None else _refine_prompt(prompt, parent.text)
        result = _dispatch(provider, model_key, call_prompt)
        record(result, mode="mcts_candidate")
        call_log.append(result)
        text = result.text if result.ok else ""
        score = evaluate(text) if text else 0.0
        return NodeState(text=text, provider=provider, model=model_key or result.model), score

    return generate_fn


def ask_deep(
    prompt: str,
    *,
    actions: list[Candidate] | None = None,
    n_steps: int = 6,
    evaluate: Callable[[str], float] | None = None,
) -> Result:
    """Búsqueda AB-MCTS: n_steps rondas alternando explorar modelos nuevos y
    refinar la mejor rama encontrada. Se queda con el candidato de mejor score.
    Lento y caro (varias llamadas secuenciales) — modo "deep", explícito.
    """
    actions = actions or DEFAULT_ACTIONS
    evaluate_fn = evaluate or (lambda text: _llm_judge(prompt, text))
    call_log: list[Result] = []

    algo = tq.ABMCTSA(dist_type="beta")  # reward está en [0,1], beta es el prior correcto
    generate_fns = {
        f"{provider}:{model_key or 'default'}": _make_generate_fn(
            provider, model_key, prompt, evaluate_fn, call_log
        )
        for provider, model_key in actions
    }

    state = algo.init_tree()
    for _ in range(n_steps):
        state = algo.step(state, generate_fns)

    best = tq.top_k(state, algo, k=1)
    total_latency = sum(r.latency_s for r in call_log)
    if not best or not best[0][0].text:
        return Result(
            text="", model="ab-mcts", provider="mcts",
            latency_s=total_latency, error="no candidate scored above empty",
        )

    best_node, best_score = best[0]
    return Result(
        text=best_node.text,
        model=f"ab-mcts({best_node.provider}/{best_node.model},score={best_score:.2f},steps={n_steps})",
        provider="mcts",
        latency_s=total_latency,
        input_tokens=sum(r.input_tokens for r in call_log),
        output_tokens=sum(r.output_tokens for r in call_log),
    )
