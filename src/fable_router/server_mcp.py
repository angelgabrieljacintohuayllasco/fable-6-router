"""MCP fachada: expone el router como herramientas Claude Code puede llamar."""
from __future__ import annotations

from fastmcp import FastMCP

from . import doctor, ensemble, ledger, mcts, router

mcp = FastMCP("fable-6-router")


@mcp.tool
def setup_check() -> str:
    """Corré esto primero. Revisa qué credenciales faltan (Vertex/gcloud,
    OpenCode Go, Codex/ChatGPT, Qwen Model Studio opcional) y da el comando
    exacto para arreglar cada una. Ninguna de esas se puede automatizar acá
    (son logins interactivos/OAuth) — este check solo te dice qué correr.
    """
    return doctor.report()


@mcp.tool
def ask(prompt: str, task_type: str | None = None) -> str:
    """Ruta el prompt al mejor modelo disponible (router A) y devuelve el texto.

    task_type opcional: code, reasoning, writing, extraction, chat.
    Si se omite, un clasificador barato (Gemini Flash) lo decide.
    """
    result = router.ask(prompt, task_type=task_type)
    if not result.ok:
        return f"[error] todos los proveedores fallaron: {result.error}"
    return result.text


@mcp.tool
def ask_ensemble(prompt: str) -> str:
    """Manda el prompt en paralelo a 3 modelos de vendors distintos (Gemini 3.1 Pro,
    GLM 5.1, GPT-5.5) y sintetiza la mejor respuesta con Gemini 3.1 Pro. Más lento
    y costoso que `ask`, úsalo cuando la calidad importa más que la velocidad.
    """
    result = ensemble.ask_ensemble(prompt)
    if not result.ok:
        return f"[error] todos los proveedores fallaron: {result.error}"
    return result.text


@mcp.tool
def ask_deep(prompt: str, n_steps: int = 6) -> str:
    """Búsqueda AB-MCTS (TreeQuest de Sakana AI): explora y refina en árbol entre
    varios modelos, evaluando cada candidato con un LLM-judge. El modo más lento
    y caro (varias llamadas secuenciales) pero el de mayor calidad — úsalo solo
    para tareas realmente difíciles donde `ask` y `ask_ensemble` no bastan.
    """
    result = mcts.ask_deep(prompt, n_steps=n_steps)
    if not result.ok:
        return f"[error] {result.error}"
    return result.text


@mcp.tool
def stats() -> str:
    """Muestra estadísticas de uso: llamadas, latencia, tokens y costo por modelo."""
    rows = ledger.stats()
    if not rows:
        return "sin datos aún"
    lines = ["provider/model | calls | ok | avg_latency_s | in_tok | out_tok | cost_usd"]
    for r in rows:
        lines.append(
            f"{r['provider']}/{r['model']} | {r['calls']} | {r['ok_calls']} | "
            f"{r['avg_latency']:.2f} | {r['total_input']} | {r['total_output']} | "
            f"{r['total_cost']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
