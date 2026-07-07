"""MCP fachada: expone el router como herramientas Claude Code puede llamar.

Los submódulos pesados (treequest ~1s, google-genai ~1s, openai ~0.5s) se
importan DENTRO de cada tool, no acá arriba: el handshake MCP con Claude Code
debe responder rápido o el cliente marca el server como colgado. Con lazy
imports el arranque queda en ~1s (solo fastmcp).
"""
from __future__ import annotations

from fastmcp import FastMCP

# treequest arrastra numpy, cuya DLL C se CUELGA si el primer import ocurre
# dentro de un AnyIO worker thread en Windows (loader lock; visto con py-spy:
# worker congelado en numpy._core.multiarray create_module). Pre-cargarlo acá,
# en el main thread antes del event loop, cuesta ~1s de arranque y elimina el
# cuelgue. El resto de imports pesados sí pueden ser lazy (son HTTP, no DLL).
import treequest as _preloaded_treequest  # noqa: F401, E402

mcp = FastMCP("fable-6-router")


@mcp.tool
def setup_check() -> str:
    """Corré esto primero. Revisa qué credenciales faltan (Vertex/gcloud,
    OpenCode Go, Codex/ChatGPT, Qwen Model Studio opcional) y da el comando
    exacto para arreglar cada una. Ninguna de esas se puede automatizar acá
    (son logins interactivos/OAuth) — este check solo te dice qué correr.
    """
    from . import doctor

    return doctor.report()


@mcp.tool
def ask(prompt: str, task_type: str | None = None) -> str:
    """Ruta el prompt al mejor modelo disponible (router A) y devuelve el texto.

    task_type opcional: code, reasoning, writing, extraction, chat.
    Si se omite, un clasificador barato (Gemini Flash) lo decide.
    """
    from . import router

    result = router.ask(prompt, task_type=task_type)
    if not result.ok:
        return f"[error] todos los proveedores fallaron: {result.error}"
    return result.text


@mcp.tool
def ask_candidates(prompt: str) -> str:
    """PREFERILA sobre ask_ensemble si VOS sos un modelo fuerte (Claude Opus/Fable).

    Manda el prompt en paralelo a varios modelos (Gemini 3.1 Pro, GLM 5.1,
    Qwen 3.7 Max, GPT-5.5) y devuelve TODAS las respuestas crudas etiquetadas,
    SIN sintetizar. Vos, el modelo que llama, sos el agregador: compará las
    respuestas, quedate con lo correcto de cada una, corregí donde una
    contradiga a las demás y entregá al usuario UNA respuesta final propia
    (tesis Sakana: el ensemble rinde lo que rinde su agregador, y el agregador
    más fuerte disponible sos vos). No muestres las respuestas crudas salvo
    que el usuario las pida.
    """
    from . import ensemble

    results = ensemble.gather_candidates(prompt)
    ok = [r for r in results if r.ok]
    if not ok:
        errors = "; ".join(f"{r.provider}: {r.error}" for r in results)
        return f"[error] todos los candidatos fallaron: {errors}"
    parts = [
        f"=== {r.provider}/{r.model} (latencia {r.latency_s:.0f}s) ===\n{r.text}"
        for r in ok
    ]
    failed = [r for r in results if not r.ok]
    if failed:
        parts.append(
            "=== candidatos caídos (ignoralos) ===\n"
            + "\n".join(f"{r.provider}/{r.model}: {r.error}" for r in failed)
        )
    return "\n\n".join(parts)


@mcp.tool
def ask_ensemble(prompt: str) -> str:
    """Ensemble con agregación INTERNA (Claude CLI o Gemini sintetizan en el
    server). Pensada pa clientes que no son un modelo fuerte (API OpenAI-compat,
    scripts). Si vos sos Claude Opus/Fable llamando esta tool, usá mejor
    `ask_candidates` y sintetizá vos — mismo costo, mejor agregador y sin
    gastar el plan de Claude dos veces.
    """
    from . import ensemble

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
    from . import mcts

    result = mcts.ask_deep(prompt, n_steps=n_steps)
    if not result.ok:
        return f"[error] {result.error}"
    return result.text


@mcp.tool
def stats() -> str:
    """Muestra estadísticas de uso: llamadas, latencia, tokens y costo por modelo."""
    from . import ledger

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
