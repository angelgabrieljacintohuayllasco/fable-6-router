"""Capa A: task type -> ordered fallback chain of (provider, model_key).
First one that answers ok wins; cache is checked before each hop.
"""
from __future__ import annotations

from .adapters import claude_cli, codex_cli, copilot, dashscope, opencode_cli, vertex
from .adapters.base import Result
from .classifier import classify
from .ledger import record
from . import cache

# (provider, model_key) — model_key is None for codex (single model per call).
# "code" prueba primero Qwen 3.7 Max real via DashScope (si DASHSCOPE_API_KEY
# está seteada); si no, cae a Qwen 3.6 Plus via OpenCode Go.
ROUTES: dict[str, list[tuple[str, str | None]]] = {
    "code": [("copilot", "sonnet"), ("dashscope", "qwen-max"), ("opencode", "qwen"), ("opencode", "glm"), ("vertex", "gemini-pro")],
    "reasoning": [("claude", None), ("copilot", "sonnet"), ("vertex", "gemini-pro"), ("codex", None), ("opencode", "glm")],
    "writing": [("claude", None), ("codex", None), ("vertex", "gemini-pro"), ("opencode", "glm")],
    "extraction": [("vertex", "gemini-flash"), ("opencode", "glm")],
    "chat": [("vertex", "gemini-flash"), ("opencode", "glm")],
}


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
    if provider == "copilot":
        return copilot.complete(model_key or "sonnet", prompt)
    raise ValueError(f"unknown provider: {provider}")


def ask(prompt: str, *, task_type: str | None = None) -> Result:
    """Route a prompt to the best model for its task type, with fallback."""
    if task_type is None:
        task_type, _difficulty = classify(prompt)
    chain = ROUTES.get(task_type, ROUTES["chat"])

    last_result: Result | None = None
    for provider, model_key in chain:
        cache_model = model_key or provider
        cached_text = cache.get(provider, cache_model, prompt)
        if cached_text is not None:
            return Result(text=cached_text, model=cache_model, provider=provider, latency_s=0.0)

        result = _dispatch(provider, model_key, prompt)
        record(result, mode="router")
        last_result = result
        if result.ok:
            cache.put(provider, cache_model, prompt, result.text)
            return result

    return last_result
