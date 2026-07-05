"""Fase 0 gate: one trivial prompt per provider. All green = adapters work."""
from __future__ import annotations

from .adapters import codex_cli, opencode_cli, vertex
from .ledger import record

PROMPT = "Responde solo con la palabra: OK"


def _report(result) -> None:
    status = "OK" if result.ok else f"FAIL: {result.error}"
    print(
        f"[{result.provider}/{result.model}] {status} "
        f"latency={result.latency_s:.2f}s in={result.input_tokens} out={result.output_tokens} "
        f"cost={result.cost_usd} text={result.text[:60]!r}"
    )
    record(result, mode="smoke")


def main() -> None:
    print("--- Vertex (Gemini 3.1 Pro) ---")
    _report(vertex.complete("gemini-pro", PROMPT))

    print("--- Vertex (Gemini Flash) ---")
    _report(vertex.complete("gemini-flash", PROMPT))

    print("--- OpenCode Go (GLM 5.1) ---")
    _report(opencode_cli.complete("glm", PROMPT))

    print("--- OpenCode Go (Qwen 3.6 Plus) ---")
    _report(opencode_cli.complete("qwen", PROMPT))

    print("--- Codex CLI (GPT-5.5, ChatGPT plan Go) ---")
    _report(codex_cli.complete(PROMPT))


if __name__ == "__main__":
    main()
