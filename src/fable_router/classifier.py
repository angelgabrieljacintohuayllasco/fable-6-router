"""Capa 0: cheap task classifier. Decides which router bucket a prompt falls into."""
from __future__ import annotations

from .adapters import vertex

TASK_TYPES = ("code", "reasoning", "writing", "extraction", "chat")

_PROMPT = """Clasifica el siguiente prompt de usuario. Responde SOLO con una línea CSV:
tipo,dificultad

tipo es uno de: code, reasoning, writing, extraction, chat
dificultad es uno de: easy, hard

Prompt del usuario:
---
{prompt}
---
Responde solo la línea CSV, nada más."""


def classify(prompt: str) -> tuple[str, str]:
    result = vertex.complete("gemini-flash", _PROMPT.format(prompt=prompt), max_output_tokens=100)
    if not result.ok:
        return "chat", "easy"  # fail open to the cheapest bucket
    line = result.text.strip().splitlines()[-1] if result.text.strip() else ""
    parts = [p.strip().lower() for p in line.split(",")]
    task_type = parts[0] if parts and parts[0] in TASK_TYPES else "chat"
    difficulty = parts[1] if len(parts) > 1 and parts[1] in ("easy", "hard") else "easy"
    return task_type, difficulty
