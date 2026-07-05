"""Fachada API — endpoint /v1/chat/completions compatible con clientes OpenAI.

Uso: `uvicorn fable_router.server_api:app --reload`
El campo `model` elige el modo: "fable-router" (A, rápido), "fable-router-ensemble"
(B, calidad), "fable-router-deep" (C, AB-MCTS, lento/caro).
"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from . import ensemble, mcts, router
from .adapters.base import Result

app = FastAPI(title="Fable 6 Router", description="Multi-model orchestrator")

MODES = {
    "fable-router": lambda prompt: router.ask(prompt),
    "fable-router-ensemble": lambda prompt: ensemble.ask_ensemble(prompt),
    "fable-router-deep": lambda prompt: mcts.ask_deep(prompt),
}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "fable-router"
    messages: list[ChatMessage]


def _last_user_prompt(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return messages[-1].content if messages else ""


def _to_openai_response(result: Result, requested_model: str) -> dict:
    finish_reason = "stop" if result.ok else "error"
    content = result.text if result.ok else f"[error] {result.error}"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.input_tokens,
            "completion_tokens": result.output_tokens,
            "total_tokens": result.input_tokens + result.output_tokens,
        },
        "fable_router": {"provider": result.provider, "resolved_model": result.model},
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> dict:
    prompt = _last_user_prompt(req.messages)
    handler = MODES.get(req.model, MODES["fable-router"])
    result = handler(prompt)
    return _to_openai_response(result, req.model)


@app.get("/v1/models")
def list_models() -> dict:
    now = int(time.time())
    return {
        "object": "list",
        "data": [{"id": name, "object": "model", "created": now, "owned_by": "fable-6-router"} for name in MODES],
    }
