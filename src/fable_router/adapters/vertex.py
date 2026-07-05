"""Gemini via Vertex AI + gcloud ADC (no API key). Gotchas documented in the
G-Mini Agent vault (13 Referencia - Gemini Vertex): location must be "global"
for Gemini 3.x, model ids carry "-preview", thinking eats maxOutputTokens so
it must stay high, and 3.1-pro has a low quota (429s need backoff).
"""
from __future__ import annotations

import os
import time

from google import genai
from google.genai import types
from google.genai.errors import APIError

from .base import Result

PROJECT = os.environ.get("FABLE_ROUTER_GCP_PROJECT", "")
LOCATION = os.environ.get("FABLE_ROUTER_GCP_LOCATION", "global")

MODELS = {
    "gemini-flash": "gemini-3-flash-preview",
    "gemini-pro": "gemini-3.1-pro-preview",
    "gemini-flash-lite": "gemini-3.1-flash-lite-preview",
}

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not PROJECT:
            raise RuntimeError(
                "FABLE_ROUTER_GCP_PROJECT no está seteado (tu project id de Google Cloud)"
            )
        _client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    return _client


def complete(
    model_key: str,
    prompt: str,
    *,
    max_output_tokens: int = 8000,
    thinking_budget: int = 256,
    max_retries: int = 4,
) -> Result:
    model_id = MODELS.get(model_key, model_key)
    client = _get_client()
    config = types.GenerateContentConfig(
        maxOutputTokens=max_output_tokens,
        thinkingConfig=types.ThinkingConfig(thinkingBudget=thinking_budget),
    )

    start = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id, contents=prompt, config=config
            )
            text = response.text or ""
            usage = response.usage_metadata
            return Result(
                text=text,
                model=model_id,
                provider="vertex",
                latency_s=time.monotonic() - start,
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                raw=response,
            )
        except APIError as exc:
            if exc.code in (429, 503) and attempt < max_retries - 1:
                last_error = exc
                time.sleep(2**attempt)
                continue
            return Result(
                text="",
                model=model_id,
                provider="vertex",
                latency_s=time.monotonic() - start,
                error=f"{exc.code}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via Result.error
            return Result(
                text="",
                model=model_id,
                provider="vertex",
                latency_s=time.monotonic() - start,
                error=str(exc),
            )

    return Result(
        text="",
        model=model_id,
        provider="vertex",
        latency_s=time.monotonic() - start,
        error=f"exhausted {max_retries} retries: {last_error}",
    )
