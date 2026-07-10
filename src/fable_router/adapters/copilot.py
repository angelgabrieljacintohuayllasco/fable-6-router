"""GitHub Copilot nativo (suscripción Copilot Pro), API OpenAI-compatible.

Flujo copiado de opencode (packages/opencode/src/plugin/github-copilot,
verificado 2026-07-09): el token OAuth de GitHub (`gho_...`) viaja DIRECTO
como Bearer contra https://api.githubcopilot.com — ya no hay exchange
copilot_internal. Headers requeridos: X-GitHub-Api-Version 2026-06-01,
Openai-Intent y x-initiator.

El token se lee de la auth de opencode ya guardada
(~/.local/share/opencode/auth.json, entry "github-copilot".refresh — se crea
con `opencode auth login` eligiendo GitHub Copilot) o de la env var
GITHUB_COPILOT_TOKEN. Nativo en vez de `opencode run` porque el CLI carga
~35k tokens de contexto de agente por llamada; esto manda solo el prompt.

Modelos verificados en cuenta Copilot Pro (GET /models): claude-sonnet-5,
gemini-3.1-pro-preview, gpt-5.6-terra, gpt-5.6-luna (que Codex con cuenta
ChatGPT NO sirve), kimi-k2.7-code. claude-opus-4.8/fable-5 aparecen pero
policy=disabled en el plan Pro.
"""
from __future__ import annotations

import json
import os
import time

from openai import APIStatusError, OpenAI

from .base import Result

BASE_URL = "https://api.githubcopilot.com"
API_VERSION = "2026-06-01"

MODELS = {
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4.5",
    "gemini-pro": "gemini-3.1-pro-preview",
    "gemini-flash": "gemini-3.5-flash",
    "terra": "gpt-5.6-terra",
    "luna": "gpt-5.6-luna",
    "kimi": "kimi-k2.7-code",
}

_client: OpenAI | None = None


def _auth_json_path() -> str:
    return os.path.expanduser("~/.local/share/opencode/auth.json")


def _token() -> str | None:
    env = os.environ.get("GITHUB_COPILOT_TOKEN")
    if env:
        return env
    try:
        with open(_auth_json_path(), encoding="utf-8") as f:
            return json.load(f)["github-copilot"]["refresh"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def is_configured() -> bool:
    return _token() is not None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=_token(),
            base_url=BASE_URL,
            default_headers={
                "User-Agent": "fable-6-router/1.0",
                "X-GitHub-Api-Version": API_VERSION,
                "Openai-Intent": "conversation-edits",
                "x-initiator": "user",
            },
        )
    return _client


def _needs_responses_api(model_id: str) -> bool:
    # Los GPT-5.6 en Copilot solo se sirven por /responses: /chat/completions
    # devuelve 400 'not accessible via the /chat/completions endpoint'.
    return model_id.startswith("gpt-5.6")


def complete(model_key: str, prompt: str, *, max_retries: int = 3) -> Result:
    model_id = MODELS.get(model_key, model_key)
    if not is_configured():
        return Result(
            text="", model=model_id, provider="copilot", latency_s=0.0,
            error=(
                "sin token de Copilot: corré `opencode auth login` y elegí "
                "GitHub Copilot, o seteá GITHUB_COPILOT_TOKEN"
            ),
        )

    client = _get_client()
    start = time.monotonic()
    for attempt in range(max_retries):
        try:
            if _needs_responses_api(model_id):
                response = client.responses.create(model=model_id, input=prompt)
                usage = response.usage
                return Result(
                    text=response.output_text or "", model=model_id, provider="copilot",
                    latency_s=time.monotonic() - start,
                    input_tokens=getattr(usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(usage, "output_tokens", 0) or 0,
                    raw=response,
                )
            response = client.chat.completions.create(
                model=model_id, messages=[{"role": "user", "content": prompt}]
            )
            choice = response.choices[0].message.content or ""
            usage = response.usage
            return Result(
                text=choice, model=model_id, provider="copilot",
                latency_s=time.monotonic() - start,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                raw=response,
            )
        except APIStatusError as exc:
            if exc.status_code in (429, 503) and attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return Result(
                text="", model=model_id, provider="copilot",
                latency_s=time.monotonic() - start, error=f"{exc.status_code}: {exc.message}",
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via Result.error
            return Result(
                text="", model=model_id, provider="copilot",
                latency_s=time.monotonic() - start, error=str(exc),
            )


def list_models() -> list[str]:
    """Ids de modelo que la cuenta puede usar (para doctor/debug)."""
    client = _get_client()
    return [m.id for m in client.models.list().data]
