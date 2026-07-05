"""Qwen via Alibaba Cloud Model Studio (DashScope), OpenAI-compatible mode.

Docs verified 2026-07-05:
- https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope
- https://www.alibabacloud.com/help/en/model-studio/first-api-call-to-qwen

Auth: `DASHSCOPE_API_KEY` (official env var name, kept as-is instead of a
project-specific one so existing DashScope tooling/docs apply directly).

Base URL: Model Studio's current OpenAI-compatible gateway is workspace-scoped
(`https://{WorkspaceId}.<region>.maas.aliyuncs.com/compatible-mode/v1`), but
the region/workspace a given free-tier key belongs to isn't something this
project can know — default here is the older global compatible-mode endpoint
(no workspace id, still active, used by most community SDKs/tutorials).
Override with `FABLE_ROUTER_DASHSCOPE_BASE_URL` if your key needs the
workspace-scoped gateway instead (see README "Configurar Qwen Model Studio").
"""
from __future__ import annotations

import os
import time

from openai import APIStatusError, OpenAI

from .base import Result

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

MODELS = {
    "qwen-max": "qwen3.7-max",
    "qwen-plus": "qwen-plus",
}

_client: OpenAI | None = None


def is_configured() -> bool:
    return bool(os.environ.get("DASHSCOPE_API_KEY"))


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ.get("FABLE_ROUTER_DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
        )
    return _client


def complete(model_key: str, prompt: str, *, max_retries: int = 3) -> Result:
    model_id = MODELS.get(model_key, model_key)
    if not is_configured():
        return Result(
            text="", model=model_id, provider="dashscope", latency_s=0.0,
            error=(
                "DASHSCOPE_API_KEY no configurada. Conseguí una key gratis en "
                "https://bailian.console.aliyun.com/ (Model Studio) y ponela en .env"
            ),
        )

    client = _get_client()
    start = time.monotonic()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id, messages=[{"role": "user", "content": prompt}]
            )
            choice = response.choices[0].message.content or ""
            usage = response.usage
            return Result(
                text=choice, model=model_id, provider="dashscope",
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
                text="", model=model_id, provider="dashscope",
                latency_s=time.monotonic() - start, error=f"{exc.status_code}: {exc.message}",
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via Result.error
            return Result(
                text="", model=model_id, provider="dashscope",
                latency_s=time.monotonic() - start, error=str(exc),
            )
