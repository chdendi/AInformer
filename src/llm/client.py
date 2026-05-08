from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import LLMConfig

log = logging.getLogger(__name__)


# DeepSeek-chat 标价（人民币 / 1M tokens），cache miss 价格
_PRICE_INPUT_CNY_PER_M = 1.92    # 输入
_PRICE_OUTPUT_CNY_PER_M = 7.92   # 输出

_USAGE_LOCK = asyncio.Lock()
_USAGE: dict[str, int] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


async def _record_usage(resp: Any) -> None:
    """Extract token usage from a chat completion response and accumulate it."""
    usage = getattr(resp, "usage", None)
    if not usage:
        return
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    async with _USAGE_LOCK:
        _USAGE["calls"] += 1
        _USAGE["prompt_tokens"] += pt
        _USAGE["completion_tokens"] += ct
        cum_pt = _USAGE["prompt_tokens"]
        cum_ct = _USAGE["completion_tokens"]
    log.info(
        "LLM usage: prompt=%d completion=%d (cum prompt=%d completion=%d)",
        pt,
        ct,
        cum_pt,
        cum_ct,
    )


def get_usage_summary() -> dict[str, Any]:
    """Snapshot of LLM token usage so far in this process. Safe to call any time."""
    pt = _USAGE["prompt_tokens"]
    ct = _USAGE["completion_tokens"]
    cost = pt / 1_000_000 * _PRICE_INPUT_CNY_PER_M + ct / 1_000_000 * _PRICE_OUTPUT_CNY_PER_M
    return {
        "calls": _USAGE["calls"],
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "cost_cny": round(cost, 4),
    }


def make_client(cfg: LLMConfig) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)


async def chat_json(
    client: AsyncOpenAI,
    cfg: LLMConfig,
    system: str,
    user: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 4000,
) -> dict[str, Any]:
    """Call chat completion and parse JSON. Retries on transient failures."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=20),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            resp = await client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            await _record_usage(resp)
            content = resp.choices[0].message.content or "{}"
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                cleaned = content.strip().strip("`")
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                return json.loads(cleaned)
    return {}


async def chat_text(
    client: AsyncOpenAI,
    cfg: LLMConfig,
    system: str,
    user: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 2000,
) -> str:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=20),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            resp = await client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            await _record_usage(resp)
            return resp.choices[0].message.content or ""
    return ""
