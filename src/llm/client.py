from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
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
_MAX_JSON_RETRY_TOKENS = 8192


@dataclass(frozen=True)
class _CompletionResult:
    parsed: dict[str, Any]
    finish_reason: str
    completion_tokens: int
    content_length: int
    reasoning_length: int


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


def _parse_json_content(content: str) -> dict[str, Any]:
    candidates = [content]
    cleaned = content.strip().strip("`")
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    candidates.append(cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    log.warning("JSON parse failed. Raw content (first 500 chars): %s", content[:500])
    return {}


def _completion_tokens(resp: Any) -> int:
    usage = getattr(resp, "usage", None)
    return (getattr(usage, "completion_tokens", 0) or 0) if usage else 0


def _finish_reason(resp: Any) -> str:
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return ""
    return getattr(choices[0], "finish_reason", "") or ""


async def _chat_completion(
    client: AsyncOpenAI,
    cfg: LLMConfig,
    system: str,
    user: str,
    *,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> _CompletionResult:
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = await client.chat.completions.create(**kwargs)
    await _record_usage(resp)
    if not resp.choices:
        raise RuntimeError("LLM response did not include a completion choice")

    message = resp.choices[0].message
    content = getattr(message, "content", None) or ""
    reasoning = getattr(message, "reasoning_content", None) or ""
    parsed = _parse_json_content(content)
    finish_reason = _finish_reason(resp)
    completion_tokens = _completion_tokens(resp)

    if not parsed:
        log.warning(
            "LLM returned unusable JSON (model=%s, json_mode=%s, finish=%s, completion=%d, content_len=%d, reasoning_len=%d)",
            cfg.model,
            json_mode,
            finish_reason,
            completion_tokens,
            len(content),
            len(reasoning),
        )
    return _CompletionResult(
        parsed=parsed,
        finish_reason=finish_reason,
        completion_tokens=completion_tokens,
        content_length=len(content),
        reasoning_length=len(reasoning),
    )


async def _request_completion(
    client: AsyncOpenAI,
    cfg: LLMConfig,
    system: str,
    user: str,
    *,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> _CompletionResult:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=20),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            return await _chat_completion(
                client,
                cfg,
                system,
                user,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
    raise RuntimeError("unreachable")


def _expanded_json_budget(max_tokens: int) -> int:
    return min(max(max_tokens * 2, 4096), _MAX_JSON_RETRY_TOKENS)


async def chat_json(
    client: AsyncOpenAI,
    cfg: LLMConfig,
    system: str,
    user: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 4000,
) -> dict[str, Any]:
    """Call a model for JSON, recovering from incomplete structured output.

    Reasoning-capable models can consume the entire completion budget before
    closing their JSON object. A retry must therefore increase the budget,
    rather than repeating the same truncated request unchanged.
    """
    first = await _request_completion(
        client,
        cfg,
        system,
        user,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
    )
    if first.parsed:
        return first.parsed

    retry_tokens = max_tokens
    if first.finish_reason == "length":
        retry_tokens = _expanded_json_budget(max_tokens)
        if retry_tokens > max_tokens:
            log.warning(
                "LLM JSON was truncated (model=%s, completion=%d, content_len=%d); retrying with max_tokens=%d",
                cfg.model,
                first.completion_tokens,
                first.content_length,
                retry_tokens,
            )
            enlarged = await _request_completion(
                client,
                cfg,
                system,
                user,
                temperature=temperature,
                max_tokens=retry_tokens,
                json_mode=True,
            )
            if enlarged.parsed:
                return enlarged.parsed

    log.warning(
        "Retrying LLM JSON without response_format (model=%s, max_tokens=%d)",
        cfg.model,
        retry_tokens,
    )
    fallback = await _request_completion(
        client,
        cfg,
        system,
        user,
        temperature=temperature,
        max_tokens=retry_tokens,
        json_mode=False,
    )
    return fallback.parsed


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
