from __future__ import annotations

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
            return resp.choices[0].message.content or ""
    return ""
