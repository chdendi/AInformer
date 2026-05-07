from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
_SEMAPHORE = asyncio.Semaphore(4)


async def tavily_search(
    query: str,
    *,
    max_results: int = 8,
    days: int = 2,
    topic: str = "news",
) -> list[dict[str, Any]]:
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        log.warning("TAVILY_API_KEY not set; skipping query: %s", query)
        return []

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "topic": topic,
        "days": days,
        "include_raw_content": False,
    }

    async with _SEMAPHORE:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(2),
                wait=wait_exponential(min=1, max=8),
                reraise=True,
            ):
                with attempt:
                    async with httpx.AsyncClient(timeout=30) as client:
                        r = await client.post(TAVILY_URL, json=payload)
                        r.raise_for_status()
                        data = r.json()
                        return [
                            {
                                "title": x.get("title", ""),
                                "url": x.get("url", ""),
                                "snippet": x.get("content", ""),
                                "score": x.get("score", 0.0),
                                "published_at": x.get("published_date", ""),
                                "source": "tavily",
                            }
                            for x in data.get("results", [])
                            if x.get("url")
                        ]
        except Exception as e:
            log.warning("Tavily query failed [%s]: %s", query, e)
            return []
    return []


async def batch_search(queries: list[str], **kwargs) -> list[dict[str, Any]]:
    results = await asyncio.gather(
        *[tavily_search(q, **kwargs) for q in queries],
        return_exceptions=False,
    )
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for batch in results:
        for item in batch:
            url = item["url"]
            if url in seen:
                continue
            seen.add(url)
            merged.append(item)
    return merged
