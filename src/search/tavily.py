from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
_SEMAPHORE = asyncio.Semaphore(1)
_REQUEST_LOCK = asyncio.Lock()
_LAST_REQUEST_AT = 0.0
_MIN_REQUEST_INTERVAL_SECONDS = 1.0
_DISABLED_REASON = ""
_PERMANENT_FAILURE_STATUSES = {400, 401, 402, 403, 422, 432}


async def _wait_for_request_slot() -> None:
    """Serialize API calls so a single workflow run stays below provider rate limits."""
    global _LAST_REQUEST_AT
    async with _REQUEST_LOCK:
        now = asyncio.get_running_loop().time()
        wait_for = _MIN_REQUEST_INTERVAL_SECONDS - (now - _LAST_REQUEST_AT)
        if wait_for > 0:
            await asyncio.sleep(wait_for)
        _LAST_REQUEST_AT = asyncio.get_running_loop().time()


def _response_detail(response: httpx.Response, api_key: str) -> str:
    body = " ".join(response.text.split())[:300]
    if api_key:
        body = body.replace(api_key, "***")
    return body or "no response body"


def _disable_for_run(status_code: int, detail: str) -> None:
    global _DISABLED_REASON
    if _DISABLED_REASON:
        return
    _DISABLED_REASON = f"HTTP {status_code}: {detail}"
    log.error(
        "Tavily disabled for this workflow run after a non-retriable response (%s). "
        "Check TAVILY_API_KEY, account credits, and provider status.",
        _DISABLED_REASON,
    )


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
    if _DISABLED_REASON:
        log.debug("Tavily skipped for query [%s]: %s", query, _DISABLED_REASON)
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
        if _DISABLED_REASON:
            return []
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(2),
                wait=wait_exponential(min=1, max=8),
                reraise=True,
            ):
                with attempt:
                    await _wait_for_request_slot()
                    async with httpx.AsyncClient(timeout=30) as client:
                        r = await client.post(TAVILY_URL, json=payload)
                        if r.status_code in _PERMANENT_FAILURE_STATUSES:
                            _disable_for_run(r.status_code, _response_detail(r, api_key))
                            return []
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
            log.warning("Tavily query failed [%s]: %r", query, e)
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
