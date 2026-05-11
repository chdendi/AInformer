"""Unified web search with automatic fallback chain.

Priority: Tavily API → DuckDuckGo search → direct news source scraping.
Each layer only activates when the layer above returns no results.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from .duckduckgo import ddg_batch_search
from .news_direct import fetch_direct_sources
from .tavily import batch_search as tavily_batch

log = logging.getLogger(__name__)


async def unified_search(
    queries: list[str],
    *,
    max_results: int = 6,
    days: int = 2,
) -> list[dict[str, Any]]:
    """Search with fallback: Tavily → DDG → direct sources.

    Only activates next layer when current layer returns zero results.
    Results are deduplicated by URL within each layer.
    """
    all_materials: list[dict[str, Any]] = []

    # Layer 1: Tavily (if API key configured)
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if tavily_key:
        log.info("Search layer 1: Tavily (%d queries)", len(queries))
        tavily_results = await tavily_batch(queries, max_results=max_results, days=days)
        all_materials.extend(tavily_results)
        if tavily_results:
            log.info("Tavily returned %d results — skipping fallback", len(tavily_results))
            return all_materials
        log.info("Tavily returned empty, falling back to DDG")
    else:
        log.info("TAVILY_API_KEY not set, skipping Tavily layer")

    # Layer 2: DuckDuckGo
    log.info("Search layer 2: DuckDuckGo (%d queries)", len(queries))
    ddg_results = await ddg_batch_search(queries, max_results=max_results)
    # Filter out items already seen from Tavily
    seen_urls = {m["url"] for m in all_materials}
    for item in ddg_results:
        if item["url"] not in seen_urls:
            all_materials.append(item)
    if ddg_results:
        log.info("DDG returned %d new results", len(ddg_results))
        return all_materials
    log.info("DDG returned empty, falling back to direct source scraping")

    # Layer 3: Direct news source scraping
    log.info("Search layer 3: direct news source scraping")
    direct_results = await fetch_direct_sources()
    seen_urls = {m["url"] for m in all_materials}
    for item in direct_results:
        if item["url"] not in seen_urls:
            all_materials.append(item)
    if direct_results:
        log.info("Direct sources returned %d new results", len(direct_results))
    else:
        log.warning("All search layers exhausted — no results found")

    return all_materials
