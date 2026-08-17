"""Unified web search with an API-free fallback chain.

Priority: DuckDuckGo search → direct news source scraping.
Each layer only activates when the layer above returns no results.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .duckduckgo import ddg_batch_search
from .news_direct import fetch_direct_sources

log = logging.getLogger(__name__)


async def unified_search(
    queries: list[str],
    *,
    max_results: int = 6,
    days: int = 2,
) -> list[dict[str, Any]]:
    """Search with fallback: DuckDuckGo → direct sources.

    Only activates next layer when current layer returns zero results.
    Results are deduplicated by URL within each layer.
    """
    # Layer 1: DuckDuckGo
    log.info("Search layer 1: DuckDuckGo (%d queries)", len(queries))
    ddg_results = await ddg_batch_search(queries, max_results=max_results)
    if ddg_results:
        log.info("DDG returned %d results", len(ddg_results))
        return ddg_results
    log.info("DDG returned empty, falling back to direct source scraping")

    # Layer 2: Direct news source scraping
    log.info("Search layer 2: direct news source scraping")
    direct_results = await fetch_direct_sources()
    if direct_results:
        log.info("Direct sources returned %d results", len(direct_results))
    else:
        log.warning("All search layers exhausted — no results found")

    return direct_results
