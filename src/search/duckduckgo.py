"""DuckDuckGo search via HTML scraping.

Uses DuckDuckGo Lite (no-JS version) to avoid blocking. When rate-limited
(HTTP 429 / 403), degrades gracefully by returning an empty list.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

DDG_LITE_URL = "https://lite.duckduckgo.com/lite/"
_SEMAPHORE = asyncio.Semaphore(3)  # be gentle to DDG
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _decode_ddg_url(raw: str) -> str:
    """Decode DDG redirect URL like //duckduckgo.com/l/?uddg=ENCODED → real URL."""
    if "uddg=" in raw:
        parsed = urlparse(raw if "://" in raw else f"https:{raw}")
        qs = parse_qs(parsed.query)
        uddg = qs.get("uddg", [""])[0]
        if uddg:
            return uddg
    if raw.startswith("//"):
        return f"https:{raw}"
    return raw


def _is_noise_url(url: str) -> bool:
    """Drop DDG/Bing ad and help links that can appear in Lite results."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.endswith("duckduckgo.com"):
        return True
    if host in {"bing.com", "www.bing.com"} and "/aclick" in parsed.path:
        return True
    return False


async def ddg_search(
    query: str,
    *,
    max_results: int = 8,
) -> list[dict[str, Any]]:
    """Search DuckDuckGo Lite and return up to `max_results` results.

    Returns list of dicts with keys: title, url, snippet.
    Returns empty list on any failure (rate limit, network error, parsing error).
    """
    async with _SEMAPHORE:
        try:
            async with httpx.AsyncClient(
                headers={
                    "User-Agent": USER_AGENT,
                },
                timeout=15.0,
                follow_redirects=True,
            ) as client:
                resp = await client.get(
                    DDG_LITE_URL,
                    params={"q": query},
                )
                if resp.status_code in (429, 403):
                    log.warning("DDG rate-limited for query: %s", query[:60])
                    return []
                if resp.status_code not in (200, 202):
                    log.warning("DDG HTTP %s for query: %s", resp.status_code, query[:60])
                    return []
                html = resp.text
        except Exception as e:
            log.warning("DDG fetch failed for query [%s]: %s", query[:60], e)
            return []

    if not html or len(html) < 200:
        log.warning("DDG empty response for query: %s", query[:60])
        return []

    try:
        soup = BeautifulSoup(html, "html.parser")
        items: list[dict[str, Any]] = []
        # DDG Lite: each result is 2-3 <tr>: one with <a class="result-link">,
        # one with <td class="result-snippet">, one with <span class="link-text">
        result_links = soup.select("a.result-link")
        for link_el in result_links:
            if len(items) >= max_results:
                break
            url = _decode_ddg_url(link_el.get("href", "").strip())
            title = link_el.get_text(strip=True)
            if not url or not title:
                continue
            if _is_noise_url(url):
                log.debug("DDG dropped noise result: %s", url[:120])
                continue

            snippet = ""
            link_row = link_el.find_parent("tr")
            if link_row:
                snippet_row = link_row.find_next_sibling("tr")
                if snippet_row:
                    snippet_td = snippet_row.select_one("td.result-snippet")
                    if snippet_td:
                        snippet = snippet_td.get_text(strip=True)

            items.append({
                "title": title,
                "url": url,
                "snippet": snippet[:600],
                "source": "ddg",
            })
        log.info("DDG query [%s]: %d results", query[:60], len(items))
        return items
    except Exception as e:
        log.warning("DDG parse failed for query [%s]: %s", query[:60], e)
        return []


async def ddg_batch_search(queries: list[str], max_results: int = 8) -> list[dict[str, Any]]:
    """Search multiple queries in parallel, dedup by URL."""
    results = await asyncio.gather(
        *[ddg_search(q, max_results=max_results) for q in queries],
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
    log.info("DDG batch: %d queries → %d unique results", len(queries), len(merged))
    return merged
