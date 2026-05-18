"""Direct news source scraping — fallback when search APIs are unavailable.

Scrapes known news sites for recent articles. Each source has its own parser.
Returns structured items compatible with the RSS/tavily item format.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import httpx
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_SEMAPHORE = asyncio.Semaphore(3)


def _clean(text: Any | None) -> str:
    if not text:
        return ""
    if hasattr(text, "get_text"):
        text = text.get_text(" ", strip=True)
    return " ".join(text.split())


async def _fetch_html(url: str, timeout: float = 15.0) -> str:
    """Fetch URL, return HTML text. Returns empty string on failure."""
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            timeout=timeout,
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        log.warning("Direct fetch failed [%s]: %s", url, e)
        return ""


# ── Per-source parsers ──────────────────────────────────────────────

async def _parse_techcrunch() -> list[dict[str, Any]]:
    """Scrape TechCrunch AI category page."""
    html = await _fetch_html("https://techcrunch.com/category/artificial-intelligence/")
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict[str, Any]] = []
    # TechCrunch post cards
    for article in soup.select("article, .post-block, .loop-card")[:10]:
        link = article.select_one("a[href]")
        title_el = article.select_one("h2, h3, .loop-card__title")
        excerpt_el = article.select_one("p, .loop-card__excerpt")
        if not link or not title_el:
            continue
        url = link.get("href", "")
        title = _clean(title_el.get_text())
        snippet = _clean(excerpt_el.get_text()) if excerpt_el else ""
        if not title or not url:
            continue
        if not url.startswith("http"):
            url = "https://techcrunch.com" + url
        items.append({
            "title": title,
            "url": url,
            "snippet": snippet[:600],
            "source": "techcrunch",
            "category_hint": "industry",
        })
    log.info("Direct: techcrunch → %d items", len(items))
    return items


async def _parse_theverge_ai() -> list[dict[str, Any]]:
    """Fetch The Verge AI section via RSS (HTML page is JS-rendered)."""
    html = await _fetch_html("https://www.theverge.com/rss/ai-artificial-intelligence/index.xml")
    if not html:
        return []
    soup = BeautifulSoup(html, "xml")
    items: list[dict[str, Any]] = []
    for entry in soup.select("entry, item")[:10]:
        title_el = entry.select_one("title")
        link_el = entry.select_one("link")
        if title_el and link_el:
            title = _clean(title_el.get_text())
            url = link_el.get("href") or link_el.get_text(strip=True) or ""
            if not title or not url:
                continue
            items.append({
                "title": title,
                "url": url,
                "snippet": "",
                "source": "verge",
                "category_hint": "industry",
            })
    log.info("Direct: theverge/ai → %d items", len(items))
    return items


async def _parse_36kr() -> list[dict[str, Any]]:
    """Scrape 36kr newsflash (快讯)."""
    html = await _fetch_html("https://36kr.com/newsflashes")
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict[str, Any]] = []
    # 36kr newsflash items
    for item in soup.select(".newsflash-item, .item-desc, a.item-title")[:12]:
        if item.name == "a":
            url = item.get("href", "")
            title = _clean(item.get_text())
            if not url.startswith("http"):
                url = "https://36kr.com" + url
        else:
            link = item.select_one("a[href]")
            if not link:
                continue
            url = link.get("href", "")
            if not url.startswith("http"):
                url = "https://36kr.com" + url
            title = _clean(link.get_text())
        desc = _clean(item.select_one(".item-desc, .desc, p"))
        if not title or not url:
            continue
        items.append({
            "title": title,
            "url": url,
            "snippet": desc[:600] if desc else "",
            "source": "36kr",
            "category_hint": "chinese",
        })
    log.info("Direct: 36kr → %d items", len(items))
    return items


async def _parse_jiqizhixin() -> list[dict[str, Any]]:
    """Scrape 机器之心 homepage."""
    html = await _fetch_html("https://www.jiqizhixin.com/")
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict[str, Any]] = []
    for article in soup.select("article, .article-item, .content-item")[:10]:
        link = article.select_one("a[href]")
        title_el = article.select_one("h2, h3, .title")
        if not link or not title_el:
            continue
        url = link.get("href", "")
        if not url.startswith("http"):
            url = "https://www.jiqizhixin.com" + url
        title = _clean(title_el.get_text())
        if not title:
            continue
        items.append({
            "title": title,
            "url": url,
            "snippet": "",
            "source": "jiqizhixin",
            "category_hint": "chinese",
        })
    log.info("Direct: jiqizhixin → %d items", len(items))
    return items


async def _parse_steam_news() -> list[dict[str, Any]]:
    """Scrape Steam news feed."""
    html = await _fetch_html("https://store.steampowered.com/news/")
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict[str, Any]] = []
    for item in soup.select(".newsitem, .event, a.news_item")[:10]:
        if item.name == "a":
            url = item.get("href", "")
            title_el = item.select_one(".newsitem_Title, .event_title, h3")
            title = _clean(title_el.get_text()) if title_el else _clean(item.get_text())
        else:
            link = item.select_one("a[href]")
            title_el = item.select_one(".newsitem_Title, .event_title, h3")
            if not link:
                continue
            url = link.get("href", "")
            title = _clean(title_el.get_text()) if title_el else ""
        if not url or not title:
            continue
        if not url.startswith("http"):
            url = "https://store.steampowered.com" + url
        items.append({
            "title": title,
            "url": url,
            "snippet": "",
            "source": "steam",
            "category_hint": "industry",
        })
    log.info("Direct: steam → %d items", len(items))
    return items


async def _parse_ign() -> list[dict[str, Any]]:
    """Scrape IGN news."""
    html = await _fetch_html("https://www.ign.com/news")
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict[str, Any]] = []
    for article in soup.select("article, .item, .content-item")[:10]:
        link = article.select_one("a[href]")
        title_el = article.select_one("h2, h3, .title")
        if not link or not title_el:
            continue
        url = link.get("href", "")
        if not url.startswith("http"):
            url = "https://www.ign.com" + url
        title = _clean(title_el.get_text())
        if not title:
            continue
        items.append({
            "title": title,
            "url": url,
            "snippet": "",
            "source": "ign",
            "category_hint": "industry",
        })
    log.info("Direct: ign → %d items", len(items))
    return items


# Map of source name → parser function
DIRECT_SOURCES: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "techcrunch": _parse_techcrunch,
    "verge": _parse_theverge_ai,
    "36kr": _parse_36kr,
    "jiqizhixin": _parse_jiqizhixin,
    "steam": _parse_steam_news,
    "ign": _parse_ign,
}


async def fetch_direct_sources(
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch recent articles from direct news sources.

    Args:
        sources: List of source keys to fetch. Defaults to all.

    Returns deduplicated list of items compatible with RSS/tavily format.
    """
    keys = sources or list(DIRECT_SOURCES.keys())
    tasks = {}
    for k in keys:
        parser = DIRECT_SOURCES.get(k)
        if parser:
            tasks[k] = parser()

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for (k, items) in zip(tasks.keys(), results):
        if isinstance(items, Exception):
            log.warning("Direct source %s failed: %s", k, items)
            continue
        for item in items:
            url = item.get("url", "")
            if url in seen:
                continue
            seen.add(url)
            merged.append(item)

    log.info("Direct sources: %d sources → %d unique items", len(tasks), len(merged))
    return merged
