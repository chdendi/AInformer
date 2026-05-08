"""Scrape github.com/trending for the daily Top-N repository list.

GitHub does not expose an official trending API, so we parse the public HTML
page. The selector targets `article.Box-row`, which has been stable since
2019. Failures (HTTP errors, layout changes) degrade gracefully: callers see
an empty list and the daily report simply omits the trending section.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

TRENDING_URL = "https://github.com/trending"
DEFAULT_LIMIT = 20
REQUEST_TIMEOUT = 15.0
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def _parse_stars_today(text: str) -> int:
    """Parse '1,234 stars today' / '128 stars this week' → int. Returns 0 on failure."""
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 0


def _parse_html(html: str, limit: int) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("article.Box-row")
    items: list[dict[str, Any]] = []

    for row in rows[:limit]:
        link = row.select_one("h2 a")
        if not link:
            continue
        href = link.get("href", "").strip()
        if not href.startswith("/"):
            continue
        full_name = href.lstrip("/")
        parts = full_name.split("/", 1)
        if len(parts) != 2:
            continue
        owner, repo = parts

        description = _clean(
            (row.select_one("p") or {}).get_text(" ", strip=True)
            if row.select_one("p")
            else ""
        )
        language = _clean(
            (row.select_one("[itemprop='programmingLanguage']") or {}).get_text(strip=True)
            if row.select_one("[itemprop='programmingLanguage']")
            else ""
        )
        stars_total_text = ""
        stars_link = row.select_one(f"a[href='/{full_name}/stargazers']")
        if stars_link:
            stars_total_text = _clean(stars_link.get_text(strip=True))
        stars_today = 0
        for span in row.select("span.d-inline-block.float-sm-right"):
            stars_today = _parse_stars_today(span.get_text(strip=True))
            if stars_today:
                break

        items.append(
            {
                "owner": owner,
                "repo": repo,
                "full_name": full_name,
                "url": f"https://github.com/{full_name}",
                "description": description,
                "language": language,
                "stars_total": stars_total_text,
                "stars_today": stars_today,
            }
        )

    return items


async def fetch_trending(limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    """Fetch GitHub Trending (daily, all languages). Returns up to `limit` items.

    Network or parsing failures log a warning and return an empty list — the
    caller is expected to treat empty results as "skip the trending section".
    """
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
        ) as client:
            resp = await client.get(TRENDING_URL, params={"since": "daily"})
            resp.raise_for_status()
            html = resp.text
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        log.warning("GitHub trending fetch failed: %s", exc)
        return []

    try:
        items = _parse_html(html, limit)
    except Exception as exc:  # noqa: BLE001 — parsing is best-effort
        log.warning("GitHub trending parse failed: %s", exc)
        return []

    log.info("GitHub trending fetched: %d repos", len(items))
    return items
