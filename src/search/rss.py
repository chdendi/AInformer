from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import httpx

log = logging.getLogger(__name__)


RSS_FEEDS: dict[str, dict[str, Any]] = {
    # 厂商官方
    "openai": {"url": "https://openai.com/news/rss.xml", "category": "industry"},
    "anthropic_news": {"url": "https://www.anthropic.com/news/rss.xml", "category": "industry"},
    "anthropic_engineering": {"url": "https://www.anthropic.com/engineering/rss.xml", "category": "tutorial"},
    "deepmind": {"url": "https://deepmind.google/blog/rss.xml", "category": "industry"},
    "huggingface_blog": {"url": "https://huggingface.co/blog/feed.xml", "category": "tutorial"},
    "meta_ai": {"url": "https://ai.meta.com/blog/rss/", "category": "industry"},
    "google_research": {"url": "https://research.google/blog/rss/", "category": "industry"},
    # 英文媒体
    "verge_ai": {"url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "category": "industry"},
    "techcrunch_ai": {"url": "https://techcrunch.com/category/artificial-intelligence/feed/", "category": "industry"},
    "venturebeat_ai": {"url": "https://venturebeat.com/category/ai/feed/", "category": "industry"},
    "arstechnica_ai": {"url": "https://feeds.arstechnica.com/arstechnica/index", "category": "industry"},
    # 社区 / 聚合
    "hn_ai": {
        "url": "https://hnrss.org/frontpage?points=150&q=AI+OR+LLM+OR+Claude+OR+OpenAI+OR+GPT+OR+Anthropic+OR+DeepSeek",
        "category": "tutorial",
    },
    # 学术
    "arxiv_ai": {"url": "https://export.arxiv.org/rss/cs.AI", "category": "chinese"},
    "arxiv_cl": {"url": "https://export.arxiv.org/rss/cs.CL", "category": "chinese"},
    "arxiv_lg": {"url": "https://export.arxiv.org/rss/cs.LG", "category": "chinese"},
    # 中文
    "jiqizhixin": {"url": "https://www.jiqizhixin.com/rss", "category": "chinese"},
    "qbitai": {"url": "https://www.qbitai.com/feed", "category": "chinese"},
    "sspai": {"url": "https://sspai.com/feed", "category": "tutorial"},
    "36kr_newsflash": {"url": "https://36kr.com/feed-newsflash", "category": "chinese"},
}

UA = "Mozilla/5.0 (compatible; AInformer/1.0; +https://github.com)"


async def _fetch(client: httpx.AsyncClient, name: str, meta: dict[str, Any]):
    try:
        r = await client.get(meta["url"], timeout=20, follow_redirects=True, headers={"User-Agent": UA})
        if r.status_code != 200:
            log.warning("RSS %s -> HTTP %s", name, r.status_code)
            return name, None
        parsed = feedparser.parse(r.text)
        return name, parsed
    except Exception as e:
        log.warning("RSS %s failed: %s", name, e)
        return name, None


def _entry_dt(entry: Any) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        val = entry.get(key)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc)
            except Exception:
                continue
    return None


async def collect_rss(within_days: int = 2) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=within_days)
    out: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=20) as client:
        results = await asyncio.gather(
            *[_fetch(client, n, m) for n, m in RSS_FEEDS.items()],
            return_exceptions=False,
        )
    for name, parsed in results:
        if not parsed:
            continue
        meta = RSS_FEEDS[name]
        for e in parsed.entries[:25]:
            dt = _entry_dt(e)
            if dt and dt < cutoff:
                continue
            title = (e.get("title") or "").strip()
            link = (e.get("link") or "").strip()
            if not title or not link:
                continue
            summary = (e.get("summary") or e.get("description") or "").strip()
            out.append({
                "title": title,
                "url": link,
                "snippet": summary[:600],
                "source": name,
                "published_at": dt.isoformat() if dt else "",
                "category_hint": meta["category"],
            })
    log.info("RSS collected %d items", len(out))
    return out
