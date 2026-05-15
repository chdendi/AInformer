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
    # Anthropic 已下线官方 RSS（2026-05 实测全部 404），暂依赖 Tavily 兜底
    "deepmind": {"url": "https://deepmind.google/blog/rss.xml", "category": "industry"},
    "huggingface_blog": {"url": "https://huggingface.co/blog/feed.xml", "category": "industry"},
    # ai.meta.com 已下线 RSS，改用 Meta 工程博客 AI Research 分类
    "meta_eng_ai": {"url": "https://engineering.fb.com/category/ai-research/feed/", "category": "industry"},
    "google_research": {"url": "https://research.google/blog/rss/", "category": "industry"},
    "nvidia_blog": {"url": "https://blogs.nvidia.com/feed/", "category": "industry"},
    "microsoft_ai": {"url": "https://blogs.microsoft.com/ai/feed/", "category": "industry"},
    "aws_ml": {"url": "https://aws.amazon.com/blogs/machine-learning/feed/", "category": "industry"},
    # 英文媒体
    "verge_ai": {"url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "category": "industry"},
    "techcrunch_ai": {"url": "https://techcrunch.com/category/artificial-intelligence/feed/", "category": "industry"},
    "venturebeat_ai": {"url": "https://venturebeat.com/category/ai/feed/", "category": "industry"},
    "arstechnica_ai": {"url": "https://feeds.arstechnica.com/arstechnica/index", "category": "industry"},
    "techreview": {"url": "https://www.technologyreview.com/feed/", "category": "industry"},
    "marktechpost": {"url": "https://www.marktechpost.com/feed/", "category": "industry"},
    # 社区 / 聚合
    "hn_ai": {
        "url": "https://hnrss.org/frontpage?points=150&q=AI+OR+LLM+OR+Claude+OR+OpenAI+OR+GPT+OR+Anthropic+OR+DeepSeek",
        "category": "industry",
    },
    # AI 专业媒体（内容密度高，每日必有新稿）
    "the_decoder": {"url": "https://the-decoder.com/feed/", "category": "industry"},
    "wired_ai": {"url": "https://www.wired.com/feed/category/artificial-intelligence/latest/rss", "category": "industry"},
    "semafor_ai": {"url": "https://www.semafor.com/rss/topic/ai", "category": "industry"},
    # 实践派评论 / 长文
    "latent_space": {"url": "https://www.latent.space/feed", "category": "opinion"},
    "simon_willison": {"url": "https://simonwillison.net/atom/everything/", "category": "opinion"},
    "huyenchip": {"url": "https://huyenchip.com/feed.xml", "category": "opinion"},
    # 观点 / 评论
    "sam_altman": {"url": "https://blog.samaltman.com/posts.atom", "category": "opinion"},
    "import_ai": {"url": "https://importai.substack.com/feed", "category": "opinion"},
    "lesswrong": {"url": "https://www.lesswrong.com/feed.xml?view=curated", "category": "opinion"},
    "thegradient": {"url": "https://thegradient.pub/feed/", "category": "opinion"},
    "interconnects": {"url": "https://www.interconnects.ai/feed", "category": "opinion"},
    "aisnakeoil": {"url": "https://www.aisnakeoil.com/feed.xml", "category": "opinion"},
    "karpathy_blog": {"url": "https://karpathy.github.io/feed.xml", "category": "opinion"},
    "darioamodei_blog": {"url": "https://darioamodei.com/feed.xml", "category": "opinion"},
    "oneusefulthing": {"url": "https://www.oneusefulthing.org/feed", "category": "opinion"},
    # 高产观点写手 — Tavily 不可用时主要靠这几位撑住 opinion 栏目
    "gary_marcus": {"url": "https://garymarcus.substack.com/feed", "category": "opinion"},
    "raschka": {"url": "https://magazine.sebastianraschka.com/feed", "category": "opinion"},
    "lilian_weng": {"url": "https://lilianweng.github.io/index.xml", "category": "opinion"},
    # Dwarkesh 暂无可用公开 RSS（dwarkesh.com /feed 超时、Substack ID 未公开），暂依赖 Tavily 兜底
    # 学术
    "arxiv_ai": {"url": "https://export.arxiv.org/rss/cs.AI", "category": "academic"},
    "arxiv_cl": {"url": "https://export.arxiv.org/rss/cs.CL", "category": "academic"},
    "arxiv_lg": {"url": "https://export.arxiv.org/rss/cs.LG", "category": "academic"},
    "bair": {"url": "https://bair.berkeley.edu/blog/feed.xml", "category": "academic"},
    # 中文
    # 机器之心已下线 RSS（站点全部返回 HTML 而非 RSS），暂依赖 qbitai/36kr/direct scrape
    "qbitai": {"url": "https://www.qbitai.com/feed", "category": "chinese"},
    "36kr_newsflash": {"url": "https://36kr.com/feed-newsflash", "category": "chinese"},
    "36kr_ai": {"url": "https://36kr.com/feed?cid=ai", "category": "chinese"},
    "qwen_blog": {"url": "https://qwenlm.github.io/blog/index.xml", "category": "chinese"},
    "leiphone": {"url": "https://www.leiphone.com/feed", "category": "chinese"},
    "sspai": {"url": "https://sspai.com/feed", "category": "chinese"},
    "geekpark": {"url": "https://www.geekpark.net/feed", "category": "chinese"},
    # 游戏 / 科技媒体（可能含 AI 交叉内容，由 LLM 筛选）
    "steam_news": {"url": "https://store.steampowered.com/feeds/news.xml", "category": "industry"},
    "ign": {"url": "https://feeds.feedburner.com/ign/all", "category": "industry"},
    "pcgamer": {"url": "https://www.pcgamer.com/uk/rss/", "category": "industry"},
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
