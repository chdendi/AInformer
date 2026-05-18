from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, timedelta
from typing import Any

from openai import AsyncOpenAI

from ..config import LLMConfig
from ..llm.client import chat_json
from ..search.web_search import unified_search
from .definitions import ANALYST_NAMES, LEADER_PROFILES, AgentSpec

log = logging.getLogger(__name__)


AGENT_SYSTEM = (
    "你是 AI 行业研究员，从搜索素材中筛选高质量 AI 资讯为中文日报供稿。\n"
    "规则：只输出 JSON。禁止编造。中文撰写，专有名词保留英文。每条给出价值点。"
)

_JSON_FMT_OPINION = """  "items": [
    {
      "tier": "leader|analyst",
      "title": "中文标题（<40字）",
      "summary": "1-2句摘要（<80字）",
      "value_note": "一句话价值点（<40字）",
      "source_name": "媒体、作者或社区名",
      "url": "原始 URL（必须来自素材）",
      "published_at": "ISO 时间或空字符串",
      "importance": "hot|star|pin",
      "quote_en": "英文原文摘句；若素材为中文可留空",
      "quote_zh": "中文观点摘句或中文翻译",
      "person": "严格匹配 leader 或 analyst 名单中的人物全名"
    }
  ]"""

_JSON_FMT_BASE = """  "items": [
    {
      "title": "中文标题（<40字）",
      "summary": "1-2句摘要（<80字）",
      "value_note": "一句话价值点（<40字）",
      "source_name": "媒体或人物名",
      "url": "原始 URL（必须来自素材）",
      "published_at": "ISO 时间或空字符串",
      "importance": "hot|star|pin"
    }
  ]"""

_MATERIAL_LIMITS = {
    "industry": 40,
    "opinion": 40,
    "academic": 35,
    "chinese": 35,
}

_SOURCE_RANK = {
    "openai": 35,
    "deepmind": 32,
    "huggingface_blog": 28,
    "meta_eng_ai": 26,
    "google_research": 26,
    "nvidia_blog": 26,
    "microsoft_ai": 24,
    "aws_ml": 22,
    "verge_ai": 22,
    "techcrunch_ai": 22,
    "the_decoder": 22,
    "venturebeat_ai": 18,
    "marktechpost": 16,
    "techreview": 14,
    "arstechnica_ai": 14,
    "qbitai": 18,
    "36kr_ai": 14,
    "ddg": -10,
}

_INDUSTRY_SIGNAL_WORDS = (
    "ai", "llm", "gpt", "claude", "gemini", "grok", "llama", "mistral",
    "openai", "anthropic", "deepmind", "nvidia", "model", "agent",
    "agents", "siri", "copilot", "chatbot", "benchmark", "release",
    "launch", "update", "open-source", "open source", "funding",
    "regulation", "policy", "deepfake", "生成式", "大模型", "智能体",
    "模型", "发布", "开源", "融资", "监管", "产品",
)

_BROAD_INDUSTRY_SOURCES = {"arstechnica_ai", "ign", "pcgamer", "steam"}


def _build_user_prompt(
    spec: AgentSpec,
    materials: list[dict[str, Any]],
    excluded: list[str],
    today: str,
    relaxed: bool = False,
) -> str:
    excluded_block = "\n".join(f"- {t}" for t in excluded[:40]) if excluded else "（无）"
    material_lines = []
    material_limit = _MATERIAL_LIMITS.get(spec.key, 30)
    for i, m in enumerate(materials[:material_limit], 1):
        pub = (m.get("published_at") or "")[:10]
        material_lines.append(
            f"[{i}] {m.get('title','')} | {m.get('url','')} | {m.get('source','?')} | {pub}\n"
            f"  {(m.get('snippet') or '')[:150]}"
        )
    materials_block = "\n\n".join(material_lines) if material_lines else "（无素材）"

    json_fmt = _JSON_FMT_OPINION if spec.key == "opinion" else _JSON_FMT_BASE

    try:
        window_start = (date.fromisoformat(today) - timedelta(days=1)).isoformat()
    except ValueError:
        window_start = today

    if relaxed:
        requirements = (
            "- 宽松模式：目标输出 2-3 条，降低质量门槛，但仍需有实质内容（非纯营销稿）。\n"
            f"- 候选素材的 published_at 在 {window_start} ~ {today} 窗口内即可入选，不必强求当日。\n"
            "- importance：star 或 pin 即可。\n"
            "- url 必须来自候选素材，禁止编造。\n"
            "- 务必跳过与\"近期已报道标题\"相似的内容（同一事件不同媒体均算重复）。\n"
            '- 如果候选素材中确实没有符合该栏目的内容，返回 {"items": []}'
        )
    else:
        requirements = (
            "- 输出 3-5 条最高质量资讯，宁缺毋滥；若当天高质量素材不足，输出 1-2 条也可以，不要因为数量不足返回空。\n"
            f"- 候选素材的 published_at 在 {window_start} ~ {today} 窗口内即可入选，不必强求当日发布。\n"
            "- importance：最多 1 hot，1-2 star，其余 pin。\n"
            "- url 必须来自候选素材，禁止编造。\n"
            "- 务必跳过与\"近期已报道标题\"相似的内容（标题相似或同一事件不同媒体均算重复）。"
        )

    mode_note = "（宽松兜底模式）" if relaxed else ""

    return f"""
报道窗口：{window_start} 至 {today}（含昨日发布的内容；超出窗口的素材请跳过）
栏目：{spec.name}{mode_note}
焦点：{spec.focus}
要求：{spec.extra_instructions}

## 近期已报道过的标题（务必跳过，避免与昨日日报重复）：
{excluded_block}

## 候选素材（{len(materials)} 条，已初筛并按可信度/时效排序）：
{materials_block}

## JSON 输出格式：
{{
{json_fmt}
}}

要求：
{requirements}
- 顶层 JSON 只能包含 items 字段，不要输出解释、诊断或其他字段。
""".strip()


def _material_date(item: dict[str, Any]) -> date | None:
    raw = (item.get("published_at") or "")[:10]
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _material_score(spec_key: str, item: dict[str, Any], today: str) -> int:
    title = item.get("title") or ""
    url = item.get("url") or ""
    if not title or not url:
        return -1000

    source = (item.get("source") or "").lower()
    category = item.get("category_hint") or ""
    text = f"{title} {item.get('snippet') or ''}".lower()

    score = _SOURCE_RANK.get(source, 0)
    if category == spec_key:
        score += 25
    if source != "ddg":
        score += 8

    item_day = _material_date(item)
    try:
        today_day = date.fromisoformat(today)
    except ValueError:
        today_day = None
    if item_day and today_day:
        age = (today_day - item_day).days
        if age == 0:
            score += 20
        elif age == 1:
            score += 16
        elif age == 2:
            score += 8
        elif age > 2:
            score -= 15
    elif source == "ddg":
        score -= 12

    if spec_key == "industry":
        hits = sum(1 for word in _INDUSTRY_SIGNAL_WORDS if word in text)
        if source in _BROAD_INDUSTRY_SOURCES and hits == 0:
            return -1000
        score += min(hits * 5, 30)

    return score


def _prepare_prompt_materials(
    spec_key: str,
    materials: list[dict[str, Any]],
    today: str,
) -> list[dict[str, Any]]:
    """Rank material before it reaches the LLM prompt window.

    Search can return stale pages or ad links, while RSS usually carries the
    freshest source-specific updates. Ranking prevents low-signal search
    results from crowding useful RSS items out of the fixed prompt window.
    """
    ranked = [
        (_material_score(spec_key, item, today), idx, item)
        for idx, item in enumerate(materials)
    ]
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [item for score, _, item in ranked if score > -100]


async def run_agent(
    spec: AgentSpec,
    rss_pool: list[dict[str, Any]],
    excluded_by_cat: dict[str, list[str]],
    today: str,
    client: AsyncOpenAI,
    cfg: LLMConfig,
) -> dict[str, Any]:
    log.info("[agent:%s] starting", spec.key)
    excluded_titles = excluded_by_cat.get(spec.key, []) + excluded_by_cat.get("_headlines", [])

    search_items: list[dict[str, Any]] = await unified_search(spec.queries, max_results=6, days=2)
    log.info("[agent:%s] search=%d", spec.key, len(search_items))

    rss_items = [m for m in rss_pool if m.get("category_hint") in spec.rss_categories] if spec.rss_categories else []
    log.info("[agent:%s] rss=%d", spec.key, len(rss_items))

    materials = _merge_materials(search_items, rss_items, prefer_rss=spec.key == "opinion")

    if not materials:
        log.warning("[agent:%s] no materials (search=%d rss=%d), returning empty",
                    spec.key, len(search_items), len(rss_items))
        return {"key": spec.key, "name": spec.name, "items": []}

    log.info("[agent:%s] materials=%d (search=%d, rss=%d), excluded_titles=%d",
             spec.key, len(materials), len(search_items), len(rss_items), len(excluded_titles))

    prompt_materials = _prepare_prompt_materials(spec.key, materials, today)
    top_sources = [m.get("source", "?") for m in prompt_materials[:8]]
    log.info(
        "[agent:%s] prompt_materials=%d/%d top_sources=%s",
        spec.key, len(prompt_materials), len(materials), top_sources,
    )

    valid_urls = {m["url"] for m in prompt_materials}
    material_corpus = _materials_corpus(prompt_materials) if spec.key == "opinion" else ""

    user_prompt = _build_user_prompt(spec, prompt_materials, excluded_titles, today)
    try:
        result = await chat_json(client, cfg, AGENT_SYSTEM, user_prompt, temperature=0.3, max_tokens=3000)
    except Exception as e:
        log.error("[agent:%s] LLM strict-mode failed: %s", spec.key, e)
        result = {}

    filtered, drops = _validate_items(result, valid_urls, spec.key, mode="strict")
    if spec.key == "opinion":
        before_strict_tier = len(filtered)
        filtered = _validate_opinion_tier(filtered, material_corpus)
        log.info(
            "[agent:opinion] strict tier-check kept %d/%d (raw=%d, fabricated=%d)",
            len(filtered), before_strict_tier, drops["raw"], drops["fabricated"],
        )

    # Fallback: if LLM returned 0 items but materials existed, retry with relaxed constraints
    if not filtered and materials:
        log.warning(
            "[agent:%s] strict mode produced 0 items (raw=%d, fabricated_url_drops=%d) — retrying relaxed",
            spec.key,
            drops["raw"],
            drops["fabricated"],
        )
        relaxed_prompt = _build_user_prompt(spec, prompt_materials, excluded_titles, today, relaxed=True)
        try:
            result2 = await chat_json(client, cfg, AGENT_SYSTEM, relaxed_prompt, temperature=0.5, max_tokens=2000)
        except Exception as e:
            log.error("[agent:%s] relaxed-mode LLM failed: %s", spec.key, e)
            result2 = {}
        filtered, drops2 = _validate_items(result2, valid_urls, spec.key, mode="relaxed")
        if spec.key == "opinion":
            before_relaxed_tier = len(filtered)
            filtered = _validate_opinion_tier(filtered, material_corpus)
            log.info(
                "[agent:opinion] relaxed tier-check kept %d/%d (raw=%d, fabricated=%d)",
                len(filtered), before_relaxed_tier, drops2["raw"], drops2["fabricated"],
            )
        else:
            log.info(
                "[agent:%s] relaxed mode raw=%d kept=%d fabricated_url_drops=%d",
                spec.key,
                drops2["raw"],
                len(filtered),
                drops2["fabricated"],
            )

    log.info("[agent:%s] returned %d items", spec.key, len(filtered))
    return {"key": spec.key, "name": spec.name, "items": filtered}


def _merge_materials(
    search_items: list[dict[str, Any]],
    rss_items: list[dict[str, Any]],
    *,
    prefer_rss: bool = False,
) -> list[dict[str, Any]]:
    """Merge search and RSS results while keeping both source types visible.

    For opinion/community voices, curated RSS feeds are often higher-signal than
    broad search results. Interleaving prevents one source from crowding the
    other out of the prompt's material window.
    """
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []

    if not prefer_rss:
        for src in (search_items, rss_items):
            for m in src:
                url = m.get("url")
                if not url or url in seen:
                    continue
                seen.add(url)
                merged.append(m)
        return merged

    primary, secondary = rss_items, search_items
    max_len = max(len(primary), len(secondary))

    for i in range(max_len):
        for src in (primary, secondary):
            if i >= len(src):
                continue
            m = src[i]
            url = m.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            merged.append(m)

    return merged


def _extract_result_items(
    result: dict[str, Any] | Any,
    spec_key: str,
    mode: str,
) -> list[Any]:
    if isinstance(result, list):
        return result
    if not isinstance(result, dict):
        return []

    direct = result.get("items")
    if isinstance(direct, list):
        return direct

    candidate_keys = (
        spec_key,
        "results",
        "articles",
        "news",
        "资讯",
        "行业新闻",
        "AI 行业新闻与产品动态",
    )
    for key in candidate_keys:
        val = result.get(key)
        if isinstance(val, list):
            log.warning("[agent:%s][%s] extracted items from non-standard key '%s'", spec_key, mode, key)
            return val
        if isinstance(val, dict) and isinstance(val.get("items"), list):
            log.warning("[agent:%s][%s] extracted nested items from key '%s'", spec_key, mode, key)
            return val["items"]

    list_values = [v for v in result.values() if isinstance(v, list)]
    if len(list_values) == 1 and all(isinstance(x, dict) for x in list_values[0]):
        log.warning("[agent:%s][%s] extracted sole list value from non-standard JSON", spec_key, mode)
        return list_values[0]

    log.warning(
        "[agent:%s][%s] LLM JSON missing top-level items; keys=%s",
        spec_key,
        mode,
        list(result.keys())[:8],
    )
    return []


def _validate_items(
    result: dict[str, Any] | Any,
    valid_urls: set[str],
    spec_key: str,
    *,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter LLM-returned items to those whose URL is in `valid_urls`.

    Returns (kept_items, {"raw": int, "fabricated": int}) so callers can log
    the gap between raw LLM output and items that survived URL validation —
    historically this gap was invisible and made silent failures hard to spot.
    """
    raw_items = _extract_result_items(result, spec_key, mode)
    kept: list[dict[str, Any]] = []
    fabricated = 0
    for it in raw_items:
        if not isinstance(it, dict):
            continue
        if it.get("url") and it["url"] not in valid_urls:
            log.debug("[agent:%s][%s] dropping fabricated url: %s", spec_key, mode, it["url"])
            fabricated += 1
            continue
        it["category"] = spec_key
        kept.append(it)
    return kept, {"raw": len(raw_items), "fabricated": fabricated}


def _materials_corpus(materials: list[dict[str, Any]]) -> str:
    """Concat title + snippet from all materials into one lowercase blob.

    Used by opinion tier validation to verify that a claimed leader's name
    actually appears somewhere in the source pool — guards against the LLM
    fabricating a person field (historically defaulted to "Sam Altman").
    """
    parts: list[str] = []
    for m in materials:
        parts.append(m.get("title") or "")
        parts.append(m.get("snippet") or "")
    return " ".join(parts).lower()


def _validate_opinion_tier(items: list[dict[str, Any]], corpus: str) -> list[dict[str, Any]]:
    """Enforce two-tier opinion taxonomy.

    leader: person ∈ LEADER_PROFILES, name mentioned in materials, Musk gated
            by AI-keyword filter.
    analyst: person ∈ ANALYST_NAMES.
    Items failing both buckets are dropped.
    """
    kept: list[dict[str, Any]] = []
    for it in items:
        tier = (it.get("tier") or "").strip().lower()
        person = (it.get("person") or "").strip()
        if not person:
            log.debug("[opinion-tier] drop empty person: %s", it.get("title"))
            continue

        if tier == "leader" or person in LEADER_PROFILES:
            profile = LEADER_PROFILES.get(person)
            if not profile:
                log.debug("[opinion-tier] drop leader not in roster: %s", person)
                continue
            if person.lower() not in corpus:
                log.debug("[opinion-tier] drop leader '%s' not mentioned in materials", person)
                continue
            kw_filter = profile.get("keyword_filter")
            if kw_filter:
                quote_blob = f"{it.get('quote_en','')} {it.get('quote_zh','')} {it.get('title','')}"
                if not any(kw.lower() in quote_blob.lower() for kw in kw_filter):
                    log.debug("[opinion-tier] drop %s — quote misses AI keywords", person)
                    continue
            it["tier"] = "leader"
            kept.append(it)
            continue

        if tier == "analyst" or person in ANALYST_NAMES:
            if person not in ANALYST_NAMES:
                log.debug("[opinion-tier] drop analyst not in roster: %s", person)
                continue
            it["tier"] = "analyst"
            kept.append(it)
            continue

        log.debug("[opinion-tier] drop unclassified person='%s' tier='%s'", person, tier)
    return kept


async def run_all_agents(
    specs: list[AgentSpec],
    rss_pool: list[dict[str, Any]],
    excluded_by_cat: dict[str, list[str]],
    today: str,
    client: AsyncOpenAI,
    cfg: LLMConfig,
) -> list[dict[str, Any]]:
    return await asyncio.gather(
        *[run_agent(s, rss_pool, excluded_by_cat, today, client, cfg) for s in specs]
    )
