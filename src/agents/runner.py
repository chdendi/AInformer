from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, timedelta
from typing import Any

from openai import AsyncOpenAI

from ..config import LLMConfig
from ..llm.client import chat_json
from ..search.web_search import unified_search
from .definitions import ANALYST_NAMES, COMMUNITY_VOICE_SOURCES, LEADER_PROFILES, AgentSpec

log = logging.getLogger(__name__)


AGENT_SYSTEM = (
    "你是 AI 行业研究员，从搜索素材中筛选高质量 AI 资讯为中文日报供稿。\n"
    "规则：只输出 JSON。禁止编造。中文撰写，专有名词保留英文。每条给出价值点。"
)

_JSON_FMT_OPINION = """  "items": [
    {
      "tier": "leader|analyst|community",
      "title": "中文标题（<40字）",
      "summary": "1-2句摘要（<80字）",
      "value_note": "一句话价值点（<40字）",
      "source_name": "媒体、作者或社区名",
      "url": "原始 URL（必须来自素材）",
      "published_at": "ISO 时间或空字符串",
      "importance": "hot|star|pin",
      "quote_en": "英文原文摘句；若素材为中文或 community 核心观点可留空",
      "quote_zh": "中文观点摘句、中文翻译或 community 核心观点",
      "person": "leader/analyst 用人物全名；community 可用作者、社区或 newsletter 名"
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

_RELAXED_MATERIAL_LIMITS = {
    "industry": 12,
    "opinion": 10,
    "academic": 10,
    "chinese": 12,
}

_RELAXED_EXCLUDED_LIMIT = 20

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
    "simon_willison": 30,
    "gary_marcus": 28,
    "interconnects": 27,
    "raschka": 27,
    "huyenchip": 26,
    "lilian_weng": 26,
    "oneusefulthing": 25,
    "karpathy_blog": 25,
    "sam_altman": 25,
    "darioamodei_blog": 25,
    "eugene_yan": 24,
    "hamel_husain": 24,
    "jay_alammar": 24,
    "semianalysis": 23,
    "deeplearning_ai_batch": 22,
    "latent_space": 22,
    "import_ai": 22,
    "thegradient": 20,
    "aisnakeoil": 20,
    "lesswrong": 18,
    "arxiv_ai": 26,
    "arxiv_cl": 24,
    "arxiv_lg": 24,
    "bair": 24,
    "hn_ai_discussions": -8,
    "lobsters_ai": -8,
    "ddg": -10,
}

_SOURCE_DISPLAY = {
    "the_decoder": "The Decoder",
    "verge_ai": "The Verge",
    "techcrunch_ai": "TechCrunch",
    "marktechpost": "MarkTechPost",
    "arstechnica_ai": "Ars Technica",
    "openai": "OpenAI",
    "deepmind": "Google DeepMind",
    "nvidia_blog": "NVIDIA Blog",
    "microsoft_ai": "Microsoft AI Blog",
    "qbitai": "量子位",
    "simon_willison": "Simon Willison",
    "gary_marcus": "Gary Marcus",
    "interconnects": "Interconnects",
    "raschka": "Sebastian Raschka",
    "huyenchip": "Chip Huyen",
    "lilian_weng": "Lilian Weng",
    "oneusefulthing": "One Useful Thing",
    "karpathy_blog": "Andrej Karpathy",
    "sam_altman": "Sam Altman",
    "darioamodei_blog": "Dario Amodei",
    "latent_space": "Latent Space",
    "import_ai": "Import AI",
    "lesswrong": "LessWrong",
    "thegradient": "The Gradient",
    "aisnakeoil": "AI Snake Oil",
    "eugene_yan": "Eugene Yan",
    "hamel_husain": "Hamel Husain",
    "jay_alammar": "Jay Alammar",
    "semianalysis": "SemiAnalysis",
    "deeplearning_ai_batch": "DeepLearning.AI The Batch",
    "hn_ai_discussions": "Hacker News",
    "lobsters_ai": "Lobsters",
    "arxiv_ai": "arXiv cs.AI",
    "arxiv_cl": "arXiv cs.CL",
    "arxiv_lg": "arXiv cs.LG",
    "bair": "BAIR Blog",
    "36kr_newsflash": "36氪快讯",
    "36kr_ai": "36氪 AI",
    "qwen_blog": "通义千问博客",
    "leiphone": "雷峰网",
    "sspai": "少数派",
    "geekpark": "极客公园",
    "ddg": "Web Search",
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

_OPINION_SOURCE_PEOPLE = {
    "simon_willison": ("analyst", "Simon Willison"),
    "gary_marcus": ("analyst", "Gary Marcus"),
    "interconnects": ("analyst", "Nathan Lambert"),
    "raschka": ("analyst", "Sebastian Raschka"),
    "huyenchip": ("analyst", "Chip Huyen"),
    "lilian_weng": ("analyst", "Lilian Weng"),
    "oneusefulthing": ("analyst", "Ethan Mollick"),
    "karpathy_blog": ("leader", "Andrej Karpathy"),
    "sam_altman": ("leader", "Sam Altman"),
    "darioamodei_blog": ("leader", "Dario Amodei"),
    "eugene_yan": ("community", "Eugene Yan"),
    "hamel_husain": ("community", "Hamel Husain"),
    "jay_alammar": ("community", "Jay Alammar"),
}

_ACADEMIC_FALLBACK_SOURCES = {"arxiv_ai", "arxiv_cl", "arxiv_lg", "bair"}
_CHINESE_FALLBACK_SOURCES = {
    "qbitai",
    "36kr_newsflash",
    "36kr_ai",
    "qwen_blog",
    "leiphone",
    "sspai",
    "geekpark",
}


def _build_user_prompt(
    spec: AgentSpec,
    materials: list[dict[str, Any]],
    excluded: list[str],
    today: str,
    relaxed: bool = False,
    *,
    material_limit: int | None = None,
    excluded_limit: int = 40,
) -> str:
    excluded_block = "\n".join(f"- {t}" for t in excluded[:excluded_limit]) if excluded else "（无）"
    material_lines = []
    actual_material_limit = material_limit or _MATERIAL_LIMITS.get(spec.key, 30)
    for i, m in enumerate(materials[:actual_material_limit], 1):
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


_HTML_RE = re.compile(r"<[^>]+>")


def _clean_material_text(text: str, limit: int = 140) -> str:
    cleaned = _HTML_RE.sub("", text or "")
    cleaned = " ".join(cleaned.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _base_diagnostics(spec_key: str) -> dict[str, Any]:
    return {
        "section": spec_key,
        "search_count": 0,
        "rss_count": 0,
        "materials_count": 0,
        "prompt_materials_count": 0,
        "top_sources": [],
        "excluded_titles_count": 0,
        "strict_raw_count": 0,
        "strict_kept_count": 0,
        "strict_fabricated_url_drops": 0,
        "strict_tier_drops": 0,
        "relaxed_attempted": False,
        "relaxed_raw_count": 0,
        "relaxed_kept_count": 0,
        "relaxed_fabricated_url_drops": 0,
        "relaxed_tier_drops": 0,
        "fallback_count": 0,
        "selection_mode": "empty",
        "empty_reason": "",
    }


def _empty_agent_result(spec: AgentSpec, diagnostics: dict[str, Any], reason: str) -> dict[str, Any]:
    diagnostics["selection_mode"] = "empty"
    diagnostics["empty_reason"] = reason
    return {"key": spec.key, "name": spec.name, "items": [], "diagnostics": diagnostics}


def _material_fallback_items(
    spec_key: str,
    materials: list[dict[str, Any]],
    *,
    keep: int = 4,
) -> list[dict[str, Any]]:
    """Emergency non-LLM fallback for high-signal news sections.

    This is intentionally limited to sections where a raw source card is better
    than silently publishing an empty column. Opinion is restricted to curated
    first-party author feeds so the person/tier fields remain deterministic.
    """
    items: list[dict[str, Any]] = []
    for m in materials:
        title = _clean_material_text(m.get("title") or "", limit=80)
        url = m.get("url") or ""
        if not title or not url:
            continue
        snippet = _clean_material_text(m.get("snippet") or "", limit=120)
        source = m.get("source") or ""

        if spec_key == "opinion":
            person_meta = _OPINION_SOURCE_PEOPLE.get(source)
            if not person_meta and source not in COMMUNITY_VOICE_SOURCES:
                continue
            if person_meta:
                tier, person = person_meta
            else:
                tier, person = "community", COMMUNITY_VOICE_SOURCES[source]
            quote_text = snippet or title
            items.append({
                "tier": tier,
                "title": title,
                "summary": snippet or f"{person} 的近期观点摘要。",
                "value_note": "观点摘要",
                "source_name": _SOURCE_DISPLAY.get(source, person),
                "url": url,
                "published_at": m.get("published_at") or "",
                "importance": "star" if not items else "pin",
                "quote_en": quote_text if tier != "community" else "",
                "quote_zh": quote_text if tier == "community" else "",
                "person": person,
                "category": spec_key,
                "selection_mode": "fallback",
            })
        elif spec_key == "academic":
            if source not in _ACADEMIC_FALLBACK_SOURCES:
                continue
            items.append({
                "title": title,
                "summary": snippet or "近期 AI 论文或评测素材，因 LLM 结构化输出异常按学术来源兜底保留。",
                "value_note": "学术来源素材兜底",
                "source_name": _SOURCE_DISPLAY.get(source, source or "Academic"),
                "url": url,
                "published_at": m.get("published_at") or "",
                "importance": "star" if len(items) < 2 else "pin",
                "category": spec_key,
                "selection_mode": "fallback",
            })
        elif spec_key == "industry":
            items.append({
                "title": title,
                "summary": snippet or "高可信来源的近期 AI 行业动态，因 LLM 结构化输出异常按素材兜底保留。",
                "value_note": "LLM 输出异常时的高可信素材兜底",
                "source_name": _SOURCE_DISPLAY.get(source, source or "Web"),
                "url": url,
                "published_at": m.get("published_at") or "",
                "importance": "star" if len(items) < 2 else "pin",
                "category": spec_key,
                "selection_mode": "fallback",
            })
        elif spec_key == "chinese":
            if source not in _CHINESE_FALLBACK_SOURCES:
                continue
            items.append({
                "title": title,
                "summary": snippet or "近期中文 AI 生态素材，因 LLM 结构化输出异常按中文来源兜底保留。",
                "value_note": "中文来源素材兜底",
                "source_name": _SOURCE_DISPLAY.get(source, source or "中文来源"),
                "url": url,
                "published_at": m.get("published_at") or "",
                "importance": "star" if len(items) < 2 else "pin",
                "category": spec_key,
                "selection_mode": "fallback",
            })
        else:
            return []

        if len(items) >= keep:
            break
    return items


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
    diagnostics = _base_diagnostics(spec.key)
    diagnostics["excluded_titles_count"] = len(excluded_titles)

    search_items: list[dict[str, Any]] = await unified_search(spec.queries, max_results=6, days=2)
    log.info("[agent:%s] search=%d", spec.key, len(search_items))
    diagnostics["search_count"] = len(search_items)

    rss_items = [m for m in rss_pool if m.get("category_hint") in spec.rss_categories] if spec.rss_categories else []
    log.info("[agent:%s] rss=%d", spec.key, len(rss_items))
    diagnostics["rss_count"] = len(rss_items)

    materials = _merge_materials(search_items, rss_items, prefer_rss=spec.key == "opinion")
    diagnostics["materials_count"] = len(materials)

    if not materials:
        log.warning("[agent:%s] no materials (search=%d rss=%d), returning empty",
                    spec.key, len(search_items), len(rss_items))
        return _empty_agent_result(spec, diagnostics, "no_materials")

    log.info("[agent:%s] materials=%d (search=%d, rss=%d), excluded_titles=%d",
             spec.key, len(materials), len(search_items), len(rss_items), len(excluded_titles))

    prompt_materials = _prepare_prompt_materials(spec.key, materials, today)
    top_sources = [m.get("source", "?") for m in prompt_materials[:8]]
    diagnostics["prompt_materials_count"] = len(prompt_materials)
    diagnostics["top_sources"] = top_sources
    log.info(
        "[agent:%s] prompt_materials=%d/%d top_sources=%s",
        spec.key, len(prompt_materials), len(materials), top_sources,
    )

    if not prompt_materials:
        log.warning("[agent:%s] all materials filtered before prompt", spec.key)
        return _empty_agent_result(spec, diagnostics, "materials_filtered_out")

    valid_urls = {m["url"] for m in prompt_materials}

    user_prompt = _build_user_prompt(spec, prompt_materials, excluded_titles, today)
    try:
        result = await chat_json(client, cfg, AGENT_SYSTEM, user_prompt, temperature=0.3, max_tokens=3000)
    except Exception as e:
        log.error("[agent:%s] LLM strict-mode failed: %s", spec.key, e)
        result = {}

    filtered, drops = _validate_items(result, valid_urls, spec.key, mode="strict")
    diagnostics["strict_raw_count"] = drops["raw"]
    diagnostics["strict_fabricated_url_drops"] = drops["fabricated"]
    if spec.key == "opinion":
        before_strict_tier = len(filtered)
        filtered = _validate_opinion_tier(filtered, prompt_materials)
        diagnostics["strict_tier_drops"] = before_strict_tier - len(filtered)
        log.info(
            "[agent:opinion] strict tier-check kept %d/%d (raw=%d, fabricated=%d)",
            len(filtered), before_strict_tier, drops["raw"], drops["fabricated"],
        )
    diagnostics["strict_kept_count"] = len(filtered)
    if filtered:
        diagnostics["selection_mode"] = "strict"

    # Fallback: if LLM returned 0 items but materials existed, retry with relaxed constraints
    if not filtered:
        diagnostics["relaxed_attempted"] = True
        log.warning(
            "[agent:%s] strict mode produced 0 items (raw=%d, fabricated_url_drops=%d) — retrying relaxed",
            spec.key,
            drops["raw"],
            drops["fabricated"],
        )
        compact_materials = prompt_materials[:_RELAXED_MATERIAL_LIMITS.get(spec.key, 12)]
        log.info(
            "[agent:%s] relaxed compact retry with %d/%d prompt materials",
            spec.key,
            len(compact_materials),
            len(prompt_materials),
        )
        relaxed_prompt = _build_user_prompt(
            spec,
            compact_materials,
            excluded_titles,
            today,
            relaxed=True,
            material_limit=len(compact_materials),
            excluded_limit=_RELAXED_EXCLUDED_LIMIT,
        )
        try:
            result2 = await chat_json(client, cfg, AGENT_SYSTEM, relaxed_prompt, temperature=0.5, max_tokens=2000)
        except Exception as e:
            log.error("[agent:%s] relaxed-mode LLM failed: %s", spec.key, e)
            result2 = {}
        compact_urls = {m["url"] for m in compact_materials}
        filtered, drops2 = _validate_items(result2, compact_urls, spec.key, mode="relaxed")
        diagnostics["relaxed_raw_count"] = drops2["raw"]
        diagnostics["relaxed_fabricated_url_drops"] = drops2["fabricated"]
        if spec.key == "opinion":
            before_relaxed_tier = len(filtered)
            filtered = _validate_opinion_tier(filtered, compact_materials)
            diagnostics["relaxed_tier_drops"] = before_relaxed_tier - len(filtered)
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
        diagnostics["relaxed_kept_count"] = len(filtered)
        if filtered:
            diagnostics["selection_mode"] = "relaxed"

    if not filtered:
        fallback = _material_fallback_items(spec.key, prompt_materials)
        diagnostics["fallback_count"] = len(fallback)
        if fallback:
            log.warning("[agent:%s] using deterministic material fallback: %d items", spec.key, len(fallback))
            filtered = fallback
            diagnostics["selection_mode"] = "fallback"

    if not filtered:
        diagnostics["selection_mode"] = "empty"
        diagnostics["empty_reason"] = "llm_and_fallback_empty"

    log.info("[agent:%s] returned %d items", spec.key, len(filtered))
    return {"key": spec.key, "name": spec.name, "items": filtered, "diagnostics": diagnostics}


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
        it["selection_mode"] = mode
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


def _validate_opinion_tier(items: list[dict[str, Any]], materials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enforce opinion taxonomy.

    leader: person ∈ LEADER_PROFILES, name mentioned in materials, Musk gated
            by AI-keyword filter.
    analyst: person ∈ ANALYST_NAMES.
    community: URL comes from a trusted opinion/community source.
    Items failing both buckets are dropped.
    """
    corpus = _materials_corpus(materials)
    source_by_url = {
        m.get("url"): (m.get("source") or "")
        for m in materials
        if m.get("url")
    }
    kept: list[dict[str, Any]] = []
    for it in items:
        tier = (it.get("tier") or "").strip().lower()
        person = (it.get("person") or "").strip()
        url = it.get("url") or ""
        source = source_by_url.get(url, "")
        source_person_meta = _OPINION_SOURCE_PEOPLE.get(source)
        if not person:
            if tier == "community" and source in COMMUNITY_VOICE_SOURCES:
                person = COMMUNITY_VOICE_SOURCES[source]
                it["person"] = person
            else:
                log.debug("[opinion-tier] drop empty person: %s", it.get("title"))
                continue

        if tier == "leader" or person in LEADER_PROFILES:
            profile = LEADER_PROFILES.get(person)
            if not profile:
                log.debug("[opinion-tier] drop leader not in roster: %s", person)
                continue
            if person.lower() not in corpus and source_person_meta != ("leader", person):
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

        if tier == "community" or source in COMMUNITY_VOICE_SOURCES:
            if source not in COMMUNITY_VOICE_SOURCES:
                log.debug("[opinion-tier] drop community item from untrusted source: %s", source)
                continue
            it["tier"] = "community"
            it["person"] = person or COMMUNITY_VOICE_SOURCES[source]
            if not it.get("quote_en") and not it.get("quote_zh") and it.get("summary"):
                it["quote_zh"] = it["summary"]
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
