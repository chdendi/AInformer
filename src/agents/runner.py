from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI

from ..config import LLMConfig
from ..llm.client import chat_json
from ..search.web_search import unified_search
from .definitions import AgentSpec

log = logging.getLogger(__name__)


AGENT_SYSTEM = (
    "你是 AI 行业研究员，从搜索素材中筛选高质量 AI 资讯为中文日报供稿。\n"
    "规则：只输出 JSON。禁止编造。中文撰写，专有名词保留英文。每条给出价值点。"
)

_JSON_FMT_OPINION = """  "items": [
    {
      "title": "中文标题（<40字）",
      "summary": "1-2句摘要（<80字）",
      "value_note": "一句话价值点（<40字）",
      "source_name": "媒体或人物名",
      "url": "原始 URL（必须来自素材）",
      "published_at": "ISO 时间或空字符串",
      "importance": "hot|star|pin",
      "quote_en": "原文引用",
      "quote_zh": "中文翻译",
      "person": "发言人姓名"
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


def _build_user_prompt(
    spec: AgentSpec,
    materials: list[dict[str, Any]],
    excluded: list[str],
    today: str,
    relaxed: bool = False,
) -> str:
    excluded_block = "\n".join(f"- {t}" for t in excluded[:20]) if excluded else "（无）"
    material_lines = []
    for i, m in enumerate(materials[:25], 1):
        pub = (m.get("published_at") or "")[:10]
        material_lines.append(
            f"[{i}] {m.get('title','')} | {m.get('url','')} | {m.get('source','?')} | {pub}\n"
            f"  {(m.get('snippet') or '')[:150]}"
        )
    materials_block = "\n\n".join(material_lines) if material_lines else "（无素材）"

    json_fmt = _JSON_FMT_OPINION if spec.key == "opinion" else _JSON_FMT_BASE

    if relaxed:
        requirements = (
            "- 宽松模式：目标输出 2-3 条，降低质量门槛，但仍需有实质内容（非纯营销稿）。\n"
            "- importance：star 或 pin 即可。\n"
            "- url 必须来自候选素材，禁止编造。\n"
            '- 如果候选素材中确实没有符合该栏目的内容，返回 {"items": []}'
        )
    else:
        requirements = (
            "- 输出 4-5 条最高质量资讯，宁缺毋滥。\n"
            "- importance：最多 1 hot，1-2 star，其余 pin。\n"
            "- url 必须来自候选素材，禁止编造。\n"
            "- 跳过与\"近期已报道标题\"相似的内容。"
        )

    mode_note = "（宽松兜底模式）" if relaxed else ""

    return f"""
今日日期：{today}
栏目：{spec.name}{mode_note}
焦点：{spec.focus}
要求：{spec.extra_instructions}

## 近期已报道过的标题（跳过）：
{excluded_block}

## 候选素材（{len(materials)} 条，已初筛）：
{materials_block}

## JSON 输出格式：
{{
{json_fmt}
}}

要求：
{requirements}
""".strip()


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

    seen: set[str] = set()
    materials: list[dict[str, Any]] = []
    for src in (search_items, rss_items):
        for m in src:
            url = m.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            materials.append(m)

    if not materials:
        log.warning("[agent:%s] no materials, returning empty", spec.key)
        return {"key": spec.key, "name": spec.name, "items": []}

    user_prompt = _build_user_prompt(spec, materials, excluded_titles, today)
    try:
        result = await chat_json(client, cfg, AGENT_SYSTEM, user_prompt, temperature=0.3, max_tokens=3000)
    except Exception as e:
        log.error("[agent:%s] LLM failed: %s", spec.key, e)
        return {"key": spec.key, "name": spec.name, "items": []}

    items = result.get("items", []) if isinstance(result, dict) else []

    valid_urls = {m["url"] for m in materials}
    filtered = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("url") and it["url"] not in valid_urls:
            log.debug("[agent:%s] dropping fabricated url: %s", spec.key, it["url"])
            continue
        it.setdefault("category", spec.key)
        it["category"] = spec.key
        filtered.append(it)

    # Fallback: if LLM returned 0 items but materials existed, retry with relaxed constraints
    if not filtered and materials:
        log.warning("[agent:%s] LLM returned 0 items, retrying with relaxed constraints", spec.key)
        relaxed_prompt = _build_user_prompt(spec, materials, excluded_titles, today, relaxed=True)
        try:
            result2 = await chat_json(client, cfg, AGENT_SYSTEM, relaxed_prompt, temperature=0.5, max_tokens=2000)
            items2 = result2.get("items", []) if isinstance(result2, dict) else []
            for it in items2:
                if not isinstance(it, dict):
                    continue
                if it.get("url") and it["url"] not in valid_urls:
                    log.debug("[agent:%s] fallback dropping fabricated url: %s", spec.key, it["url"])
                    continue
                it.setdefault("category", spec.key)
                it["category"] = spec.key
                filtered.append(it)
            log.info("[agent:%s] fallback returned %d items", spec.key, len(filtered))
        except Exception as e:
            log.error("[agent:%s] fallback LLM failed: %s", spec.key, e)

    log.info("[agent:%s] returned %d items", spec.key, len(filtered))
    return {"key": spec.key, "name": spec.name, "items": filtered}


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
