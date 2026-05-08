from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI

from ..config import LLMConfig
from ..llm.client import chat_json
from ..search.tavily import batch_search
from .definitions import AgentSpec

log = logging.getLogger(__name__)


AGENT_SYSTEM = (
    "你是一个资深的 AI 行业研究员。你的任务是从给定的搜索素材中筛选、提炼、结构化最值得关注的 AI 资讯，"
    "为一份高质量中文 AI 日报供稿。\n\n"
    "严格要求：\n"
    "1. 只输出 JSON，禁止任何 markdown 包裹。\n"
    "2. 不允许编造素材中不存在的链接、引用、人物或事件。\n"
    "3. 所有内容用中文撰写，专有名词、产品名、人名保留英文原文。\n"
    "4. 凡引用 AI 领袖原话，必须保留英文原文，并附中文翻译。\n"
    "5. 对每条资讯都要给出价值点（为什么读者应该看）。\n"
)


def _build_user_prompt(
    spec: AgentSpec,
    materials: list[dict[str, Any]],
    excluded: list[str],
    today: str,
) -> str:
    excluded_block = "\n".join(f"- {t}" for t in excluded[:30]) if excluded else "（无）"
    material_lines = []
    for i, m in enumerate(materials[:40], 1):
        material_lines.append(
            f"[{i}] 来源={m.get('source','?')} | 时间={m.get('published_at','?')}\n"
            f"    标题：{m.get('title','')}\n"
            f"    链接：{m.get('url','')}\n"
            f"    摘要：{(m.get('snippet') or '')[:200]}"
        )
    materials_block = "\n\n".join(material_lines) if material_lines else "（无素材）"

    return f"""
今日日期：{today}
你负责的栏目：**{spec.name}**
栏目焦点：{spec.focus}
额外要求：{spec.extra_instructions}

## 已在最近 7 天日报中出现过的标题（务必跳过这些主题，避免重复）
{excluded_block}

## 候选搜索素材（共 {len(materials)} 条，已按热度/时效初筛）
{materials_block}

## 输出格式（严格 JSON）
{{
  "items": [
    {{
      "title": "中文标题（< 40 字）",
      "summary": "1-2 句中文摘要，突出核心信息",
      "value_note": "1 句话价值点：为什么读者要关心",
      "source_name": "媒体或人物名",
      "url": "原始 URL（必须来自素材）",
      "published_at": "ISO 时间或空字符串",
      "importance": "hot | star | pin",
      "quote_en": "（仅 opinion 栏目；无则空字符串）",
      "quote_zh": "（仅 opinion 栏目；无则空字符串）",
      "person": "（仅 opinion 栏目；无则空字符串）"
    }}
  ]
}}

要求：
- 输出 5-6 条最高质量的资讯（栏目最终只展示 4 张卡片，多 1-2 条作为去重 buffer）。质量优先于数量，宁缺毋滥。
- importance：本栏目最多 1 条 "hot"（重大里程碑），1-2 条 "star"（重要更新），其余 "pin"。
- 不要输出与"已在最近 7 天日报中出现过的标题"相似的内容。
- url 必须从候选素材中选取，不要编造。
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

    tavily_items: list[dict[str, Any]] = await batch_search(spec.queries, max_results=6, days=2)
    log.info("[agent:%s] tavily=%d", spec.key, len(tavily_items))

    rss_items = [m for m in rss_pool if m.get("category_hint") in spec.rss_categories] if spec.rss_categories else []
    log.info("[agent:%s] rss=%d", spec.key, len(rss_items))

    seen: set[str] = set()
    materials: list[dict[str, Any]] = []
    for src in (tavily_items, rss_items):
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
        result = await chat_json(client, cfg, AGENT_SYSTEM, user_prompt, temperature=0.3, max_tokens=1800)
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
