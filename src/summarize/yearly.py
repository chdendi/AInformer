from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI

from ..config import LLMConfig, tz
from ..llm.client import chat_json
from ..render.summary_pages import write_yearly
from .aggregator import flatten_items, load_daily_in_range, load_monthly_in_year, top_sources

log = logging.getLogger(__name__)


SYSTEM = (
    "你是一份高质量中文 AI 年报的总编辑。基于过去一年的日报与月报数据，"
    "提炼全年主线脉络与十大事件，撰写一份结构化年报。\n"
    "严格要求：只输出 JSON。所有内容用中文。不允许编造素材外的事件或链接。"
)


def _format_for_prompt(items: list[dict[str, Any]], monthly: list[dict[str, Any]]) -> str:
    monthly_block = []
    for m in monthly:
        monthly_block.append(
            f"### {m.get('label')}\n"
            f"主题: {m.get('tagline','')}\n"
            f"综述: {m.get('overview','')[:400]}\n"
            f"里程碑数: {len(m.get('milestones', []))}"
        )
    monthly_text = "\n\n".join(monthly_block) if monthly_block else "（无月报）"

    headline_items = sorted(
        [it for it in items if it.get("importance") in ("hot", "star")],
        key=lambda x: x.get("report_date", ""),
    )[:200]
    items_block = "\n".join(
        f"- [{it.get('report_date','')}] [{it.get('importance','pin')}] {it.get('title','')} | {it.get('source_name','')} | {it.get('url','')}"
        for it in headline_items
    )

    return f"## 月报概览\n{monthly_text}\n\n## 全年高优资讯（节选 {len(headline_items)} 条）\n{items_block}"


async def synthesize_yearly(
    client: AsyncOpenAI,
    cfg: LLMConfig,
    year: int,
    items: list[dict[str, Any]],
    monthly: list[dict[str, Any]],
) -> dict[str, Any]:
    user = f"""
年份：{year}
全年共有 {len(items)} 条日报资讯，{len(monthly)} 个月报。

数据：

{_format_for_prompt(items, monthly)}

请输出如下 JSON：
{{
  "tagline": "一句话本年主题（< 30 字）",
  "overview": "年度综述：用一段连贯的中文（350-550 字），梳理全年 AI 行业的主线、转折、关键变化",
  "themes": [
    {{"title": "主线主题", "summary": "3-5 句深度描述", "references": [{{"label":"事件简述","url":"URL"}}]}}
  ],
  "milestones": [
    {{"title":"事件标题","summary":"1-2 句描述","date":"YYYY-MM-DD","url":"URL"}}
  ]
}}

要求：
- themes 输出 5-8 条全年主线（如：开源闭源拉锯、Agent 能力突破、训练范式演进、成本与算力变化、监管与版权…）。
- milestones 严格输出年度十大事件（10 条），按时间顺序。
- 引用 URL 必须来自素材。
""".strip()
    return await chat_json(client, cfg, SYSTEM, user, temperature=0.4, max_tokens=5000)


async def build_yearly(year: int, cfg: LLMConfig) -> dict[str, Any]:
    daily = load_daily_in_range(year, None)
    items = flatten_items(daily)
    monthly = load_monthly_in_year(year)
    log.info("Yearly %d: %d daily reports, %d items, %d monthly", year, len(daily), len(items), len(monthly))

    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    if items or monthly:
        synth = await synthesize_yearly(client, cfg, year, items, monthly)
    else:
        synth = {"tagline": "本年无可用日报数据", "overview": "暂无内容。", "themes": [], "milestones": []}

    monthly_links = [
        {
            "label": m.get("label", ""),
            "slug": m.get("slug", ""),
            "tagline": m.get("tagline", ""),
            "total_items": m.get("total_items", 0),
        }
        for m in monthly
    ]

    summary = {
        "year": year,
        "label": str(year),
        "slug": str(year),
        "title": f"{year} 年度 AI 回顾",
        "tagline": synth.get("tagline", ""),
        "overview": synth.get("overview", ""),
        "themes": synth.get("themes", []),
        "milestones": synth.get("milestones", []),
        "monthly_links": monthly_links,
        "month_count": len(monthly),
        "total_items": len(items),
        "top_sources": top_sources(items, 15),
        "generated_at": datetime.now(tz()).isoformat(),
    }
    return summary


async def run_yearly(year: int, cfg: LLMConfig) -> None:
    summary = await build_yearly(year, cfg)
    out = write_yearly(summary)
    log.info("Wrote %s", out)
