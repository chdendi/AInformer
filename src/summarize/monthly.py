from __future__ import annotations

import asyncio
import logging
from calendar import monthrange
from datetime import date, datetime
from typing import Any

from openai import AsyncOpenAI

from ..config import LLMConfig, tz
from ..llm.client import chat_json
from ..render.daily import WEEKDAY_ZH
from ..render.summary_pages import write_monthly
from .aggregator import flatten_items, load_daily_in_range, top_sources

log = logging.getLogger(__name__)


SYSTEM = (
    "你是一份高质量中文 AI 月报的总编辑。基于过去一个月每天的日报数据，"
    "提炼出本月的主线脉络、关键里程碑、重要观点，撰写一份结构化月报。\n"
    "严格要求：只输出 JSON。所有内容用中文。不允许编造素材外的事件或链接。"
)


def _format_items_for_prompt(items: list[dict[str, Any]], limit_per_cat: int = 30) -> str:
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        cat = it.get("category", "other")
        by_cat.setdefault(cat, []).append(it)

    blocks = []
    for cat, lst in by_cat.items():
        lst_sorted = sorted(
            lst,
            key=lambda x: (0 if x.get("importance") == "hot" else (1 if x.get("importance") == "star" else 2), x.get("report_date", "")),
        )[:limit_per_cat]
        lines = [f"### {cat} ({len(lst)} 条，节选 {len(lst_sorted)})"]
        for it in lst_sorted:
            lines.append(
                f"- [{it.get('report_date','')}] [{it.get('importance','pin')}] "
                f"{it.get('title','')} | {it.get('source_name','')} | {it.get('url','')}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


async def synthesize_monthly(
    client: AsyncOpenAI,
    cfg: LLMConfig,
    year: int,
    month: int,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    user = f"""
月份：{year}-{month:02d}
本月共有 {len(items)} 条日报资讯。

数据明细：

{_format_items_for_prompt(items)}

请输出如下 JSON：
{{
  "tagline": "一句话本月主题（< 30 字）",
  "overview": "本月综述：用一段连贯的中文（260-400 字），梳理本月 AI 行业的主线、节奏、变化",
  "themes": [
    {{
      "title": "主题名（如：开源大模型加速追赶闭源）",
      "summary": "2-4 句详细描述",
      "references": [{{"label": "事件简述", "url": "URL（必须来自素材）"}}]
    }}
  ],
  "milestones": [
    {{
      "title": "里程碑标题",
      "summary": "1-2 句描述",
      "date": "YYYY-MM-DD（如可考据）",
      "url": "URL（来自素材）"
    }}
  ]
}}

要求：
- themes 输出 4-6 个主题，每个主题给 2-5 个 references。
- milestones 输出 8-12 条本月最有标志意义的事件，按时间顺序。
- 引用 URL 必须来自素材，不要编造。
""".strip()
    return await chat_json(client, cfg, SYSTEM, user, temperature=0.4, max_tokens=4000)


async def build_monthly(year: int, month: int, cfg: LLMConfig) -> dict[str, Any]:
    daily = load_daily_in_range(year, month)
    items = flatten_items(daily)
    log.info("Monthly %s-%02d: %d daily reports, %d items", year, month, len(daily), len(items))

    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    if items:
        synth = await synthesize_monthly(client, cfg, year, month, items)
    else:
        synth = {"tagline": "本月无可用日报数据", "overview": "暂无内容。", "themes": [], "milestones": []}

    daily_links = []
    for d in daily:
        date_str = d.get("date", "")
        try:
            wd = WEEKDAY_ZH[date.fromisoformat(date_str).weekday()]
        except Exception:
            wd = ""
        sections = d.get("sections") or {}
        total = sum(len(v) for v in sections.values())
        hot = sum(1 for v in sections.values() for it in v if it.get("importance") == "hot")
        daily_links.append({
            "date": date_str,
            "weekday_zh": wd,
            "slug": date_str.replace("-", ""),
            "theme": d.get("today_theme", ""),
            "total": total,
            "hot_count": hot,
        })

    summary = {
        "year": year,
        "month": month,
        "label": f"{year}-{month:02d}",
        "slug": f"{year}{month:02d}",
        "title": f"{year} 年 {month} 月 AI 月度回顾",
        "tagline": synth.get("tagline", ""),
        "overview": synth.get("overview", ""),
        "themes": synth.get("themes", []),
        "milestones": synth.get("milestones", []),
        "daily_links": daily_links,
        "day_count": len(daily),
        "total_items": len(items),
        "top_sources": top_sources(items, 10),
        "generated_at": datetime.now(tz()).isoformat(),
    }
    return summary


async def run_monthly(year: int, month: int, cfg: LLMConfig) -> None:
    summary = await build_monthly(year, month, cfg)
    out = write_monthly(summary)
    log.info("Wrote %s", out)
