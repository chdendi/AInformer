from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from .config import LLMConfig
from .llm.client import chat_json

log = logging.getLogger(__name__)


SYNTH_SYSTEM = (
    "你是一份高质量中文 AI 日报的总编辑。基于五个栏目编辑提交的稿件，"
    "撰写日报的开篇综述、头条精选。\n"
    "严格要求：只输出 JSON，所有内容用中文，专有名词保留英文。不允许编造素材外的事实或链接。"
)


def _format_sections_for_prompt(sections: dict[str, list[dict[str, Any]]]) -> str:
    blocks = []
    section_zh = {
        "tutorial": "AI 使用姿势与教程",
        "industry": "AI 行业新闻与产品动态",
        "opinion": "AI 领袖发言与深度观点",
        "chinese": "中文 AI 生态与学术动态",
        "academic": "AI 学术与评测",
    }
    for key, items in sections.items():
        lines = [f"### {section_zh.get(key, key)}"]
        for i, it in enumerate(items, 1):
            summary = (it.get("summary") or "")[:80]
            lines.append(
                f"{i}. [{it.get('importance','pin')}] {it.get('title','')} | {it.get('source_name','')} | {it.get('url','')}\n"
                f"   摘要：{summary}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


async def synthesize_overview(
    client: AsyncOpenAI,
    cfg: LLMConfig,
    today: str,
    sections: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    user = f"""
日期：{today}

五栏目稿件如下：

{_format_sections_for_prompt(sections)}

请输出如下 JSON：
{{
  "lede": "首部综述：用一段连贯的中文（150-220 字），总结今日最轰动的 3-5 条新闻。要点之间自然过渡，不要罗列编号。",
  "today_theme": "一句话（< 30 字）今日主题",
  "headlines": [
    {{
      "title": "头条标题（< 35 字）",
      "summary": "2-3 句详细摘要",
      "url": "对应原文 URL（必须来自素材）",
      "source_name": "来源",
       "category": "tutorial|industry|opinion|chinese|academic"
    }}
  ]
}}

要求：
- headlines 选 3-5 条（覆盖不同栏目最好），且必须出现在五栏目稿件中。
- 不要编造素材外的事实。
""".strip()
    return await chat_json(client, cfg, SYNTH_SYSTEM, user, temperature=0.4, max_tokens=2000)
