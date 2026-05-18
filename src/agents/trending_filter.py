"""LLM-driven AI-relevance filter for GitHub Trending.

Takes the raw daily Top-N from `src.search.github_trending.fetch_trending`
and asks the model to keep the most AI/ML-related ones, plus write a Chinese
summary and one-line value note for each. Designed for a single LLM call to
keep token cost bounded — input is small (~20 short repo descriptions).
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from ..config import LLMConfig
from ..llm.client import chat_json

log = logging.getLogger(__name__)


TRENDING_SYSTEM = (
    "你是一份中文 AI 日报的栏目编辑，专门挑选 GitHub Trending 中真正与 AI/ML 相关的开源项目。\n"
    "严格只输出 JSON。所有输出用中文，专有名词、库名、命令保留英文原文。"
)


def _format_repos_for_prompt(repos: list[dict[str, Any]]) -> str:
    lines = []
    for i, r in enumerate(repos, 1):
        desc = (r.get("description") or "").strip() or "（无描述）"
        lang = r.get("language") or "—"
        stars_today = r.get("stars_today") or 0
        lines.append(
            f"{i}. {r.get('full_name')} · {lang} · 今日 +{stars_today} ⭐\n"
            f"   {desc}"
        )
    return "\n".join(lines)


async def filter_trending_with_llm(
    client: AsyncOpenAI,
    cfg: LLMConfig,
    repos: list[dict[str, Any]],
    keep: int = 4,
) -> list[dict[str, Any]]:
    """Pick the top `keep` AI-relevant repos and add Chinese editorial fields.

    Returns enriched dicts that retain the original scrape fields (`owner`,
    `repo`, `url`, `description`, `language`, `stars_today`, `stars_total`)
    plus `summary_zh` (中文项目摘要), `value_note` (中文一句话点评) and
    `ai_topic`（AI 相关子领域标签）.

    If the LLM call fails or returns no valid items, falls back to the top
    repos by daily stars with a generic Chinese summary.
    """
    if not repos:
        log.warning("[trending] empty repo list — nothing to filter")
        return []

    log.info(
        "[trending] LLM filter input: %d repos, sample=%s",
        len(repos),
        [r.get("full_name") for r in repos[:5]],
    )
    by_full_name = {r["full_name"]: r for r in repos}

    user = f"""
今日 GitHub Trending（daily, all language）共 {len(repos)} 个仓库：

{_format_repos_for_prompt(repos)}

任务：从中挑选 {keep} 个与 AI / 机器学习 / LLM / agent / Coding agent / 多模态 / RAG / 推理框架
**或 AI 工程基础设施** 相关的项目。

判定标准（满足任一即视为 AI 相关）：
- 模型层：LLM 训练/微调框架、推理引擎、模型权重、多模态模型、扩散模型
- 应用层：Agent 框架、RAG、prompt 管理、AI workflow 编排、AI 应用脚手架、向量数据库
- 评测层：AI benchmark / eval 工具
- **Coding Agent 生态**：Claude Code / Cursor / Aider / Continue 的 skills / rules / extensions / 配置集
- **MCP 生态**：MCP server、MCP client、MCP 工具集
- **Agent 周边基础设施**：agent memory、agent observability、agent testing、tool-use 框架
- **AI 数字人 / 具身**：digital human、AI avatar、robotics policy、VLA 模型
- 描述里出现 LLM/agent/MCP/Claude/GPT/Gemini/RAG/embedding 等关键词且名副其实的项目

反例（这些不算 AI 相关，请排除）：
- 通用包管理器（uv、pnpm 等）、通用编译器、纯 CLI 工具（除非是 AI agent 的 CLI）
- 数据库、缓存、消息队列等纯基础设施（不为 AI 服务的）
- 游戏、桌面美化、纯前端组件库
- 仅在 README 写"AI-powered"营销词但实质是 SaaS 落地页

**判断倾向**：宁可错收一个边界项目，不要漏掉一个真正的 AI 工程项目。
描述里有 "skills" / "agent" / "claude" / "MCP" / "LLM" 等关键词，且仓库结构看起来不是营销页，
就应该判为 AI 相关。

输出 JSON：
{{
  "items": [
    {{
      "full_name": "owner/repo",
      "ai_topic": "子领域标签（如 LLM 推理 / Agent 框架 / 多模态 / RAG / 训练 / Eval / MCP 工具）",
      "summary_zh": "中文摘要（35-70 字）：把英文简介改写成自然中文，说明项目做什么",
      "value_note": "一句话中文点评（25-50 字）：它解决了什么、为什么值得开发者关注"
    }}
  ]
}}

要求：
- items 数量正好 {keep}（如果 AI 相关项目不足 {keep}，按实际数量返回，可少不可多凑）。
- full_name 必须严格来自上面列表中的项目。
- summary_zh 必须是中文摘要，不要逐词硬翻；库名、协议名、模型名、命令可保留英文。
- value_note 不要复述描述原文，要给"读者价值"判断。
""".strip()

    try:
        resp = await chat_json(
            client,
            cfg,
            TRENDING_SYSTEM,
            user,
            temperature=0.3,
            max_tokens=700,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("[trending] LLM filter failed (%s) — falling back to top stars_today", exc)
        return _fallback_top_stars(repos, keep)

    picked = resp.get("items") or []
    log.info("[trending] LLM returned %d picks: %s", len(picked), [
        (it.get("full_name") if isinstance(it, dict) else str(it)) for it in picked
    ])

    enriched: list[dict[str, Any]] = []
    dropped_unknown: list[str] = []
    for it in picked:
        full_name = (it.get("full_name") or "").strip()
        base = by_full_name.get(full_name)
        if not base:
            dropped_unknown.append(full_name)
            continue
        enriched.append(
            {
                **base,
                "ai_topic": (it.get("ai_topic") or "").strip(),
                "summary_zh": (
                    it.get("summary_zh") or _fallback_summary_zh(base)
                ).strip(),
                "value_note": (it.get("value_note") or "").strip(),
            }
        )
    if dropped_unknown:
        log.warning(
            "[trending] dropped %d picks with unknown full_name: %s",
            len(dropped_unknown),
            dropped_unknown,
        )

    if not enriched:
        log.warning("[trending] LLM returned 0 valid picks — falling back to top stars_today")
        return _fallback_top_stars(repos, keep)

    log.info("[trending] LLM filter kept %d / %d", len(enriched), len(repos))
    return enriched[:keep]


def _fallback_top_stars(repos: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """Sort repos by stars_today desc and return top `keep` with fallback notes.

    Used when the LLM call errors out or judges every repo non-AI. The intent
    is "never silently empty trending" — empty value_note / ai_topic are
    conditionally rendered, so cards still display cleanly with just the repo
    line, Chinese fallback summary and stars.
    """
    ranked = sorted(repos, key=lambda r: r.get("stars_today") or 0, reverse=True)
    fallback = [
        {
            **r,
            "ai_topic": "",
            "summary_zh": _fallback_summary_zh(r),
            "value_note": "",
        }
        for r in ranked[:keep]
    ]
    log.info(
        "[trending] fallback kept %d / %d by stars_today: %s",
        len(fallback),
        len(repos),
        [(r["full_name"], r.get("stars_today")) for r in fallback],
    )
    return fallback


def _fallback_summary_zh(repo: dict[str, Any]) -> str:
    """Best-effort Chinese summary when the LLM does not provide one."""
    full_name = repo.get("full_name") or repo.get("repo") or "该项目"
    language = repo.get("language") or ""
    if language:
        return f"{full_name} 是一个 {language} 项目，今日在 GitHub Trending 获得较高关注。"
    return f"{full_name} 今日在 GitHub Trending 获得较高关注，建议结合仓库说明进一步评估。"
