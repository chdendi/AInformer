"""LLM-driven AI-relevance filter for GitHub Trending.

Takes the raw daily Top-N from `src.search.github_trending.fetch_trending`
and asks the model to keep the most AI/ML-related ones, plus write a Chinese
summary and one-line value note for each. Designed for a single LLM call to
keep token cost bounded — input is small (~20 short repo descriptions).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from openai import AsyncOpenAI

from ..config import LLMConfig
from ..llm.client import chat_json

log = logging.getLogger(__name__)


GENERIC_TRENDING_SUMMARY_RE = re.compile(
    r"是一个\s*[^，。]{1,40}\s*项目，今日在 GitHub Trending 获得较高关注"
    r"|今日在 GitHub Trending 获得较高关注"
)

AI_TOPIC_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("MCP 工具", ("mcp", "model context protocol")),
    ("Agent 浏览器基础设施", ("playwright", "chromium", "browser automation", "bot detection", "fingerprint")),
    ("Coding Agent 生态", (
        "claude code",
        "codex",
        "cursor",
        "copilot",
        "aider",
        "opencode",
        "antigravity",
        "coding agent",
        "coding agents",
        "agent skills",
        "skill registry",
    )),
    ("Agent 框架", ("agent", "agents", "workflow", "tool-use", "tool use")),
    ("个人 AI 助手", ("personal ai", "ai assistant", "super intelligence", "digital human", "avatar")),
    ("RAG / 知识库", ("rag", "retrieval", "embedding", "vector", "knowledge graph")),
    ("LLM 推理", ("llm", "inference", "serving", "vllm", "ollama", "transformer")),
    ("模型训练", ("training", "fine-tuning", "finetuning", "lora", "dataset")),
    ("多模态", ("multimodal", "vision-language", "vla", "tts", "speech", "voice")),
    ("AI 工程基础设施", (" ai ", "ai-powered", "artificial intelligence", "machine learning", "deep learning")),
]

VALUE_NOTE_BY_TOPIC = {
    "MCP 工具": "降低模型接入外部工具的成本，适合关注 Agent 工具生态的开发者。",
    "Agent 浏览器基础设施": "可提升网页自动化和 Agent 浏览器操作稳定性，适合测试与反检测场景。",
    "Coding Agent 生态": "把可复用技能沉淀为工具规范，能降低团队使用编码 Agent 的试错成本。",
    "Agent 框架": "面向复杂任务编排和工具调用，适合评估 Agent 工程化落地能力。",
    "个人 AI 助手": "个人助手方向仍在快速试错，隐私、易用性和本地能力值得重点观察。",
    "RAG / 知识库": "围绕知识检索和上下文压缩，直接影响企业 AI 应用的可用性与成本。",
    "LLM 推理": "推理效率和部署体验是模型落地关键，适合关注性能与成本优化。",
    "模型训练": "训练与微调工具会影响模型迭代效率，适合关注开源模型工程实践。",
    "多模态": "多模态能力正从演示走向工具化，适合观察端侧和交互场景。",
    "AI 工程基础设施": "属于 AI 应用落地的基础组件，值得结合 README 进一步评估成熟度。",
}


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

    If the LLM call fails or returns no valid items, falls back to AI-keyword
    ranked repos with summaries derived from each repository description.
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
        ai_topic = _summary_or_fallback_topic(base, it.get("ai_topic"))
        enriched.append(
            {
                **base,
                "ai_topic": ai_topic,
                "summary_zh": _summary_or_fallback(base, it.get("summary_zh")),
                "value_note": _summary_or_fallback_value_note(
                    base,
                    it.get("value_note"),
                    ai_topic,
                ),
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
    """Return fallback repos with description-based editorial fields.

    Used when the LLM call errors out or judges every repo non-AI. The intent
    is "never silently empty trending", while still avoiding language-only
    placeholder summaries.
    """
    ranked = _rank_fallback_repos(repos)
    fallback = [
        {
            **r,
            "ai_topic": _fallback_ai_topic(r),
            "summary_zh": _fallback_summary_zh(r),
            "value_note": _fallback_value_note(r),
        }
        for r in ranked[:keep]
    ]
    log.info(
        "[trending] fallback kept %d / %d by AI relevance and stars_today: %s",
        len(fallback),
        len(repos),
        [(r["full_name"], r.get("stars_today")) for r in fallback],
    )
    return fallback


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate_sentence(text: str, limit: int = 86) -> str:
    text = _clean_text(text).strip("，,。.；;：: ")
    if len(text) <= limit:
        return text + ("。" if text and text[-1] not in "。.!！?" else "")
    return text[: limit - 1].rstrip("，,。.；;：: ") + "。"


def _repo_text(repo: dict[str, Any]) -> str:
    return _clean_text(
        " ".join(
            str(repo.get(k) or "")
            for k in ("full_name", "repo", "description", "language")
        )
    ).lower()


def _fallback_ai_topic(repo: dict[str, Any]) -> str:
    text = f" {_repo_text(repo)} "
    for topic, keywords in AI_TOPIC_RULES:
        if any(keyword in text for keyword in keywords):
            return topic
    return ""


def _ai_relevance_score(repo: dict[str, Any]) -> int:
    text = f" {_repo_text(repo)} "
    score = 0
    for idx, (_, keywords) in enumerate(AI_TOPIC_RULES):
        if any(keyword in text for keyword in keywords):
            score += max(1, len(AI_TOPIC_RULES) - idx)
    return score


def _rank_fallback_repos(repos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [r for r in repos if _ai_relevance_score(r) > 0]
    pool = scored if scored else repos
    return sorted(
        pool,
        key=lambda r: (_ai_relevance_score(r), r.get("stars_today") or 0),
        reverse=True,
    )


def _is_generic_summary(text: str) -> bool:
    cleaned = _clean_text(text)
    return not cleaned or bool(GENERIC_TRENDING_SUMMARY_RE.search(cleaned))


def _summary_or_fallback(repo: dict[str, Any], summary: Any) -> str:
    cleaned = _clean_text(summary)
    if _is_generic_summary(cleaned):
        return _fallback_summary_zh(repo)
    return cleaned


def _summary_or_fallback_topic(repo: dict[str, Any], topic: Any) -> str:
    return _clean_text(topic) or _fallback_ai_topic(repo)


def _summary_or_fallback_value_note(
    repo: dict[str, Any],
    value_note: Any,
    topic: Any = None,
) -> str:
    cleaned = _clean_text(value_note)
    return cleaned or _fallback_value_note(repo, _clean_text(topic))


def _fallback_summary_zh(repo: dict[str, Any]) -> str:
    """Best-effort Chinese summary when the LLM does not provide one.

    The fallback must still describe what the repository does. A language-only
    sentence is not a summary and reads poorly in the published daily report.
    """
    full_name = repo.get("full_name") or repo.get("repo") or "该项目"
    desc = _clean_text(repo.get("description"))
    lower = desc.lower()

    if desc:
        if "personal ai" in lower and "super intelligence" in lower:
            return "面向个人使用的 AI 助手项目，强调隐私、简单和高能力，定位为个人智能工作入口。"
        if "academic research skills" in lower and "claude code" in lower:
            return "为 Claude Code 提供学术研究工作流技能，覆盖资料研究、写作、审稿、修改到定稿流程。"
        if "skill registry" in lower and ("coding agent" in lower or "coding agents" in lower):
            return "面向专业 AI 编码 Agent 的安全技能注册表，可扩展 Antigravity、Claude Code、Cursor、Copilot 等工具。"
        if "stealth chromium" in lower and "playwright" in lower:
            return "带源码级指纹补丁的隐身 Chromium，可作为 Playwright 替代方案通过自动化检测。"
        if "code knowledge graph" in lower:
            return "为 AI 编码工具预索引本地代码知识图谱，减少 token 消耗和工具调用次数。"
        if "agent skills" in lower or ("skills" in lower and "agent" in lower):
            return "提供可复用的 Agent Skills 集合，帮助把专业任务流程封装成 AI Agent 可调用能力。"
        if "mcp" in lower:
            return "围绕 MCP 生态提供工具或服务，帮助大模型更方便地接入外部系统与工作流。"
        if re.search(r"rag|retrieval|embedding|vector", lower):
            return "面向 RAG、检索或向量化场景的工具项目，用于增强大模型访问外部知识的能力。"
        if re.search(r"llm|inference|serving|transformer", lower):
            return "面向大模型开发或推理部署的工具项目，重点改善模型运行效率与工程体验。"
        if re.search(r"tts|speech|voice", lower):
            return "面向语音合成或语音交互的 AI 工具项目，适合关注多模态应用落地。"
        return _truncate_sentence(f"仓库简介显示，{full_name} 主要提供：{desc}", limit=96)

    topic = _fallback_ai_topic(repo)
    if topic:
        return f"{full_name} 属于{topic}方向，但仓库未提供简介，需要打开 README 进一步确认具体能力。"
    return f"{full_name} 仓库未提供简介，建议结合 README 和代码结构进一步评估具体用途。"


def _fallback_value_note(repo: dict[str, Any], topic: str = "") -> str:
    topic = topic or _fallback_ai_topic(repo)
    if topic:
        note = VALUE_NOTE_BY_TOPIC.get(topic)
        if note:
            return note
    fallback_topic = _fallback_ai_topic(repo)
    if fallback_topic and fallback_topic != topic:
        note = VALUE_NOTE_BY_TOPIC.get(fallback_topic)
        if note:
            return note
    if _clean_text(repo.get("description")):
        return "仓库简介提供了明确使用场景，值得结合 README 进一步判断成熟度。"
    return ""
