from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSpec:
    key: str
    name: str
    focus: str
    queries: list[str]
    rss_categories: list[str]
    extra_instructions: str


def _queries_with_month(base: list[str], month_token: str) -> list[str]:
    return [f"{q} {month_token}" for q in base]


def _queries_mixed_month(base: list[str], month_token: str, skip_month: set[int] | None = None) -> list[str]:
    """Append month_token to queries, except those whose index is in skip_month.

    Useful for benchmark-style queries where adding a month makes the query miss
    the canonical leaderboard pages (which use 'latest' rather than dated URLs).
    """
    skip = skip_month or set()
    return [q if i in skip else f"{q} {month_token}" for i, q in enumerate(base)]


def build_agent_specs(month_token: str) -> list[AgentSpec]:
    """month_token like '2026-05' for query freshness hints."""

    tutorial = AgentSpec(
        key="tutorial",
        name="AI 使用姿势与教程",
        focus="开发者使用 AI 工具的最新技巧、配置、工作流、教程、prompt 模板。",
        queries=_queries_with_month(
            [
                "Claude Code tips workflow tutorial",
                "Cursor AI best practices",
                "GitHub Copilot new features",
                "prompt engineering techniques",
                "Codex CLI tutorial",
                "v0 dev OR bolt.new OR Replit Agent tutorial",
                "Devin OR Augment Code OR Cline review",
                "AI coding assistant comparison",
                "AI Agent framework LangChain CrewAI Dify",
                "AI 编程 技巧 教程",
                "Claude Code 使用 心得",
            ],
            month_token,
        ),
        rss_categories=["tutorial"],
        extra_instructions=(
            "聚焦实操价值：能落地的工作流、可复用的 prompt、工具使用 tip。"
            "排除纯产品发布广告，保留有具体使用方法描述的内容。"
        ),
    )

    industry = AgentSpec(
        key="industry",
        name="AI 行业新闻与产品动态",
        focus="大模型发布、产品更新、融资收购、政策法规、benchmark 等行业新闻。",
        queries=_queries_with_month(
            [
                "OpenAI announcement",
                "Anthropic Claude release",
                "Google DeepMind Gemini update",
                "Meta AI Llama release",
                "xAI Grok release",
                "Mistral AI release",
                "AI product launch this week",
                "AI funding round series",
                "AI regulation policy",
                "open source LLM new release",
                "Hugging Face trending model",
                "Midjourney OR Sora OR Runway OR Kling update",
                "AI benchmark new result",
            ],
            month_token,
        ),
        rss_categories=["industry"],
        extra_instructions=(
            "突出行业格局变化与产品里程碑：有版本号 / 有融资金额 / 有政策原文链接的内容优先。"
        ),
    )

    opinion = AgentSpec(
        key="opinion",
        name="AI 领袖发言与深度观点",
        focus="Sam Altman、Dario Amodei、Demis Hassabis、Andrej Karpathy、Yann LeCun、Jim Fan 等关键人物的最新发言、博客、长文观点。",
        queries=_queries_with_month(
            [
                '"Sam Altman" said OR posted OR essay',
                '"Dario Amodei" Anthropic safety',
                '"Andrej Karpathy" tweet OR post OR talk',
                '"Yann LeCun" debate OR opinion',
                '"Jim Fan" robotics OR NVIDIA',
                '"Jensen Huang" keynote AI',
                '"Demis Hassabis" DeepMind interview',
                '"Mustafa Suleyman" Microsoft AI',
                '"Fei-Fei Li" World Labs',
                '"Ilya Sutskever" SSI',
                '"Elon Musk" Grok xAI',
                "AI safety alignment essay",
                "AGI timeline opinion",
            ],
            month_token,
        ),
        rss_categories=["opinion"],
        extra_instructions=(
            "必须包含人物姓名 + 原文引用（英文保留原文 + 中文翻译）。"
            "排除营销稿、产品介绍。聚焦观点、判断、争论。"
        ),
    )

    chinese = AgentSpec(
        key="chinese",
        name="中文 AI 生态与学术动态",
        focus="国内 AI 公司动态（DeepSeek、月之暗面、智谱、阶跃、字节、阿里、百度、腾讯）、中文产品更新、学术突破。",
        queries=_queries_with_month(
            [
                "DeepSeek 最新",
                "Kimi 月之暗面 更新",
                "智谱 GLM 发布",
                "阶跃星辰 模型",
                "通义千问 阿里 AI 发布",
                "字节豆包 AI 更新",
                "百度 文心 发布",
                "腾讯混元 发布",
                "中国 AI 创业 融资",
                "AI 论文 突破",
                "arXiv LLM breakthrough paper",
                "machine learning research highlight",
            ],
            month_token,
        ),
        rss_categories=["chinese"],
        extra_instructions=(
            "兼顾中文厂商动态与重要学术突破。"
            "对 arXiv 论文，给出一句话价值点（解决了什么 / 比 SOTA 高多少）。"
        ),
    )

    academic = AgentSpec(
        key="academic",
        name="AI 学术论文",
        focus="重要 AI 论文与学术突破：arXiv 新论文、HuggingFace papers、SOTA 进展、新 benchmark 论文。",
        queries=_queries_with_month(
            [
                "arXiv LLM breakthrough paper",
                "arXiv new paper SOTA",
                "HuggingFace daily papers trending",
                "machine learning research highlight",
                "transformer architecture new paper",
                "RLHF DPO alignment paper",
                "diffusion model paper new",
                "multimodal LLM paper",
                "agent reasoning paper arXiv",
                "AI 论文 突破 arXiv",
                "新 benchmark 论文 评测",
                "scaling law new paper",
            ],
            month_token,
        ),
        rss_categories=["academic"],
        extra_instructions=(
            "聚焦真正有突破或方法创新的论文，给出一句话价值点："
            "解决了什么问题 / 比 SOTA 高多少 / 对工业界的潜在影响。"
            "排除综述、二次解读、营销稿。优先有数字、有比较、有代码或权重开源的工作。"
        ),
    )

    benchmark = AgentSpec(
        key="benchmark",
        name="模型评测雷达",
        focus="模型评测榜单变化与能力对比：LMSYS Arena、Aider、SWE-bench、Artificial Analysis、Open LLM Leaderboard 等的最新排名变化。",
        queries=_queries_mixed_month(
            [
                "LMSYS Chatbot Arena leaderboard",
                "Aider leaderboard coding",
                "SWE-bench verified leaderboard",
                "Artificial Analysis model comparison",
                "Open LLM Leaderboard HuggingFace",
                "MMLU score new model",
                "GPQA benchmark result new",
                "HumanEval coding benchmark new",
                "model benchmark ranking change",
                "AI 模型 评测 榜单 变化",
            ],
            month_token,
            skip_month={0, 1, 2, 3, 4},
        ),
        rss_categories=["benchmark"],
        extra_instructions=(
            "必须包含具体数字变化：分数、排名、相对前一版本的提升幅度。"
            "排除没有数字支撑的笼统评价；优先官方榜单更新与第三方对比报告。"
            "若同一榜单本期无变化，可省略；不要凑数。"
        ),
    )

    return [tutorial, industry, opinion, chinese, academic, benchmark]
