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
        name="AI 学术与评测",
        focus="重要 AI 论文、学术突破与模型评测榜单变化。兼顾 arXiv 新论文和 LMSYS/Aider/SWE-bench 等主流榜单的排名变动。",
        queries=_queries_mixed_month(
            [
                # 论文搜索（带月 token 增加时效性）
                "arXiv LLM breakthrough paper",
                "arXiv new paper SOTA",
                "HuggingFace daily papers trending",
                "machine learning research highlight",
                "transformer architecture new paper",
                "RLHF DPO alignment paper",
                "diffusion model paper new",
                "multimodal LLM paper",
                "agent reasoning paper arXiv",
                "scaling law new paper",
                "AI 论文 突破 arXiv",
                "新 benchmark 论文 评测",
                # 评测榜单（不加月 token，保留搜索 "latest" leaderboard）
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
            skip_month={12, 13, 14, 15, 16},
        ),
        rss_categories=["academic"],
        extra_instructions=(
            "论文部分：聚焦真正有突破或方法创新的论文，给出一句话价值点（解决了什么 / 比 SOTA 高多少）。"
            "评测部分：必须包含具体数字变化（分数、排名、提升幅度），无数字支撑的笼统评价跳过。"
            "排除综述、二次解读、营销稿。若榜单本期无变化可省略。共输出 4-5 条，论文与评测各 2-3 条为宜。"
        ),
    )

    return [industry, opinion, chinese, academic]
