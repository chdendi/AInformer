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


# L1 头部：CEO + 首席科学家。person 字段必须严格匹配这些名字之一。
# Musk 加 keyword_filter — quote_en/quote_zh 必须含 xAI/Grok/AI 字样，
# 否则会把他的政治/SpaceX 言论也吸进来。
LEADER_PROFILES: dict[str, dict] = {
    "Sam Altman":        {"query": '"Sam Altman" said OR posted OR essay'},
    "Dario Amodei":      {"query": '"Dario Amodei" Anthropic safety'},
    "Demis Hassabis":    {"query": '"Demis Hassabis" DeepMind interview'},
    "Jensen Huang":      {"query": '"Jensen Huang" keynote AI'},
    "Mustafa Suleyman":  {"query": '"Mustafa Suleyman" Microsoft AI'},
    "Andrej Karpathy":   {"query": '"Andrej Karpathy" tweet OR post OR talk'},
    "Yann LeCun":        {"query": '"Yann LeCun" debate OR opinion'},
    "Ilya Sutskever":    {"query": '"Ilya Sutskever" SSI'},
    "Fei-Fei Li":        {"query": '"Fei-Fei Li" World Labs'},
    "Jim Fan":           {"query": '"Jim Fan" robotics OR NVIDIA'},
    "Elon Musk":         {"query": '"Elon Musk" Grok xAI', "keyword_filter": ("xAI", "Grok", "AI ", " AI")},
}

# L2 写手：高产 blogger / newsletter 作者
ANALYST_NAMES: set[str] = {
    "Simon Willison", "Gary Marcus", "Lilian Weng", "Sebastian Raschka",
    "Chip Huyen", "Nathan Lambert", "Ethan Mollick", "Arvind Narayanan",
    "Sayash Kapoor",
}


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
            "突出行业格局变化、产品里程碑、公司组织调整、平台治理、重要模型/工具发布。"
            "有版本号 / 融资金额 / 政策原文链接的内容优先，但不要把它们当作硬性门槛；"
            "可信媒体或官方来源报道的实质产品更新也应保留。"
        ),
    )

    opinion_queries: list[str] = [p["query"] for p in LEADER_PROFILES.values()]
    opinion_queries += [
        '"Simon Willison" LLM blog',
        '"Gary Marcus" essay AI',
        '"Lilian Weng" blog',
        '"Sebastian Raschka" newsletter',
        '"Chip Huyen" blog',
        '"Nathan Lambert" Interconnects',
        '"Ethan Mollick" One Useful Thing',
        "Latent Space podcast essay",
        "Import AI newsletter Jack Clark",
        "AI safety alignment essay",
        "AGI timeline opinion essay",
    ]
    leaders_str = "、".join(LEADER_PROFILES.keys())
    analysts_str = "、".join(sorted(ANALYST_NAMES))
    opinion = AgentSpec(
        key="opinion",
        name="AI 观点与社区声音",
        focus=(
            "两层供稿：L1 领袖发言（CEO + 首席科学家原话、博客、采访），"
            "L2 深度分析（高产 blogger / newsletter 作者的长文观点）。"
        ),
        queries=_queries_with_month(opinion_queries, month_token),
        rss_categories=["opinion"],
        extra_instructions=(
            f"每条必须标 tier 字段：\n"
            f"- tier=\"leader\"：person 必须严格等于这 11 人之一 — {leaders_str}；"
            f"Elon Musk 仅限其 xAI/Grok/AI 相关言论，与政治/SpaceX 无关的发言一律跳过。\n"
            f"- tier=\"analyst\"：person 必须严格等于这些写手之一 — {analysts_str}。\n"
            "quote_en 可以是：(a) 本人 blog/tweet/采访/演讲中的原文摘句；"
            "(b) 第三方报道里的直接引语（带引号的原话）。"
            "纯新闻转述（如『The Verge 报道 OpenAI 取消了 X』这种没有当事人原话的内容）跳过。"
            "排除营销稿、产品介绍。person 字段不要随手填，找不到合规人物就少给几条；"
            "若实在没有合规素材，返回 {\"items\": []} 即可。"
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
