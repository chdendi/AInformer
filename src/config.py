from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
DATA_DIR = DOCS_DIR / "data"
DAILY_DATA_DIR = DATA_DIR / "daily"
MONTHLY_DATA_DIR = DATA_DIR / "monthly"
YEARLY_DATA_DIR = DATA_DIR / "yearly"
DAILY_HTML_DIR = DOCS_DIR / "daily"
MONTHLY_HTML_DIR = DOCS_DIR / "monthly"
YEARLY_HTML_DIR = DOCS_DIR / "yearly"


def tz() -> ZoneInfo:
    return ZoneInfo(os.environ.get("TZ", "Asia/Shanghai"))


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return cls(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        )


SITE_BASE_URL = os.environ.get("SITE_BASE_URL", "").rstrip("/")
SITE_TITLE = "AInformer · 每日 AI 简报"
SITE_DESC = "由 DeepSeek + Tavily 自动生成的中文 AI 资讯日报，覆盖工具教程、行业动态、观点声音、中文生态。"


def ensure_dirs() -> None:
    for d in (
        DAILY_DATA_DIR,
        MONTHLY_DATA_DIR,
        YEARLY_DATA_DIR,
        DAILY_HTML_DIR,
        MONTHLY_HTML_DIR,
        YEARLY_HTML_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
