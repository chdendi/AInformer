from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ..config import DAILY_DATA_DIR, DAILY_HTML_DIR, SITE_BASE_URL, tz
from .engine import render

WEEKDAY_ZH = ["一", "二", "三", "四", "五", "六", "日"]

CATEGORY_LABELS = {
    "tutorial": "使用姿势",
    "industry": "行业新闻",
    "opinion": "领袖观点",
    "chinese": "中文 / 学术",
    "academic": "学术与评测",
}
IMPORTANCE_LABELS = {"hot": "🔥 头条", "star": "⭐ 重要", "pin": "📎 关注"}


def category_label(c: str) -> str:
    return CATEGORY_LABELS.get(c, c)


def importance_label(i: str) -> str:
    return IMPORTANCE_LABELS.get(i, "📎 关注")


def home_url() -> str:
    if SITE_BASE_URL:
        return SITE_BASE_URL + "/index.html"
    return "../index.html"


def render_daily(data: dict[str, Any]) -> str:
    date_str = data["date"]
    dt = datetime.fromisoformat(date_str)
    sections = data.get("sections") or {k: [] for k in CATEGORY_LABELS}
    for k in CATEGORY_LABELS:
        sections.setdefault(k, [])
    headlines = data.get("headlines") or []
    total = sum(len(v) for v in sections.values())
    hot_count = sum(1 for v in sections.values() for it in v if it.get("importance") == "hot")

    generated_at = data.get("generated_at", "")
    gen_short = ""
    if generated_at:
        try:
            gen_short = datetime.fromisoformat(generated_at).astimezone(tz()).strftime("%Y-%m-%d %H:%M")
        except Exception:
            gen_short = generated_at

    report = {
        "date": date_str,
        "weekday_zh": WEEKDAY_ZH[dt.weekday()],
        "today_theme": data.get("today_theme", ""),
        "lede": data.get("lede", ""),
        "headlines": headlines,
        "daily_takeaway": data.get("daily_takeaway") or {},
        "sections": sections,
        "trending": data.get("trending") or [],
        "total": total,
        "hot_count": hot_count,
        "generated_at_short": gen_short,
    }

    return render(
        "daily.html.j2",
        report=report,
        generated_at=gen_short or generated_at,
        home_url=home_url(),
        category_label=category_label,
        importance_label=importance_label,
    )


def write_daily_html(data: dict[str, Any]) -> Path:
    DAILY_HTML_DIR.mkdir(parents=True, exist_ok=True)
    slug = data["date"].replace("-", "")
    out = DAILY_HTML_DIR / f"{slug}.html"
    out.write_text(render_daily(data), encoding="utf-8")
    return out


def write_daily_json(data: dict[str, Any]) -> Path:
    DAILY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    slug = data["date"].replace("-", "")
    out = DAILY_DATA_DIR / f"{slug}.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
