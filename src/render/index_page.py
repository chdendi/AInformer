from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from ..config import (
    DAILY_DATA_DIR,
    DOCS_DIR,
    MONTHLY_DATA_DIR,
    YEARLY_DATA_DIR,
    tz,
)
from .daily import WEEKDAY_ZH, home_url
from .engine import render


def _read_json_dir(path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for p in sorted(path.glob("*.json"), reverse=True):
        try:
            out.append(json.loads(p.read_text("utf-8")))
        except Exception:
            continue
    return out


def _daily_card(d: dict[str, Any]) -> dict[str, Any]:
    sections = d.get("sections") or {}
    total = sum(len(v) for v in sections.values())
    hot = sum(1 for v in sections.values() for it in v if it.get("importance") == "hot")
    date_str = d.get("date", "")
    weekday = ""
    try:
        weekday = WEEKDAY_ZH[datetime.fromisoformat(date_str).weekday()]
    except Exception:
        pass
    return {
        "date": date_str,
        "weekday_zh": weekday,
        "slug": date_str.replace("-", ""),
        "theme": d.get("today_theme", ""),
        "lede": d.get("lede", ""),
        "total": total,
        "hot_count": hot,
    }


def _monthly_card(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": m.get("label", ""),
        "slug": m.get("slug") or m.get("label", "").replace("-", ""),
        "title": m.get("title", ""),
        "tagline": m.get("tagline", ""),
        "total": m.get("total_items", 0),
    }


def _yearly_card(y: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": y.get("label", ""),
        "slug": y.get("slug") or y.get("label", ""),
        "title": y.get("title", ""),
        "tagline": y.get("tagline", ""),
    }


def write_index() -> None:
    daily_raw = _read_json_dir(DAILY_DATA_DIR)
    monthly_raw = _read_json_dir(MONTHLY_DATA_DIR)
    yearly_raw = _read_json_dir(YEARLY_DATA_DIR)

    daily = [_daily_card(d) for d in daily_raw]
    monthly = [_monthly_card(m) for m in monthly_raw]
    yearly = [_yearly_card(y) for y in yearly_raw]

    today = datetime.now(tz()).strftime("%Y-%m-%d")
    generated_at = datetime.now(tz()).strftime("%Y-%m-%d %H:%M")

    html = render(
        "index.html.j2",
        daily=daily,
        monthly=monthly,
        yearly=yearly,
        today=today,
        generated_at=generated_at,
        home_url=home_url(),
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
