from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import (
    MONTHLY_DATA_DIR,
    MONTHLY_HTML_DIR,
    YEARLY_DATA_DIR,
    YEARLY_HTML_DIR,
    tz,
)
from .daily import home_url
from .engine import render


def render_monthly(summary: dict[str, Any]) -> str:
    generated_at = datetime.now(tz()).strftime("%Y-%m-%d %H:%M")
    return render(
        "monthly.html.j2",
        summary=summary,
        home_url=home_url(),
        generated_at=generated_at,
    )


def render_yearly(summary: dict[str, Any]) -> str:
    generated_at = datetime.now(tz()).strftime("%Y-%m-%d %H:%M")
    return render(
        "yearly.html.j2",
        summary=summary,
        home_url=home_url(),
        generated_at=generated_at,
    )


def write_monthly(summary: dict[str, Any]) -> Path:
    MONTHLY_HTML_DIR.mkdir(parents=True, exist_ok=True)
    MONTHLY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    slug = summary["slug"]
    (MONTHLY_DATA_DIR / f"{slug}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    out = MONTHLY_HTML_DIR / f"{slug}.html"
    out.write_text(render_monthly(summary), encoding="utf-8")
    return out


def write_yearly(summary: dict[str, Any]) -> Path:
    YEARLY_HTML_DIR.mkdir(parents=True, exist_ok=True)
    YEARLY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    slug = summary["slug"]
    (YEARLY_DATA_DIR / f"{slug}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    out = YEARLY_HTML_DIR / f"{slug}.html"
    out.write_text(render_yearly(summary), encoding="utf-8")
    return out
