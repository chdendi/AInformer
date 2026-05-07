from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from ..config import DAILY_DATA_DIR, MONTHLY_DATA_DIR


def load_daily_in_range(year: int, month: int | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not DAILY_DATA_DIR.exists():
        return out
    for p in sorted(DAILY_DATA_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
        date_str = d.get("date", "")
        try:
            dt = date.fromisoformat(date_str)
        except Exception:
            continue
        if dt.year != year:
            continue
        if month is not None and dt.month != month:
            continue
        out.append(d)
    return out


def load_monthly_in_year(year: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not MONTHLY_DATA_DIR.exists():
        return out
    for p in sorted(MONTHLY_DATA_DIR.glob(f"{year}*.json")):
        try:
            out.append(json.loads(p.read_text("utf-8")))
        except Exception:
            continue
    return out


def flatten_items(daily_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in daily_reports:
        date_str = d.get("date", "")
        for cat, items in (d.get("sections") or {}).items():
            for it in items or []:
                row = dict(it)
                row.setdefault("category", cat)
                row["report_date"] = date_str
                out.append(row)
        for h in d.get("headlines") or []:
            row = dict(h)
            row.setdefault("category", "headline")
            row["report_date"] = date_str
            row["importance"] = "hot"
            out.append(row)
    return out


def top_sources(items: list[dict[str, Any]], n: int = 10) -> list[tuple[str, int]]:
    c = Counter()
    for it in items:
        s = it.get("source_name", "")
        if s:
            c[s] += 1
    return c.most_common(n)
