from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import (
    DAILY_DATA_DIR,
    DOCS_DIR,
    MONTHLY_DATA_DIR,
    SITE_BASE_URL,
    YEARLY_DATA_DIR,
)

API_DIR = DOCS_DIR / "api" / "v1"
API_DAILY_DIR = API_DIR / "daily"
API_MONTHLY_DIR = API_DIR / "monthly"
API_YEARLY_DIR = API_DIR / "yearly"

API_TITLE = "AInformer API"
API_VERSION = "1.0.0"
API_DESC = "AI 资讯日报 JSON API，提供每日/每月/每年 AI 行业简报的结构化数据。"


def _read_json_dir(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(path.glob("*.json"), reverse=True):
        try:
            out.append(json.loads(p.read_text("utf-8")))
        except Exception:
            continue
    return out


def _data_url(path: str) -> str:
    if SITE_BASE_URL:
        return f"{SITE_BASE_URL}/{path.lstrip('/')}"
    return f"/{path.lstrip('/')}"


def _build_daily_index() -> dict[str, Any]:
    items = _read_json_dir(DAILY_DATA_DIR)
    entries: list[dict[str, Any]] = []
    for d in items:
        sections = d.get("sections") or {}
        total = sum(len(v) for v in sections.values())
        entries.append({
            "date": d.get("date", ""),
            "theme": d.get("today_theme", ""),
            "generated_at": d.get("generated_at", ""),
            "total_items": total,
            "data_url": _data_url(f"data/daily/{d['date'].replace('-', '')}.json"),
            "html_url": _data_url(f"daily/{d['date'].replace('-', '')}.html"),
        })

    latest = entries[0] if entries else None
    return {
        "updated_at": datetime.now().astimezone().isoformat(),
        "total_days": len(entries),
        "latest": latest,
        "dates": entries,
    }


def _build_monthly_index() -> dict[str, Any]:
    items = _read_json_dir(MONTHLY_DATA_DIR)
    entries: list[dict[str, Any]] = []
    for m in items:
        entries.append({
            "label": m.get("label", ""),
            "title": m.get("title", ""),
            "tagline": m.get("tagline", ""),
            "total_items": m.get("total_items", 0),
            "generated_at": m.get("generated_at", ""),
            "data_url": _data_url(f"data/monthly/{m.get('slug', m.get('label', '').replace('-', ''))}.json"),
            "html_url": _data_url(f"monthly/{m.get('slug', m.get('label', '').replace('-', ''))}.html"),
        })

    latest = entries[0] if entries else None
    return {
        "updated_at": datetime.now().astimezone().isoformat(),
        "total_months": len(entries),
        "latest": latest,
        "months": entries,
    }


def _build_yearly_index() -> dict[str, Any]:
    items = _read_json_dir(YEARLY_DATA_DIR)
    entries: list[dict[str, Any]] = []
    for y in items:
        entries.append({
            "label": y.get("label", ""),
            "title": y.get("title", ""),
            "tagline": y.get("tagline", ""),
            "total_items": y.get("total_items", 0),
            "generated_at": y.get("generated_at", ""),
            "data_url": _data_url(f"data/yearly/{y.get('slug', y.get('label', ''))}.json"),
            "html_url": _data_url(f"yearly/{y.get('slug', y.get('label', ''))}.html"),
        })

    latest = entries[0] if entries else None
    return {
        "updated_at": datetime.now().astimezone().isoformat(),
        "total_years": len(entries),
        "latest": latest,
        "years": entries,
    }


def _build_manifest(
    daily_index: dict[str, Any],
    monthly_index: dict[str, Any],
    yearly_index: dict[str, Any],
) -> dict[str, Any]:
    endpoints: dict[str, Any] = {
        "daily": {
            "index": _data_url("api/v1/daily/index.json"),
            "latest": _data_url("api/v1/daily/latest.json"),
            "description": "每日 AI 资讯简报列表，latest 为最新一期的完整引用",
        },
        "monthly": {
            "index": _data_url("api/v1/monthly/index.json"),
            "latest": _data_url("api/v1/monthly/latest.json"),
            "description": "每月 AI 资讯月报列表",
        },
        "yearly": {
            "index": _data_url("api/v1/yearly/index.json"),
            "latest": _data_url("api/v1/yearly/latest.json"),
            "description": "每年 AI 资讯年报列表",
        },
    }

    stats = {
        "total_daily_reports": daily_index.get("total_days", 0),
        "total_monthly_reports": monthly_index.get("total_months", 0),
        "total_yearly_reports": yearly_index.get("total_years", 0),
        "latest_daily_date": (daily_index.get("latest") or {}).get("date"),
        "latest_monthly_label": (monthly_index.get("latest") or {}).get("label"),
        "latest_yearly_label": (yearly_index.get("latest") or {}).get("label"),
    }

    return {
        "api": API_TITLE,
        "version": API_VERSION,
        "description": API_DESC,
        "site_url": SITE_BASE_URL or "",
        "updated_at": datetime.now().astimezone().isoformat(),
        "stats": stats,
        "endpoints": endpoints,
    }


def build_api() -> None:
    for d in (API_DAILY_DIR, API_MONTHLY_DIR, API_YEARLY_DIR):
        d.mkdir(parents=True, exist_ok=True)

    daily_index = _build_daily_index()
    monthly_index = _build_monthly_index()
    yearly_index = _build_yearly_index()
    manifest = _build_manifest(daily_index, monthly_index, yearly_index)

    def _write(path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    _write(API_DAILY_DIR / "index.json", daily_index)
    _write(API_MONTHLY_DIR / "index.json", monthly_index)
    _write(API_YEARLY_DIR / "index.json", yearly_index)
    _write(API_DIR / "manifest.json", manifest)

    def _empty_latest(kind: str) -> dict[str, Any]:
        return {
            "available": False,
            "message": f"No {kind} report has been generated yet.",
            "updated_at": datetime.now().astimezone().isoformat(),
        }

    _write(API_DAILY_DIR / "latest.json", daily_index.get("latest") or _empty_latest("daily"))
    _write(API_MONTHLY_DIR / "latest.json", monthly_index.get("latest") or _empty_latest("monthly"))
    _write(API_YEARLY_DIR / "latest.json", yearly_index.get("latest") or _empty_latest("yearly"))

    print("✅ API index files generated under docs/api/v1/")
