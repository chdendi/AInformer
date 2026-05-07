from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .config import DAILY_DATA_DIR

log = logging.getLogger(__name__)


_PUNCT_RE = re.compile(r"[\s\-_·•—–:：、，,。\.\?\!？！\"'\(\)（）\[\]【】《》<>~`]+")


def normalize_title(t: str) -> str:
    return _PUNCT_RE.sub("", (t or "").lower()).strip()


def load_recent_reports(days: int = 7, today: date | None = None) -> list[dict[str, Any]]:
    if today is None:
        today = date.today()
    out: list[dict[str, Any]] = []
    for n in range(1, days + 1):
        d = today - timedelta(days=n)
        path = DAILY_DATA_DIR / f"{d.strftime('%Y%m%d')}.json"
        if not path.exists():
            continue
        try:
            out.append(json.loads(path.read_text("utf-8")))
        except Exception as e:
            log.warning("Failed to read %s: %s", path, e)
    return out


def collect_excluded(reports: list[dict[str, Any]]) -> tuple[set[str], set[str], list[str]]:
    """Return (urls, normalized_titles, raw_titles_for_prompt)."""
    urls: set[str] = set()
    titles_norm: set[str] = set()
    raw: list[str] = []
    for r in reports:
        for section in (r.get("sections") or {}).values():
            for it in section or []:
                if it.get("url"):
                    urls.add(it["url"])
                t = it.get("title", "")
                if t:
                    titles_norm.add(normalize_title(t))
                    raw.append(t)
        for h in r.get("headlines") or []:
            t = h.get("title", "")
            if t:
                titles_norm.add(normalize_title(t))
                raw.append(t)
    return urls, titles_norm, raw


def filter_new(items: list[dict[str, Any]], excluded_urls: set[str], excluded_titles: set[str]) -> list[dict[str, Any]]:
    out = []
    for it in items:
        url = it.get("url", "")
        if url and url in excluded_urls:
            continue
        norm = normalize_title(it.get("title", ""))
        if norm and norm in excluded_titles:
            continue
        out.append(it)
    return out


def dedupe_within(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_url: set[str] = set()
    seen_title: set[str] = set()
    out = []
    for it in items:
        url = it.get("url", "")
        norm = normalize_title(it.get("title", ""))
        if url and url in seen_url:
            continue
        if norm and norm in seen_title:
            continue
        if url:
            seen_url.add(url)
        if norm:
            seen_title.add(norm)
        out.append(it)
    return out
