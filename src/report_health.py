from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from .config import DAILY_DATA_DIR

SECTION_KEYS = ("industry", "opinion", "chinese", "academic")


def _load_daily_reports(limit: int) -> list[dict[str, Any]]:
    if not DAILY_DATA_DIR.exists():
        return []

    reports: list[dict[str, Any]] = []
    for path in sorted(DAILY_DATA_DIR.glob("*.json"), reverse=True):
        try:
            reports.append(json.loads(path.read_text("utf-8")))
        except Exception:
            continue
        if limit > 0 and len(reports) >= limit:
            break
    return list(reversed(reports))


def build_health(reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_section: dict[str, dict[str, Any]] = {}
    dates = [r.get("date", "") for r in reports]

    for key in SECTION_KEYS:
        counts: list[int] = []
        empty_dates: list[str] = []
        fallback_dates: list[str] = []
        empty_reasons: Counter[str] = Counter()

        for report in reports:
            date = report.get("date", "")
            items = (report.get("sections") or {}).get(key) or []
            meta = (report.get("section_meta") or {}).get(key) or {}
            count = len(items)
            counts.append(count)

            if count == 0:
                empty_dates.append(date)
                empty_reasons[meta.get("empty_reason") or "unknown"] += 1

            if meta.get("selection_mode") == "fallback" or any(
                item.get("selection_mode") == "fallback" for item in items
            ):
                fallback_dates.append(date)

        total_days = len(reports)
        empty_days = len(empty_dates)
        by_section[key] = {
            "days": total_days,
            "empty_days": empty_days,
            "empty_rate": round(empty_days / total_days, 4) if total_days else 0,
            "avg_items": round(sum(counts) / total_days, 2) if total_days else 0,
            "min_items": min(counts) if counts else 0,
            "max_items": max(counts) if counts else 0,
            "fallback_days": len(fallback_dates),
            "empty_dates": empty_dates,
            "fallback_dates": fallback_dates,
            "empty_reasons": dict(empty_reasons),
        }

    return {
        "days": len(reports),
        "date_start": dates[0] if dates else "",
        "date_end": dates[-1] if dates else "",
        "sections": by_section,
    }


def _print_text(health: dict[str, Any]) -> None:
    print(f"Report health: {health['days']} days ({health['date_start']} ~ {health['date_end']})")
    for key, row in health["sections"].items():
        empty_rate = row["empty_rate"] * 100
        print(
            f"- {key:>8}: avg={row['avg_items']:.2f}, "
            f"empty={row['empty_days']}/{row['days']} ({empty_rate:.1f}%), "
            f"fallback={row['fallback_days']}"
        )
        if row["empty_dates"]:
            print(f"          empty_dates={', '.join(row['empty_dates'])}")
        if row["empty_reasons"]:
            reasons = ", ".join(f"{k}:{v}" for k, v in row["empty_reasons"].items())
            print(f"          empty_reasons={reasons}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect AInformer section coverage over recent daily reports.")
    parser.add_argument("--days", type=int, default=7, help="Number of latest daily reports to inspect; <=0 means all.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    health = build_health(_load_daily_reports(args.days))
    if args.json:
        print(json.dumps(health, ensure_ascii=False, indent=2))
    else:
        _print_text(health)


if __name__ == "__main__":
    main()
