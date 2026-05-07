from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import date, datetime

from .agents.definitions import build_agent_specs
from .agents.runner import run_all_agents
from .config import LLMConfig, ensure_dirs, tz
from .dedupe import collect_excluded, dedupe_within, filter_new, load_recent_reports
from .llm.client import make_client
from .render.daily import write_daily_html, write_daily_json
from .render.index_page import write_index
from .search.rss import collect_rss
from .synthesize import synthesize_overview

log = logging.getLogger("ainformer.daily")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default="", help="YYYY-MM-DD; defaults to today in TZ")
    p.add_argument("--dedupe-days", type=int, default=7)
    p.add_argument("--dry-run", action="store_true", help="Skip LLM call, write empty placeholder")
    return p.parse_args()


async def main_async(target_date: date, dedupe_days: int, dry: bool) -> None:
    ensure_dirs()
    cfg = LLMConfig.from_env()
    client = make_client(cfg)

    today_str = target_date.isoformat()
    log.info("=== Building daily report for %s ===", today_str)

    log.info("Loading last %d daily reports for dedupe...", dedupe_days)
    recent = load_recent_reports(dedupe_days, today=target_date)
    excluded_urls, excluded_titles, excluded_raw = collect_excluded(recent)
    log.info("Excluded: %d urls, %d titles", len(excluded_urls), len(excluded_titles))

    log.info("Collecting RSS pool...")
    rss_pool = await collect_rss(within_days=2)
    rss_pool = filter_new(rss_pool, excluded_urls, excluded_titles)
    log.info("RSS pool after dedup: %d items", len(rss_pool))

    month_token = target_date.strftime("%Y-%m")
    specs = build_agent_specs(month_token)

    if dry:
        sections = {s.key: [] for s in specs}
        synth = {"lede": "（dry-run）", "today_theme": "占位", "headlines": [], "daily_takeaway": {}}
    else:
        log.info("Running %d agents in parallel...", len(specs))
        agent_results = await run_all_agents(specs, rss_pool, excluded_raw, today_str, client, cfg)

        sections: dict[str, list] = {}
        for r in agent_results:
            items = filter_new(r["items"], excluded_urls, excluded_titles)
            items = dedupe_within(items)
            sections[r["key"]] = items

        log.info("Synthesizing overview...")
        synth = await synthesize_overview(client, cfg, today_str, sections)

    report = {
        "date": today_str,
        "generated_at": datetime.now(tz()).isoformat(),
        "today_theme": synth.get("today_theme", ""),
        "lede": synth.get("lede", ""),
        "headlines": synth.get("headlines", []),
        "daily_takeaway": synth.get("daily_takeaway", {}),
        "sections": sections,
    }

    json_path = write_daily_json(report)
    html_path = write_daily_html(report)
    write_index()

    total = sum(len(v) for v in sections.values())
    print(f"\n✅ Daily report generated for {today_str}")
    print(f"   JSON: {json_path.relative_to(json_path.parents[3])}")
    print(f"   HTML: {html_path.relative_to(html_path.parents[3])}")
    print(f"   Total: {total} items")
    for k, v in sections.items():
        hot = sum(1 for it in v if it.get("importance") == "hot")
        print(f"   - {k:>10}: {len(v):>2} items ({hot} hot)")


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    if args.date:
        target = date.fromisoformat(args.date)
    else:
        target = datetime.now(tz()).date()
    asyncio.run(main_async(target, args.dedupe_days, args.dry_run))


if __name__ == "__main__":
    main()
