from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import date, datetime

from .agents.definitions import build_agent_specs
from .agents.runner import run_all_agents
from .config import LLMConfig, ensure_dirs, tz
from .dedupe import collect_excluded, dedupe_within, filter_new, load_recent_reports, normalize_title
from .llm.client import get_usage_summary, make_client
from .render.daily import write_daily_html, write_daily_json
from .render.index_page import write_index
from .search.rss import collect_rss
from .synthesize import synthesize_overview

log = logging.getLogger("ainformer.daily")


# Cross-section dedup priority: smaller number wins.
# Rationale: specialty sections (opinion quotes, academic papers, benchmark
# numbers) are intrinsically tied to their column's core attribute, so they
# should keep the item. `industry` is the broadest catch-all and is most
# likely to absorb stories that genuinely belong elsewhere, so it goes last.
SECTION_PRIORITY = {
    "opinion": 1,    # quote-driven, most exclusive
    "academic": 2,   # paper-driven
    "benchmark": 3,  # eval-number driven
    "tutorial": 4,   # hands-on tutorials
    "chinese": 5,    # regional bucket
    "industry": 6,   # broadest, fallback bucket
}


def dedupe_across_sections(sections: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Drop items that appear in more than one section, keeping the highest-priority section.

    Item identity key is `url` when present, otherwise `normalize_title(title)`.
    Sections are visited in `SECTION_PRIORITY` order (ascending = higher priority);
    the first section to claim a key keeps the item, later sections drop it.

    The output preserves all original section keys (even if a section becomes empty)
    and does not mutate item dicts. Sections not listed in `SECTION_PRIORITY` are
    processed last in their original order, so unknown sections also lose ties to
    known specialty columns.
    """
    seen: set[str] = set()
    result: dict[str, list[dict]] = {k: [] for k in sections}

    ordered_keys = sorted(
        sections.keys(),
        key=lambda k: (SECTION_PRIORITY.get(k, 999), k),
    )

    removed = 0
    for key in ordered_keys:
        kept: list[dict] = []
        for item in sections[key]:
            url = (item.get("url") or "").strip()
            if url:
                ident = f"url:{url}"
            else:
                title = item.get("title") or ""
                norm = normalize_title(title)
                if not norm:
                    continue
                ident = f"title:{norm}"
            if ident in seen:
                removed += 1
                continue
            seen.add(ident)
            kept.append(item)
        result[key] = kept

    msg = f"Cross-section dedup: removed {removed} duplicates"
    log.info(msg)
    print(msg)
    return result


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
    excluded_urls, excluded_titles, excluded_by_cat = collect_excluded(recent)
    log.info(
        "Excluded: %d urls, %d titles (per-cat: %s)",
        len(excluded_urls),
        len(excluded_titles),
        {k: len(v) for k, v in excluded_by_cat.items()},
    )

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
        agent_results = await run_all_agents(specs, rss_pool, excluded_by_cat, today_str, client, cfg)

        sections: dict[str, list] = {}
        for r in agent_results:
            items = filter_new(r["items"], excluded_urls, excluded_titles)
            items = dedupe_within(items)
            sections[r["key"]] = items

        sections = dedupe_across_sections(sections)

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

    usage = get_usage_summary()
    print(
        f"   💰 LLM usage: {usage['calls']} calls · "
        f"prompt={usage['prompt_tokens']:,} · completion={usage['completion_tokens']:,} · "
        f"≈ ¥{usage['cost_cny']}"
    )


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
