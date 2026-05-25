from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import date, datetime

from .agents.definitions import build_agent_specs
from .agents.runner import run_all_agents
from .agents.trending_filter import filter_trending_with_llm
from .build_api import build_api
from .config import LLMConfig, ensure_dirs, tz
from .dedupe import collect_excluded, dedupe_within, filter_new, load_recent_reports, normalize_title
from .llm.client import get_usage_summary, make_client
from .render.daily import write_daily_html, write_daily_json
from .render.index_page import write_index
from .search.github_trending import fetch_trending
from .search.rss import collect_rss
from .synthesize import synthesize_overview

log = logging.getLogger("ainformer.daily")


# Cross-section dedup priority: smaller number wins.
# Rationale: specialty sections (opinion/community voices, academic papers, benchmark
# numbers) are intrinsically tied to their column's core attribute, so they
# should keep the item. `industry` is the broadest catch-all and is most
# likely to absorb stories that genuinely belong elsewhere, so it goes last.
SECTION_PRIORITY = {
    "opinion": 1,    # view/quote-driven, most exclusive
    "academic": 2,   # paper + benchmark driven
    "chinese": 3,    # regional bucket
    "industry": 4,   # broadest, fallback bucket
}

# Per-section visible card cap. Agents are instructed to return 5-6 candidates
# so cross-section dedup has 1-2 buffer items to absorb before this cap kicks
# in. Lowering this further is safe; raising it requires loosening the prompt.
SECTION_CARD_CAP = 4
TRENDING_FETCH_LIMIT = 20
TRENDING_KEEP = 4


def dedupe_across_sections(
    sections: dict[str, list[dict]],
    *,
    return_drops: bool = False,
) -> dict[str, list[dict]] | tuple[dict[str, list[dict]], dict[str, int]]:
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
    drops_by_section: dict[str, int] = {k: 0 for k in sections}

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
                drops_by_section[key] += 1
                continue
            seen.add(ident)
            kept.append(item)
        result[key] = kept

    msg = f"Cross-section dedup: removed {removed} duplicates"
    log.info(msg)
    print(msg)
    if return_drops:
        return result, drops_by_section
    return result


def cap_sections(
    sections: dict[str, list[dict]],
    cap: int = SECTION_CARD_CAP,
    *,
    return_drops: bool = False,
) -> dict[str, list[dict]] | tuple[dict[str, list[dict]], dict[str, int]]:
    """Trim each section to at most `cap` items. Order is preserved.

    Items kept are sorted by importance (hot → star → pin) within their
    original ordering so the visible cards skew toward higher-signal entries
    when the agent returns more than `cap` candidates.
    """
    importance_rank = {"hot": 0, "star": 1, "pin": 2}
    capped: dict[str, list[dict]] = {}
    drops_by_section: dict[str, int] = {k: 0 for k in sections}
    dropped_total = 0
    for key, items in sections.items():
        if len(items) <= cap:
            capped[key] = list(items)
            continue
        ranked = sorted(
            enumerate(items),
            key=lambda pair: (importance_rank.get(pair[1].get("importance", "pin"), 2), pair[0]),
        )
        kept = sorted(ranked[:cap], key=lambda pair: pair[0])
        capped[key] = [it for _, it in kept]
        dropped = len(items) - cap
        dropped_total += dropped
        drops_by_section[key] = dropped

    if dropped_total:
        msg = f"Section cap={cap}: dropped {dropped_total} surplus items"
        log.info(msg)
        print(msg)
    if return_drops:
        return capped, drops_by_section
    return capped


def _mark_empty_reason(meta: dict, reason: str) -> None:
    if not meta.get("empty_reason"):
        meta["empty_reason"] = reason
    meta["selection_mode"] = "empty"


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

    trending: list[dict] = []
    section_meta: dict[str, dict] = {}

    if dry:
        sections = {s.key: [] for s in specs}
        section_meta = {
            s.key: {
                "section": s.key,
                "selection_mode": "empty",
                "empty_reason": "dry_run",
                "final_count": 0,
            }
            for s in specs
        }
        synth = {"lede": "（dry-run）", "today_theme": "占位", "headlines": []}
    else:
        log.info("Running %d agents + GitHub trending in parallel...", len(specs))
        agent_results, trending_raw = await asyncio.gather(
            run_all_agents(specs, rss_pool, excluded_by_cat, today_str, client, cfg),
            fetch_trending(limit=TRENDING_FETCH_LIMIT),
        )

        sections: dict[str, list] = {}
        for r in agent_results:
            meta = dict(r.get("diagnostics") or {"section": r["key"]})
            agent_n = len(r["items"])
            items = filter_new(r["items"], excluded_urls, excluded_titles)
            after_hist = len(items)
            items = dedupe_within(items)
            after_within = len(items)
            meta["agent_count"] = agent_n
            meta["after_historical_dedupe_count"] = after_hist
            meta["historical_dedupe_drops"] = agent_n - after_hist
            meta["after_within_dedupe_count"] = after_within
            meta["within_dedupe_drops"] = after_hist - after_within
            if agent_n and not after_hist:
                _mark_empty_reason(meta, "historical_dedupe_dropped_all")
            elif after_hist and not after_within:
                _mark_empty_reason(meta, "within_dedupe_dropped_all")
            log.info(
                "[section:%s] agent=%d → after-historical-dedup=%d (-%d) → after-within-dedup=%d (-%d)",
                r["key"], agent_n, after_hist, agent_n - after_hist,
                after_within, after_hist - after_within,
            )
            sections[r["key"]] = items
            section_meta[r["key"]] = meta

        pre_cross_counts = {key: len(items) for key, items in sections.items()}
        sections, cross_drops = dedupe_across_sections(sections, return_drops=True)
        for key, drops in cross_drops.items():
            meta = section_meta.setdefault(key, {"section": key})
            meta["cross_section_dedupe_drops"] = drops
            meta["after_cross_section_dedupe_count"] = len(sections.get(key, []))
            if pre_cross_counts.get(key, 0) and not sections.get(key):
                _mark_empty_reason(meta, "cross_section_dedupe_dropped_all")

        pre_cap_counts = {key: len(items) for key, items in sections.items()}
        sections, cap_drops = cap_sections(sections, cap=SECTION_CARD_CAP, return_drops=True)
        for key, drops in cap_drops.items():
            meta = section_meta.setdefault(key, {"section": key})
            meta["cap_drops"] = drops
            meta["final_count"] = len(sections.get(key, []))
            if sections.get(key):
                meta["empty_reason"] = ""
            elif not meta.get("empty_reason"):
                if pre_cap_counts.get(key, 0):
                    _mark_empty_reason(meta, "section_cap_dropped_all")
                else:
                    _mark_empty_reason(meta, "no_publishable_items")

        for key, items in sections.items():
            if not items:
                reason = section_meta.get(key, {}).get("empty_reason") or "unknown"
                log.warning("[section:%s] FINAL empty (%s) — investigate section_meta", key, reason)

        if trending_raw:
            log.info("Filtering %d trending repos via LLM...", len(trending_raw))
            trending = await filter_trending_with_llm(client, cfg, trending_raw, keep=TRENDING_KEEP)
            if not trending:
                log.warning("Trending LLM filter returned 0 even though fetch had %d repos", len(trending_raw))
        else:
            log.warning("Trending fetch returned empty — skipping trending section (check github_trending logs above)")

        log.info("Synthesizing overview...")
        synth = await synthesize_overview(client, cfg, today_str, sections)

    report = {
        "date": today_str,
        "generated_at": datetime.now(tz()).isoformat(),
        "today_theme": synth.get("today_theme", ""),
        "lede": synth.get("lede", ""),
        "headlines": synth.get("headlines", []),

        "sections": sections,
        "section_meta": section_meta,
        "trending": trending,
    }

    json_path = write_daily_json(report)
    html_path = write_daily_html(report)
    write_index()
    build_api()

    total = sum(len(v) for v in sections.values())
    print(f"\n✅ Daily report generated for {today_str}")
    print(f"   JSON: {json_path.relative_to(json_path.parents[3])}")
    print(f"   HTML: {html_path.relative_to(html_path.parents[3])}")
    print(f"   Total: {total} items + {len(trending)} trending repos")
    for k, v in sections.items():
        hot = sum(1 for it in v if it.get("importance") == "hot")
        print(f"   - {k:>10}: {len(v):>2} items ({hot} hot)")
    if trending:
        print(f"   - {'trending':>10}: {len(trending):>2} repos")

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
