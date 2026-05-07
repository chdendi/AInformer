from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import date

from .config import LLMConfig, ensure_dirs, tz
from .render.index_page import write_index
from .summarize.monthly import run_monthly

log = logging.getLogger("ainformer.monthly")


def _previous_month(today: date) -> tuple[int, int]:
    if today.month == 1:
        return today.year - 1, 12
    return today.year, today.month - 1


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=0)
    p.add_argument("--month", type=int, default=0)
    args = p.parse_args()

    from datetime import datetime
    today = datetime.now(tz()).date()
    if args.year and args.month:
        year, month = args.year, args.month
    else:
        year, month = _previous_month(today)

    ensure_dirs()
    cfg = LLMConfig.from_env()
    asyncio.run(run_monthly(year, month, cfg))
    write_index()
    print(f"✅ Monthly report generated for {year}-{month:02d}")


if __name__ == "__main__":
    main()
