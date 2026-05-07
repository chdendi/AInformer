from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime

from .config import LLMConfig, ensure_dirs, tz
from .render.index_page import write_index
from .summarize.yearly import run_yearly

log = logging.getLogger("ainformer.yearly")


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=0)
    args = p.parse_args()

    today = datetime.now(tz()).date()
    year = args.year if args.year else today.year - 1

    ensure_dirs()
    cfg = LLMConfig.from_env()
    asyncio.run(run_yearly(year, cfg))
    write_index()
    print(f"✅ Yearly report generated for {year}")


if __name__ == "__main__":
    main()
