from __future__ import annotations

import logging
import os

from .build_api import build_api
from .config import ensure_dirs
from .render.index_page import write_index


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_dirs()
    write_index()
    build_api()
    print("✅ index.html + API regenerated")


if __name__ == "__main__":
    main()
