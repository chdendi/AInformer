from __future__ import annotations

import logging
import os

from .config import ensure_dirs
from .render.index_page import write_index


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_dirs()
    write_index()
    print("✅ index.html regenerated")


if __name__ == "__main__":
    main()
