#!/usr/bin/env bash
# Usage: bash scripts/publish_gh_pages.sh restore|publish [site_dir] [commit_message]
# Restore or publish without changing the source branch or its staging area.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/site_publish.py" "$@"
