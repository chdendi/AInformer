#!/usr/bin/env bash
# Publish the generated site to the gh-pages branch, preserving history.
#
# The site root on gh-pages is the contents of docs/ (index.html, daily/,
# data/, ...). main keeps the generator source; reports live on gh-pages.
#
#   scripts/publish_gh_pages.sh restore [site_dir]
#       Copy the current gh-pages tree into site_dir. Files already in
#       site_dir stay in place, so a newer local report is not deleted.
#       When the branch does not exist yet, seed site_dir from the newest
#       commit that still contains docs/index.html.
#
#   scripts/publish_gh_pages.sh publish [site_dir] [commit_message]
#       Overlay site_dir onto gh-pages and push. Files already on the branch
#       are left in place when this run did not regenerate them, so past
#       daily/monthly/yearly reports and dedupe JSON survive.
#
# GH_PAGES_REMOTE (default origin) and GH_PAGES_BRANCH (default gh-pages)
# choose the destination. The push is never forced.

set -euo pipefail

export GIT_TERMINAL_PROMPT=0

usage() {
  echo "usage: $0 restore|publish [site_dir] [commit_message]" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

ACTION=$1
shift

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

SITE_REL=docs
if [[ $# -ge 1 ]]; then
  SITE_REL=$1
  shift
fi
COMMIT_MSG="chore: publish site to gh-pages"
if [[ $# -ge 1 ]]; then
  COMMIT_MSG=$1
fi

REMOTE=${GH_PAGES_REMOTE:-origin}
BRANCH=${GH_PAGES_BRANCH:-gh-pages}

mkdir -p "$SITE_REL"
SITE_DIR=$(cd "$SITE_REL" && pwd)

# Cleaned up on exit so a failed publish does not leave a worktree behind.
WORK_TO_CLEAN=""
ORPHAN_TO_CLEAN=""

cleanup_publish() {
  if [[ -n "${WORK_TO_CLEAN}" ]]; then
    git worktree remove --force "$WORK_TO_CLEAN" >/dev/null 2>&1 || rm -rf "$WORK_TO_CLEAN"
    git worktree prune >/dev/null 2>&1 || true
    WORK_TO_CLEAN=""
  fi
  if [[ -n "${ORPHAN_TO_CLEAN}" ]]; then
    git branch -D "$ORPHAN_TO_CLEAN" >/dev/null 2>&1 || true
    ORPHAN_TO_CLEAN=""
  fi
}
trap cleanup_publish EXIT

# Return 0 when the branch exists, 2 when it does not, 1 on any other error.
# Invoke with set +e; a missing branch is not a fatal shell error.
remote_branch_state() {
  local err rc
  err=$(mktemp)
  git fetch --quiet --depth=1 "$REMOTE" \
    "+refs/heads/${BRANCH}:refs/remotes/${REMOTE}/${BRANCH}" \
    >/dev/null 2>"$err"
  rc=$?
  if [[ $rc -eq 0 ]]; then
    rm -f "$err"
    return 0
  fi
  if grep -q "couldn't find remote ref" "$err"; then
    rm -f "$err"
    return 2
  fi
  cat "$err" >&2
  rm -f "$err"
  return 1
}

read_branch_state() {
  local rc
  set +e
  remote_branch_state
  rc=$?
  set -e
  if [[ $rc -eq 1 ]]; then
    echo "Failed to fetch ${REMOTE}/${BRANCH}" >&2
    exit 1
  fi
  BRANCH_STATE=$rc
}

seed_from_history() {
  local rev="" tmp c
  if [[ -f "$(git rev-parse --git-path shallow)" ]]; then
    git fetch --unshallow "$REMOTE" || git fetch --deepen=2000 "$REMOTE" || true
  fi

  while read -r c; do
    [[ -z "$c" ]] && continue
    if git cat-file -e "${c}:docs/index.html" 2>/dev/null; then
      rev=$c
      break
    fi
  done < <(git log --all --format=%H -- docs/index.html || true)

  if [[ -z "$rev" ]]; then
    echo "No historical docs/ tree found; continuing without seeded history"
    return 0
  fi

  echo "Seeding ${SITE_DIR} from ${rev}:docs"
  tmp=$(mktemp -d)
  git archive "$rev" docs | tar -x -C "$tmp"
  mkdir -p "$SITE_DIR"
  tar -C "$tmp/docs" -cf - . | tar -C "$SITE_DIR" -xf -
  rm -rf "$tmp"
}

restore_site() {
  read_branch_state
  if [[ "$BRANCH_STATE" -eq 2 ]]; then
    if compgen -G "$SITE_DIR/data/daily/*.json" >/dev/null; then
      echo "${REMOTE}/${BRANCH} is absent; keeping reports already in ${SITE_DIR}"
      return 0
    fi
    seed_from_history
    return 0
  fi

  local tmp
  tmp=$(mktemp -d)
  git archive "refs/remotes/${REMOTE}/${BRANCH}" | tar -x -C "$tmp"
  tar -C "$tmp" -cf - . | tar -C "$SITE_DIR" -xf -
  rm -rf "$tmp"

  if ! compgen -G "$SITE_DIR/data/daily/*.json" >/dev/null; then
    echo "Restored ${BRANCH} but no daily JSON history was found in ${SITE_DIR}" >&2
    return 1
  fi
  echo "Restored ${BRANCH} history into ${SITE_DIR}"
}

publish_once() {
  local count

  if [[ ! -e "$SITE_DIR/index.html" && ! -d "$SITE_DIR/data" ]]; then
    echo "Refusing to publish an empty site from ${SITE_DIR}" >&2
    return 1
  fi

  read_branch_state
  WORK_TO_CLEAN=$(mktemp -d)
  rmdir "$WORK_TO_CLEAN"

  if [[ "$BRANCH_STATE" -eq 0 ]]; then
    git worktree add --detach "$WORK_TO_CLEAN" "refs/remotes/${REMOTE}/${BRANCH}"
  else
    ORPHAN_TO_CLEAN="gh-pages-seed-$$-${RANDOM}"
    git worktree add --detach "$WORK_TO_CLEAN" HEAD
    git -C "$WORK_TO_CLEAN" checkout --orphan "$ORPHAN_TO_CLEAN"
    git -C "$WORK_TO_CLEAN" rm -rf --quiet --ignore-unmatch .
    find "$WORK_TO_CLEAN" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
  fi

  tar -C "$SITE_DIR" --exclude '.git' -cf - . | tar -C "$WORK_TO_CLEAN" -xf -

  count=0
  if [[ -d "$WORK_TO_CLEAN/data/daily" ]]; then
    count=$(find "$WORK_TO_CLEAN/data/daily" -name '*.json' | wc -l | tr -d ' ')
  fi
  echo "Publishing site with ${count} daily JSON files"
  if [[ "$BRANCH_STATE" -eq 0 && "$count" -eq 0 ]]; then
    echo "Refusing to publish without daily history while ${BRANCH} already exists" >&2
    cleanup_publish
    return 1
  fi

  git -C "$WORK_TO_CLEAN" add -A
  if git -C "$WORK_TO_CLEAN" diff --cached --quiet; then
    echo "No site changes to publish"
    cleanup_publish
    return 0
  fi

  git -C "$WORK_TO_CLEAN" \
    -c user.name="${GIT_AUTHOR_NAME:-ainformer-bot}" \
    -c user.email="${GIT_AUTHOR_EMAIL:-ainformer-bot@users.noreply.github.com}" \
    commit -m "$COMMIT_MSG"

  if git -C "$WORK_TO_CLEAN" push "$REMOTE" "HEAD:${BRANCH}"; then
    echo "Published site to ${BRANCH}"
    cleanup_publish
    return 0
  fi

  echo "Push to ${REMOTE}/${BRANCH} was rejected" >&2
  cleanup_publish
  return 10
}

publish_site() {
  local attempt rc
  for attempt in 1 2 3 4 5; do
    set +e
    publish_once
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
      return 0
    fi
    if [[ $rc -ne 10 ]]; then
      return "$rc"
    fi
    echo "Publish attempt ${attempt} failed; retrying..." >&2
    sleep $((attempt * 2))
  done
  echo "Failed to publish to ${BRANCH}" >&2
  return 1
}

case "$ACTION" in
  restore) restore_site ;;
  publish) publish_site ;;
  *)
    usage
    exit 2
    ;;
esac
