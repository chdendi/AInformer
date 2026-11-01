"""Restore and publish gh-pages using Git and the Python standard library.

Publishing uses a temporary Git index and commit-tree. The source checkout,
its staging area, and its branch are never modified. Pushes are fast-forward
only; a changed restore baseline requires restoring and generating again.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile


def git(root: Path, *args: str, env: dict | None = None, check: bool = True):
    return subprocess.run(
        ["git", *args], cwd=root, env=env, check=check, capture_output=True,
    )


def copy_site(source: Path, target: Path) -> None:
    """Overlay site files, keeping archived reports absent from the source."""
    def ignore(directory, names):
        for name in names:
            if name != ".git" and (Path(directory) / name).is_symlink():
                raise ValueError("Site archives must not contain symbolic links")
        return {".git"} & set(names)

    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True, ignore=ignore)


def copy_config(root: Path, target: Path) -> None:
    for name in (".nojekyll", "_headers"):
        config = root / "site" / name
        if config.is_file():
            shutil.copy2(config, target / name)


def extract(root: Path, revision: str, target: Path, prefix: str = "") -> None:
    args = ["archive", revision]
    if prefix:
        args.append(prefix)
    archive = git(root, *args).stdout
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
                raise ValueError("Unsafe path in site archive")
            if prefix:
                path = path.relative_to(prefix)
            dest = target.joinpath(*path.parts)
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, dest.open("wb") as output:
                    shutil.copyfileobj(source, output)
            else:
                raise ValueError("Site archives must contain only regular files")


def fetch_site(root: Path, remote: str, branch: str) -> str | None:
    result = git(root, "ls-remote", "--exit-code", "--heads", remote,
                 f"refs/heads/{branch}", check=False)
    if result.returncode == 2:
        return None
    result.check_returncode()
    ref = f"refs/remotes/{remote}/{branch}"
    git(root, "fetch", "--quiet", "--depth=1", remote,
        f"+refs/heads/{branch}:{ref}")
    return git(root, "rev-parse", ref).stdout.decode().strip()


def validate_site(site: Path) -> None:
    if not (site / "index.html").is_file() or not (site / "index.html").stat().st_size:
        raise ValueError("Refusing to publish an empty site: index.html is required")
    reports = list((site / "data" / "daily").glob("*.json"))
    if not reports:
        raise ValueError("No daily JSON history found; restore the archive before publishing")
    for report in reports:
        data = json.loads(report.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data.get("date"):
            raise ValueError(f"Invalid daily report: {report.name}")
        if not (site / "daily" / f"{report.stem}.html").is_file():
            raise ValueError(f"Missing HTML for daily report: {report.name}")


def legacy_revision(root: Path, remote: str, site_revision: str | None) -> str | None:
    marker = root / "site" / "legacy-ref"
    if marker.exists():
        revision = marker.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("site/legacy-ref must contain a full historical commit SHA")
        if git(root, "cat-file", "-e", f"{revision}:docs/index.html", check=False).returncode:
            git(root, "fetch", "--quiet", "--depth=1", remote, revision)
        git(root, "cat-file", "-e", f"{revision}:docs/index.html")
        return revision
    if site_revision:
        return None
    # Compatibility with repositories that have not recorded a migration ref.
    revisions = git(root, "log", "--all", "--format=%H", "--", "docs/index.html")
    for revision in revisions.stdout.decode().splitlines():
        if git(root, "cat-file", "-e", f"{revision}:docs/index.html", check=False).returncode == 0:
            return revision
    return None


def restore(root: Path, site: Path, remote: str, branch: str, baseline: Path) -> None:
    revision = fetch_site(root, remote, branch)
    legacy = legacy_revision(root, remote, revision)
    with tempfile.TemporaryDirectory(prefix="ainformer-restore-") as temp:
        staged = Path(temp) / "site"
        staged.mkdir()
        if legacy:
            extract(root, legacy, staged, "docs")
            print(f"Seeding historical reports from {legacy}:docs")
        if revision:
            extract(root, revision, staged)
        copy_config(root, staged)
        copy_site(staged, site)
    validate_site(site)
    baseline.write_text(json.dumps({"revision": revision}), encoding="utf-8")
    print(f"Restored {branch} history into {site}")


def publish(root: Path, site: Path, remote: str, branch: str, baseline: Path, message: str) -> None:
    validate_site(site)
    revision = fetch_site(root, remote, branch)
    if baseline.exists() and json.loads(baseline.read_text(encoding="utf-8"))["revision"] != revision:
        raise ValueError("Site archive changed since restore; restore and regenerate before publishing")
    with tempfile.TemporaryDirectory(prefix="ainformer-publish-") as temp:
        staged = Path(temp) / "site"
        staged.mkdir()
        if revision:
            extract(root, revision, staged)
        copy_site(site, staged)
        copy_config(root, staged)
        validate_site(staged)
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp) / "index"))
        gitdir = git(root, "rev-parse", "--absolute-git-dir").stdout.decode().strip()
        tree_args = (f"--git-dir={gitdir}", f"--work-tree={staged}")
        git(staged, *tree_args, "read-tree", "--empty", env=env)
        git(staged, *tree_args, "add", "--all", "--force", env=env)
        tree = git(staged, *tree_args, "write-tree", env=env).stdout.decode().strip()
        if revision and tree == git(root, "rev-parse", f"{revision}^{{tree}}").stdout.decode().strip():
            print("No site changes to publish")
            return
        args = ["-c", "user.name=ainformer-bot", "-c",
                "user.email=ainformer-bot@users.noreply.github.com", "commit-tree", tree]
        if revision:
            args.extend(["-p", revision])
        args.extend(["-m", message])
        commit = git(root, *args).stdout.decode().strip()
        git(root, "push", remote, f"{commit}:refs/heads/{branch}")
    baseline.write_text(json.dumps({"revision": commit}), encoding="utf-8")
    print(f"Published site to {branch}: {commit}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("restore", "publish"))
    parser.add_argument("site_dir", nargs="?", default="docs")
    parser.add_argument("message", nargs="?", default="chore: publish site to gh-pages")
    args = parser.parse_args()
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").stdout.decode().strip())
    remote = os.environ.get("GH_PAGES_REMOTE", "origin")
    branch = os.environ.get("GH_PAGES_BRANCH", "gh-pages")
    source_branch = git(root, "branch", "--show-current").stdout.decode().strip()
    if branch in {"main", "master", source_branch}:
        raise ValueError("Refusing to publish to a source branch")
    git(root, "check-ref-format", "--branch", branch)
    site = (root / args.site_dir).resolve()
    if site == root or root.is_relative_to(site) or site.is_relative_to(root / ".git"):
        raise ValueError("Site directory must be separate from source and Git metadata")
    gitdir = Path(git(root, "rev-parse", "--absolute-git-dir").stdout.decode().strip())
    key = hashlib.sha256(f"{site}:{remote}:{branch}".encode()).hexdigest()[:16]
    baseline = gitdir / f"ainformer-site-{key}.json"
    if args.action == "restore":
        restore(root, site, remote, branch, baseline)
    else:
        publish(root, site, remote, branch, baseline, args.message)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Site publishing failed: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr.decode(errors="replace"), file=sys.stderr)
        sys.exit(1)
