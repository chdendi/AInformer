from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "publish_gh_pages.sh"


def run(
    cmd: list[str],
    cwd: Path,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env)
    if check and proc.returncode != 0:
        raise AssertionError(
            f"{cmd} -> {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["git", *args], cwd, check=check)


def init_repo(repo: Path) -> None:
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")


def write_report(repo: Path, slug: str, index: str, site_dir: str = "docs") -> None:
    site = repo / site_dir
    daily = site / "daily"
    data = site / "data" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    (daily / f"{slug}.html").write_text(f"<p>{slug}</p>", encoding="utf-8")
    report_date = f"{slug[:4]}-{slug[4:6]}-{slug[6:]}"
    (data / f"{slug}.json").write_text('{"date":"%s"}' % report_date, encoding="utf-8")
    (site / "index.html").write_text(index, encoding="utf-8")


class PublishGhPagesTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.tmp = Path(self.tmp_dir.name)
        self.repo = self.tmp / "repo"
        self.bare = self.tmp / "remote.git"
        self.repo.mkdir()
        init_repo(self.repo)
        (self.repo / "README.md").write_text("source\n", encoding="utf-8")
        (self.repo / ".gitignore").write_text("docs/\n", encoding="utf-8")
        config = self.repo / "site"
        config.mkdir()
        (config / ".nojekyll").write_text("", encoding="utf-8")
        (config / "_headers").write_text("/api/*\n  Access-Control-Allow-Origin: *\n", encoding="utf-8")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-m", "source and static site config")
        git(self.tmp, "init", "--bare", str(self.bare))
        git(self.repo, "remote", "add", "origin", str(self.bare))
        git(self.repo, "push", "-u", "origin", "main")

    def publish(self, site_dir: str = "docs", check: bool = True):
        return run(["bash", str(SCRIPT), "publish", site_dir, "test publish"], self.repo, check=check)

    def restore(self, site_dir: str = "docs", check: bool = True):
        return run(["bash", str(SCRIPT), "restore", site_dir], self.repo, check=check)

    def remote_head(self, branch: str = "gh-pages") -> str:
        return git(self.bare, "rev-parse", f"refs/heads/{branch}").stdout.strip()

    def remote_file(self, path: str) -> str:
        return git(self.bare, "show", f"refs/heads/gh-pages:{path}").stdout

    def source_state(self) -> tuple[str, ...]:
        return tuple(
            git(self.repo, *args).stdout
            for args in (
                ("symbolic-ref", "HEAD"),
                ("rev-parse", "HEAD"),
                ("ls-files", "--stage"),
                ("diff", "--cached"),
                ("diff",),
            )
        )

    def test_gitignore_ignores_all_generated_files_and_tracks_site_config(self):
        def ignored(path: str) -> bool:
            return run(
                ["git", "check-ignore", "--no-index", "-q", path], ROOT, check=False
            ).returncode == 0

        for path in (
            "docs/index.html", "docs/daily/20260101.html", "docs/data/daily/20260101.json",
            "docs/api/v1/manifest.json", "docs/.nojekyll", "docs/_headers",
        ):
            with self.subTest(path=path):
                self.assertTrue(ignored(path))
        for path in ("site/.nojekyll", "site/_headers", "site/legacy-ref"):
            with self.subTest(path=path):
                self.assertFalse(ignored(path))

    def test_publish_preserves_archives_metadata_safety_and_source_staging(self):
        write_report(self.repo, "20260101", "old-index")
        (self.repo / "docs" / ".git").mkdir()
        (self.repo / "docs" / ".git" / "config").write_text("private metadata", encoding="utf-8")
        (self.repo / "docs" / "daily" / ".git").write_text("gitdir: private", encoding="utf-8")
        readme = self.repo / "README.md"
        readme.write_text("staged source change\n", encoding="utf-8")
        git(self.repo, "add", "README.md")
        readme.write_text("unstaged source change\n", encoding="utf-8")
        source_before = self.source_state()
        main_before = self.remote_head("main")

        self.publish()

        self.assertEqual(self.remote_file("daily/20260101.html"), "<p>20260101</p>")
        self.assertEqual(self.remote_file("index.html"), "old-index")
        self.assertEqual(self.remote_file(".nojekyll"), "")
        self.assertEqual(self.remote_file("_headers"), (self.repo / "site" / "_headers").read_text())
        paths = git(self.bare, "ls-tree", "-r", "--name-only", "gh-pages").stdout.splitlines()
        self.assertFalse(any(".git" in Path(path).parts for path in paths))
        self.assertNotIn("README.md", paths)
        self.assertNotIn("site/_headers", paths)

        shutil.rmtree(self.repo / "docs")
        write_report(self.repo, "20260102", "new-index")
        self.publish()

        self.assertEqual(self.remote_file("daily/20260101.html"), "<p>20260101</p>")
        self.assertEqual(self.remote_file("daily/20260102.html"), "<p>20260102</p>")
        self.assertEqual(self.remote_file("index.html"), "new-index")
        published = self.remote_head()
        self.publish()
        self.assertEqual(self.remote_head(), published, "Identical publishing must not create a commit")

        shutil.rmtree(self.repo / "docs")
        self.restore()
        self.assertTrue((self.repo / "docs" / "data" / "daily" / "20260101.json").exists())
        self.assertTrue((self.repo / "docs" / "data" / "daily" / "20260102.json").exists())
        self.assertEqual((self.repo / "docs" / "index.html").read_text(), "new-index")
        self.assertEqual(self.source_state(), source_before)
        self.assertEqual(self.remote_head("main"), main_before)

    def test_restore_seeds_deleted_history_when_gh_pages_is_missing(self):
        write_report(self.repo, "20260101", "archived-index")
        git(self.repo, "add", "-f", "docs")
        git(self.repo, "commit", "-m", "historical generated reports")
        git(self.repo, "rm", "-r", "docs")
        git(self.repo, "commit", "-m", "stop tracking generated reports")
        git(self.repo, "push", "origin", "main")
        source_before = self.source_state()
        main_before = self.remote_head("main")

        self.restore()

        self.assertEqual((self.repo / "docs" / "index.html").read_text(), "archived-index")
        self.assertTrue((self.repo / "docs" / "data" / "daily" / "20260101.json").exists())
        self.assertEqual(self.source_state(), source_before)
        self.assertEqual(self.remote_head("main"), main_before)
        self.assertEqual(git(self.bare, "show-ref", "--verify", "--quiet", "refs/heads/gh-pages", check=False).returncode, 1)

    def test_legacy_ref_bridges_existing_branch_without_overwriting_current_reports(self):
        write_report(self.repo, "20260101", "branch-index")
        branch_report = self.repo / "docs" / "daily" / "20260101.html"
        branch_report.write_text("branch report wins", encoding="utf-8")
        self.publish()

        write_report(self.repo, "20260101", "legacy-index")
        branch_report.write_text("legacy report loses", encoding="utf-8")
        write_report(self.repo, "20260102", "legacy-index")
        git(self.repo, "add", "-f", "docs")
        git(self.repo, "commit", "-m", "newer legacy archive on main")
        legacy_ref = git(self.repo, "rev-parse", "HEAD").stdout.strip()
        git(self.repo, "rm", "-r", "docs")
        (self.repo / "site" / "legacy-ref").write_text(legacy_ref + "\n", encoding="utf-8")
        git(self.repo, "add", "site/legacy-ref")
        git(self.repo, "commit", "-m", "pin legacy archive before removing generated files")
        git(self.repo, "push", "origin", "main")
        main_before = self.remote_head("main")

        self.restore()

        self.assertEqual((self.repo / "docs" / "daily" / "20260101.html").read_text(), "branch report wins")
        self.assertTrue((self.repo / "docs" / "data" / "daily" / "20260102.json").exists())
        self.assertEqual((self.repo / "docs" / "index.html").read_text(), "branch-index")
        self.publish()
        self.assertEqual(self.remote_file("daily/20260101.html"), "branch report wins")
        self.assertEqual(self.remote_file("daily/20260102.html"), "<p>20260102</p>")
        self.assertEqual(self.remote_head("main"), main_before)

    def test_stale_restored_site_cannot_roll_back_a_newer_branch_index(self):
        write_report(self.repo, "20260101", "original-index")
        self.publish()
        self.restore("stale-site")
        self.restore()
        write_report(self.repo, "20260102", "newest-index")
        self.publish()
        current_head = self.remote_head()
        main_before = self.remote_head("main")

        stale = self.publish("stale-site", check=False)

        self.assertEqual(self.remote_file("index.html"), "newest-index")
        self.assertEqual(self.remote_file("daily/20260102.html"), "<p>20260102</p>")
        if stale.returncode != 0:
            self.assertEqual(self.remote_head(), current_head, "Rejected stale publishing must not advance the branch")
            self.assertTrue(stale.stderr.strip(), "A stale-input rejection should explain the error")
        self.assertEqual(self.remote_head("main"), main_before)

    def test_publish_rejects_empty_or_incomplete_input_without_advancing_any_ref(self):
        write_report(self.repo, "20260101", "valid-index")
        self.publish()
        current_head = self.remote_head()
        main_before = self.remote_head("main")
        source_before = self.source_state()

        for case in ("empty", "index-only", "daily-json-only"):
            with self.subTest(case=case):
                shutil.rmtree(self.repo / "docs")
                site = self.repo / "docs"
                site.mkdir()
                if case == "index-only":
                    (site / "index.html").write_text("incomplete", encoding="utf-8")
                elif case == "daily-json-only":
                    data = site / "data" / "daily"
                    data.mkdir(parents=True)
                    (data / "20260102.json").write_text('{"date":"2026-01-02"}', encoding="utf-8")

                result = self.publish(check=False)

                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr.strip())
                self.assertEqual(self.remote_head(), current_head)
                self.assertEqual(self.remote_head("main"), main_before)
                self.assertEqual(self.source_state(), source_before)

    def test_restore_fails_on_fetch_error_even_with_local_history(self):
        write_report(self.repo, "20260101", "local-index")
        source_before = self.source_state()
        main_before = self.remote_head("main")
        git(self.repo, "remote", "set-url", "origin", str(self.tmp / "unavailable.git"))

        result = self.restore(check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr.strip())
        self.assertEqual((self.repo / "docs" / "index.html").read_text(), "local-index")
        self.assertEqual(self.source_state(), source_before)
        self.assertEqual(self.remote_head("main"), main_before)

    def test_restore_fails_when_no_daily_history_is_recoverable(self):
        for branch_exists in (False, True):
            with self.subTest(branch_exists=branch_exists):
                if branch_exists:
                    git(self.repo, "push", "origin", "main:gh-pages")
                source_before = self.source_state()
                main_before = self.remote_head("main")

                result = self.restore(check=False)

                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr.strip())
                self.assertEqual(self.source_state(), source_before)
                self.assertEqual(self.remote_head("main"), main_before)

    def test_publish_cannot_target_the_source_branch(self):
        write_report(self.repo, "20260101", "valid-index")
        source_before = self.source_state()
        main_before = self.remote_head("main")

        result = run(
            ["bash", str(SCRIPT), "publish", "docs", "unsafe destination"],
            self.repo,
            check=False,
            env={**os.environ, "GH_PAGES_BRANCH": "main"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.remote_head("main"), main_before)
        self.assertEqual(self.source_state(), source_before)
