from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "publish_gh_pages.sh"


def run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
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


def write_report(repo: Path, slug: str, index: str) -> None:
    daily = repo / "docs" / "daily"
    data = repo / "docs" / "data" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    (daily / f"{slug}.html").write_text(f"<p>{slug}</p>", encoding="utf-8")
    (data / f"{slug}.json").write_text('{"date":"%s"}' % slug, encoding="utf-8")
    (repo / "docs" / "index.html").write_text(index, encoding="utf-8")
    (repo / "docs" / ".nojekyll").write_text("", encoding="utf-8")
    (repo / "docs" / "_headers").write_text("ok\n", encoding="utf-8")


class PublishGhPagesTests(unittest.TestCase):
    def test_gitignore_tracks_only_site_config(self):
        def ignored(path: str) -> bool:
            proc = run(
                ["git", "check-ignore", "--no-index", "-q", path],
                ROOT,
                check=False,
            )
            return proc.returncode == 0

        self.assertTrue(ignored("docs/index.html"))
        self.assertTrue(ignored("docs/daily/20260101.html"))
        self.assertTrue(ignored("docs/data/daily/20260101.json"))
        self.assertTrue(ignored("docs/api/v1/manifest.json"))
        self.assertFalse(ignored("docs/.nojekyll"))
        self.assertFalse(ignored("docs/_headers"))

    def test_publish_preserves_history_and_restore_brings_it_back(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            repo = tmp / "repo"
            bare = tmp / "remote.git"
            repo.mkdir()
            init_repo(repo)
            write_report(repo, "20260101", "old-index")
            (repo / "README.md").write_text("source\n", encoding="utf-8")
            git(repo, "add", ".")
            git(repo, "commit", "-m", "init")
            git(tmp, "init", "--bare", str(bare))
            git(repo, "remote", "add", "origin", str(bare))
            git(repo, "push", "-u", "origin", "main")

            first = run(
                ["bash", str(SCRIPT), "publish", "docs", "seed"],
                repo,
            )
            self.assertIn("Published site to gh-pages", first.stdout)

            check = tmp / "check"
            git(tmp, "clone", "--branch", "gh-pages", str(bare), str(check))
            self.assertEqual((check / "daily" / "20260101.html").read_text(encoding="utf-8"), "<p>20260101</p>")
            self.assertEqual((check / "index.html").read_text(encoding="utf-8"), "old-index")
            self.assertTrue((check / ".nojekyll").exists())
            self.assertFalse((check / "README.md").exists())
            self.assertFalse((check / "src").exists())

            (repo / "docs" / "daily" / "20260101.html").unlink()
            (repo / "docs" / "data" / "daily" / "20260101.json").unlink()
            write_report(repo, "20260102", "new-index")
            second = run(
                ["bash", str(SCRIPT), "publish", "docs", "day 2"],
                repo,
            )
            self.assertIn("Published site to gh-pages", second.stdout)

            git(check, "pull", "--ff-only")
            self.assertEqual((check / "daily" / "20260101.html").read_text(encoding="utf-8"), "<p>20260101</p>")
            self.assertTrue((check / "data" / "daily" / "20260101.json").exists())
            self.assertEqual((check / "daily" / "20260102.html").read_text(encoding="utf-8"), "<p>20260102</p>")
            self.assertEqual((check / "index.html").read_text(encoding="utf-8"), "new-index")
            self.assertFalse((check / "README.md").exists())

            for child in (repo / "docs").iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    subprocess.run(["rm", "-rf", str(child)], check=True)
            restored = run(["bash", str(SCRIPT), "restore"], repo)
            self.assertIn("Restored gh-pages history", restored.stdout)
            self.assertTrue((repo / "docs" / "daily" / "20260101.html").exists())
            self.assertTrue((repo / "docs" / "data" / "daily" / "20260102.json").exists())
            self.assertEqual((repo / "docs" / "index.html").read_text(encoding="utf-8"), "new-index")

    def test_restore_seeds_deleted_history_when_gh_pages_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            repo = tmp / "repo"
            bare = tmp / "remote.git"
            repo.mkdir()
            init_repo(repo)
            write_report(repo, "20260101", "archived-index")
            git(repo, "add", ".")
            git(repo, "commit", "-m", "add reports")
            git(repo, "rm", "-r", "docs")
            git(repo, "commit", "-m", "drop generated docs")
            git(tmp, "init", "--bare", str(bare))
            git(repo, "remote", "add", "origin", str(bare))
            git(repo, "push", "-u", "origin", "main")

            restored = run(["bash", str(SCRIPT), "restore"], repo)
            self.assertIn("Seeding", restored.stdout)
            self.assertEqual(
                (repo / "docs" / "index.html").read_text(encoding="utf-8"),
                "archived-index",
            )
            self.assertTrue((repo / "docs" / "data" / "daily" / "20260101.json").exists())

    def test_publish_rejects_an_empty_site(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            repo = tmp / "repo"
            bare = tmp / "remote.git"
            repo.mkdir()
            init_repo(repo)
            (repo / "README.md").write_text("source\n", encoding="utf-8")
            git(repo, "add", ".")
            git(repo, "commit", "-m", "init")
            git(tmp, "init", "--bare", str(bare))
            git(repo, "remote", "add", "origin", str(bare))
            git(repo, "push", "-u", "origin", "main")

            proc = run(["bash", str(SCRIPT), "publish", "docs", "empty"], repo, check=False)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Refusing to publish an empty site", proc.stderr)
            branches = git(repo, "ls-remote", "--heads", "origin")
            self.assertNotIn("gh-pages", branches.stdout)
