#!/usr/bin/env python3
"""Local CI daemon for vla-factory (Issue #7).

Single process on the local GPU machine. Polls GitCode for open PRs,
runs ``ci/run_gate.sh`` for any with a new head SHA, and reports results
as a PR comment.

No VPS, no webhook — all polling is direct to the GitCode API. The only
network requirement is outbound HTTPS to gitcode.com.

Covers **all PR authors**, not just the fork owner — the GitCode PR list
API returns every open PR regardless of who opened it.

Config (environment variables)::

    VLAF_GITCODE_TOKEN    GitCode personal access token (required)
    VLAF_UPSTREAM         default openeuler/vla-factory
    VLAF_BASE_DIR         parent dir for the CI checkout (auto-cloned if missing)
    VLAF_ENV_BASE         /home/you/envs/base/bin/python
    VLAF_ENV_ACT          /home/you/envs/act/bin/python   (optional)
    VLAF_ENV_PI           /home/you/envs/pi/bin/python     (optional)
    VLAF_POLL_INTERVAL    seconds between polls (default 30)
    VLAF_DB_PATH          seen-SHA tracking DB (default ~/.vlaf_ci.db)

Run::

    VLAF_GITCODE_TOKEN=... \
    VLAF_BASE_DIR=$HOME/vla-factory-ci \
    VLAF_ENV_BASE=$HOME/envs/base/bin/python \
    python3 ci/runner/daemon.py
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Config ───────────────────────────────────────────────────────────

GITCODE_TOKEN = os.environ.get("VLAF_GITCODE_TOKEN", "")
UPSTREAM = os.environ.get("VLAF_UPSTREAM", "openeuler/vla-factory")
BASE_DIR = Path(os.environ.get("VLAF_BASE_DIR", str(Path.home() / "vla-factory-ci")))
CLONE_URL = f"https://gitcode.com/{UPSTREAM}.git"
POLL_INTERVAL = int(os.environ.get("VLAF_POLL_INTERVAL", "30"))
DB_PATH = os.environ.get("VLAF_DB_PATH", str(Path.home() / ".vlaf_ci.db"))
API_BASE = "https://api.gitcode.com/api/v5"

ENVS: list[tuple[str, str]] = []
for label in ("base", "act", "pi"):
    py = os.environ.get(f"VLAF_ENV_{label.upper()}")
    if py:
        ENVS.append((label, py))
if not ENVS:
    ENVS = [("base", sys.executable)]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vlaf-ci")

sys.path.insert(0, str(Path(__file__).parent))
from pr_reporter import edit_comment, format_result, format_running, post_comment  # noqa: E402
from parse_results import parse_report_dir, tier_line  # noqa: E402


# ── SQLite: track seen PR SHAs ───────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_prs (
    pr_number   INTEGER NOT NULL,
    head_sha    TEXT    NOT NULL,
    status      TEXT,
    comment_id  INTEGER,
    created_at  TEXT    NOT NULL,
    PRIMARY KEY (pr_number, head_sha)
);
"""


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def is_seen(pr_number: int, head_sha: str) -> bool:
    conn = db()
    try:
        return conn.execute(
            "SELECT 1 FROM seen_prs WHERE pr_number = ? AND head_sha = ?",
            (pr_number, head_sha),
        ).fetchone() is not None
    finally:
        conn.close()


def mark_seen(pr_number: int, head_sha: str, status: str, comment_id: int | None) -> None:
    conn = db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO seen_prs (pr_number, head_sha, status, comment_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (pr_number, head_sha, status, comment_id, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        conn.commit()
    finally:
        conn.close()


# ── GitCode API ──────────────────────────────────────────────────────

def fetch_open_prs() -> list[dict]:
    """Return all open PRs on the upstream repo (any author)."""
    url = f"{API_BASE}/repos/{UPSTREAM}/pulls"
    try:
        resp = requests.get(url, params={"state": "open", "per_page": 100,
                                         "access_token": GITCODE_TOKEN},
                            timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning("failed to fetch open PRs: %s", e)
        return []


# ── Git operations ───────────────────────────────────────────────────

def ensure_repo() -> Path:
    """Clone the repo into BASE_DIR if not present, return the path."""
    if (BASE_DIR / ".git").exists():
        return BASE_DIR
    BASE_DIR.parent.mkdir(parents=True, exist_ok=True)
    log.info("cloning %s → %s ...", CLONE_URL, BASE_DIR)
    subprocess.run(["git", "clone", "--quiet", CLONE_URL, str(BASE_DIR)],
                   check=True, capture_output=True, timeout=300)
    log.info("clone done.")
    return BASE_DIR


def sync_pr(repo_dir: Path, pr_number: int) -> str | None:
    """Fetch the PR head ref and detach HEAD to it. Returns the SHA or None."""
    for ref in (f"refs/pull/{pr_number}/head", f"refs/merge-requests/{pr_number}/head"):
        try:
            subprocess.run(["git", "fetch", "--quiet", "--force", "origin", ref],
                           cwd=repo_dir, check=True, capture_output=True, timeout=60)
            break
        except subprocess.CalledProcessError:
            continue
    else:
        log.error("fetch failed for PR #%d", pr_number)
        return None

    subprocess.run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"],
                   cwd=repo_dir, check=True)
    subprocess.run(["git", "clean", "-qfd"], cwd=repo_dir, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir).decode().strip()


# ── Test execution ───────────────────────────────────────────────────

# All available tiers: (pytest marker, junit stem)
ALL_TIERS = [
    ("not l1 and not l2 and not l3", "l0"),
    ("l1", "l1"),
    ("l2", "l2"),
]

# Default tier assignment per environment label.
# Rationale:
#   base → L0 only (L0 needs no model deps; running it once is enough)
#   act  → L1 + L2  (lerobot parity + CPU overfit smoke)
#   pi   → L1 only  (openpi parity; act can't run these, pi picks them up)
# L1 tests self-skip (importorskip) when their upstream is absent, so
# running L1 in both act and pi is safe — each env covers a disjoint subset.
DEFAULT_ENV_TIERS: dict[str, list[str]] = {
    "base": ["l0"],
    "act": ["l1", "l2"],
    "pi": ["l1"],
}


def env_tiers(label: str) -> list[tuple[str, str]]:
    """Return the (marker, name) pairs this environment should run.

    Per-env override via environment variable ``VLAF_ENV_<LABEL>_TIERS``::

        VLAF_ENV_BASE_TIERS="l0,l1"   # also run L1 in base
        VLAF_ENV_ACT_TIERS="l1"       # skip L2 in act
    """
    tier_map = {name: (marker, name) for marker, name in ALL_TIERS}
    override = os.environ.get(f"VLAF_ENV_{label.upper()}_TIERS", "")
    if override:
        names = [n.strip() for n in override.split(",") if n.strip()]
    else:
        names = DEFAULT_ENV_TIERS.get(label, ["l0", "l1", "l2"])
    return [tier_map[n] for n in names if n in tier_map]


def run_gate(repo_dir: Path, label: str, python_bin: str, report_dir: Path) -> bool:
    """Run the assigned tiers for this env directly via pytest.

    Writes ``<tier>.xml`` junit files into *report_dir*.

    pytest exit codes: 0 = pass, 5 = no tests collected (treated as pass),
    anything else = failure.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    tiers = env_tiers(label)
    all_ok = True
    for marker, name in tiers:
        cmd = [python_bin, "-m", "pytest", str(repo_dir / "test"), "-q", "-m", marker,
               f"--junitxml={report_dir}/{name}.xml"]
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, timeout=1200)
        if result.returncode not in (0, 5):
            log.warning("  [%s] tier %s failed (exit %d): %s", label, name,
                        result.returncode, result.stderr.decode(errors="replace")[-300:])
            all_ok = False
    return all_ok


def collect_results(report_base: Path) -> list[dict]:
    rows = []
    for label, _ in ENVS:
        env_dir = report_base / label
        summaries = parse_report_dir(env_dir) if env_dir.exists() else {}
        rows.append({
            "env": label,
            "l0": tier_line(summaries.get("l0", {})),
            "l1": tier_line(summaries.get("l1", {})),
            "l2": tier_line(summaries.get("l2", {})),
            "ok": all(s.get("ok", False) for s in summaries.values()),
            "time": f"{sum(s.get('time', 0) for s in summaries.values()):.0f}s",
        })
    return rows


# ── Process one PR ───────────────────────────────────────────────────

def process_pr(pr: dict) -> bool:
    pr_number = pr["number"]
    head = pr.get("head", {})
    branch = head.get("ref", "?")
    sha = head.get("sha", "")
    user = head.get("user")
    author = user.get("login") if isinstance(user, dict) else (user or "?")

    log.info("=== PR #%d by %s  branch=%s  sha=%s ===", pr_number, author, branch, sha[:12])
    start = time.time()

    # 1. Post "running" comment.
    comment_id = None
    try:
        comment_id = post_comment(pr_number, format_running(branch, sha))
    except Exception as e:
        log.error("failed to post comment: %s", e)

    # 2. Fetch + checkout.
    actual_sha = sync_pr(BASE_DIR, pr_number)
    if actual_sha is None:
        if comment_id:
            edit_comment(comment_id, f"## CI — fetch failed\n\nPR #{pr_number} merge ref 无法拉取,可能有冲突。")
        return False
    sha = actual_sha

    # 3. Run gate in each environment.
    report_base = BASE_DIR / "ci" / "_reports"
    if report_base.exists():
        shutil.rmtree(report_base)

    all_ok = True
    for label, python_bin in ENVS:
        log.info("  [%s] running gate ...", label)
        if not run_gate(BASE_DIR, label, python_bin, report_base / label):
            all_ok = False

    elapsed = time.time() - start

    # 4. Post results.
    rows = collect_results(report_base)
    body = format_result(branch, sha, rows, all_ok, elapsed)
    if comment_id:
        try:
            edit_comment(comment_id, body)
        except Exception as e:
            log.error("failed to edit comment: %s", e)
    else:
        try:
            post_comment(pr_number, body)
        except Exception:
            pass

    log.info("PR #%d done: %s (%.0fs)", pr_number, "PASS" if all_ok else "FAIL", elapsed)
    return all_ok


# ── Main loop ────────────────────────────────────────────────────────

def main():
    if not GITCODE_TOKEN:
        sys.exit("VLAF_GITCODE_TOKEN is required")

    repo = ensure_repo()

    log.info("started: upstream=%s  repo=%s  envs=%s  poll=%ds",
             UPSTREAM, repo, [e[0] for e in ENVS], POLL_INTERVAL)

    while True:
        prs = fetch_open_prs()
        new_count = 0

        for pr in prs:
            head = pr.get("head", {})
            pr_number = pr.get("number")
            sha = head.get("sha")
            if not pr_number or not sha:
                continue
            if is_seen(pr_number, sha):
                continue

            new_count += 1
            user = head.get("user")
            author = user.get("login") if isinstance(user, dict) else (user or "?")
            log.info("new: PR #%d by %s  sha=%s", pr_number, author, sha[:12])

            try:
                success = process_pr(pr)
                mark_seen(pr_number, sha, "done" if success else "failed", None)
            except Exception as e:
                log.exception("PR #%d crashed: %s", pr_number, e)
                mark_seen(pr_number, sha, "crashed", None)

        if new_count == 0:
            log.debug("no new PRs (scanned %d)", len(prs))
        else:
            log.info("processed %d new PR(s)", new_count)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
