#!/usr/bin/env python3
"""Local CI daemon for vla-factory (Issue #7).

Single process on the local GPU machine. Polls GitCode for open PRs,
runs the tiered pytest gate for any PR with a new head SHA, and reports results
as a PR comment.

Tasks execute asynchronously on thread pools (see VLAF_CI_WORKERS /
VLAF_CMD_WORKERS): each new (PR, SHA) gets a dedicated checkout under
``<VLAF_BASE_DIR>/pr-<N>-<sha8>``, so concurrent CI runs never share a
worktree, and comment commands run on their own pool so a long review
never blocks CI (or vice versa). The main loop only polls and dispatches.

No VPS, no webhook — all polling is direct to the GitCode API. The only
network requirement is outbound HTTPS to gitcode.com.

Covers **all PR authors**, not just the fork owner — the GitCode PR list
API returns every open PR regardless of who opened it.

Config (environment variables)::

    VLAF_GITCODE_TOKEN    GitCode personal access token (required)
    VLAF_UPSTREAM         default openeuler/vla-factory
    VLAF_BASE_DIR         parent dir for the CI checkouts (auto-cloned if missing)
    VLAF_ENV_BASE         /home/you/envs/base/bin/python  (required; L0)
    VLAF_ENV_ACT          /home/you/envs/act/bin/python   (required; L1/L2)
    VLAF_ENV_PI           /home/you/envs/pi/bin/python    (required; L1)
    HF_TOKEN              Hugging Face token authorized for PaliGemma (required; PI L1)
    VLAF_POLL_INTERVAL    seconds between polls (default 30)
    VLAF_DB_PATH          seen-SHA tracking DB (default ~/.vlaf_ci.db)
    VLAF_CI_WORKERS       concurrent CI runs (default 5)
    VLAF_CMD_WORKERS      concurrent comment-command tasks (default 5)
    VLAF_TIER_TIMEOUT     per-tier pytest timeout in seconds (default 1200)
    VLAF_MAX_RETRIES      dispatch attempts for a crashed (pr, sha) (default 3)
    VLAF_AGENT_CMD        headless agent for /vla-factory review, run via
                          sh -c; e.g. 'claude -p "$(cat {prompt})" > {output}'
    VLAF_SKILL_DIR        vlafactory-code-review dir override; by default the
                          skills submodule (github.com/2012geek/skills) is
                          init'd / synced to latest main before each review

PR comment commands (any commenter, any open PR)::

    /vla-factory help     list commands
    /vla-factory review   run the vlafactory-code-review skill on the PR
    /vla-factory retest   re-run CI on the PR's current head

Run::

    VLAF_GITCODE_TOKEN=... \
    VLAF_BASE_DIR=$HOME/vla-factory-ci \
    VLAF_ENV_BASE=$HOME/envs/base/bin/python \
    VLAF_ENV_ACT=$HOME/envs/act/bin/python \
    VLAF_ENV_PI=$HOME/envs/pi/bin/python \
    HF_TOKEN=hf_... \
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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Config ───────────────────────────────────────────────────────────

GITCODE_TOKEN = os.environ.get("VLAF_GITCODE_TOKEN", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
UPSTREAM = os.environ.get("VLAF_UPSTREAM", "openeuler/vla-factory")
BASE_DIR = Path(os.environ.get("VLAF_BASE_DIR", str(Path.home() / "vla-factory-ci")))
CLONE_URL = f"https://gitcode.com/{UPSTREAM}.git"
POLL_INTERVAL = int(os.environ.get("VLAF_POLL_INTERVAL", "30"))
DB_PATH = os.environ.get("VLAF_DB_PATH", str(Path.home() / ".vlaf_ci.db"))
API_BASE = "https://api.gitcode.com/api/v5"
CI_WORKERS = int(os.environ.get("VLAF_CI_WORKERS", "5"))
CMD_WORKERS = int(os.environ.get("VLAF_CMD_WORKERS", "5"))

ENVS: list[tuple[str, str]] = []
for label in ("base", "act", "pi"):
    py = os.environ.get(f"VLAF_ENV_{label.upper()}")
    if py:
        ENVS.append((label, py))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vlaf-ci")

sys.path.insert(0, str(Path(__file__).parent))
import commands as slash_commands  # noqa: E402
from pr_reporter import edit_comment, format_result, format_running, post_comment  # noqa: E402
from parse_results import parse_report_dir, tier_line  # noqa: E402


# ── SQLite: track seen PR SHAs ───────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_prs (
    pr_number   INTEGER NOT NULL,
    head_sha    TEXT    NOT NULL,
    status      TEXT,
    comment_id  INTEGER,
    attempts    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL,
    PRIMARY KEY (pr_number, head_sha)
);
CREATE TABLE IF NOT EXISTS handled_comments (
    pr_number     INTEGER PRIMARY KEY,
    last_note_id  INTEGER NOT NULL
);
"""


def db() -> sqlite3.Connection:
    # timeout: worker threads write concurrently; wait instead of failing
    # with 'database is locked'.
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # Migration for DBs created before the attempts column existed.
    try:
        conn.execute("ALTER TABLE seen_prs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    return conn


def is_seen(pr_number: int, head_sha: str) -> bool:
    """True for SHAs that are terminal (done/failed) or already in flight.

    Rows in 'running' state were claimed by a dispatched worker and must
    not be re-submitted while that task executes. Rows left 'crashed'
    (daemon died or the task raised) are NOT seen, so the next poll
    retries them instead of treating a crash as a completed run.
    """
    conn = db()
    try:
        return conn.execute(
            "SELECT 1 FROM seen_prs WHERE pr_number = ? AND head_sha = ? "
            "AND status IN ('done', 'failed', 'running')",
            (pr_number, head_sha),
        ).fetchone() is not None
    finally:
        conn.close()


def mark_seen(pr_number: int, head_sha: str, status: str, comment_id: int | None,
              bump_attempt: bool = False) -> None:
    """Upsert a (pr, sha) row. A None comment_id preserves any stored one,
    so a crash in main() doesn't erase the id recorded by process_pr.
    bump_attempt=True increments the attempt counter — used only when the
    dispatcher claims a run, so attempts counts dispatches, not status
    transitions within one run."""
    conn = db()
    try:
        conn.execute(
            "INSERT INTO seen_prs (pr_number, head_sha, status, comment_id, attempts, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (pr_number, head_sha) DO UPDATE SET "
            "status = excluded.status, "
            "comment_id = COALESCE(excluded.comment_id, comment_id), "
            "attempts = attempts + ?",
            (pr_number, head_sha, status, comment_id, int(bump_attempt),
             datetime.now(timezone.utc).isoformat(timespec="seconds"), int(bump_attempt)),
        )
        conn.commit()
    finally:
        conn.close()


MAX_RETRIES = int(os.environ.get("VLAF_MAX_RETRIES", "3"))


def should_dispatch(pr_number: int, head_sha: str) -> bool:
    """True if this (pr, sha) still needs a CI dispatch.

    done/failed/running are covered by is_seen. A crashed run is retried
    only until VLAF_MAX_RETRIES dispatches — an SHA that keeps crashing
    (broken checkout, env problems) must not loop forever.
    """
    if is_seen(pr_number, head_sha):
        return False
    conn = db()
    try:
        row = conn.execute(
            "SELECT attempts FROM seen_prs WHERE pr_number = ? AND head_sha = ? "
            "AND status = 'crashed'",
            (pr_number, head_sha),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return True
    if row["attempts"] >= MAX_RETRIES:
        log.warning("PR #%d sha %s gave up after %d crashed attempts — not retrying",
                    pr_number, head_sha[:12], row["attempts"])
        return False
    return True


def stored_comment_id(pr_number: int, head_sha: str) -> int | None:
    """Comment id recorded by a previous (possibly crashed) run, if any."""
    conn = db()
    try:
        row = conn.execute(
            "SELECT comment_id FROM seen_prs WHERE pr_number = ? AND head_sha = ?",
            (pr_number, head_sha),
        ).fetchone()
        return row["comment_id"] if row else None
    finally:
        conn.close()


def last_handled_note(pr_number: int) -> int:
    conn = db()
    try:
        row = conn.execute(
            "SELECT last_note_id FROM handled_comments WHERE pr_number = ?",
            (pr_number,),
        ).fetchone()
        return row["last_note_id"] if row else 0
    finally:
        conn.close()


def mark_handled_note(pr_number: int, note_id: int) -> None:
    conn = db()
    try:
        conn.execute(
            "INSERT INTO handled_comments (pr_number, last_note_id) VALUES (?, ?) "
            "ON CONFLICT(pr_number) DO UPDATE SET "
            "last_note_id = MAX(last_note_id, excluded.last_note_id)",
            (pr_number, note_id),
        )
        conn.commit()
    finally:
        conn.close()


def recover_stale_running() -> int:
    """Reset 'running' rows to 'crashed' at startup, so tasks interrupted
    by a daemon restart are retried instead of blocking their SHA forever.
    Returns the number of recovered rows."""
    conn = db()
    try:
        cur = conn.execute(
            "UPDATE seen_prs SET status = 'crashed' WHERE status = 'running'")
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


# ── GitCode API ──────────────────────────────────────────────────────

def fetch_open_prs() -> list[dict]:
    """Return all open PRs on the upstream repo (any author).

    Follows page-number pagination so repos with more than 100 open PRs
    are fully scanned. The token travels in the Authorization header, not
    the URL, so it never appears in proxy / debug logs.
    """
    url = f"{API_BASE}/repos/{UPSTREAM}/pulls"
    prs: list[dict] = []
    page = 1
    while True:
        try:
            resp = requests.get(
                url,
                params={"state": "open", "per_page": 100, "page": page},
                headers={"Authorization": f"Bearer {GITCODE_TOKEN}"},
                timeout=15,
            )
            resp.raise_for_status()
            batch = resp.json()
        except Exception as e:
            log.warning("failed to fetch open PRs (page %d): %s", page, e)
            return prs
        if not isinstance(batch, list) or not batch:
            return prs
        prs.extend(batch)
        if len(batch) < 100:
            return prs
        page += 1


# ── PR comment commands (/vla-factory ...) ────────────────────────────

def fetch_pr_comments(pr_number: int) -> list[dict]:
    """Return issue-style comments on a PR, oldest first.

    Follows page-number pagination so PRs with more than 100 comments are
    fully scanned. An empty/non-list page ENDS pagination but keeps the
    notes accumulated so far — exactly 100*k comments must not vanish
    because page k+1 came back empty (mirror of fetch_open_prs).
    """
    url = f"{API_BASE}/repos/{UPSTREAM}/pulls/{pr_number}/comments"

    def collected() -> list[dict]:
        return [{"note_id": nid, "body": body, "author": author}
                for nid, body, author in sorted(notes, key=lambda x: x[0])]

    notes: list[tuple[int, str, str]] = []
    page = 1
    while True:
        try:
            resp = requests.get(url, params={"per_page": 100, "page": page},
                                headers={"Authorization": f"Bearer {GITCODE_TOKEN}"},
                                timeout=15)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as e:
            log.warning("failed to fetch comments on PR #%d (page %d): %s",
                        pr_number, page, e)
            # transport error mid-pagination: keep what we have
            return collected()
        if not isinstance(batch, list) or not batch:
            return collected()
        for c in batch:
            nid = c.get("note_id") or c.get("id")
            if nid is None:
                continue
            user = c.get("user")
            author = user.get("login") if isinstance(user, dict) else                 (user if isinstance(user, str) else "")
            notes.append((int(nid), c.get("body") or "", author))
        if len(batch) < 100:
            return collected()
        page += 1
        if page > 20:  # hard cap: 2000 comments per PR is plenty
            return collected()


def poll_commands(pr_number: int) -> list[tuple[str, callable]]:
    """Check one PR for new ``/vla-factory`` commands; return executable tasks.

    The watermark check-and-mark happens here, synchronously in the caller
    (the main loop), BEFORE any task is queued — so the same comment can
    never be dispatched twice, even if a previous dispatch is still queued
    or running when the next poll fires.

    On the first poll of a PR (no handled_comments row), the watermark is
    initialized to the newest existing comment, so only commands typed
    after the daemon started watching are executed — no history replay.
    """
    comments = fetch_pr_comments(pr_number)
    if not comments:
        return []
    seen_before = last_handled_note(pr_number)
    if seen_before == 0:
        log.info("watching comments on PR #%d from note %d",
                 pr_number, comments[-1]["note_id"])
        mark_handled_note(pr_number, comments[-1]["note_id"])
        return []

    post = _post_body(pr_number)
    tasks = []
    for c in comments:
        if c["note_id"] <= seen_before:
            continue
        mark_handled_note(pr_number, c["note_id"])
        name = slash_commands.parse_command(c["body"])
        if name is None:
            continue
        entry = slash_commands.COMMANDS.get(name)
        if entry is None:
            tasks.append((name, lambda n=name, p=post: p(
                f"未知命令 `{n}`。\n\n{slash_commands.help_text()}")))
            continue
        handler, _ = entry
        # handlers receive the comment author as their `args` parameter —
        # e.g. /vla-factory review checks it against VLAF_REVIEW_ALLOWLIST.
        tasks.append((name, lambda h=handler, n=name, p=post, a=c.get("author", ""):
                      run_command(n, h, pr_number, p, a)))
    return tasks


def run_command(name: str, handler, pr_number: int, post,
                author: str = "") -> None:
    """Execute one dispatched command; failures are logged and reported,
    never propagated — a bad command must not kill its worker thread."""
    try:
        handler(pr_number, author, post)
    except Exception as e:
        log.exception("command /vla-factory %s failed on PR #%d: %s", name, pr_number, e)
        post(f"## 命令 `{name}` 执行失败\n\n`{e}`")


def handle_commands(pr_number: int) -> None:
    """Synchronous wrapper around poll_commands (runs tasks inline).

    Kept for one-off/debug use; the daemon main loop dispatches to the
    command pool instead.
    """
    try:
        for _, fn in poll_commands(pr_number):
            fn()
    except Exception as e:
        log.exception("command handling failed on PR #%d: %s", pr_number, e)
        _post_body(pr_number)(f"## 命令处理失败\n\n`{e}`")


def _post_body(pr_number: int):
    """Return a post(body) callable that creates a PR comment."""
    def post(body: str) -> None:
        try:
            post_comment(pr_number, body)
        except (Exception, SystemExit) as e:
            log.error("failed to post command reply on PR #%d: %s", pr_number, e)
    return post


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


def pr_checkout(pr_number: int, sha: str) -> Path:
    """Dedicated checkout dir per (PR, head SHA).

    CI tasks run concurrently on a thread pool; sharing one worktree would
    have their git fetch/checkout/clean clobber each other. Each task gets
    ``pr-<N>-<sha8>``; the local clone off BASE_DIR hardlinks objects, so
    creation is cheap. Origin is repointed at the upstream URL for fetches.
    """
    repo_dir = BASE_DIR / f"pr-{pr_number}-{sha[:8]}"
    if not (repo_dir / ".git").exists():
        ensure_repo()
        subprocess.run(["git", "clone", "--quiet", str(BASE_DIR), str(repo_dir)],
                       check=True, capture_output=True, timeout=300)
        subprocess.run(["git", "remote", "set-url", "origin", CLONE_URL],
                       cwd=repo_dir, check=True, capture_output=True, timeout=60)
    return repo_dir


def cleanup_old_checkouts(pr_number: int, keep: Path) -> None:
    """Drop superseded checkout dirs for this PR (previous SHAs).

    Never touches dirs whose SHA is still claimed 'running' — a concurrent
    task for the same PR on a different SHA may be using them.
    """
    prefix = f"pr-{pr_number}-"
    running_keys = set()
    conn = db()
    try:
        rows = conn.execute(
            "SELECT head_sha FROM seen_prs WHERE pr_number = ? AND status = 'running'",
            (pr_number,),
        ).fetchall()
    finally:
        conn.close()
    for row in rows:
        running_keys.add(row["head_sha"][:8])
    for d in BASE_DIR.glob(prefix + "*"):
        if d == keep or d.name[len(prefix):] in running_keys:
            continue
        shutil.rmtree(d, ignore_errors=True)


CHECKOUT_TTL_DAYS = float(os.environ.get("VLAF_CHECKOUT_TTL_DAYS", "7"))


def gc_old_checkouts() -> int:
    """Delete per-PR checkout dirs not touched for VLAF_CHECKOUT_TTL_DAYS.

    cleanup_old_checkouts only fires when a newer SHA supersedes the dir;
    a merged/closed PR leaves the poll list, so its last checkout would
    otherwise sit on disk forever. Age is taken from .git/FETCH_HEAD
    (rewritten by every sync_pr); dirs claimed by a running task are
    never touched. Returns the number of removed dirs.
    """
    cutoff = time.time() - CHECKOUT_TTL_DAYS * 86400
    running_keys = set()
    conn = db()
    try:
        rows = conn.execute(
            "SELECT head_sha FROM seen_prs WHERE status = 'running'").fetchall()
    finally:
        conn.close()
    for row in rows:
        running_keys.add(row["head_sha"][:8])
    removed = 0
    for d in BASE_DIR.glob("pr-*-*"):
        parts = d.name.split("-")
        if not d.is_dir() or len(parts) != 3:
            continue
        if parts[2] in running_keys:
            continue
        stamp = d / ".git" / "FETCH_HEAD"
        try:
            mtime = (stamp if stamp.exists() else d).stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
            log.info("gc: removed stale checkout %s", d.name)
    return removed


def sync_pr(repo_dir: Path, pr_number: int) -> str | None:
    """Fetch the PR head ref and detach HEAD to it. Returns the SHA or None."""
    # GitCode is Gitea-based: merge-requests is its native namespace, so try
    # it first; the GitHub-style pull ref is kept as a fallback for mirrors.
    for ref in (f"refs/merge-requests/{pr_number}/head", f"refs/pull/{pr_number}/head"):
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

# All available tiers: (pytest marker, junit stem). Each tier has a dedicated
# test directory; markers are retained for report metadata and selection.
# Empty reserved tiers are reported as
# "— (skip)" (see collect_results); only a missing report dir or actual
# test failures make an environment FAIL.
ALL_TIERS = [
    ("not l1 and not l2 and not l3", "l0"),
    ("l1", "l1"),
    ("l2", "l2"),
]

# Fixed tier assignment per required environment. Every PR update must execute
# L0, L1, and L2; changing it per machine would silently reduce CI coverage.
# L1 self-skips the cases whose upstream is absent, so act and pi safely cover
# their respective Lerobot and OpenPI contracts.
DEFAULT_ENV_TIERS: dict[str, list[str]] = {
    "base": ["l0"],
    "act": ["l1", "l2"],
    "pi": ["l1"],
}

TIER_TEST_PATHS = {
    "l0": "test/l0",
    "l1": "test/l1",
    "l2": "test/l2",
    "l3": "test/l3",
}


def env_tiers(label: str) -> list[tuple[str, str]]:
    """Return the fixed (marker, name) pairs assigned to an environment."""
    tier_map = {name: (marker, name) for marker, name in ALL_TIERS}
    names = DEFAULT_ENV_TIERS.get(label, [])
    return [tier_map[n] for n in names if n in tier_map]


TIER_TIMEOUT = int(os.environ.get("VLAF_TIER_TIMEOUT", "1200"))


def _write_timeout_xml(path: Path, tier: str, timeout: int) -> None:
    """Synthetic junit for a timed-out tier: 1 failed test named tier-timeout.

    Without it the tier's xml is missing and collect_results would silently
    treat the tier as skipped.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<testsuites><testsuite name="{tier}" tests="1" failures="1" errors="0" '
        f'skipped="0" time="{timeout}">'
        f'<testcase classname="ci" name="tier-timeout" time="{timeout}">'
        f'<failure message="tier exceeded {timeout}s timeout" type="Timeout"/>'
        f"</testcase></testsuite></testsuites>"
    )


def _write_empty_xml(path: Path, tier: str) -> None:
    """Record an assigned tier with no test directory as a zero-test result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<testsuites><testsuite name="{tier}" tests="0" failures="0" errors="0" '
        'skipped="0" time="0.0"/></testsuites>'
    )


def _tail(data, n: int = 300) -> str:
    if not data:
        return ""
    if isinstance(data, bytes):
        data = data.decode(errors="replace")
    return data.strip()[-n:]


def run_gate(repo_dir: Path, label: str, python_bin: str, report_dir: Path) -> bool:
    """Run the assigned tiers for this env directly via pytest.

    Writes ``<tier>.xml`` junit files into *report_dir*.

    pytest exit codes: 0 = pass, 5 = no tests collected (treated as pass),
    anything else = failure. A tier exceeding VLAF_TIER_TIMEOUT produces a
    synthetic failed junit instead of raising — an escaping TimeoutExpired
    would mark the run 'crashed' and requeue it, looping the same slow
    tier forever. A missing/unusable python binary fails the run the same
    way instead of crash-looping.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    tiers = env_tiers(label)
    all_ok = True
    for marker, name in tiers:
        test_path = repo_dir / TIER_TEST_PATHS[name]
        if not test_path.exists():
            log.info("  [%s] tier %s has no test directory", label, name)
            _write_empty_xml(report_dir / f"{name}.xml", name)
            continue
        t0 = time.time()
        cmd = [python_bin, "-m", "pytest", str(test_path), "-q", "-m", marker,
               f"--junitxml={report_dir}/{name}.xml"]
        log.info("  [%s] tier %s start (marker: %s)", label, name, marker)
        try:
            result = subprocess.run(cmd, cwd=repo_dir, capture_output=True,
                                    timeout=TIER_TIMEOUT)
        except subprocess.TimeoutExpired as e:
            log.error("  [%s] tier %s TIMEOUT after %ds; tail of output:\n%s",
                      label, name, TIER_TIMEOUT, _tail(e.stdout) or _tail(e.stderr))
            _write_timeout_xml(report_dir / f"{name}.xml", name, TIER_TIMEOUT)
            all_ok = False
            continue
        except OSError as e:
            log.error("  [%s] tier %s could not run %s: %s", label, name, python_bin, e)
            all_ok = False
            continue
        dt = time.time() - t0
        if result.returncode not in (0, 5):
            log.warning("  [%s] tier %s failed (exit %d, %.0fs): %s", label, name,
                        result.returncode, dt, _tail(result.stderr))
            all_ok = False
        else:
            log.info("  [%s] tier %s done in %.0fs (exit %d)", label, name,
                     dt, result.returncode)
    return all_ok


def collect_results(report_base: Path) -> list[dict]:
    rows = []
    for label, _ in ENVS:
        env_dir = report_base / label
        summaries = parse_report_dir(env_dir) if env_dir.exists() else {}
        # An env is ok if it produced junit files and none of them report a
        # failure. Env tiers whose markers collect zero tests legitimately
        # yield zero-total junit files (pytest exit 5); those count as a
        # skip, not a failure — failing them made whole runs show FAIL even
        # when every collected test passed. A *missing* report dir means the
        # env never ran at all (crash / misconfiguration) and stays a FAIL,
        # since all([]) being True must not turn a never-executed env green.
        ran = bool(summaries)
        rows.append({
            "env": label,
            "l0": tier_line(summaries.get("l0", {})),
            "l1": tier_line(summaries.get("l1", {})),
            "l2": tier_line(summaries.get("l2", {})),
            "ok": ran and all(s.get("ok", False) for s in summaries.values()),
            "time": f"{sum(s.get('time', 0) for s in summaries.values()):.0f}s",
            "failed_tests": [t for s in summaries.values() for t in s.get("failed_tests", [])],
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

    # 1. Post "running" comment — or, when retrying a crashed run, edit the
    # comment left behind instead of stacking a new one on the PR.
    comment_id = stored_comment_id(pr_number, sha)
    try:
        if comment_id:
            edit_comment(comment_id, format_running(branch, sha))
        else:
            comment_id = post_comment(pr_number, format_running(branch, sha))
    except Exception as e:
        log.error("failed to post comment: %s", e)
    # Persist the running state (with comment_id) so a crash mid-run can be
    # detected, retried, and its stale comment edited on the next attempt.
    mark_seen(pr_number, sha, "running", comment_id)

    # 2. Fetch + checkout into this task's dedicated dir.
    repo_dir = pr_checkout(pr_number, sha)
    cleanup_old_checkouts(pr_number, repo_dir)
    actual_sha = sync_pr(repo_dir, pr_number)
    if actual_sha is None:
        if comment_id:
            edit_comment(comment_id, f"## CI — fetch failed\n\nPR #{pr_number} merge ref 无法拉取,可能有冲突。")
        return False
    sha = actual_sha

    # 3. Run gate in each environment.
    report_base = repo_dir / "ci" / "_reports"
    if report_base.exists():
        shutil.rmtree(report_base)

    all_ok = True
    for label, python_bin in ENVS:
        log.info("  [%s] running gate ...", label)
        if not run_gate(repo_dir, label, python_bin, report_base / label):
            all_ok = False

    elapsed = time.time() - start

    # 4. Post results. An env whose tiers produced zero tests fails the run
    # (rows carry ok=False for it), not just the table cell.
    rows = collect_results(report_base)
    all_ok = all_ok and all(r["ok"] for r in rows)
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


# ── CI task (runs on a worker thread) ───────────────────────────────

def ci_task(pr: dict) -> None:
    """Run process_pr for one (PR, SHA); record the terminal status.

    The (pr, sha) row was already claimed 'running' by the dispatcher, so
    the poll loop won't resubmit while this executes.
    """
    pr_number = pr["number"]
    sha = pr.get("head", {}).get("sha", "")
    try:
        success = process_pr(pr)
        mark_seen(pr_number, sha, "done" if success else "failed", None)
    except Exception as e:
        log.exception("PR #%d crashed: %s", pr_number, e)
        mark_seen(pr_number, sha, "crashed", None)


# ── Main loop ────────────────────────────────────────────────────────

def main():
    if not GITCODE_TOKEN:
        sys.exit("VLAF_GITCODE_TOKEN is required")
    if not HF_TOKEN:
        sys.exit(
            "HF_TOKEN is required for PI L1 parity. Accept access to "
            "google/paligemma-3b-pt-224, then configure it with bash scripts/run_ci.sh."
        )
    missing_envs = [
        label for label in ("base", "act", "pi")
        if not os.environ.get(f"VLAF_ENV_{label.upper()}")
    ]
    invalid_envs = [
        label for label in ("base", "act", "pi")
        if os.environ.get(f"VLAF_ENV_{label.upper()}")
        and not Path(os.environ[f"VLAF_ENV_{label.upper()}"]).exists()
    ]
    if missing_envs or invalid_envs:
        detail = []
        if missing_envs:
            detail.append(f"missing {', '.join(missing_envs)}")
        if invalid_envs:
            detail.append(f"invalid paths for {', '.join(invalid_envs)}")
        sys.exit(
            "Configured interpreters are required for full CI coverage: "
            f"{'; '.join(detail)}. "
            "Run bash scripts/ci/build_ci_envs.sh base act pi, then "
            "configure them with bash scripts/run_ci.sh."
        )

    ensure_repo()
    recovered = recover_stale_running()
    if recovered:
        log.info("recovered %d interrupted CI run(s) for retry", recovered)

    log.info("started: upstream=%s  repo=%s  envs=%s  poll=%ds  workers: ci=%d cmd=%d",
             UPSTREAM, BASE_DIR, [e[0] for e in ENVS], POLL_INTERVAL,
             CI_WORKERS, CMD_WORKERS)

    with ThreadPoolExecutor(max_workers=CI_WORKERS, thread_name_prefix="ci") as ci_pool, \
         ThreadPoolExecutor(max_workers=CMD_WORKERS, thread_name_prefix="cmd") as cmd_pool:

        while True:
            try:
                gc_old_checkouts()
            except Exception as e:
                log.warning("checkout gc failed: %s", e)

            prs = fetch_open_prs()

            # Command pass first: comment commands are quick API calls and
            # must not queue behind CI runs (which can take minutes per PR).
            # Watermarking happens in poll_commands, so a command can never
            # be dispatched twice regardless of queue delays.
            for pr in prs:
                pr_number = pr.get("number")
                if not pr_number:
                    continue
                try:
                    tasks = poll_commands(pr_number)
                except Exception as e:
                    log.exception("command poll failed on PR #%d: %s", pr_number, e)
                    continue
                for name, fn in tasks:
                    log.info("dispatch /vla-factory %s on PR #%d", name, pr_number)
                    cmd_pool.submit(fn)

            new_count = 0
            for pr in prs:
                head = pr.get("head", {})
                pr_number = pr.get("number")
                sha = head.get("sha")
                if not pr_number or not sha:
                    continue
                if not should_dispatch(pr_number, sha):
                    continue

                new_count += 1
                # Claim before dispatch: the 'running' row makes is_seen
                # true, so the next polls won't resubmit this SHA while the
                # task is queued or executing.
                mark_seen(pr_number, sha, "running", None, bump_attempt=True)
                user = head.get("user")
                author = user.get("login") if isinstance(user, dict) else (user or "?")
                log.info("new: PR #%d by %s  sha=%s", pr_number, author, sha[:12])
                ci_pool.submit(ci_task, pr)

            if new_count == 0:
                log.debug("no new PRs (scanned %d)", len(prs))
            else:
                log.info("dispatched %d new CI task(s)", new_count)

            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
