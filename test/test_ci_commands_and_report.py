"""Tests for the CI daemon result collection and /vla-factory comment commands."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

CI_DIR = Path(__file__).resolve().parents[1] / "scripts" / "ci"
sys.path.insert(0, str(CI_DIR))

import commands  # noqa: E402
import parse_results  # noqa: E402


# ── /vla-factory command parsing ──────────────────────────────────────

@pytest.mark.parametrize("body,expected", [
    ("/vla-factory help", "help"),
    ("/vla-factory review", "review"),
    ("some comment\n/vla-factory review\nmore text", "review"),
    ("/VLA-Factory HELP", "help"),
    ("the `/vla-factory review` command did things", None),  # inline, not whole line
    ("/vla-factory", None),
    ("plain comment", None),
    ("", None),
])
def test_parse_command(body, expected):
    assert commands.parse_command(body) == expected


def test_help_text_lists_registered_commands():
    text = commands.help_text()
    for name in commands.COMMANDS:
        assert f"/vla-factory {name}" in text


def test_review_uses_vlafactory_skill_from_skills_repo():
    assert commands.SKILL_SUBMODULE_URL == "https://github.com/2012geek/skills.git"
    assert commands.REVIEW_SKILL_RELPATH == "skills/vlafactory-code-review"


def test_help_command_posts_help(monkeypatch):
    posted = []
    commands.COMMANDS["help"][0](9, "", posted.append)
    assert len(posted) == 1
    for name in commands.COMMANDS:
        assert f"/vla-factory {name}" in posted[0]


def test_test_command_retriggers_ci(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "DB_PATH", str(tmp_path / "ci.db"))
    import sqlite3
    conn = sqlite3.connect(commands.DB_PATH)
    conn.executescript(commands._SEEN_SCHEMA)
    conn.execute("INSERT INTO seen_prs VALUES (9, 'sha1', 'done', 5, 1, 't')")
    conn.commit(); conn.close()

    posted = []
    commands.cmd_test(9, "", posted.append)

    assert "重新触发" in posted[0]
    conn = sqlite3.connect(commands.DB_PATH)
    left = conn.execute("SELECT COUNT(*) FROM seen_prs").fetchone()[0]
    conn.close()
    assert left == 0  # rows cleared → next poll re-dispatches


def test_test_command_refuses_while_running(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "DB_PATH", str(tmp_path / "ci.db"))
    import sqlite3
    conn = sqlite3.connect(commands.DB_PATH)
    conn.executescript(commands._SEEN_SCHEMA)
    conn.execute("INSERT INTO seen_prs VALUES (9, 'sha1', 'running', 5, 1, 't')")
    conn.commit(); conn.close()

    posted = []
    commands.cmd_test(9, "", posted.append)

    assert "正在运行" in posted[0]
    conn = sqlite3.connect(commands.DB_PATH)
    left = conn.execute("SELECT COUNT(*) FROM seen_prs").fetchone()[0]
    conn.close()
    assert left == 1  # in-flight bookkeeping untouched


# ── JSON unwrapping for agent outputs ────────────────────────────────

def test_count_issues(tmp_path):
    f = tmp_path / "issue-0.json"
    assert commands._count_issues(f) == 0  # missing file
    f.write_text('[{"a": 1}, {"b": 2}]')
    assert commands._count_issues(f) == 2
    f.write_text('{"issues": [{"a": 1}]}')
    assert commands._count_issues(f) == 1
    f.write_text('not json at all')
    assert commands._count_issues(f) == 0


def test_review_agents_run_concurrently(tmp_path, monkeypatch):
    """Three independent reviewers should not consume three serial budgets."""
    # This test exercises the daemon orchestration, not submodule checkout.
    # CI may intentionally run unit tests without initializing submodules, so
    # provide the minimal reviewer entrypoint that run_review() requires.
    skill_dir = tmp_path / "fake-skill"
    reviewer = skill_dir / "scripts" / "gitcode-reviewer.js"
    reviewer.parent.mkdir(parents=True)
    reviewer.write_text("// fake reviewer; subprocess execution is mocked\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VLAF_SKILL_DIR", str(skill_dir))
    monkeypatch.setenv("VLAF_AGENT_CMD", "fake-agent {prompt} {output}")
    monkeypatch.setenv("VLAF_REVIEW_WORKERS", "3")
    monkeypatch.setattr(commands, "resolve_node", lambda: "node")
    monkeypatch.setattr(commands.shutil, "which", lambda name: f"/fake/{name}")

    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_run(cmd, *, capture_output, text, timeout, env, cwd):
        nonlocal active, peak
        cwd = Path(cwd)
        if cmd[0] == "node" and "--auto-review" in cmd:
            agents = []
            for index in range(3):
                prompt = cwd / f"prompt-{index}.md"
                prompt.write_text(f"review {index}")
                agents.append({
                    "index": index,
                    "name": f"agent-{index}",
                    "promptPath": prompt.name,
                    "issuePath": f"issue-{index}.json",
                })
            (cwd / "prompts.json").write_text(json.dumps({"agents": agents}))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == "sh":
            output = cwd / cmd[2].split()[-1]
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            output.write_text("[]")
            with lock:
                active -= 1
            return subprocess.CompletedProcess(cmd, 0, "", "")
        assert "--collect-issues-from" in cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(commands.subprocess, "run", fake_run)
    posted = []

    commands.run_review(17, posted.append)

    assert peak == 3
    assert "3/3" in posted[-1]
    assert "未发现问题" in posted[-1]


def test_resolve_node_env_override(tmp_path, monkeypatch):
    fake = tmp_path / "node"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setattr(commands.shutil, "which", lambda name: None)
    monkeypatch.setenv("VLAF_NODE_BIN", str(fake))
    assert commands.resolve_node() == str(fake)


def test_resolve_node_falls_back_to_which(tmp_path, monkeypatch):
    monkeypatch.delenv("VLAF_NODE_BIN", raising=False)
    monkeypatch.setattr(commands.shutil, "which", lambda name: "/usr/bin/node")
    assert commands.resolve_node() == "/usr/bin/node"


@pytest.mark.parametrize("raw,expected", [
    ('[{"a": 1}]', '[{"a": 1}]'),
    ('```json\n[{"a": 1}]\n```', '[{"a": 1}]'),
    ('```\n[]\n```', '[]'),
    ('  [\n]  ', '[\n]'),
])
def test_extract_json(raw, expected):
    assert commands._extract_json(raw) == expected


# ── collect_results env gating ───────────────────────────────────────

def _write_junit(path: Path, tests: int, failures: int, skipped: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<testsuites><testsuite name="x" tests="{tests}" failures="{failures}" '
        f'errors="0" skipped="{skipped}" time="1.0"></testsuite></testsuites>'
    )


@pytest.fixture()
def daemon(tmp_path, monkeypatch):
    import daemon as daemon_mod
    # DB_PATH/ENVS are resolved at import time, so setenv is useless here —
    # patch the module attributes directly.
    monkeypatch.setattr(daemon_mod, "DB_PATH", str(tmp_path / "ci.db"))
    monkeypatch.setattr(daemon_mod, "ENVS", [("base", "py"), ("act", "py")])
    return daemon_mod


def test_zero_collected_tier_is_skip_not_fail(daemon, tmp_path):
    # The bug scenario: base passes L0, act's L1 marker collects zero tests
    # → junit exists with tests=0 → run must be green overall.
    report_base = tmp_path / "reports"
    _write_junit(report_base / "base" / "l0.xml", tests=3, failures=0)
    _write_junit(report_base / "act" / "l1.xml", tests=0, failures=0)

    rows = daemon.collect_results(report_base)
    by_env = {r["env"]: r for r in rows}
    assert by_env["base"]["ok"] is True
    assert by_env["act"]["ok"] is True
    assert by_env["act"]["l1"] == "— (skip)"


def test_env_with_real_failures_still_fails(daemon, tmp_path):
    report_base = tmp_path / "reports"
    _write_junit(report_base / "base" / "l0.xml", tests=3, failures=0)
    _write_junit(report_base / "act" / "l1.xml", tests=2, failures=1)

    rows = daemon.collect_results(report_base)
    by_env = {r["env"]: r for r in rows}
    assert by_env["act"]["ok"] is False


def test_failed_test_names_flow_to_report(daemon, tmp_path):
    import pr_reporter
    report_base = tmp_path / "reports"
    xml = report_base / "base" / "l0.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text(
        '<testsuites><testsuite name="x" tests="2" failures="1" errors="0" '
        'skipped="0" time="1.0">'
        '<testcase classname="test.foo" name="test_a"/>'
        '<testcase classname="test.foo" name="test_b">'
        '<failure message="AssertionError: assert 1 == 2" type="AssertionError"/>'
        '</testcase></testsuite></testsuites>')
    for f in report_base.glob("act/*.xml"):
        f.unlink()

    rows = daemon.collect_results(report_base)
    body = pr_reporter.format_result("b", "aa11", rows, False, 3.0)

    assert "`test.foo::test_b`" in body
    assert "AssertionError: assert 1 == 2" in body
    assert "失败用例" in body


def test_env_that_never_ran_fails(daemon, tmp_path):
    report_base = tmp_path / "reports"
    _write_junit(report_base / "base" / "l0.xml", tests=3, failures=0)
    # no report dir at all for act

    rows = daemon.collect_results(report_base)
    by_env = {r["env"]: r for r in rows}
    assert by_env["act"]["ok"] is False


# ── skills submodule sync ────────────────────────────────────────────

def _git(*args, cwd):
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


@pytest.fixture()
def skill_remote(tmp_path):
    """A local 'remote' for the skills repo with two commits on main."""
    remote = tmp_path / "remote.git"
    _git("init", "--quiet", "--bare", "-b", "main", cwd=remote)
    seed = tmp_path / "seed"
    _git("init", "--quiet", "-b", "main", cwd=seed)
    (seed / "v1").write_text("1")
    _git("add", ".", cwd=seed)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet", "-m", "v1", cwd=seed)
    _git("push", "--quiet", str(remote), "main", cwd=seed)
    (seed / "v2").write_text("2")
    _git("add", ".", cwd=seed)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet", "-m", "v2", cwd=seed)
    _git("push", "--quiet", str(remote), "main", cwd=seed)
    return remote


def test_ensure_skill_repo_clones_when_missing(skill_remote, tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "SKILL_SUBMODULE_URL", str(skill_remote))
    repo_dir = tmp_path / "checkout"
    repo_dir.mkdir()
    _git("init", "--quiet", cwd=repo_dir)

    root = commands.ensure_skill_repo(repo_dir)

    assert (root / ".git").exists()
    assert (root / "v1").exists()


def test_ensure_skill_repo_updates_to_latest_main(skill_remote, tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "SKILL_SUBMODULE_URL", str(skill_remote))
    repo_dir = tmp_path / "checkout"
    repo_dir.mkdir()
    _git("init", "--quiet", cwd=repo_dir)
    # Clone at the first commit (v1, before v2 exists on main).
    early = tmp_path / "early"
    early.mkdir()
    _git("clone", "--quiet", str(skill_remote), "seed", cwd=early)
    repo = early / "seed"
    _git("reset", "--hard", "HEAD~1", cwd=repo)
    _git("remote", "remove", "origin", cwd=repo)
    shutil.move(str(repo), str(repo_dir / "skills"))
    _git("remote", "add", "origin", str(skill_remote), cwd=repo_dir / "skills")
    assert not (repo_dir / "skills" / "v2").exists()

    root = commands.ensure_skill_repo(repo_dir)

    assert (root / "v2").exists()  # pulled latest main
    assert commands._run_git(["rev-parse", "HEAD"], root).stdout == \
           commands._run_git(["rev-parse", "origin/main"], root).stdout


def test_ensure_skill_repo_offline_keeps_checkout(skill_remote, tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "SKILL_SUBMODULE_URL", str(skill_remote))
    repo_dir = tmp_path / "checkout"
    repo_dir.mkdir()
    _git("init", "--quiet", cwd=repo_dir)
    commands.ensure_skill_repo(repo_dir)
    # Point origin at a dead URL: fetch fails, existing content must survive.
    _git("remote", "set-url", "origin", str(tmp_path / "does-not-exist.git"),
         cwd=repo_dir / "skills")

    root = commands.ensure_skill_repo(repo_dir)

    assert (root / "v1").exists()


def test_ensure_skill_repo_via_submodule_config(skill_remote, tmp_path, monkeypatch):
    """The happy path in production: parent repo has submodule config."""
    monkeypatch.setattr(commands, "SKILL_SUBMODULE_URL", str(skill_remote))
    parent = tmp_path / "parent"
    parent.mkdir()
    _git("init", "--quiet", cwd=parent)
    (parent / ".gitmodules").write_text(
        '[submodule "skills"]\n\tpath = skills\n'
        f"\turl = {skill_remote}\n")
    _git("add", ".gitmodules", cwd=parent)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit",
         "--quiet", "-m", "add submodule", cwd=parent)

    root = commands.ensure_skill_repo(parent)

    assert (root / "v2").exists()  # init + sync to latest main


# ── handled-comment watermark ────────────────────────────────────────

def test_comment_watermark_monotonic(daemon):
    daemon.mark_handled_note(5, 10)
    daemon.mark_handled_note(5, 7)  # out-of-order arrival must not rewind
    assert daemon.last_handled_note(5) == 10
    assert daemon.last_handled_note(6) == 0


# ── command dispatch ─────────────────────────────────────────────────

def test_handle_commands_swallows_fetch_errors(daemon, monkeypatch):
    def boom(pr_number):
        raise RuntimeError("net down")
    monkeypatch.setattr(daemon, "fetch_pr_comments", boom)
    daemon.handle_commands(7)  # must not raise


def test_handle_commands_first_poll_initializes_watermark(daemon, monkeypatch):
    monkeypatch.setattr(daemon, "fetch_pr_comments", lambda pr: [
        {"note_id": 1, "body": "/vla-factory help"}])
    daemon.handle_commands(7)
    assert daemon.last_handled_note(7) == 1  # swallowed, not dispatched


def test_handle_commands_dispatches_new_command(daemon, monkeypatch):
    real_last_handled = daemon.last_handled_note
    monkeypatch.setattr(daemon, "fetch_pr_comments", lambda pr: [
        {"note_id": 1, "body": "old comment"},
        {"note_id": 2, "body": "/vla-factory help"}])
    monkeypatch.setattr(daemon, "last_handled_note", lambda pr: 1)
    posted = []
    monkeypatch.setattr(daemon, "post_comment",
                        lambda pr, body: posted.append((pr, body)) or 99)

    daemon.handle_commands(7)

    assert real_last_handled(7) == 2
    assert len(posted) == 1 and posted[0][0] == 7
    assert "/vla-factory help" in posted[0][1]


def test_handle_commands_unknown_command_replies_help(daemon, monkeypatch):
    monkeypatch.setattr(daemon, "fetch_pr_comments", lambda pr: [
        {"note_id": 5, "body": "/vla-factory bogus"}])
    monkeypatch.setattr(daemon, "last_handled_note", lambda pr: 4)
    posted = []
    monkeypatch.setattr(daemon, "post_comment",
                        lambda pr, body: posted.append((pr, body)) or 99)

    daemon.handle_commands(7)

    assert len(posted) == 1
    assert "未知命令" in posted[0][1]
    assert "/vla-factory help" in posted[0][1]


def test_poll_commands_no_double_dispatch(daemon, monkeypatch):
    monkeypatch.setattr(daemon, "fetch_pr_comments", lambda pr: [
        {"note_id": 1, "body": "hi"},
        {"note_id": 2, "body": "/vla-factory help"}])

    first = daemon.poll_commands(7)          # initializes watermark, swallows
    assert first == []
    second = daemon.poll_commands(7)         # no new comments
    assert second == []
    assert daemon.last_handled_note(7) == 2


# ── async dispatch bookkeeping ───────────────────────────────────────

def test_running_claim_blocks_resubmission(daemon):
    daemon.mark_seen(8, "aa11", "running", None)
    assert daemon.is_seen(8, "aa11") is True   # in flight → not resubmitted


def test_recover_stale_running(daemon):
    daemon.mark_seen(8, "aa11", "running", None)
    assert daemon.recover_stale_running() == 1
    assert daemon.is_seen(8, "aa11") is False  # crashed → retried next poll


def test_crashed_is_retried(daemon):
    daemon.mark_seen(8, "aa11", "crashed", None)
    assert daemon.is_seen(8, "aa11") is False


def test_retry_cap(daemon):
    sha = "aa11"
    for _ in range(3):  # three claimed dispatches, all crashed
        daemon.mark_seen(8, sha, "running", None, bump_attempt=True)
        daemon.mark_seen(8, sha, "crashed", None)
    assert daemon.should_dispatch(8, sha) is False  # gave up
    assert daemon.should_dispatch(8, "other") is True


def test_status_transitions_do_not_bump_attempts(daemon):
    daemon.mark_seen(8, "aa11", "running", None, bump_attempt=True)
    daemon.mark_seen(8, "aa11", "running", 42)
    daemon.mark_seen(8, "aa11", "failed", None)
    daemon.mark_seen(8, "aa11", "crashed", None)
    conn = daemon.db()
    try:
        row = conn.execute(
            "SELECT attempts, status FROM seen_prs WHERE pr_number=8 AND head_sha='aa11'"
        ).fetchone()
    finally:
        conn.close()
    assert row["attempts"] == 1 and row["status"] == "crashed"
    assert daemon.should_dispatch(8, "aa11") is True


# ── run_gate resilience ──────────────────────────────────────────────

def test_run_gate_timeout_writes_failed_junit(daemon, tmp_path, monkeypatch):
    def hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd="pytest", timeout=1, output=b"slow test ...")
    monkeypatch.setattr(daemon.subprocess, "run", hang)
    reports = tmp_path / "reports" / "base"

    ok = daemon.run_gate(tmp_path, "base", "python", reports)

    assert ok is False
    s = parse_results.parse_junit(reports / "l0.xml")
    assert s["failed"] == 1 and s["ok"] is False
    assert "timeout" in (reports / "l0.xml").read_text().lower()


def test_run_gate_missing_python_fails_not_raises(daemon, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no such python")
    monkeypatch.setattr(daemon.subprocess, "run", boom)

    ok = daemon.run_gate(tmp_path, "base", "/no/such/python", tmp_path / "r")

    assert ok is False


def test_terminal_states_block(daemon):
    for status in ("done", "failed"):
        daemon.mark_seen(8, f"{status}1", status, None)
        assert daemon.is_seen(8, f"{status}1") is True


# ── per-PR checkout isolation ────────────────────────────────────────

@pytest.fixture()
def base_repo(tmp_path, monkeypatch, daemon):
    base = tmp_path / "ci-base"
    base.mkdir()
    _git("init", "--quiet", "-b", "main", cwd=base)
    (base / "README").write_text("x")
    _git("add", ".", cwd=base)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet", "-m", "init", cwd=base)
    monkeypatch.setattr(daemon, "BASE_DIR", base)
    return base


def test_pr_checkout_is_per_sha(daemon, base_repo):
    a1 = daemon.pr_checkout(7, "aaaaaaaa1111")
    a2 = daemon.pr_checkout(7, "aaaaaaaa1111")   # same SHA → reused
    b = daemon.pr_checkout(7, "bbbbbbbb2222")    # new SHA → new dir
    other = daemon.pr_checkout(9, "aaaaaaaa1111")  # other PR → new dir

    assert a1 == a2
    assert a1 != b and a1 != other
    for d in (a1, b, other):
        assert (d / ".git").exists()
        assert d.parent == base_repo


def test_cleanup_old_checkouts(daemon, base_repo):
    keep = base_repo / "pr-7-cccccccc"
    old = base_repo / "pr-7-aaaaaaaa"
    running = base_repo / "pr-7-bbbbbbbb"
    for d in (keep, old, running):
        d.mkdir()
    daemon.mark_seen(7, "bbbbbbbb_full_sha", "running", None)

    daemon.cleanup_old_checkouts(7, keep)

    assert keep.exists()
    assert not old.exists()
    assert running.exists()  # in-flight task's dir is never touched


def test_gc_old_checkouts_by_age(daemon, base_repo):
    import os
    stale = base_repo / "pr-16-aaaaaaaa"
    fresh = base_repo / "pr-17-bbbbbbbb"
    claimed = base_repo / "pr-18-cccccccc"
    for d in (stale, fresh, claimed):
        (d / ".git").mkdir(parents=True)
        (d / ".git" / "FETCH_HEAD").write_text("x")
    old_ts = time.time() - 10 * 86400
    for f in (stale, stale / ".git" / "FETCH_HEAD"):
        os.utime(f, (old_ts, old_ts))
    os.utime(claimed / ".git" / "FETCH_HEAD", (old_ts, old_ts))
    daemon.mark_seen(18, "cccccccc_full_sha", "running", None)

    removed = daemon.gc_old_checkouts()

    assert removed == 1
    assert not stale.exists()
    assert fresh.exists()
    assert claimed.exists()  # old but claimed running → untouched


# ── review fixes: auth / cooldown / reentry / env / quoting ─────────

def test_review_allowlist_refuses_stranger(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "DB_PATH", str(tmp_path / "ci.db"))
    monkeypatch.setenv("VLAF_REVIEW_ALLOWLIST", "alice, bob")
    posted = []
    commands.cmd_review(9, "mallory", posted.append)
    assert len(posted) == 1 and "无权" in posted[0]


def test_review_allowlist_passes_member(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "DB_PATH", str(tmp_path / "ci.db"))
    monkeypatch.setenv("VLAF_REVIEW_ALLOWLIST", "alice")
    monkeypatch.setenv("VLAF_REVIEW_COOLDOWN_MIN", "0")
    calls = []
    monkeypatch.setattr(commands, "run_review", lambda pr, post: calls.append(pr))
    commands.cmd_review(9, "alice", lambda b: None)
    assert calls == [9]


def test_review_cooldown_blocks_recent(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "DB_PATH", str(tmp_path / "ci.db"))
    monkeypatch.delenv("VLAF_REVIEW_ALLOWLIST", raising=False)
    monkeypatch.setenv("VLAF_REVIEW_COOLDOWN_MIN", "30")
    commands._mark_reviewed(9)  # just now
    posted = []
    commands.cmd_review(9, "", posted.append)
    assert len(posted) == 1 and "冷却" in posted[0]


def test_review_cooldown_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "DB_PATH", str(tmp_path / "ci.db"))
    monkeypatch.delenv("VLAF_REVIEW_ALLOWLIST", raising=False)
    monkeypatch.setenv("VLAF_REVIEW_COOLDOWN_MIN", "0")
    commands._mark_reviewed(9)
    calls = []
    monkeypatch.setattr(commands, "run_review", lambda pr, post: calls.append(pr))
    commands.cmd_review(9, "", lambda b: None)
    assert calls == [9]


def test_run_review_reentry_guard(monkeypatch):
    with commands._REVIEW_MUTEX:
        commands._ACTIVE_REVIEWS.add(9)
    posted = []
    commands.run_review(9, posted.append)
    assert len(posted) == 1 and "进行中" in posted[0]
    assert 9 in commands._ACTIVE_REVIEWS  # untouched: not our run
    with commands._REVIEW_MUTEX:
        commands._ACTIVE_REVIEWS.discard(9)


def test_agent_paths_shell_quoted(tmp_path, monkeypatch):
    # A path with spaces/metachars must survive sh -c rendering quoted.
    weird = tmp_path / "pro mpt (v1).md"
    rendered = 'claude -p < {prompt} > {output}'.format(
        prompt=__import__("shlex").quote(str(weird)),
        output=__import__("shlex").quote(str(tmp_path / "out.json")))
    assert f"'{weird}'" in rendered


def test_fetch_pr_comments_keeps_accumulated_on_empty_page(daemon, monkeypatch):
    page1 = [{"note_id": i, "body": "x", "user": {"login": "u"}} for i in range(1, 101)]
    page2 = []  # exactly 100 comments → page 2 empty

    class FakeResp:
        def __init__(self, payload): self._p = payload
        def raise_for_status(self): pass
        def json(self): return self._p

    pages = [page1, page2]
    monkeypatch.setattr(daemon.requests, "get",
                        lambda url, params=None, headers=None, timeout=None:
                        FakeResp(pages.pop(0) if pages else []))

    comments = daemon.fetch_pr_comments(7)

    assert len(comments) == 100          # page-1 notes preserved
    assert comments[0]["author"] == "u"  # author captured


def test_poll_commands_passes_author(daemon, monkeypatch):
    monkeypatch.setattr(daemon, "fetch_pr_comments", lambda pr: [
        {"note_id": 1, "body": "old"},
        {"note_id": 2, "body": "/vla-factory help", "author": "alice"}])
    monkeypatch.setattr(daemon, "last_handled_note", lambda pr: 1)
    seen_authors = []
    monkeypatch.setattr(daemon, "run_command",
                        lambda name, handler, pr, post, author="":
                        seen_authors.append(author))
    monkeypatch.setattr(daemon, "_post_body", lambda pr: (lambda b: None))

    tasks = daemon.poll_commands(7)
    for _, fn in tasks:
        fn()

    assert seen_authors == ["alice"]
