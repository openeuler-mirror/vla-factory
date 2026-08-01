#!/usr/bin/env python3
"""GitCode PR comment management for CI result reporting.

Handles create / edit of PR comments via the GitCode API v5. Used by the
runner daemon to post "CI running" then edit the same comment with final
results — one comment per CI run, not a growing thread.

Can also be used standalone to post a one-off comment::

    python3 ci/runner/pr_reporter.py post   9 "hello"
    python3 ci/runner/pr_reporter.py edit   <comment_id> "updated"
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

API_BASE = "https://api.gitcode.com/api/v5"
UPSTREAM = os.environ.get("VLAF_UPSTREAM", "openeuler/vla-factory")


def _token() -> str:
    t = os.environ.get("VLAF_GITCODE_TOKEN", "")
    if not t:
        sys.exit("VLAF_GITCODE_TOKEN is required")
    return t


def _api(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{API_BASE}{path}?access_token={_token()}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode(errors="replace")[:200]
        raise RuntimeError(f"GitCode API {method} {path} → HTTP {e.code}: {err_body}") from e


def post_comment(pr_number: int, body: str) -> int:
    """Create a PR comment; returns the numeric note_id (for later editing)."""
    resp = _api("POST", f"/repos/{UPSTREAM}/pulls/{pr_number}/comments", {"body": body})
    # GitCode returns two IDs: a hash "id" and a numeric "note_id".
    # PATCH/DELETE need the numeric note_id.
    nid = resp.get("note_id")
    if not nid:
        raise RuntimeError(f"no note_id in response: {resp}")
    return nid


def edit_comment(comment_id: int, body: str) -> None:
    """Edit an existing PR comment by ID."""
    _api("PATCH", f"/repos/{UPSTREAM}/pulls/comments/{comment_id}", {"body": body})


# ── Result formatting ────────────────────────────────────────────────

def format_running(branch: str, sha: str) -> str:
    return (
        "## CI 测试运行中\n\n"
        f"branch: `{branch}` · commit: `{sha[:12]}`\n\n"
        "测试正在本地 GPU 机器上执行，完成后此处更新结果。"
    )


def format_result(
    branch: str,
    sha: str,
    results: list[dict],
    overall_success: bool,
    elapsed: float,
) -> str:
    """Format the final CI report as a markdown comment.

    ``results`` is a list of dicts, one per environment::

        [{"env": "base", "l0": "249 passed", "l1": "— (skip)", "l2": "— (skip)", "ok": True, "time": "21s"}, ...]
    """
    icon = "all tests passed" if overall_success else "some tests FAILED"
    status = f"## CI 测试报告 — {'pass' if overall_success else 'FAIL'}\n\n"
    status += f"branch: `{branch}` · commit: `{sha[:12]}` · {icon} · {elapsed:.0f}s\n\n"

    # Results table
    status += "| 环境 | L0 单元 | L1 parity | L2 冒烟 | 耗时 |\n"
    status += "|------|---------|-----------|---------|------|\n"
    for r in results:
        ok_mark = "pass" if r["ok"] else "**FAIL**"
        status += (
            f"| {r['env']} | {r.get('l0', '—')} | {r.get('l1', '—')} "
            f"| {r.get('l2', '—')} | {r.get('time', '—')} |\n"
        )

    return status


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: pr_reporter.py post <pr_number> <body>  |  edit <comment_id> <body>")
    cmd = sys.argv[1]
    if cmd == "post":
        pr = int(sys.argv[2])
        body = sys.argv[3] if len(sys.argv) > 3 else sys.stdin.read()
        cid = post_comment(pr, body)
        print(f"comment {cid} posted on PR #{pr}")
    elif cmd == "edit":
        cid = int(sys.argv[2])
        body = sys.argv[3] if len(sys.argv) > 3 else sys.stdin.read()
        edit_comment(cid, body)
        print(f"comment {cid} updated")
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
