#!/usr/bin/env python3
"""PR comment commands for the CI daemon (``/vla-factory <cmd>``).

Anyone can type a command as a PR comment::

    /vla-factory help   — list available commands
    /vla-factory review — run the vlafactory-code-review skill on this PR
    /vla-factory retest — re-trigger CI on the PR's current head

The daemon polls PR comments (see daemon.poll_commands), parses command
lines, and dispatches here. ``review`` drives the bundled
``vlafactory-code-review``
skill (the ``skills`` git submodule → github.com/2012geek/skills): ensure
the submodule is initialized (or plain-cloned as fallback) and checked out
at the latest main, generate one multi-role prompt, execute it with a
headless agent (VLAF_AGENT_CMD), then collect + post the findings as
inline PR comments. ``test`` resets the daemon's seen-SHA bookkeeping for
the PR so the next poll re-dispatches CI on the current head.

Config (environment variables, in addition to the daemon's)::

    VLAF_AGENT_CMD    headless agent command template, executed via `sh -c`
                      so redirection / $(...) work; placeholders {prompt}
                      {output}. Defaults to the hard-coded claude template
                      below — the only supported agent while others are
                      verified one by one.
    VLAF_SKILL_DIR    vlafactory-code-review dir; when set, submodule sync
                      is skipped and this dir is used as-is (default:
                      <repo>/skills/skills/vlafactory-code-review)
    VLAF_DB_PATH      seen-SHA DB shared with the daemon (same default)
    VLAF_AGENT_TIMEOUT          per-reviewer timeout in seconds (default 600)
    VLAF_REVIEW_STEP_TIMEOUT    prompt-generation/post timeout (default 180)
    VLAF_REVIEW_WORKERS         reviewer workers per PR (default 1)
    VLAF_REVIEW_GLOBAL_WORKERS  reviewer process cap for the daemon (default 4)
    VLAF_REVIEW_COOLDOWN_MIN    per-PR review cooldown in minutes, 0 disables
                                (default 30) — review burns paid agent quota,
                                so unbounded triggers are refused
    VLAF_REVIEW_ALLOWLIST       comma-separated logins allowed to run
                                /vla-factory review (empty = anyone; cooldown
                                still applies)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger("vlaf-ci")

COMMAND_PREFIX = "/vla-factory"

# Hard-coded default headless agent: claude in print mode, prompt fed via
# stdin and stdout redirected to the issue JSON. NEVER "$(cat {prompt})" —
# the prompt embeds the full PR diff and a single argv is capped at
# MAX_ARG_STRLEN (~128KB on Linux): big PRs die with "argument list too
# long".
DEFAULT_AGENT_CMD = ('claude -p --no-session-persistence '
                     '--disable-slash-commands --tools "" '
                     '< {prompt} > {output}')


# What child processes inherit from the daemon environment. The agent runs
# PR-author-controlled text — it gets the bare minimum (PATH/HOME for the
# CLI and its credentials file, proxies for egress), never daemon secrets.
_ENV_WHITELIST = ("PATH", "HOME", "LANG", "TERM", "TMPDIR",
                  "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                  "NO_PROXY", "no_proxy", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
                  "NODE_EXTRA_CA_CERTS")
_ENV_PREFIX_WHITELIST = ("LC_",)


def _positive_env_int(name: str, default: int) -> int:
    """Read a positive integer setting, falling back on invalid input."""
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        log.warning("ignoring invalid %s; using %d", name, default)
        return default
    if value < 1:
        log.warning("ignoring non-positive %s; using %d", name, default)
        return default
    return value


# Command workers may review several PRs concurrently. Keep model-process
# fan-out bounded across the whole daemon, not merely within one PR.
_REVIEW_AGENT_SLOTS = threading.BoundedSemaphore(
    _positive_env_int("VLAF_REVIEW_GLOBAL_WORKERS", 4)
)

# One review per PR at a time: run_review wipes its workdir on entry, so a
# second concurrent review of the same PR would delete the first one's
# prompts while its agent is still running.
_REVIEW_MUTEX = threading.Lock()
_ACTIVE_REVIEWS: set[int] = set()

# Same DB the daemon tracks seen SHAs in (mirrored DDL; keep in sync with
# daemon.SCHEMA).
DB_PATH = os.environ.get("VLAF_DB_PATH", str(Path.home() / ".vlaf_ci.db"))
_REVIEW_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_history (
    pr_number       INTEGER PRIMARY KEY,
    last_review_at  TEXT NOT NULL
);
"""
_SEEN_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_prs (
    pr_number   INTEGER NOT NULL,
    head_sha    TEXT    NOT NULL,
    status      TEXT,
    comment_id  INTEGER,
    attempts    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL,
    PRIMARY KEY (pr_number, head_sha)
);
"""

# name → (handler, one-line description for help)
COMMANDS: dict[str, tuple] = {}


def command(name: str, description: str):
    def register(fn):
        COMMANDS[name] = (fn, description)
        return fn
    return register


# ── Parsing ──────────────────────────────────────────────────────────

_CMD_RE = re.compile(rf"^{re.escape(COMMAND_PREFIX)}\s+([a-zA-Z-]+)\s*$", re.M | re.I)


def parse_command(body: str) -> str | None:
    """Return the command name if *body* contains a ``/vla-factory <cmd>`` line.

    Only whole-line matches count, so prose like "the /vla-factory review
    command" doesn't trigger. Unknown command names are returned as-is so
    the dispatcher can reply with the help text.
    """
    m = _CMD_RE.search(body or "")
    return m.group(1).lower() if m else None


# ── Commands ─────────────────────────────────────────────────────────

@command("help", "显示可用命令")
def cmd_help(pr_number: int, args: str, post) -> None:
    post(help_text())


def _review_cooldown_seconds() -> int:
    try:
        minutes = float(os.environ.get("VLAF_REVIEW_COOLDOWN_MIN", "30"))
    except ValueError:
        minutes = 30
    return max(0.0, minutes) * 60


def _review_allowlist() -> set[str]:
    raw = os.environ.get("VLAF_REVIEW_ALLOWLIST", "")
    return {name.strip() for name in raw.split(",") if name.strip()}


def _review_in_cooldown(pr_number: int) -> tuple[bool, int]:
    """(in_cooldown, seconds_left) against review_history."""
    cooldown = _review_cooldown_seconds()
    if cooldown <= 0:
        return False, 0
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.executescript(_REVIEW_SCHEMA)
        row = conn.execute(
            "SELECT last_review_at FROM review_history WHERE pr_number = ?",
            (pr_number,)).fetchone()
    finally:
        conn.close()
    if not row:
        return False, 0
    try:
        elapsed = time.time() - float(row["last_review_at"] if not isinstance(row, tuple) else row[0])
    except (TypeError, ValueError):
        return False, 0
    left = int(cooldown - elapsed)
    return left > 0, max(left, 0)


def _mark_reviewed(pr_number: int) -> None:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.executescript(_REVIEW_SCHEMA)
        conn.execute(
            "INSERT INTO review_history (pr_number, last_review_at) VALUES (?, ?) "
            "ON CONFLICT(pr_number) DO UPDATE SET last_review_at = excluded.last_review_at",
            (pr_number, str(time.time())))
        conn.commit()
    finally:
        conn.close()


@command("review", "对本 PR 运行 vlafactory-code-review 技能，将问题以行内评论形式贴出")
def cmd_review(pr_number: int, args: str, post) -> None:
    """``args`` carries the comment author's login (set by the dispatcher)."""
    allow = _review_allowlist()
    if allow and args not in allow:
        post(f"## 无权触发检视\n\n`/vla-factory review` 仅限指定人员使用"
             f"（VLAF_REVIEW_ALLOWLIST）。")
        return
    in_cd, left = _review_in_cooldown(pr_number)
    if in_cd:
        post(f"## 检视冷却中\n\n本 PR 刚执行过检视，请 {left // 60 + 1} 分钟后再试"
             f"（VLAF_REVIEW_COOLDOWN_MIN 可调）。")
        return
    _mark_reviewed(pr_number)  # stamp at start: parallel triggers all pay the cooldown
    run_review(pr_number, post)


@command("retest", "重新触发本 PR 当前 head 的 CI 测试")
def cmd_test(pr_number: int, args: str, post) -> None:
    """Reset the daemon's seen-SHA rows for this PR.

    The next poll (≤ VLAF_POLL_INTERVAL) finds the current head unseen and
    dispatches a fresh CI run; the report updates the existing CI comment.
    Refuses while a run is in flight — a second concurrent run would race
    on the same per-PR checkout dir.
    """
    # BEGIN IMMEDIATE takes the write lock before the check, making
    # check-then-delete atomic against the main loop claiming a freshly
    # pushed SHA ('running') between our SELECT and DELETE.
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    try:
        conn.executescript(_SEEN_SCHEMA)
        conn.execute("BEGIN IMMEDIATE")
        try:
            running = conn.execute(
                "SELECT 1 FROM seen_prs WHERE pr_number = ? AND status = 'running'",
                (pr_number,),
            ).fetchone()
            if running:
                conn.execute("ROLLBACK")
                post("## CI 正在运行\n\n本 PR 的 CI 任务仍在执行或排队中，请等完成后再触发。")
                return
            conn.execute("DELETE FROM seen_prs WHERE pr_number = ?", (pr_number,))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    post("已重新触发 CI：下一轮轮询将对当前 head 运行测试，结果会更新到原 CI 报告评论。")


# ── Help text ────────────────────────────────────────────────────────

def help_text() -> str:
    lines = [f"## {COMMAND_PREFIX} 可用命令", ""]
    for name, (_, desc) in sorted(COMMANDS.items()):
        lines.append(f"- `{COMMAND_PREFIX} {name}` — {desc}")
    return "\n".join(lines) + "\n"


# ── Skill submodule sync ─────────────────────────────────────────────

SKILL_SUBMODULE_PATH = "skills"
SKILL_SUBMODULE_URL = "https://github.com/2012geek/skills.git"
REVIEW_SKILL_RELPATH = "skills/vlafactory-code-review"  # inside skills repo

# Usual install roots probed for node when it is not on PATH — nvm/conda
# installs live outside a daemon's (often non-login) PATH.
_NODE_CANDIDATES = [
    "/usr/local/bin/node",
    "/usr/bin/node",
    "~/.local/bin/node",
    "~/miniconda3/bin/node",
    "~/miniconda3/envs/*/bin/node",
    "~/anaconda3/bin/node",
    "~/anaconda3/envs/*/bin/node",
    "~/.nvm/versions/node/*/bin/node",
    "~/.volta/bin/node",
]


def resolve_node() -> str | None:
    """Locate a node executable for the review skill scripts (or None).

    Order: VLAF_NODE_BIN → PATH → common install roots.
    """
    env_bin = os.environ.get("VLAF_NODE_BIN", "")
    if env_bin and Path(env_bin).exists():
        return env_bin
    found = shutil.which("node")
    if found:
        return found
    import glob as _glob
    for pattern in _NODE_CANDIDATES:
        for match in sorted(_glob.glob(os.path.expanduser(pattern))):
            if os.path.isfile(match) and os.access(match, os.X_OK):
                return match
    return None


def _run_git(args: list[str], cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, timeout=timeout)


def ensure_skill_repo(repo_dir: Path) -> Path:
    """Make sure the skills submodule exists and is at latest main.

    Returns the submodule root. Missing submodule → ``git submodule update
    --init`` (plain clone as fallback, e.g. when the parent checkout has no
    submodule config). Present submodule → fetch origin/main and hard-swap
    to it, so reviews always run the newest skill version regardless of the
    pinned gitlink. A fetch failure (offline machine) keeps the current
    checkout and logs a warning — review proceeds with what's on disk.
    """
    root = repo_dir / SKILL_SUBMODULE_PATH

    if (root / ".git").exists():
        _sync_to_latest_main(root)
        return root

    if root.exists() and any(root.iterdir()):
        shutil.rmtree(root)  # leftover junk, no .git — machine-managed checkout
    r = _run_git(["submodule", "update", "--init", SKILL_SUBMODULE_PATH], repo_dir)
    if r.returncode != 0 or not (root / ".git").exists():
        log.warning("submodule init failed (%s), falling back to clone: %s",
                    r.returncode, (r.stderr or "").strip()[-200:])
        r = _run_git(["clone", "--quiet", "--depth", "1", SKILL_SUBMODULE_URL,
                      str(root)], cwd=repo_dir)
        if r.returncode != 0:
            raise RuntimeError(f"无法初始化 skills 子模块: {(r.stderr or '').strip()[-300:]}")
        return root

    _sync_to_latest_main(root)
    return root


_SKILL_SYNC_LOCK = threading.Lock()
# The shared submodule is force-checked-out during sync; concurrent reviews
# must not read a half-swapped tree (see _run_review_impl snapshot).


def _sync_to_latest_main(root: Path) -> None:
    r = _run_git(["fetch", "--quiet", "origin", "main"], root)
    if r.returncode != 0:
        log.warning("skills fetch failed, keeping current checkout: %s",
                    (r.stderr or "").strip()[-200:])
        return
    _run_git(["checkout", "--quiet", "--force", "FETCH_HEAD"], root)


# ── Review pipeline ──────────────────────────────────────────────────

def run_review(pr_number: int, post) -> None:
    """Run the vlafactory-code-review skill and post the outcome.

    One review per PR at a time — _run_review_impl wipes the shared
    workdir on entry, so concurrent runs of the same PR would destroy
    each other's prompts.
    """
    with _REVIEW_MUTEX:
        if pr_number in _ACTIVE_REVIEWS:
            post("## 检视进行中\n\n本 PR 已有检视任务在执行，请等待完成后再触发。")
            return
        _ACTIVE_REVIEWS.add(pr_number)
    try:
        _run_review_impl(pr_number, post)
    finally:
        with _REVIEW_MUTEX:
            _ACTIVE_REVIEWS.discard(pr_number)


def _run_review_impl(pr_number: int, post) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    custom_dir = os.environ.get("VLAF_SKILL_DIR", "")
    skill_source = None
    if not custom_dir:
        try:
            with _SKILL_SYNC_LOCK:
                skill_root = ensure_skill_repo(repo_root)
            skill_source = skill_root / REVIEW_SKILL_RELPATH
        except Exception as e:
            post(f"## 检视失败\n\n初始化/更新 skills 子模块失败：`{e}`")
            return

    node = resolve_node()
    if not node:
        post(
            "## 检视失败\n\n"
            "CI 机器上未找到 `node`（vlafactory-code-review 技能使用 Node.js）"
            "的）。安装方式任选其一，完成后重新执行 `/vla-factory review`：\n\n"
            "1. `conda install -y nodejs`（daemon 所在的 miniconda 环境，最简单）\n"
            "2. nvm 安装后**重启 daemon**（nvm 的 node 不在 daemon 的 PATH 里，"
            "但 daemon 会自动探测 `~/.nvm/versions/node/*/bin/node`）\n"
            "3. 任意方式安装后设置 `VLAF_NODE_BIN` 指向 node 可执行文件路径"
        )
        return

    agent_cmd = os.environ.get("VLAF_AGENT_CMD", "") or DEFAULT_AGENT_CMD
    agent_bin = agent_cmd.split()[0]
    if not shutil.which(agent_bin):
        post(
            "## 检视失败\n\n"
            f"CI 机器上未找到 `{agent_bin}`（当前检视 agent 命令：`{agent_cmd}`）。\n\n"
            "安装并登录后重试，或在 `~/.vlaf_ci.conf` 用 `VLAF_AGENT_CMD` 指定其他命令模板"
            "（占位符 `{prompt}` `{output}`，经 shell 执行）。"
        )
        return

    upstream = os.environ.get("VLAF_UPSTREAM", "openeuler/vla-factory")
    owner, _, repo = upstream.partition("/")

    workdir = Path.cwd() / ".tmp" / "gitcode-review" / f"pr-{pr_number}"
    # Clean slate: prompts/issues from a previous (failed or older-head) run
    # must never leak into this one.
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    prompts_json = workdir / "prompts.json"

    # Snapshot the skill into the workdir: the shared submodule can be
    # force-checked-out by a concurrent review's sync, and the agent/node
    # steps must not read a torn tree (a custom VLAF_SKILL_DIR is
    # operator-managed and used in place).
    if skill_source is not None:
        skill_dir = workdir / "skill"
        shutil.copytree(skill_source, skill_dir,
                        ignore=shutil.ignore_patterns("node_modules", ".git"))
    else:
        skill_dir = Path(custom_dir)
    reviewer = skill_dir / "scripts" / "gitcode-reviewer.js"
    if not reviewer.exists():
        post(f"## 检视失败\n\n未找到 vlafactory-code-review 技能: `{reviewer}`")
        return

    # Prompt generation/posting are network bookkeeping; model execution gets
    # its own larger budget. Review agents run concurrently with bounded fanout.
    agent_timeout = _positive_env_int("VLAF_AGENT_TIMEOUT", 600)
    step_timeout = _positive_env_int("VLAF_REVIEW_STEP_TIMEOUT", 180)
    review_workers = _positive_env_int("VLAF_REVIEW_WORKERS", 1)

    # Deliberately minimal child environments: the agent executes PR-author
    # controlled text, so it must never see the daemon's secrets — only the
    # node helper steps get the GitCode token, the agent gets bare basics
    # (PATH/HOME so claude resolves, proxy vars so egress works).
    base_env = {k: v for k, v in os.environ.items()
                if k in _ENV_WHITELIST or k.startswith(_ENV_PREFIX_WHITELIST)}
    node_env = {**base_env,
                "GITCODE_TOKEN": os.environ.get("VLAF_GITCODE_TOKEN", ""),
                "GITCODE_OWNER": owner, "GITCODE_REPO": repo,
                "GITCODE_BASE_URL": "https://api.gitcode.com"}
    agent_env = base_env

    def run(cmd: list[str], timeout: int = agent_timeout,
            env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env or node_env, cwd=workdir)

    post("## 检视运行中\n\n正在调用 vlafactory-code-review 技能，完成后此处更新结果。")

    # 1. Generate one multi-role prompt (auto-review covers all roles).
    # The manifest path is passed explicitly: the reviewer's default is
    # .tmp/gitcode-review/pr-N relative to its cwd, which nests one level
    # below workdir since workdir is also the process cwd.
    lang = os.environ.get("VLAF_REVIEW_LANG", "zh")
    review_guide = os.environ.get("VLAF_REVIEW_GUIDE", "")

    prompt_cmd = [node, str(reviewer), "--pr", str(pr_number),
                  "--auto-review", "--prompts-to", str(prompts_json),
                  "--force", "--comment-language", lang]
    if review_guide:
        prompt_cmd.extend(["--review-guide", review_guide])
    try:
        r = run(prompt_cmd, timeout=step_timeout)
    except subprocess.TimeoutExpired:
        post(f"## 检视失败\n\n生成检视 prompts 超时（{step_timeout} 秒）。")
        return

    # The bundle is written before final diagnostics. A written, parseable
    # bundle therefore wins over the exit code; only fail when there is
    # nothing usable.
    prompts = _load_prompts(workdir) if prompts_json.exists() else []
    if not prompts:
        detail = (r.stderr or r.stdout)[-800:] if r.returncode != 0 else \
            (r.stdout or r.stderr)[-500:]
        post(f"## 检视失败\n\n生成检视 prompts 失败（exit {r.returncode}）：\n\n```\n"
             f"{detail}\n```")
        return
    if r.returncode != 0:
        log.warning("review PR #%d: reviewer exited %d AFTER writing a valid "
                    "bundle (%d prompt(s)) — continuing",
                    pr_number, r.returncode, len(prompts))

    # 2. Execute the prompt with the headless agent → issue-0.json.
    # The template runs through `sh -c` so real-world headless CLIs work,
    # e.g. redirection / command substitution:
    #   VLAF_AGENT_CMD='claude -p "$(cat {prompt})" > {output}'
    log.info("review PR #%d: %d agent prompt(s): %s",
             pr_number, len(prompts), ", ".join(n for n, _, _ in prompts))
    failed = []
    succeeded = 0
    found = 0
    last_error = ""

    def execute_agent(name: str, prompt_file: Path, issue_file: Path):
        t0 = time.time()
        # Paths are shell-quoted on substitution: a path with shell
        # metacharacters must not be interpreted by `sh -c`.
        rendered = agent_cmd.format(prompt=shlex.quote(str(prompt_file)),
                                    output=shlex.quote(str(issue_file)))
        log.info("review PR #%d: agent '%s' running ... (timeout %ds)",
                 pr_number, name, agent_timeout)
        try:
            with _REVIEW_AGENT_SLOTS:
                ar = run(["sh", "-c", rendered], timeout=agent_timeout,
                         env=agent_env)
            if ar.returncode != 0:
                error = (ar.stderr or ar.stdout)[-300:]
                log.warning("review PR #%d: agent '%s' failed (exit %d, %.0fs): %s",
                            pr_number, name, ar.returncode, time.time() - t0,
                            error)
                return name, False, 0, error

            _coerce_json_output(issue_file)
            count = _count_issues(issue_file)
            log.info("review PR #%d: agent '%s' done in %.0fs → %d issue(s)",
                     pr_number, name, time.time() - t0, count)
            return name, True, count, ""
        except subprocess.TimeoutExpired as e:
            log.warning("review PR #%d: agent '%s' timed out: %s", pr_number, name, e)
            return name, False, 0, str(e)

    max_workers = min(review_workers, len(prompts))
    with ThreadPoolExecutor(max_workers=max_workers,
                            thread_name_prefix=f"review-pr-{pr_number}") as pool:
        futures = [pool.submit(execute_agent, *spec) for spec in prompts]
        for future in as_completed(futures):
            name, ok, count, error = future.result()
            if ok:
                succeeded += 1
                found += count
            else:
                failed.append(name)
                last_error = error

    # 3. Collect + post the findings as inline comments. --comment-language
    # is mandatory when posting (resolveCommentLanguage throws otherwise and
    # cannot prompt interactively under the daemon).
    try:
        r = run([node, str(reviewer), "--pr", str(pr_number),
                 "--collect-issues-from", str(workdir),
                 "--post", "--approve-all", "--skip-validation",
                 "--comment-language", lang], timeout=step_timeout)
    except subprocess.TimeoutExpired:
        post(f"## 检视失败\n\n汇总/发布检视结果超时（{step_timeout} 秒）。")
        return
    log.info("review PR #%d: collect+post exit %d, %d issue(s) found in total",
             pr_number, r.returncode, found)
    if r.returncode != 0:
        post(f"## 检视失败\n\n汇总/发布检视结果失败（exit {r.returncode}）：\n\n```\n"
             f"{(r.stderr or r.stdout)[-800:]}\n```")
        return

    if not succeeded:
        post(f"## 检视失败\n\n所有 {len(prompts)} 个 agent 均执行失败"
             f"（VLAF_AGENT_CMD：`{agent_cmd}`）。最后一次错误输出：\n\n"
             f"```\n{last_error}\n```")
        return
    parts = [f"## 检视完成 — {succeeded}/{len(prompts)} 个 agent 成功，"
             f"共发现 {found} 个问题"]
    if failed:
        parts.append(f"\n失败的 agent: {', '.join(failed)}")
    parts.append("\n问题已作为行内评论贴出。" if found else "\n未发现问题。")
    post("\n".join(parts))
    log.info("review PR #%d complete: %d issue(s) posted as inline comments",
             pr_number, found)


def _count_issues(issue_file: Path) -> int:
    """Number of issues in an agent's output JSON (0 = none/invalid)."""
    try:
        data = json.loads(issue_file.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and isinstance(data.get("issues"), list):
        return len(data["issues"])
    return 0


def _load_prompts(workdir: Path) -> list[tuple[str, Path, Path]]:
    """Read prompts.json and pair each agent with its prompt/issue file.

    prompts.json is written by gitcode-reviewer.js --auto-review
    --prompts-to as ``{"agents": [{index, name, promptPath, issuePath}]}``.
    """
    prompts_file = workdir / "prompts.json"
    try:
        data = json.loads(prompts_file.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("failed to read %s: %s", prompts_file, e)
        return []
    out = []
    for agent in data.get("agents", []):
        prompt = workdir / agent.get("promptPath", "")
        issue = workdir / agent.get("issuePath", f"issue-{agent.get('index')}.json")
        if prompt.exists():
            out.append((agent.get("name", f"agent-{agent.get('index')}"), prompt, issue))
    return out


def _coerce_json_output(issue_file: Path) -> None:
    """Best-effort: if the agent wrapped its JSON in prose/fences, unwrap it.

    Agents occasionally emit markdown fences around the issue array; strip
    them so --collect-issues-from can parse the file. Missing files are
    left alone (the agent found nothing worth reporting).
    """
    if not issue_file.exists():
        return
    try:
        issue_file.write_text(_extract_json(issue_file.read_text()))
    except OSError:
        pass


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()
