# vla-factory CI System

> Related issue: #7 (test pipeline construction)

---

## 1. Background and constraints

### Why not GitCode native CI

On 2026-07-29 we pushed a probe `.gitcode/workflows/l0.yml` with an
`on: push` trigger to the fork — **no job ever ran**. Further digging
showed that **no repository** under the openeuler organization carries
`.gitcode/workflows/` or `.github/workflows/`; their CI is driven by a
repo-root `Jenkinsfile` + shared `ci-scripts/` (Jenkins configured outside
the repo). Joining that system requires an onboarding request to
sig-infrastructure — it is not something a single committed file enables.

### GitCode API capabilities (verified by probing)

| Capability | Status | Notes |
|------------|--------|-------|
| PR list / detail API | ✅ | `GET /pulls?state=open`, returns PRs from all authors |
| PR comment API | ✅ | Create returns `note_id` (numeric, used for editing) and `id` (hash) |
| Commit status API | ❌ 404 | `/statuses/{sha}`, `/commits/{sha}/statuses`, `/check-runs` all absent |
| Webhook on fork | ✅ | `push_events` / `merge_requests_events` (boolean flags), basic auth only |
| Webhook on upstream | ❌ 403 | Not possible without admin |

---

## 2. Design evolution

### Attempt 1: webhook on fork → abandoned

The initial design hung a `push_events` webhook on the fork, with a VPS
receiving events and querying the PR list. Probing revealed the fundamental
limit: **a webhook on the fork only captures the fork owner's pushes** —
PRs opened by other contributors from their own forks never trigger it.
The team has multiple contributors (hezhenhao2, leningchen_admin, …);
missing anyone's PR is unacceptable.

### Attempt 2: VPS + webhook + polling fallback → simplified

To cover all authors, a polling thread (scanning all open PRs every 30s)
was added on top of the webhook. But once polling covers every PR, the
webhook's second-level latency advantage is marginal — at the cost of a
public port, a basic-auth secret, and a dependency with an undocumented
payload format. Not worth it.

### Final design: pure polling, no VPS, no webhook

Drop the webhook and the VPS middle layer. A single Python process on the
local GPU machine polls the GitCode PR API directly, covers PRs from
**all contributors**, and needs only outbound HTTPS.

---

## 3. Final architecture

```
┌─ Local GPU machine (single-process daemon) ─────────────┐
│                                                          │
│  Every 30s:                                              │
│    1. GET GitCode /pulls?state=open                      │
│       → open PRs from all authors                        │
│    2. Diff against local SQLite; new head SHA → process  │
│    3. git fetch PR head → checkout --detach              │
│    4. Run pytest per environment × tier → junit XML      │
│    5. Parse results → format markdown table              │
│    6. POST / edit the PR comment                         │
│    7. Record SHA in SQLite (dedup)                       │
│                                                          │
└──────────────────────────────────────────────────────────┘
         │  outbound HTTPS
         ▼
    GitCode API
```

### Design decisions

| Choice | Rationale |
|--------|-----------|
| Pure polling, no webhook | A fork webhook only covers the fork owner; polling covers **all contributors** |
| No VPS middle layer | Without a webhook there is no public port to host; the local machine talks to the GitCode API directly |
| Local SQLite dedup | `UNIQUE(pr_number, head_sha)` guarantees one run per (PR, SHA) |
| Results as PR comments | GitCode has no commit status API — no other option |
| Edit one comment | Never append new comments; a push must not spam the thread |

### Environment × tier assignment

Each test environment runs only the tiers it can cover, no duplication:

| Env | Deps | Tiers | Covered cases |
|-----|------|-------|---------------|
| **base** | core + `[dev]` | L0 | all framework contract tests |
| **act** | + lerobot | L1 + L2 | lerobot parity + overfit smoke (planned, see §4) |
| **pi** | + openpi | L1 | openpi parity (planned, see §4) |

L1 tests self-route via `pytest.importorskip`: the act environment runs the
lerobot-related cases, the pi environment runs the openpi-related ones,
without interference. Missing dependencies produce skips, not errors.

> openpi and lerobot cannot coexist in one environment (openpi pins an old
> lerobot via a uv git source), hence the separate environments. See
> `scripts/ci/build_ci_envs.sh`.

---

## 4. Test tiers

Tests are layered by **what they verify** and collected from tier directories.

| Tier | Marker | Verifies | Cost | Trigger |
|------|--------|----------|------|---------|
| **L0** unit | `test/l0/` | our own code: module behavior, edge cases, error paths | seconds, CPU | every PR |
| **L1** parity | `test/l1/` + `@pytest.mark.l1` | adopted upstream semantics: transform chains, normalization formulas, PEFT mounting | seconds, CPU | every PR |
| **L2** smoke | `test/l2/` + `@pytest.mark.l2` | end-to-end connectivity: single-episode overfit | minutes, CPU/GPU | every PR |

The markers are registered under `[tool.pytest.ini_options] markers` in
`pyproject.toml`. Tiers are selected by directory, so a test's execution
level is visible from its path as well as its marker. A tier
that collects zero tests (junit `tests=0`, pytest exit 5) is a **skip**,
not a FAIL; an environment only FAILs when its report directory is missing
entirely (the env never ran — crash / misconfiguration).

### L0 — framework tests

These verify **our own code**. The base environment runs the general tests;
cases that require optional model dependencies skip explicitly when those
dependencies are absent. Tests are organized around stable behavioral
contracts. Historical phase verification scripts are not retained, and this
document does not duplicate a per-file case count that quickly becomes stale.

| Subsystem | Main contracts |
|-----------|----------------|
| Data | Reader / codec registration and discovery, semantic inference, real data reads, transforms, sample windows, and DataLoader behavior |
| Model | ModelMetadata field classification, built-in declarations, external plugins, optional checkpoint consistency, and ACT / PI0 / PI05 adapters |
| Assembly | Compatibility diagnostics, mappings, ModelIOSpec, TransformPipelinePlan, persistence, and interface-drift rejection |
| Training | Strategy registration and strict config, LoRA mounting, and real train → checkpoint output |
| Inference / Deployment | Execution policies, dual action widths, train → infer round trip, platform adapters, PolicyRunner, and RPC transport |
| User Interface | Recipe parsing and rejection paths, inspect output, CLI registration, and failure-side-effect boundaries |

Test counts change with parametrization and optional dependencies, so CI treats
pytest output—not this document—as the source of truth.

### L1 — parity tests

Verify that **adopted upstream semantics** match the official
implementations. Golden values are embedded in the test code (constants /
reference implementations), no external `.npz`. Each upstream contract is
pinned to a source commit; missing dependencies auto-skip via
`importorskip`.

| File | Cases | Upstream | Verified contract |
|------|-------|----------|-------------------|
| `test/l1/utils.py` | — (helper) | — | `assert_tensor_parity`: reports first mismatching element position / both values / shape / dtype |
| `test/l1/test_normalize_parity.py` | 10 | openpi (eps 1e-6) + lerobot (eps 1e-8) | eps is a per-model upstream contract; config eps reaches the arithmetic; two orders of magnitude apart; openpi pin has not drifted |
| `test/l1/test_openpi_pipeline_parity.py` | ~10 | openpi (`PI0Pytorch`) | Full-chain pi0/pi05 parity: state/actions element-wise equal, image role matching, letterbox padding, prompt token alignment |
| `test/l1/test_act_pipeline_parity.py` | 6 | lerobot (`processor_act`) | Full-chain ACT parity: state/actions/images element-wise equal, channels-first layout, ImageNet normalization equivalence |
| `test/l1/test_peft_parity.py` | 10 | peft (tensor-level) + openpi (contract-level) | LoRA mount surface tensors identical, scaling formula == openpi, adapter stays float32 on bf16 base, merge writes delta |

### L2 — end-to-end smoke (planned, not yet merged)

> Same as L1: the L2 files live on `dev_ci-backup` and are not yet merged.
> Requires the `[act]` extra (ACT runs on CPU); pi0/pi05 need a GPU.

Verify **end-to-end connectivity**: train a single episode to near-zero
loss. A joint assertion — data, normalization, model input contract,
trainable-parameter set — any broken link fails it.

| File (planned) | Coverage |
|------|----------|
| `test/integration/test_overfit_smoke.py` | Loss drops by an order of magnitude (vs degenerate baseline), training artifact rebuilds the same episode through the inference path, actual trainable-parameter set == the set declared by `ModelMetadata.components`, language-conditioning path alive |
| `test/integration/thresholds.yaml` | Per-model overfit thresholds |

---

## 5. Usage

### First-time setup

```bash
# 1. Prepare the three required test environments
#    The daemon clones the repo automatically on first start — no manual clone
bash scripts/ci/build_ci_envs.sh base act pi
```

### Day-to-day start

```bash
bash scripts/run_ci.sh
```

### Local CI-equivalent run

```bash
python3 scripts/ci/run_local_ci.py
```

This uses the same configured base/act/pi interpreters and daemon test matrix.
It refuses a dirty worktree by default because remote CI tests a clean detached
commit; use `--allow-dirty` only while iterating before a commit.

Interactive configuration (first run only; saved to `~/.vlaf_ci.conf` and
reused afterwards):

```
============================================================
  vla-factory CI daemon configuration
  (defaults in brackets; press Enter to accept)
============================================================

  GitCode token [13rYhn...]:              ← auto-read from config.json
  Hugging Face token (PaliGemma access required):
  CI directory (auto-cloned if missing) [~/vla-factory-ci]:
  Poll interval (seconds) [30]:

  Test environments (all required for L0/L1/L2 coverage):

  base python (L0) [/.../python3.12]:     ← current interpreter
  act python (L1+L2) [/.../envs/act/bin/python]:
  pi python (L1) [/.../envs/pi/bin/python]:
```

Once configured, the daemon polls and automatically tests + comments on
new PRs.

### Persistent service (systemd)

```ini
# /etc/systemd/system/vlaf-ci.service
[Unit]
Description=vla-factory CI daemon
After=network.target

[Service]
Type=simple
EnvironmentFile=/home/you/.vlaf_ci.conf
WorkingDirectory=/home/you/vla-factory-ci
ExecStart=/usr/bin/python3 scripts/ci/daemon.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vlaf-ci
```

### Configuration reference

| Variable | Description | Default |
|----------|-------------|---------|
| `VLAF_GITCODE_TOKEN` | GitCode access token (required) | read from config.json |
| `HF_TOKEN` | Hugging Face token authorized for `google/paligemma-3b-pt-224` (required; PI L1) | configured value |
| `VLAF_BASE_DIR` | CI directory (auto-cloned if missing) | `~/vla-factory-ci` |
| `VLAF_ENV_BASE` | base environment python (required; L0) | configured path |
| `VLAF_ENV_ACT` | act environment python (required; L1/L2) | configured path |
| `VLAF_ENV_PI` | pi environment python (required; L1) | configured path |
| `VLAF_POLL_INTERVAL` | poll interval in seconds | 30 |
| `VLAF_DB_PATH` | dedup DB path | `~/.vlaf_ci.db` |
| `VLAF_UPSTREAM` | upstream repository | `openeuler/vla-factory` |
| `VLAF_CI_WORKERS` | CI task pool concurrency | 5 |
| `VLAF_CMD_WORKERS` | comment-command pool concurrency | 5 |
| `VLAF_TIER_TIMEOUT` | per-tier pytest timeout (s) | 1200 |
| `VLAF_MAX_RETRIES` | crash retry cap per (pr, sha) | 3 |
| `VLAF_CHECKOUT_TTL_DAYS` | per-PR checkout GC (days) | 7 |

### /vla-factory comment commands (review knobs)

Anyone can trigger these by a whole-line comment on an open PR (auth and
rate limiting below):

- `/vla-factory help` — list commands
- `/vla-factory retest` — re-run CI on the current head (refused while running)
- `/vla-factory review` — run the vlafactory-code-review skill from the
  skills submodule; findings are posted as inline comments

| Variable | Description | Default |
|----------|-------------|---------|
| `VLAF_AGENT_CMD` | headless agent template (`{prompt}`/`{output}` shell-quoted on substitution) | claude headless (tools and slash commands disabled) |
| `VLAF_AGENT_TIMEOUT` | per-reviewer execution timeout (s) | 600 |
| `VLAF_REVIEW_STEP_TIMEOUT` | prompt-gen / collect+post timeout (s) | 180 |
| `VLAF_REVIEW_WORKERS` | reviewers per PR | 1 |
| `VLAF_REVIEW_GLOBAL_WORKERS` | daemon-wide reviewer cap | 4 |
| `VLAF_REVIEW_COOLDOWN_MIN` | per-PR review cooldown in minutes (0 = off) | 30 |
| `VLAF_REVIEW_ALLOWLIST` | logins allowed to trigger review (comma-separated, empty = anyone) | empty |
| `VLAF_REVIEW_LANG` | review comment language (en/zh) | zh |
| `VLAF_SKILL_DIR` | skill dir override (default: skills submodule, synced to latest main) | empty |
| `VLAF_NODE_BIN` | explicit node binary path | auto-detected |

### PR comment format

The daemon posts one "running" comment per PR, then edits it into the
result table when the run finishes.

With all three configured environments, the result table reports their
assigned tiers:

```markdown
## CI 测试报告 — pass

branch: `dev_ci` · commit: `517028f8f6fe` · all tests passed · 32s

| 环境 | L0 单元 | L1 parity | L2 冒烟 | 耗时 |
|------|---------|-----------|---------|------|
| base | 159 passed, 3 skipped | — (skip) | — (skip) | 21s |
```

The daemon runs this three-environment matrix for every new PR head.

---

## 6. Components and file layout

```
scripts/
  install.sh               model env install (--model optional; all+confirm)
  run_ci.sh                CI daemon entry (thin shell → ci/run_ci.py)
  ci/
    run_ci.py              interactive launcher (configure → save → start daemon)
    daemon.py              core daemon (poll GitCode → run tests → comment)
    pr_reporter.py         GitCode PR comment tool (create/edit/format)
    parse_results.py       junit XML → structured result summaries
    build_ci_envs.sh       one command to build the base/act/pi venvs
```

**Core daemon loop** (the main loop only polls and dispatches; tasks run
asynchronously on two thread pools):

```python
while True:
    gc_old_checkouts()                          # TTL-reclaim merged PRs' dirs
    prs = fetch_open_prs()                      # GET /pulls?state=open (all authors)
    for pr in prs:
        poll_commands(pr.number)                # comment commands → cmd pool
    for pr in prs:
        if is_seen(pr.number, pr.sha):          # done/failed/running all count
            continue
        mark_seen(pr.number, pr.sha, "running") # claim before dispatch
        ci_pool.submit(ci_task, pr)             # fetch → pytest → comment
    sleep(30)
```

`ci_task` runs in a dedicated checkout (`pr-<N>-<sha8>`; concurrent runs
never share a worktree) with the environment × tier assignment: base runs
L0, act runs L1+L2, pi runs L1. Each tier calls `pytest --junitxml`
directly and does not depend on scripts from the PR branch (which may be
out of sync). A single tier exiting 5 (nothing collected) is not a
failure — zero-collected tiers are skips; an environment only FAILs when
its report directory is missing entirely. A tier that exceeds its timeout
writes a synthetic failed test (`tier-timeout`) instead of retrying
forever, and an SHA that keeps crashing stops after the retry cap.

---

## 7. Security and risk

| Threat / risk | Mitigation |
|---------------|------------|
| Token leak | Stored only in environment variables / systemd unit, never in the repo; API requests carry it in the `Authorization: Bearer` header, never the URL (avoids proxy/log leaks) |
| Executing malicious PR code | `checkout --detach` + `git clean -qfd`; the working tree is disposable |
| PR comment API rate limits | SHA dedup + editing one comment (no appending) |
| GitCode PR API changes | Only the standard `GET /pulls?state=open`, the most stable endpoint |
| Merge ref missing (conflict) | Caught → comment reports "fetch failed" |
| Daemon dies, PRs missed | SQLite treats only `done`/`failed` as complete; `running`/`crashed` rows are retried after restart, reusing the recorded comment_id to edit the original "running" comment (no leftovers) |
| openpi + lerobot env conflict | Three isolated environments (`build_ci_envs.sh`) |
