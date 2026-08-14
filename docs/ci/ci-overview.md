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
| **base** | core + `[dev]` | L0 | all of L0 (162 cases¹) |
| **act** | + lerobot | L1 + L2 | lerobot parity + overfit smoke (planned, see §4) |
| **pi** | + openpi | L1 | openpi parity (planned, see §4) |

¹ Measured with `pytest --collect-only` on master: 161 test functions plus
1 parametrize expansion. The number drifts as tests are added; re-measure
when updating this document.

L1 tests self-route via `pytest.importorskip`: the act environment runs the
lerobot-related cases, the pi environment runs the openpi-related ones,
without interference. Missing dependencies produce skips, not errors.

> openpi and lerobot cannot coexist in one environment (openpi pins an old
> lerobot via a uv git source), hence the separate environments. See
> `scripts/ci/build_ci_envs.sh`.

---

## 4. Test tiers

Tests are layered by **what they verify**, isolated with pytest markers.

| Tier | Marker | Verifies | Cost | Trigger |
|------|--------|----------|------|---------|
| **L0** unit | none (`not l1 and not l2 and not l3`) | our own code: module behavior, edge cases, error paths | seconds, CPU | every PR |
| **L1** parity | `@pytest.mark.l1` | adopted upstream semantics: transform chains, normalization formulas, PEFT mounting | seconds, CPU | every PR |
| **L2** smoke | `@pytest.mark.l2` | end-to-end connectivity: single-episode overfit | minutes, CPU/GPU | every PR |

The markers are registered under `[tool.pytest.ini_options] markers` in
`pyproject.toml`. No test on master currently carries `l1`/`l2` (the parity
and smoke tests live on the `dev_ci-backup` branch, see below), so **for now
L0 equals the full suite** and the L1/L2 tiers collect zero tests — the
daemon reports FAIL, not pass, for an environment whose tiers all collect
zero, so an empty run can never show up green.

### L0 — unit tests (162 cases)

Verify **our own code**. No model extras required; the base environment
runs everything. Master's `test/` currently holds 13 files with 161 test
functions (162 collected after parametrize expansion):

**Config / CLI**

| File | Cases | Coverage |
|------|-------|----------|
| `test_protocols_registry_config.py` | 4 | Protocol contracts (Observation/ActionSpec/VLA hierarchy/ModelMetadata), registry (register/lookup/duplicate/unknown model), YAML→TrainRecipe parsing, all `examples/*.yaml` parseable |
| `test_cli_deploy.py` | 2 | `deploy` command registration / invalid-argument exit; `serve` not registered |

**Model layer (model/)**

| File | Cases | Coverage |
|------|-------|----------|
| `test_checkpoint_validation.py` | 12 | Optional checkpoint config parsing and ModelMetadata consistency checks; metadata-based camera mapping |
| `test_act_model.py` | 15 | ACT lerobot adapter: protocol compliance, registry integration, observation_to, factory wrapper (compute_loss/predict/multi-camera/save-load), profile defaults & recipe overrides |
| `test_pi0_model.py` | 4 | pi0 adapter (fake openpi): metadata, camera_mapping translation, loss/predict delegation, empty-camera placeholder |
| `test_pi05_model.py` | 13 | pi05 deltas vs pi0: factory variant construction, discrete-state prompt, task fallback chain, quantile normalize/unnormalize roundtrip |
| `test_lora_strategy.py` | 8 | LoRA strategy logic (fake peft): single/multi subtree wrapping, merge unwrap, target-component validation, strict field validation |

**Training (training/)**

| File | Cases | Coverage |
|------|-------|----------|
| `test_phase4_engine.py` | 8 | Training engine: strategy dispatch (full/freeze/selective + unknown raises), recipe→training-args mapping, 3-step CPU training loop |
| `test_training_strategy_registry.py` | 5 | Strategy registration, unknown-name diagnostics, config field/type validation, legacy-field rejection, and one-class extension example |

**Data pipeline (data/)**

| File | Cases | Coverage |
|------|-------|----------|
| `test_data_pipeline.py` | 41 | End-to-end data pipeline (bundled 3-episode lerobot dataset): LeRobotV3 reader, PyAV codec decode, deterministic all-episode `SampleWindow` construction, transforms, VLADataset, DataLoader batching |
| `test_robotwin_reader.py` | 7 | RoboTwin reader + codec happy path (synthetic dataset): can_read, schema, episode length/range, state/action reads, frame decode, norm_stats |

**Deploy / inference (deploy/)**

| File | Cases | Coverage |
|------|-------|----------|
| `test_inference_engine.py` | 31 | Inference engine: ObsDict build/freeze, 3 execution policies (synchronous / receding-horizon / temporal-ensembling), obs normalization, train↔infer consistency (30 functions + 1 parametrize expansion) |
| `test_policy_runtime.py` | 7 | PolicyRunner orchestration: fake transport+engine, predict/send, action-adapter, reset control |
| `test_robotwin_server.py` | 13 | RoboTwin platform adapter + length-prefixed transport: get_action roundtrip, obs parsing, numpy codec roundtrip |

### L1 — parity tests (planned, not yet merged)

> The L1 parity files (`test/parity/*.py`) currently live on the
> `dev_ci-backup` branch and are **not yet merged into master**. Running
> `pytest -m l1` on master collects zero tests; the tier shows `— (skip)`
> and its environment is reported FAIL (see §3) — so do not configure
> act/pi environments for the daemon until the parity files land. The
> table below is the plan that takes effect once they merge.

Verify that **adopted upstream semantics** match the official
implementations. Golden values are embedded in the test code (constants /
reference implementations), no external `.npz`. Each upstream contract is
pinned to a source commit; missing dependencies auto-skip via
`importorskip`.

| File (planned) | Cases | Upstream | Verified contract |
|------|-------|----------|-------------------|
| `test/parity/utils.py` | — (helper) | — | `assert_tensor_parity`: reports first mismatching element position / both values / shape / dtype |
| `test_normalize_parity.py` | 10 | openpi (eps 1e-6) + lerobot (eps 1e-8) | eps is a per-model upstream contract; config eps reaches the arithmetic; two orders of magnitude apart; openpi pin has not drifted |
| `test_openpi_pipeline_parity.py` | ~10 | openpi (`PI0Pytorch`) | Full-chain pi0/pi05 parity: state/actions element-wise equal, image role matching, letterbox padding, prompt token alignment |
| `test_act_pipeline_parity.py` | 6 | lerobot (`processor_act`) | Full-chain ACT parity: state/actions/images element-wise equal, channels-first layout, ImageNet normalization equivalence |
| `test_peft_parity.py` | 10 | peft (tensor-level) + openpi (contract-level) | LoRA mount surface tensors identical, scaling formula == openpi, adapter stays float32 on bf16 base, merge writes delta |

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
# 1. Prepare test environments (at minimum base; L1 needs act/pi extras)
#    The daemon clones the repo automatically on first start — no manual clone
bash scripts/ci/build_ci_envs.sh base          # minimum: L0
# bash scripts/ci/build_ci_envs.sh base act pi # full coverage: L0 + L1 + L2
```

### Day-to-day start

```bash
python3 scripts/run_ci.sh
```

Interactive configuration (first run only; saved to `~/.vlaf_ci.conf` and
reused afterwards):

```
============================================================
  vla-factory CI daemon configuration
  (defaults in brackets; press Enter to accept)
============================================================

  GitCode token [13rYhn...]:              ← auto-read from config.json
  CI directory (auto-cloned if missing) [~/vla-factory-ci]:
  Poll interval (seconds) [30]:

  Test environments (leave act/pi empty to skip; L0 only):

  base python (L0) [/.../python3.12]:     ← current interpreter
  act python (L1+L2, empty to skip) []:
  pi python (L1, empty to skip) []:
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
| `VLAF_BASE_DIR` | CI directory (auto-cloned if missing) | `~/vla-factory-ci` |
| `VLAF_ENV_BASE` | base environment python (required) | current interpreter |
| `VLAF_ENV_ACT` | act environment python (optional) | empty |
| `VLAF_ENV_PI` | pi environment python (optional) | empty |
| `VLAF_POLL_INTERVAL` | poll interval in seconds | 30 |
| `VLAF_DB_PATH` | dedup DB path | `~/.vlaf_ci.db` |
| `VLAF_UPSTREAM` | upstream repository | `openeuler/vla-factory` |
| `VLAF_ENV_BASE_TIERS` | tiers for base env (override) | `l0` |
| `VLAF_ENV_ACT_TIERS` | tiers for act env (override) | `l1,l2` |
| `VLAF_ENV_PI_TIERS` | tiers for pi env (override) | `l1` |

### PR comment format

The daemon posts one "running" comment per PR, then edits it into the
result table when the run finishes.

With the current setup (base environment only, L0 = full suite) the actual
output looks like:

```markdown
## CI 测试报告 — pass

branch: `dev_ci` · commit: `517028f8f6fe` · all tests passed · 32s

| 环境 | L0 单元 | L1 parity | L2 冒烟 | 耗时 |
|------|---------|-----------|---------|------|
| base | 159 passed, 3 skipped | — (skip) | — (skip) | 21s |
```

Once the parity/smoke tests merge and act/pi environments are configured,
the table grows to multiple environments and tiers.

---

## 6. Components and file layout

```
scripts/
  install.sh               model environment install (pi0/pi05)
  run_ci.sh                CI daemon entry (thin shell → ci/run_ci.py)
  ci/
    run_ci.py              interactive launcher (configure → save → start daemon)
    daemon.py              core daemon (poll GitCode → run tests → comment)
    pr_reporter.py         GitCode PR comment tool (create/edit/format)
    parse_results.py       junit XML → structured result summaries
    build_ci_envs.sh       one command to build the base/act/pi venvs
```

**Core daemon loop**:

```python
while True:
    prs = fetch_open_prs()              # GET /pulls?state=open (all authors)
    for pr in prs:
        if is_seen(pr.number, pr.sha):  # SQLite dedup
            continue
        process_pr(pr)                  # fetch → pytest → comment
        mark_seen(pr.number, pr.sha)    # record as processed
    sleep(30)
```

`process_pr` executes the environment × tier assignment: base runs L0, act
runs L1+L2, pi runs L1. Each tier calls `pytest --junitxml` directly and
does not depend on scripts from the PR branch (which may be out of sync).
A single tier exiting 5 (nothing collected) is not a failure, but **an
environment whose tiers all collect zero tests is reported FAIL** — empty
summaries must never pass, so a misconfigured environment cannot be
silently swallowed.

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
