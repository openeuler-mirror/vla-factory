# VLA adaptation harness — first executable slice

## Single entry point

Run the complete adaptation workflow with no command-line parameters:

```bash
python run.py
```

`run.py` asks for the upstream repository, VLA/checkpoint source (Hugging Face
ID/URL, ModelScope URL, or local directory), Factory revision,
benchmark, job location, and maximum repair attempts. It persists those inputs,
runs both candidate Agents, runs Judge validation, and feeds a structured Judge
failure report back to the responsible Agent for a bounded repair attempt. It
always writes one final conclusion to
`<job-dir>/artifacts/final-verdict.json`: `PASS`, `FAIL`, or `INCOMPLETE`.

Resume an interrupted job without repeating those prompts:

```bash
python run.py --resume runs/<job-id>
```

The saved repair budget is reused. For a legacy job that has no saved workflow
state, or to change the remaining budget explicitly, add
`--max-repairs <count>`.

Design background: the execution architecture lives in
[`docs/architecture/auto-adapt-harness-execution.cn.md`](../docs/architecture/auto-adapt-harness-execution.cn.md)
and the workflow diagram in
[`docs/graph/auto-adapt-workflow.cn.svg`](../docs/graph/auto-adapt-workflow.cn.svg).

This is the first, intentionally small execution loop. It has no LLM Judge.
Its verdict is native-loss parity between an upstream policy and a VLA-Factory
bundle on a Judge-selected, recorded benchmark sample.

## Agent runner

Both candidate-producing roles are invoked only as:

```text
opencode run --format json --model csi-provider/GLM-5.2 --dir <workspace> <role-specific prompt>
```

This is OpenCode's non-interactive, JSON event mode. During `python run.py`
setup, the interactive model menu defaults to `csi-provider/GLM-5.2` and lists
all currently registered CSI models. The selection is stored in
`manifest.json`, so resumed tasks and parity diagnosis keep the same model.
Direct `python -m adapt_vla_model init` callers can use `--opencode-model` with any
registered model. The harness does not accept an arbitrary agent command.

The initial Reference and Adapter tasks run concurrently in their separate
workspaces. Candidate-specific repair tasks remain ordered and operate only on
the retained workspace for the failed role.

Judge records Factory source compliance before running its installer. After
that immutable checkpoint, Reference and Factory preflight run concurrently in
their isolated runtimes. Judge then creates the one shared parity sample;
Reference and Factory loss capture run concurrently against that sample, and
comparison waits for both results. If parity diagnosis selects both roles, the
two role-local repairs also run concurrently and rejoin before the next Judge
cycle.

Transient OpenCode provider failures are infrastructure failures, not candidate
repair failures. For an explicitly retryable response, HTTP 408/429/5xx, or a
connection timeout/reset, Harness restarts the same OpenCode task after 5, 15,
and 30 seconds while retaining its workspace. These retries do not consume the
candidate repair budget. If all four infrastructure attempts fail, the workflow
ends as `INCOMPLETE` and records the provider failure separately instead of
reporting an invalid candidate.

All workflow code lives in the `adapt_vla_model/` package: `prompts.py` holds the role-local Agent boundaries (Reference / Adapter / Diagnosis), `runtime.py` the shared Manifest and OpenCode invocation code, and `judge.py` the Judge itself (`python -m adapt_vla_model`, run from `agent/`). `contract.py` is the Policy protocol copied into every bundle, `benchmarks/` holds Judge-owned benchmark discovery plus the benchmark data itself, and `parity/` holds the loss-capture scripts executed by path inside candidate venvs.

## Judge-owned benchmark

The default benchmark is `lerobot-leisaac-pickup-cude-30-v2` (6D SO101 data;
it needs only a local LeRobot dataset, not a simulation environment); agents
never choose the task, data, seed, metric, or tolerance of any benchmark.
`robotwin-beat-block-hammer` is also available for the RoboTwin protocol; its
complete Judge-owned definition is:

```text
RoboTwin / beat_block_hammer / demo_randomized / unseen / seed 0 / 100 episodes
```

```text
adapt_vla_model/benchmarks/robotwin/beat_block_hammer/
├── spec.json              # identity, task, seed, comparison
├── REFERENCE_BUNDLE_CONTRACT.md # Reference endpoint contract
└── run.py                 # Judge-only execution and result parsing
```

To add a benchmark, add one directory under `adapt_vla_model/benchmarks/<platform>/`
with those three files and, where its platform boundary differs, a
benchmark-specific Factory Bundle Contract. Judge
discovers it by `spec.json:name`; Agent prompts receive only their role's
contract.
The default benchmark `lerobot-leisaac-pickup-cude-30-v2` uses the
platform-owned local dataset cache, not a dataset copy in an Agent workspace:

```bash
export LEROBOT_DATASET_ROOT="$HOME/.cache/huggingface/lerobot"
```

Select it only when creating the job; its name is recorded in the manifest and
cannot be changed by either agent:

```bash
python -m adapt_vla_model init /tmp/vla-job \
  --job-id smolvla-so101 --upstream-repo https://example/upstream.git \
  --vla smolvla --checkpoint lerobot/smolvla_base --factory-base main \
  --benchmark lerobot-leisaac-pickup-cude-30-v2
```

Set platform-owned environment variables before creating a job:

```bash
export ROBOTWIN_ROOT=/path/to/RoboTwin
export ROBOTWIN_PY=/path/to/robotwin-python       # Python from the RoboTwin environment
export VLA_FACTORY_PATH=/path/to/vla-factory      # optional; defaults to this repo's parent
```

The Judge reads its fixed parity sample from `$ROBOTWIN_ROOT/data/`; it never
copies benchmark data into an Agent workspace.

`init` records the selected benchmark's installed revision in the Manifest. It
accepts no task, data, seed, metric, or tolerance override.

```bash
python -m adapt_vla_model init /tmp/vla-job \
  --job-id pi05-demo --upstream-repo https://example/upstream.git \
  --vla pi05 --checkpoint /models/pi05 --factory-base main
```

`init` also clones the upstream repository and creates a detached VLA-Factory worktree under the job. This isolates both Agent workspaces and pins their Git revisions in `workspace-lock.json`:

```bash
python -m adapt_vla_model agent /tmp/vla-job reference
python -m adapt_vla_model agent /tmp/vla-job adapter
```

The Reference role must place a `runtime.json` in its assigned `.judge/` bundle:

```json
{
  "command": ["python", "serve_robotwin_policy.py", "--port", "{port}"],
  "cwd": "/work/policy",
  "protocol": "robotwin_tcp"
}
```

Reference also provides `policy_impl.py`, `policy_spec.json`, `install_<model>.sh`, and `smoke_test_<model>.py`.

Adapter instead provides a Factory-native checkpoint bundle:

```text
.judge/factory_checkpoint/
├── model.pt
├── weights.json
└── inference_metadata/{assembly.json,recipe.yaml}
```

with `factory_bundle.json`, an installer, and a smoke test that loads
`InferenceEngine(factory_checkpoint)`. Adapter does not provide a policy
implementation or runtime command: Judge starts its fixed `adapt_vla_model/factory_policy_server.py`,
which invokes the VLA-Factory RoboTwin deployment chain. The Agent must run its
own installer and smoke test before handoff; Judge repeats them and keeps its
own logs. For API discovery, Judge also provides a complete isolated upstream
source snapshot at `.judge/upstream_source/` (excluding only `.git` and the
Reference Agent's private `.judge` artifacts):
it is source code and repository metadata, never checkpoint weights.

Agent self-tests and the corresponding Judge preflight use the same persistent,
role-local uv and pip caches, normalized proxy variables, configured package
index, and CPU Torch backend. Judge still deletes and recreates the candidate
venv for every preflight: sharing a download cache does not reuse `site-packages`.
Installers must use uv/pip's resolver and the supplied cache variables; they
must not scan uv's internal `archive-v0`, assemble `site-packages` from cached
trees, discover host caches, or hard-code a different mirror or proxy.

The files actually live beside their source workspace, not in duplicate top-level folders:

```text
<job>/workspaces/upstream/.judge/runtime.json
<job>/workspaces/factory/.judge/factory_bundle.json
```

Judge allocates a distinct port for each candidate and replaces `{port}`. Both services must implement RoboTwin's native TCP `ModelClient` protocol. Judge then runs the fixed client once per candidate:

```bash
python -m adapt_vla_model judge /tmp/vla-job
```

Run the standard-library checks with:

```bash
python -m unittest -v
```
