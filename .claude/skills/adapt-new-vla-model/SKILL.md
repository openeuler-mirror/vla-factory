---
name: adapt-new-vla-model
description: Adapt a new upstream VLA model (lerobot ACT / openpi PI0 style) into VLA Factory — dependency routing into pyproject + scripts/install.sh, Observation coverage analysis and ModelMetadata fact declaration so the resolver plans data_to_model, robot_to_model / model_to_robot plan checks, and L1 pipeline parity tests. Use whenever integrating, wrapping, or registering a new model family (e.g. "adapt SmolVLA into vla-factory", "add a pi0-FAST variant"), whenever writing or editing anything under vla_factory/model/adapters/, when declaring or debugging ModelMetadata facts and the transform plans they produce, when adding a model extra to pyproject.toml or a branch to scripts/install.sh, or when writing test/l1 pipeline-parity tests — even if the request covers only one of these steps.
---

# Adapt a New VLA Model

The framework
owns no architecture: you write a thin adapter (`vla_factory/model/adapters/<name>.py`)
that wraps the upstream model by composition, declare its interface as
`ModelMetadata` facts, and let the composition resolver derive the pipelines.
Worked examples to mirror: `adapters/act.py` (lerobot `ACTPolicy`, pip-friendly
route), `adapters/pi0.py` + `adapters/openpi.py` (openpi `PI0Pytorch`,
uv-required route), and `adapters/diffusion_policy.py` (real-stanford
`DiffusionUnetHybridImagePolicy`: from-scratch, no prompt, multi-frame
history — the third interface shape).

`diffusion_policy` (PR #27) and the OpenVLA adapter referenced below
(PR #23) are **open PRs, not on master**. Wherever this skill describes
their mechanisms (`checkpoint_image_transform`,
`assemble_token_action_sequence`, identity stats, the `diffusion_policy`
install branch), treat it as proposed design and read the code from the PR
branch — re-check that the PR has landed before relying on those
interfaces.

Stable rules live in `.claude/CLAUDE.md` ("Extending VLA Factory" +
"Conventions"); this skill is the step order and the per-step checks.

---

## Step 0 — Recon the upstream repository

Before writing anything, extract from the upstream repo and record:

- Python dependencies and their pin style (loose bounds vs strict `==` vs
  git-only source; in-place patches of other packages);
- what `forward` / predict actually consumes, and its exact input format
  (openpi: `Observation.from_dict`; lerobot: a batch dict with
  `observation.images.*` / `observation.state` / `action` / `action_is_pad`);
- the official preprocessing chain (transform list / processor factory) and
  its constants: image value range, layout, resize policy, normalization mode
  and eps, tokenizer repo and max length;
- action contract: dim, chunk length, control mode, denormalization;
- checkpoint format and how upstream loads it.

Summary to carry into the steps below:

```text
Model:                    Closest existing adapter:
Upstream repo + ref:      Dep pin style (A pip / B uv):
Forward consumes:         Official chain entry:
Image facts:              Vector facts (mode + eps):
Action contract:          Checkpoint format:
```

---

## Step 1 — Dependencies: route them into the install path

The dep analysis from Step 0 picks one of two routes.

**Route A — pip-friendly** (PyPI packages, compatible pins, no patches — the
ACT case):

1. add `[project.optional-dependencies].<name>` in `pyproject.toml`;
2. add a branch in `scripts/install.sh`'s `install_one_model()` plus the
   model name in the arg-parse `case` and the usage header;
3. `ModelMetadata.install_hint = 'pip install -e ".[<name>]"'`.

**Route B — uv-required** (strict `==` pins, git-only source, or in-place
patches of other packages — the openpi case): a plain
`pip install -e ".[<name>]"` will not work. Still declare the extra (docs +
`all`), but make `install.sh` the real installer: pin upstream to a known-good
git commit, fetch the tarball into `.local-deps/<name>` with a `.vla-pin`
freshness file, apply any `sed` fixes, install with uv flags, apply patches
after install. `install_hint = 'bash scripts/install.sh --model <name>'`.
Watch for the empty-wheel trap: an upstream with no `__init__.py` anywhere
(PEP 420 namespace layout — real-stanford diffusion_policy) installs from
`git+…` as an empty package. Patch the tarball's `setup.py`
(`find_packages` → `find_namespace_packages`), failing loudly if the sed
anchor moved upstream (worked example: the `diffusion_policy` branch of
`scripts/install.sh`).

Both routes:

- `install_hint` must be the exact command that works today, in the script's
  current argparse form — the factory pastes it into every `ImportError`.
- torch routing stays auto-detected (cu126/cu128 from compute cap); never
  hardcode a local index, mirror, or username.
- Heavy upstream imports go **inside the factory**, never at module top, so
  the registry keeps loading for users without the extra; a broken adapter
  surfaces as `RegistryLoadError`, not a silent "not registered".
- An upstream bug needing a source patch: prefer a small guarded patch at
  import time (see `_patch_lerobot_groot` in `adapters/act.py`) with the
  affected versions named in its docstring.

Check:

```bash
vlafactory-cli list                                  # registry loads without the extra
bash scripts/install.sh --model <name>               # fresh venv installs
vlafactory-cli inspect model --name <name>           # tunables print with sources
```

---

## Step 2 — Model input: Observation coverage

List every field the upstream forward consumes and compare against
`Observation` (`vla_factory/model/model_interface.py`): `images`,
`image_masks`, `state`, `tokenized_prompt`, `tokenized_prompt_mask`,
`token_ar_mask`, `token_loss_mask`.

**Case A — `Observation` covers it.** Assemble and call upstream directly:
translate `Observation` → upstream batch inside the wrapper (example:
`ACTModelWrapper._obs_to_lerobot_batch`). Map camera names through the
build-time correspondence (`assembly.camera_mapping` / the config image keys
captured at factory time), never dict insertion order. The adapter does no
preprocessing of its own.

**Case B — the model needs a field `Observation` does not carry as-is**
(uint8 HWC dataset images vs float CHW normalized; raw state vs z-scored;
task text vs token ids; dataset width vs model width). The field must be
**produced by the pipeline**, not computed inside the adapter. Declare the
fact on `ModelMetadata`; the resolver (`vla_factory/assembly/resolve/pipelines.py`)
reads it and plans the step into `data_to_model`:

| Required input | Declare on `ModelMetadata` | Planned step(s) |
|---|---|---|
| float images in model range | `image_input_range` | `image_to_float` |
| normalized images | `image_normalize_mode="imagenet"` | `image_normalize` |
| CHW / HWC layout | `image_layout` | `image_layout` |
| fixed slot resolution | `vision_slots[].resolution` + `image_resize_mode` (`pad` letterbox / `stretch`) | `resize_images` |
| model-width state/action | `dim_policy` + `dim_policy_max` (targets land in `ModelIOSpec`) | `pad_dimensions` |
| normalized state/action | `vector_normalization` + `vector_normalization_eps` (lerobot 1e-8 vs openpi 1e-6 — a classic parity bug source) | `normalize_vector` |
| tokenized prompt | `requires_prompt`, `language_template`, `tokenizer_repo`, `tokenizer_max_length`, `prompt_includes_state` | `task_tokenize` |

Two escalation sub-cases:

- The computation has **no registered step yet** → add a transform under
  `vla_factory/assembly/transform/` (`@register`, `compile_call` reads
  `PlanContext`, `from_call` builds it, `inverse_call` pairs the inverse for
  `model_to_robot`) and make the resolver select it from a fact. Never a
  model-declared step list — `model.config.transforms` is rejected by design.
- `Observation` itself lacks the field (a genuinely new modality) → stop and
  report; extending `Observation` is a framework design decision, not an
  adapter-local one (see Stop conditions).

A third shape (OpenVLA/Prismatic): the upstream ships an HF
**processor that owns the whole preprocessing chain** — image geometry +
per-tower normalization (OpenVLA's fused DINOv2+SigLIP channel stack),
prompt template, action discretization. The framework delegation mode for
this is **proposed in PR #23, not yet on master** — the bullets below
describe that proposal, not current capability; either way, do not
improvise the chain inside the adapter:

- declare `image_normalize_mode="checkpoint_processor"` → the resolver
  plans one `checkpoint_image_transform` step applying the base
  checkpoint's own processor to the primary camera (requires a resolved
  `camera_mapping.primary`); the plan serializes the checkpoint repo and
  the step loads the processor lazily at build time. Mutually exclusive
  with `image_resize_mode` — the processor owns geometry. That mutex is
  what closes the old trap: a declared resize mode + non-native
  resolution used to plan a framework `resize_images` before the
  processor re-resized (double interpolation, silent kernel
  substitution — cv2 `INTER_LINEAR` vs the processor's own resize);
- a prompt/sequence assembly `task_tokenize` cannot express (action
  tokens interleaved as the answer turn, label masking) goes in the
  `assemble_token_action_sequence` step, fed by the read-only
  `language_template` fact — never a template string duplicated in the
  wrapper;
- **stats come from the assembly, never the base checkpoint** (unlike the
  two bullets above, this rule is master-current —
  `vector_normalization="quantile"` is how pi05 normalizes today): a
  pretrained checkpoint bundles norm stats of *its* training data
  (openvla-7b ships ~25 OXE datasets' q01/q99); declare it so the plan
  normalizes with the fine-tuning dataset's stats — the way upstream's own
  fine-tune script recomputes them. A hand-picked key into the base
  checkpoint's stats is the smell.

Adapter hard rules, enforced at construction:

- every shape/slot comes from `assembly.model_io_spec` /
  `assembly.camera_mapping`, never from a schema or an array shape;
- every `params` key must be read — `TrackedConfig.assert_all_consumed`
  fails on a declared-but-unread knob;
- no shape re-validation inside the adapter: what reaches the wrapper was
  produced by a pipeline planned toward the same `model_io_spec` the
  wrapper was built from, so it is correct by construction, and a wrapper
  validator restates that contract in a second place that can drift from
  it. Contract checking happens in tests, not at runtime: the untrusted
  boundary is validated by the framework (the inference engine checks
  deploy-time observations against the checkpoint `DataSchema`), and the
  model-side input contract — shapes, layout, time axis — is asserted in
  the L1 parity test by feeding the assembled inputs to the real upstream
  (let upstream itself be the validator);
- action horizon follows the paradigm: `pretrained_finetune` declares the
  named `action_horizon` fact, `from_scratch` declares
  `params["action_horizon"]`; both or neither is a broken entry.

Check (no GPU, no weights, no upstream import — `resolve` is a dry run):

```bash
vlafactory-cli resolve --config examples/<name>_*.yaml
vlafactory-cli inspect --config examples/<name>_*.yaml
```

Read the printed `data_to_model` plan: every fact from the table above must
appear as exactly the step you intended, with the upstream's constants
(range, eps, resolution, pad width), in the resolver's dependency order
(`plan_data_to_model`, `resolve/pipelines.py` — geometry on raw pixels before
range/layout conversion, padding as the final shape reconciliation). The
tail of the plan branches on `prompt_includes_state`:

```text
resize_images → image_to_float → image_layout → image_normalize → normalize_vector
plain prompt:           → pad_dimensions → task_tokenize
prompt_includes_state:  → task_tokenize → pad_dimensions
```

`task_tokenize` sits **before** `pad_dimensions` only when
`prompt_includes_state` (the state is digitized into the prompt, so it must
be read at dataset width — padding first would digitize pad zeros into the
tokens); for a plain prompt the resolver appends it after padding, token
ids being independent of vector shapes.

---

## Step 3 — `robot_to_model` and `model_to_robot`

`robot_to_model` is the deployment alias of `data_to_model` — identical plan
content, because platform adapters emit the checkpoint's `DataSchema` keys and
vector order. Step 2 done correctly means `robot_to_model` needs nothing new;
what to check here:

- fewer robot cameras than vision slots → `missing_slot_policy`
  (`zero_pad` / `drop` / `error`) matches upstream's behavior on missing cams;
- `control_mode_pref` vs the robot profile's declared mode;
- the platform adapter maps by the checkpoint `DataSchema` — no name-based
  camera/joint guessing beyond what `RobotProfile` declares.

`model_to_robot` is the **planned inverse** built from the same facts
(`inverse_call`), never the forward list reversed. Check:

- denormalization uses the same mode and eps as the forward `normalize_vector`;
- padding is stripped back to dataset width (e.g. model 32 → dataset 8):
  `model_output_dim` (raw network output) and `execution_action_dim` (the
  command after `model_to_robot`) must stay distinct in the adapter;
- `predict_actions` returns `[B, horizon, action_dim]` in DataSchema action
  space. Upstream-internal decoding (flow sampling, CVAE) stays inside the
  upstream model — only pipeline steps have inverses here;
- when upstream's decode applies its *own* unnormalization (discretized
  action models — OpenVLA's ActionTokenizer), mount identity stats
  (q01=-1 / q99=+1) under a synthetic key in the checkpoint's norm_stats
  so the decode returns actions in NORMALIZED space and the planned
  `unnormalize_action` inverse finishes the round trip — the same
  model-emits-normalized division of labor as pi0 (worked example, PR #23
  not yet on master: `_inject_identity_action_stats`,
  `adapters/openvla.py`).

Check: same `vlafactory-cli resolve` output — read both plans, confirm every
forward step that has an inverse is paired, and that the inverse ends at the
`DataSchema` action interface.

---

## Step 4 — L1 parity tests

Write `test/l1/test_<name>_pipeline_parity.py` (`pytest.mark.l1`; the tier is
selected by directory: `pytest test/l1`). Mirror
`test/l1/test_act_pipeline_parity.py` (lerobot-shaped chain: preprocessing
lives in the dataset loader, policy side is normalization only) or
`test/l1/test_openpi_pipeline_parity.py` (openpi-shaped chain: resize /
to-float / tokenize all in the transform chain), or
`test/l1/test_diffusion_policy_parity.py` (no runnable official chain to
build — pin the transcribed constants instead: upstream commit + file:line
header, plus a pin guard test asserting `install.sh`'s `<NAME>_REF` still
equals the cited commit, so bumping the pin without re-reading the cited
expressions goes red).

Structure — three module-scoped fixtures:

- `raw` — one real sample + schema + norm_stats, shared verbatim by both
  sides (isolate the transform chain, not the stats producer);
- `upstream` — the official chain built **independently**: construct the
  upstream config and processor directly, never through our factory or our
  merged config, or you are comparing us with ourselves;
- `ours` — `resolve_assembly(recipe)` → `build_pipeline(assembly.data_to_model)`
  → apply steps → `collate_fn` → adapter's Observation translation.

Compare **at the upstream model's real input boundary** (openpi
`Observation.from_dict`; lerobot `make_act_pre_post_processors` output). One
comparison there covers every constant at once — eps, value range, resolution,
pad width, step order — without asserting them one by one.

Assertion rules:

- default bit-identical via `test/l1/utils.py::assert_tensor_parity`
  (`rtol=atol=0`); tolerance only where two implementations legitimately
  differ (e.g. jax vs cv2 interpolation), with the measured numbers and the
  reason in a comment — never a silent blanket tolerance;
- assert layout where a wrong axis would be silent (CHW `[B,3,H,W]`);
- assert absent fields too (a non-language model must not receive tokenize
  steps);
- if the model postprocesses, cover the `model_to_robot` inverse the same way
  (see `test/l1/test_normalize_parity.py`);
- when upstream preprocessing is *transcribed* — whether into the wrapper
  or into a framework step (OpenVLA's `assemble_token_action_sequence`,
  PR #23 not yet on master, mirrors upstream's `RLDSBatchTransform`) —
  build the reference from the upstream classes themselves; they are
  importable. Comparing a transcription with itself proves nothing.

Guards:

- `pytest.importorskip("<exact.upstream.module>")` — precise to the module
  this file imports: dotted-path `find_spec` raises on a missing parent, and
  upstream module paths move across versions;
- module-level `pytest.skip` when `test/data/lerobot_train_data_3_episodes`
  is absent, so the default suite stays green.

No model weights are involved — L1 parity validates the transform chain
against the official chain, structurally (weights enter only L2+ smoke runs).
Document in the file docstring what is deliberately **not** covered (batch > 1,
stats production, specific masks) — the next reader needs the boundary.

Finish the integration per `.claude/CLAUDE.md` — done means **registered and
discoverable**, every item checkable:

- registered: `@register_vla(ModelMetadata(...))` on the factory in
  `adapters/<name>.py`, and `vlafactory-cli list` prints `<name>` in a
  clean environment (no model extra installed — the registry must load
  without it);
- documented: example recipe `examples/<name>_*.yaml`, and a support-table
  row in **both** `README.md` and `README.cn.md`;
- tested: L0 smoke `test/l0/test_<name>_model.py` (load + forward,
  guarded skip without the extra) plus the Step 4 parity test above.

```bash
pytest test/l1/test_<name>_pipeline_parity.py   # in the model's venv
pytest test/l0/test_<name>_model.py
```

---

## Stop conditions

Stop and report instead of guessing when:

- upstream behavior (preprocessing constant, output semantics) is unclear or
  undocumented;
- `Observation` lacks a field the model needs — extending it is a framework
  design decision; propose it, don't hack around it in the adapter;
- action semantics cannot be mapped safely (mode, horizon, width);
- the required change conflicts with the layer design docs under `docs/modules/`
  or would drag in unrelated repo-wide migration.

Report:

```text
Blocker:            Affected step (1–4):
Reason:             Smallest viable next step:
```

---

## Completion report

Before declaring the adaptation complete:

```text
Model + upstream ref:          Files changed (adapter / pyproject / install.sh / recipe / tests):
Install route (A/B) + hint:    Facts declared (per Step 2 table):
params keys (+ all consumed):  data_to_model steps (resolved):
model_to_robot inverse:        L1 results (bit-identical? documented tolerances?):
Unsupported features:          Validation commands run:
```

Do not claim full model support if any of the four steps is partial — say
which one and what remains.
