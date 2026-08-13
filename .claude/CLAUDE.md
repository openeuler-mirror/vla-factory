# CLAUDE.md

This file is the Claude Code entrypoint for repository guidance. It orients
you to **how VLA Factory is structured, how it runs, and how to extend it**
without re-deriving the architecture from scratch on every task. It is the
single source of truth for the project's development rules — there is no
separate rules file.

For the full architecture rationale (the "why"), see
[`docs/architecture/vla-factory-architecture.md`](../docs/architecture/vla-factory-architecture.md)
(EN) / `.cn.md` (中文). For data-pipeline internals, see
[`docs/modules/data-module.md`](../docs/modules/data-module.md). For the
distilled methodology of *adapting a new model* (openpi/pi0 style), see the
`adapt_new_model` skill — this file is the rules, that skill is the
hard-won experience.

---

## Quick orientation

VLA Factory is a **recipe-driven** fine-tuning framework for robot
Vision-Language-Action (VLA) models. A single YAML recipe describes the
model, data, action space, fine-tuning strategy, training params, and output
location; the framework runs the full loop: data pipeline → model
construction → training → checkpoint artifact → inference/deploy service.

The core positioning (read this twice): **the framework owns no model
architecture code.** It is an engineering glue layer that wraps external
model ecosystems (lerobot's `ACTPolicy`, openpi's `PI0Pytorch`, …) through
thin adapters, and unifies data semantics + training artifacts + deployment
entrypoints. It should preserve upstream model semantics behind explicit
adaptation boundaries, not become a rewrite of every upstream VLA repo. When
you reach for a model implementation, go to its upstream repo — never
reimplement it here.

Execution path:

```text
YAML recipe → TrainRecipe → model registry (entries/<name>.py)
  → data reader / codec (Canonical IR) → assembly transforms / training sampler
  → VLADataset / DataLoader → VLATrainer (HF Trainer based)
  → checkpoint + inference_metadata
  → InferenceEngine + platform adapters (simulator / lerobot host / robotwin)
```

The composition layer (`assembly/`) resolves the data schema × model metadata ×
robot profile into a `ResolvedAssembly`, and **the training and inference loops
consume it** (architecture §7.4 stage 4, data × model half): `train()` resolves
once, builds the model and the pipeline from that product, and saves it as
`inference_metadata/assembly.json`; `InferenceEngine` executes that saved
contract. `robot_to_model` and the JointMapping consumers are still deferred.

---

## Code structure

- **`vla_factory/recipe/cli.py`** — argparse CLI: `train`, `preprocess`, `list`,
  `resolve`, `inspect`, `evaluate`, `infer`, `deploy`. Entry: `vlafactory-cli`
  (installed) or `python -m vla_factory` (from source). This is where user
  commands land. `resolve` dry-runs the composition; `inspect` prints one
  dimension's declared facts + sources (`inspect data/model/robot`, or
  `inspect --config` for all three) — both run with no GPU / no optional extras.
- **`vla_factory/recipe/`** — user-expression layer: recipe parsing & model defaults.
  - `recipe.py` — `TrainRecipe` and sub-dataclasses, incl. `RobotConfig` and
    `AssemblyConfig` (composition selection + controlled override).
  - `parser.py` — `parse_recipe(path|dict) → TrainRecipe`. Unknown keys are
    ignored and there is no legacy-shape translation: a field the resolver now
    derives was removed outright, so an out-of-date recipe has stale keys that
    simply do nothing (run `resolve` to see what the composition derived).
  - `defaults.py` — `model_params()` + `resolve_recipe()`: the single merge
    point. Deep-merges the model's declared `ModelMetadata.params` under the
    recipe's per-run `model.config` (recipe wins), and enforces the tunable
    allow-list — a `model.config` key the model never declared is an error with
    `difflib` candidates (gate 1 of three; see `model-module.cn.md` §4.6).
    OmegaConf-based, no Hydra runtime.
- **`vla_factory/model/`**
  - `interfaces/` — `ModelMetadata` (frozen descriptor: backend, action head
    type, trainable components), `VLAModel` / `VLAModelPyTorch` /
    `VLAModelJAX` protocols, `Observation`/`ActionSpec`. Framework-agnostic
    contracts that flow between data and model layers.
  - `checkpoint_validation.py` — optionally compares a checkpoint's redundant
    `config.json` shapes with `ModelMetadata`. It is diagnostic only: it never
    supplies or overrides resolver/model facts, so checkpoints from different
    locations remain interchangeable within one model family.
  - `registry/` — `@register_vla(metadata)` decorator + `get_entry()` /
    `list_entries()`. Entries auto-discovered from
    `registry/entries/*.py` on first lookup (see "Extending").
- **`vla_factory/model/registry/entries/`** — the actual adapters, one file
  per model: `act.py`, `pi0.py`, `pi05.py`. **These are the canonical worked examples**
  for adding a new model — diff against them, don't start from scratch.
- **`vla_factory/data/`** — read-only Canonical IR only (no sample building):
  `formats/` (format readers: `lerobot_v3.py`, registry + `get_reader()`),
  `codec/` (video decode: `pyav.py`, `resolve_codec()`), `manifest.py`
  (`DataSchema` entry-table form: `cameras[]`/`state.dims[]`/`action.dims[]`
  with per-fact source labels; legacy flat fields are read-only derived props
  until phase 4), and `semantics.py` (deterministic inference rules for camera
  `semantic` and action `mode` — unique-match-only, §8.5).
- **`vla_factory/robot/`** — robot body descriptions (`RobotProfile`):
  `profile.py` + `registry.py` (`get_robot_profile()` / `list_robot_profiles()`)
  and bundled `profiles/*.yaml` (e.g. `lekiwi.yaml`). Static body facts only —
  no transport / platform session info.
- **`vla_factory/assembly/`** — composition resolution layer
  (data × model × robot). `resolver/` holds `resolve_assembly()` →
  `ResolvedAssembly`, the serializable mapping/pipeline-plan types and the
  structured `ResolutionError`. `from_recipe.py` is the one orchestration entry
  (`resolve_from_recipe(recipe)`: registry → optional checkpoint check → robot
  profile → dataset descriptions → pure resolver) used by train, inference and
  the CLI. `artifact.py` reads/writes `assembly.json` (versioned envelope) and
  holds the declaration-drift check. `transforms/` holds `TransformPipeline` +
  `TransformRegistry` (`@register` steps: `resize_images`, `pad_dimensions`,
  `image_to_float`, `image_layout`, `normalize`, `task_tokenize`, …); a step is
  planned by `compile_call` (resolver) and built by `from_call`
  (`build_pipeline`) — there is no declaration→step shortcut.
- **`vla_factory/training/`** — training orchestration + sample building.
  `train.py` (recipe → trained model), `pytorch_trainer.py` (`VLATrainer`,
  wraps HF `Trainer`), `strategies/` (`apply_strategy`: full / freeze /
  selective; LoRA is WIP), plus the sample-construction pipeline moved out of
  `data/`: `dataset.py` (`VLADataset` / `collate_fn`), `loader.py`
  (`create_dataloaders`), `sampling/` (`SlidingWindowSampler`),
  `manifest.py` (`build_manifest`).
- **`vla_factory/inference/`** — split into a transport-agnostic inference core
  and pluggable sub-layers (see `docs/modules/deploy-module.md`):
  - `infer.py` — the inference core: `InferenceEngine` + `ObsDict`, the
    `ActionChunk`/`ActionCommand` contracts, the execution policies
    (synchronous / temporal_ensembling / receding_horizon) + `PolicyExecutor`,
    and the `ReplayPolicy` stand-in.
  - `policy_runtime.py` — the two serving forms: `PolicyRunner` (client-shaped
    loop driving an injected transport; shared by `simulator` and `lerobot`
    host) and `RemotePolicyModel` (server-shaped RPC handler).
  - `platforms/` — per-platform observation/action adapters (`simulator`,
    `lerobot` host, `robotwin`, `groot`) behind the `PlatformObservationAdapter`
    protocol (`base.py`).
  - `transports/` — pure connection/framing/serialization, no orchestration:
    `zmq.py` (LeKiwi PUSH/PULL client), `length_prefixed_json.py`
    (RoboTwin-compatible TCP RPC).
  - `connectors/` — dependency-free callbacks imported by the robot env
    (`robotwin.py` + bootstrap `robotwin.yml`), runnable without the model deps.
- **`vla_factory/utils/constants.py`** — on-disk artifact layout:
  `inference_metadata/{assembly.json,recipe.yaml,schema.json,norm_stats.json}`,
  `final/model.pt`. The engine reads only the first two; `schema.json` /
  `norm_stats.json` are readable copies for `inspect` and external tooling.
- **`vla_factory/utils/vocabulary.py`** — the single source for the three
  cross-dimension controlled vocabularies (architecture §4.5): `CAMERA_SEMANTICS`,
  `CONTROL_MODES` (`joint_pos`/`joint_delta`/`joint_vel`), `ACTION_HEADS`, plus
  the data/model source-annotation types. Referenced by data, model and robot —
  do not re-declare these in a dimension.
- **`examples/`** — ready recipes. `reference.yaml` is the fully-annotated
  template (every field documented); `act_lekiwi.yaml`, `pi0.yaml`.
- **`scripts/install.sh`** — uv-based env setup for openpi (see Installing).
- **`test/`** — pytest: `test_act_model.py`, `test_pi0_model.py`,
  `test_data_pipeline.py`, `test_checkpoint_validation.py`,
  `test_inference_engine.py`, `test_phase4_engine.py`,
  `test_protocols_registry_config.py`.

---

## How it runs

A `train` invocation goes: `parse_recipe` → `resolve_recipe` (merge model
defaults) → **`resolve_from_recipe` (the composition, before any side effect —
no output dir is created or wiped until it succeeds)** → `get_entry(model_name)`
→ `factory(recipe, assembly)` builds the wrapped model from `model.path`
(pretrained) or from-scratch → `apply_strategy` freezes params per
`finetuning_strategy` + `ModelMetadata.components` →
`create_dataloaders(recipe, assembly)` (reader + codec + the `data_to_model`
plan instantiated + sampler) → `VLATrainer` runs HF `Trainer` → saves
`final/model.pt` + `inference_metadata/{assembly,recipe,schema,norm_stats}`.

`deploy --platform {simulator,lerobot,robotwin}` loads a checkpoint's
`inference_metadata/{assembly.json,recipe.yaml}` and **executes** the saved
assembly (descriptions, IO spec, both pipeline plans; no re-resolution, and a
checkpoint without `assembly.json` is refused rather than re-derived),
builds the `InferenceEngine`, and serves it to a platform adapter:
via `PolicyRunner` (`policy_runtime.py` driving `transports/zmq.py`) for
`simulator`/`lerobot` real robot, or a
length-prefixed-JSON TCP server (`transports/length_prefixed_json.py` +
`RemotePolicyModel`) for `robotwin` — the sim connects as a client through the
dependency-free `connectors/robotwin.py`. `infer`/`evaluate` reuse the same
engine on a dataset sample / per-episode L1.

The transform pipeline is the contract bridge: each model's declaration lists
the `TransformRegistry` steps it needs in `ModelMetadata.params["transforms"]`,
while interface values (image range, normalize mode, vector normalization,
vector widths and fixed image resolutions) come from named model facts. ACT's
optional `input_image_size` is an explicit from-scratch model tunable; absent it,
the IO spec keeps the dataset-native size. The resolver builds `ModelIOSpec`
first, then compiles `data_to_model` / `model_to_robot` calls toward that target.
Steps never report shapes back through `output_widths`/`output_image_sizes`, and
shape arguments repeated in a step config are rejected. Training and inference
only instantiate saved plans (`build_pipeline`); the deployed postprocessor is
the planned inverse, never the forward list reversed.

---

## Installing

Two install paths, by ecosystem friction:

- **ACT** (lerobot's `ACTPolicy`, standard PyPI, pip-friendly):
  `pip install -e ".[act]"`.
- **pi0 / pi05** (openpi's `PI0Pytorch`): `bash scripts/install.sh pi0`.
  A plain `pip install -e ".[pi0]"` **does not work** — openpi's strict `==`
  pins + in-place `transformers` patch require the **uv** installer
  (PubGrub resolver). The script auto-detects the local CUDA driver and
  routes torch/torchvision through the matching PyTorch CUDA wheel index
  (cu126 / cu128); it pins openpi to a known-good git commit for
  reproducibility (no release tags upstream).

Dev deps: `pip install -e ".[dev]"` (pytest, pytest-cov, tensorboard).

---

## Key ideas and plugging in

**Recipe carries choices, not relations.** One YAML says which model, which
dataset, which robot, how to train, and where to write — plus, in `assembly:`,
the controlled overrides for a relation the resolver cannot pin down alone. What
it does *not* carry is anything derivable from the three descriptions: action
widths, chunk length, observation window, image size and camera mapping all come
from the composition, because a recipe that restates them is a second answer
that can disagree. `data:` therefore holds only the dataset itself, and the train/val
split is a framework constant (`training/manifest.py`) rather than a knob whose
only effect is shrinking the training set — nothing evaluates the held-out half
during training. CLI overrides (`--steps`, `--batch-size`, `--output-dir`)
tweak without editing the file. `examples/reference.yaml` documents every field;
`docs/modules/recipe-module.cn.md` documents the three zones and the deprecation
window.

**Facts vs tunables — the container is the attribute.** A model ships one
declaration, `ModelMetadata`. Named fields are facts the composition resolver
reads and a recipe can never override; `params` holds that model's tunable
defaults, deep-merged under the recipe's `model.config` where **recipe wins**.
A model author classifies nothing: framework facts have names and types,
everything else goes in `params`. Three guards keep the surface honest —
an undeclared `model.config` key is rejected by `resolve_recipe()`, a declared
key nothing reads is rejected by the factory (`utils/tracked_config.py`), and a
fact set inside a step config is rejected by `assembly/transforms/base.py`. Each
guard exists because that failure was previously silent. `inspect model` prints
every tunable with its effective value and source.

**Registry, not if/else.** Each model self-registers with `@register_vla`
inside `entries/<name>.py`; `get_entry(name)` looks it up. Entries are
auto-discovered on first lookup via `pkgutil.iter_modules`. A broken entry
raises `RegistryLoadError` (never masked as "not registered") — so optional
deps must be deferred to *factory-call time* (see `act.py`: lerobot imported
inside the factory, so `list_entries()` works even without the `[act]` extra).

**ModelMetadata is the only model-interface truth.** Camera roles, dimensions,
resolution, normalization, and temporal behavior come from the registry entry.
`checkpoint_validation.py` may compare a checkpoint `config.json` against that
declaration before loading, but the observed values never flow into resolver
output. A missing checkpoint config skips the optional check; a contradiction
fails clearly. Add a registry entry only when the model-family interface differs,
not merely for another checkpoint location.

**Thin adapters, composition over inheritance.** Each entry holds an
upstream model instance and translates `vla_factory.Observation` ↔ the
upstream's batch format; it does not rewrite the upstream model. (Contrast
RLinf, which *inherits* `PI0Pytorch` to add RL heads for its own use case —
vla_factory only fine-tunes, so a thin wrapper is the correct boundary.)

**The assembly is the downstream's only entry.** Training and inference read
every data × model × robot relation off the `ResolvedAssembly` — widths and
horizon from `model_io_spec`, slots from `camera_mapping`, pipelines from the
plans — and never re-derive one from a schema, a model name or an array shape
(architecture §4.2.6). The registry is still consulted for two things: the
factory (code cannot be serialized) and the checkpoint's declaration-drift check
(`assembly/artifact.py`). A checkpoint carries its assembly, so serving it never
re-resolves — a declaration that changed since training is a loud failure rather
than a pipeline that quietly feeds the model wrong-valued inputs.

**Model IO precedes pipeline planning.** Resolution establishes
all five Mappings first, builds `ModelIOSpec` directly from `ModelMetadata`,
resolved model tunables, `DataSchema` and `CameraMapping`, and only then plans
transforms. State/Action Mapping entries represent real correspondences only;
never expand target-width padding into source-less Mapping entries. Padding and
resize calls consume the source/target shapes in `PlanContext`; never recover an
interface by scanning calls or adding a transform-side `output_*` shape hook.
At inference, keep `model_output_dim` (network output) distinct from
`execution_action_dim` (the command leaving `model_to_robot`).

---

## Extending VLA Factory: a new model

Follow `entries/pi0.py` (or `act.py`) as the worked example. Steps:

1. **Entry module** — create `vla_factory/model/registry/entries/<name>.py`.
   This is the whole model declaration; there is no second file:
   - `@register_vla(ModelMetadata(name=..., backend="pytorch",
     action_head_type=..., components={...}))` on a factory
     `(recipe, assembly) -> VLAModel`. Take every shape and slot from
     `assembly.model_io_spec` / `assembly.camera_mapping` — deriving
     `action_dim` from a schema or reading `camera_mapping` off the recipe is
     the composition's job, and doing it here is how the two used to disagree.
     `recipe` is only for what is not a relation: `model.path` and
     `model.config`.
   - Defer heavy/optional upstream imports **inside the factory**, not at
     module top, so registry load stays robust (rationale in
     `registry.py:RegistryLoadError`).
   - Wrap the upstream model by composition; translate
     `Observation` ↔ upstream batch format; delegate `forward` /
     `predict_actions`. Do not reimplement architecture.
   - Declare facts on the named fields (vision slots, `dim_policy`,
     `image_input_range`, `vector_normalization`, ...) and everything tunable —
     upstream hyperparameters plus the default `transforms` step list — in
     `params`. Every `params` key must be read by the factory or by a registered
     framework consumer, or model construction fails.
   - The action horizon follows the paradigm and the resolver enforces it:
     `pretrained_finetune` declares the named `action_horizon` (a family fact),
     `from_scratch` declares `params["action_horizon"]` (the user's choice).
     Declaring both, neither, or the wrong one for the paradigm is a broken
     entry.
2. **pyproject extra** — add `[project.optional-dependencies].<name>` for the
   upstream ecosystem deps. If it needs uv (strict pins / patches), wire it
   into `scripts/install.sh` rather than expecting `pip install -e ".[name]"`.
3. **Example recipe** — add `examples/<name>_*.yaml`; list it in the README
   support table.
4. **Tests** — mirror `test/test_<name>_model.py` (load + forward smoke).

When you hit a decision point during adaptation (which upstream to wrap,
how to handle broken deps, consistency vs local convenience), consult the
`adapt_new_model` skill — it distills the openpi/pi0 integration (Issue #3)
into five decision tests.

---

## Conventions (the rules)

These are **stable framework rules**, not per-task choices — apply them
everywhere, including existing code. (The experience/rationale behind them
lives in the `adapt_new_model` skill; this section is the rule, that skill
is the why + worked examples.)

- **General framework, not a single machine.** Code and scripts must not
  assume one CUDA version, user, GPU, or network. Install scripts
  auto-detect the environment (CUDA → cu126/cu128 index); never hardcode a
  local value (paths, usernames, GPU IDs, env names). Unsupported
  environments should fail early with an actionable message. Pin upstream git
  commits (no release tags → known-good commit) for reproducibility across
  machines. Comments describe general facts ("driver 550+ runs cu126
  wheels"), not local ones ("verified on my box"). Decision test: would this
  run on a fresh machine with a different GPU and no proxy?
- **uv-based dependency workflow.** Use the repository's uv workflow for
  Python dependencies; don't introduce new pip-only workflows unless
  explicitly justified. Keep backend-specific heavy dependencies (openpi,
  lerobot) isolated from the core framework environment (see Installing).
- **English for code; docs may be multilingual.** Code comments, docstrings,
  log messages, error strings, install hints, and recipe field comments in
  `*.py` / `*.toml` / `*.sh` are English — as are identifiers, type names,
  config keys, CLI options, and test names. Only `README*.md` and `docs/`
  (architecture/module docs, `.cn.md`) may carry other languages. Never mix
  multiple languages inside one comment, identifier, config schema, or error
  message. Code readers are contributors who read English; mixing languages
  in code breaks grep-ability and consistency.
- **Framework owns no architecture.** Wrap upstream; translate at the
  adapter boundary; never copy model architecture code into the tree.
- **Optional deps defer to factory-call time**, never at module import, so
  registry load and `list_entries()` stay robust across users who skip extras.
- **All other imports go at module top.** The factory-call-time rule above is
  the *exception*, reserved for optional heavy ecosystem deps (openpi, lerobot,
  peft, transformers) and documented circular-import workarounds (name the
  cycle in a comment). Lightweight internal modules (`vla_factory.utils.*`,
  config/data helpers) are never imported inside a function.
- **Config YAML is static values only** — no computed/interpolated fields in
  recipes (model-default merging + dot-path are done in code via OmegaConf,
  not in the YAML itself).
- **Smallest change, touched-scope consistency.** Make the smallest change
  that satisfies the task; don't refactor unrelated modules or change shared
  contracts unless the task requires it. When consistency and scope conflict:
  new code follows current conventions, touched old code may be migrated when
  safe, untouched old code is not modified for cosmetic consistency alone.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest                       # all
pytest test/test_act_model.py  # one file
```

New behavior needs a test (unit or smoke). If a test needs a heavy model
extra or GPU, guard the import with `importlib.util.find_spec` and
`pytest.skip` so the default suite stays green without the extra installed.

---

## Further reading

- Architecture (the why): `docs/architecture/vla-factory-architecture.md` · `.cn.md`
- Layered architecture + flow diagrams: `docs/graph/*.svg`
- Data module design: `docs/modules/data-module.md` · `.cn.md`
- Deploy module design: `docs/modules/deploy-module.md` · `.cn.md`
- Fully-annotated recipe: `examples/reference.yaml`
- Adapting a new model (experience + decision tests): `adapt_new_model` skill
- README (user-facing, multilingual): `README.md` · `README.cn.md`
