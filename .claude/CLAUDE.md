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
YAML recipe → TrainRecipe → model registry (adapters/<name>.py)
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
contract. Platform adapters emit the checkpoint DataSchema, so
`robot_to_model` is the deployment semantic alias of `data_to_model` (equal
plan content); `model_to_robot` returns DataSchema actions for the adapter to
execute. Do not infer robot camera/joint bindings from names.

---

## Code structure

- **`vla_factory/frontend/`** — user-facing entry points and their shared input
  contract. `recipe.py` owns `TrainRecipe`, strict YAML parsing, and model
  tunable merging. `cli.py` organizes `train`, `preprocess`, `list`, `resolve`,
  `inspect`, `evaluate`, `infer`, and `deploy`; future WebUI or Agent frontends
  live beside it rather than at package root. Entry: `vlafactory-cli`
  (installed) or `python -m vla_factory` (from source). `resolve` and `inspect`
  run with no GPU / no optional model extras.
- **`vla_factory/model/`**
  - `model_interface.py` — the public reading entry: `ModelMetadata`,
    `VisionSlot`, `Observation`, and the framework-agnostic `VLAModel` /
    backend protocols.
  - `checkpoint_validation.py` — optionally compares a checkpoint's redundant
    `config.json` shapes with `ModelMetadata`. It is diagnostic only: it never
    supplies or overrides resolver/model facts, so checkpoints from different
    locations remain interchangeable within one model family.
  - `registry.py` — `ModelRegistry`, `@register_vla(metadata)`, `get_entry()`
    and `list_entries()`. Built-ins are discovered under `adapters/`; external
    packages use the `vla_factory.models` entry-point group.
  - `adapters/` — upstream model bindings. `act.py`, `pi0.py`, and `pi05.py`
    own each family declaration/factory; `openpi.py` holds code deliberately
    shared by PI0 and PI0.5. These are the worked extension examples.
- **`vla_factory/data/`** — read-only Canonical IR only (no sample building):
  start at `data_schema.py` (`describe_dataset()` plus `DataSchema` /
  `NormStats` / `Episode` / `Frame` / `VideoRef`). Here, data schema means the
  data layer's whole format-neutral representation, not just the `DataSchema`
  class. `reader/` contains `FormatReader`,
  `ReaderRegistry`, and storage-format implementations; `codec/` contains
  `VideoCodec`, `CodecRegistry`, and decoders. Built-ins use registration
  decorators; external packages use `vla_factory.readers` /
  `vla_factory.codecs` entry points. `DataSchema` uses entry-table form
  (`cameras[]`/`state.dims[]`/`action.dims[]`)
  with per-fact source labels and read-only derived properties for common
  widths/keys. `semantics.py` contains deterministic inference rules for camera
  `semantic` and action `mode` — unique-best-match-only, §8.5).
- **`vla_factory/robot/`** — robot body descriptions (`RobotProfile`):
  `profile.py` owns the pure, validated description types; `registry.py` owns
  YAML loading and bundled lookup (`load_robot_profile()` /
  `get_robot_profile()` / `list_robot_profiles()`); `profiles/*.yaml` contains
  built-ins such as `lekiwi.yaml`. Import the public surface from
  `vla_factory.robot`. Static body facts only — no transport / platform session
  info, runtime safety enforcement, or observation/action adaptation.
- **`vla_factory/assembly/`** — composition resolution layer
  (data × model × robot). Start at `resolve_assembly.py`: it owns the public
  `resolve_assembly(recipe)` orchestration entry, the `ResolvedAssembly` result,
  direct `assembly.json` persistence, and the model-interface consistency check.
  `resolve/` contains pure `resolve_from_facts(...)`, mapping/IO/pipeline rules,
  and structured `ResolutionError`; there is no artifact envelope or format
  version. `transform/` holds `TransformPipeline` +
  `TransformRegistry` (`@register` steps: `resize_images`, `pad_dimensions`,
  `image_to_float`, `image_layout`, `normalize`, `task_tokenize`, …); a step is
  planned by `compile_call` (resolver) and built by `from_call`
  (`build_pipeline`) — there is no declaration→step shortcut.
- **`vla_factory/training/`** — training orchestration + sample building. Start
  at `train.py`, which only orders the lifecycle. `dataset.py` owns
  `SampleWindow`, full-dataset window construction, `VLADataset`, and
  `collate_fn`; `dataloader.py` executes a resolved assembly as one training
  loader; `trainer.py` owns `VLATrainer` and HuggingFace argument mapping;
  `checkpoint.py` owns contract/final-weight persistence. `strategies/` is a
  registered extension point: each `FinetuningStrategy` strictly parses its
  own `finetuning.config`, prepares the model, and finalizes its inference state
  (`full` / `freeze` / `selective` / `lora`). Strategies select/wrap parameters;
  methods that change loss, sampling, or the loop are a different future layer.
- **`vla_factory/inference/`** — split into a transport-agnostic inference core
  and pluggable sub-layers (see `docs/modules/deploy-module.md`):
  - `inference_engine.py` — `InferenceEngine` + `ObsDict`; loads and executes
    the checkpoint's saved assembly and always returns an `ActionChunk`.
  - `execution.py` — `ActionChunk` / `ActionCommand`, the three execution
    policies, `PolicyExecutor`, and the `ReplayPolicy` stand-in.
  - `checkpoint.py` — inference metadata and model-weight discovery/loading.
  - `evaluate_dataset.py` — single-sample inference and per-episode L1
    evaluation; no deployment runtime concerns.
  - `deploy.py` — the public deployment orchestration entry plus the two
    serving forms: `PolicyRunner` and `RemotePolicyModel`.
  - `platforms/` — per-platform observation/action adapters (`simulator`,
    `lerobot` host, `robotwin`, `groot`) behind the `PlatformObservationAdapter`
    protocol (`base.py`).
  - `transports/` — pure connection/framing/serialization, no orchestration:
    `zmq.py` (LeKiwi PUSH/PULL client), `length_prefixed_json.py`
    (RoboTwin-compatible TCP RPC).
  - `connectors/` — dependency-free callbacks imported by the robot env
    (`robotwin.py` + bootstrap `robotwin.yml`), runnable without the model deps.
- **`vla_factory/utils/constants.py`** — on-disk artifact layout:
  `inference_metadata/{assembly.json,recipe.yaml}`, `final/model.pt`.
  `assembly.json` is the single source for schema, normalization statistics,
  mappings, ModelIOSpec, and pipeline plans.
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

A `train` invocation goes: `parse_recipe` → `merge_model_config` (merge model
defaults) → **`resolve_assembly` (the composition, before any side effect —
no output dir is created or wiped until it succeeds)** → `get_entry(model_name)`
→ `factory(recipe, assembly)` builds the wrapped model from `model.path`
(pretrained) or from-scratch → the registered `FinetuningStrategy` strictly
parses `finetuning.config` and prepares the model →
`create_dataloader(recipe, assembly)` (reader + codec + the `data_to_model`
plan instantiated + sampler) → `VLATrainer` runs HF `Trainer` → saves
`final/model.pt` + `inference_metadata/{assembly.json,recipe.yaml}`.

`deploy --platform {simulator,lerobot,robotwin}` loads a checkpoint's
`inference_metadata/{assembly.json,recipe.yaml}` and **executes** the saved
assembly (descriptions, IO spec, both pipeline plans; no re-resolution, and a
checkpoint without `assembly.json` is refused rather than re-derived),
builds the `InferenceEngine`, and serves it to a platform adapter:
via `PolicyRunner` (`deploy.py` driving `transports/zmq.py`) for
`simulator`/`lerobot` real robot, or a
length-prefixed-JSON TCP server (`transports/length_prefixed_json.py` +
`RemotePolicyModel`) for `robotwin` — the sim connects as a client through the
dependency-free `connectors/robotwin.py`. `infer`/`evaluate` reuse the same
engine on a dataset sample / per-episode L1.

The transform pipeline is the contract bridge. Models declare immutable input
requirements as named `ModelMetadata` facts (image range/layout/resize policy,
normalization, tokenizer behavior, vector widths and fixed image resolutions),
and the assembly resolver selects and orders the corresponding
`TransformRegistry` operations. There is no `model.config.transforms` field and
no per-run step-list override. ACT's optional `input_image_size` remains an
explicit from-scratch model tunable; absent it, the IO spec keeps the
dataset-native size. The resolver builds `ModelIOSpec` first, then derives
`data_to_model` / `model_to_robot` calls toward that target. Training and
inference only instantiate saved plans (`build_pipeline`); the deployed
postprocessor is the planned inverse, never the forward list reversed.

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
dataset, which robot, how to train, and where to write — plus, in `overrides:`,
the controlled overrides for a relation the resolver cannot pin down alone. What
it does *not* carry is anything derivable from the three descriptions: action
widths, chunk length, observation window, image size and camera mapping all come
from the composition, because a recipe that restates them is a second answer
that can disagree. `data:` therefore holds only the dataset itself. Training
currently evaluates no validation metric, so every episode contributes training
windows; a split returns only together with a real evaluation loop. CLI
overrides (`--steps`, `--batch-size`, `--output-dir`)
tweak without editing the file. `examples/reference.yaml` documents every field;
`docs/modules/frontend-module.cn.md` documents the three zones and the strict,
no-compatibility parser contract.

**Facts vs tunables — the container is the attribute.** A model ships one
declaration, `ModelMetadata`. Named fields are facts the composition resolver
reads and a recipe can never override; `params` holds that model's tunable
defaults, deep-merged under the recipe's `model.config` where **recipe wins**.
A model author classifies nothing: framework facts have names and types,
everything else goes in `params`. Three guards keep the surface honest —
an undeclared `model.config` key is rejected by `merge_model_config()`, a declared
key nothing reads is rejected by the factory (`utils/tracked_config.py`), and
`model.config.transforms` is rejected explicitly because pipeline operations
are resolver output rather than tunables. Each guard exists because that failure
was previously silent. `inspect model` prints every tunable with its effective
value and source.

**Registry, not if/else.** Each built-in model self-registers with
`@register_vla` inside `adapters/<name>.py`; external packages publish a
`vla_factory.models` entry point. `get_entry(name)` looks it up. Built-in
adapters are discovered on first lookup via `pkgutil.iter_modules`. A broken adapter
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
factory (code cannot be serialized) and `ResolvedAssembly.check_model_compatibility()`.
A checkpoint carries its assembly, so serving it never
re-resolves — a declaration that changed since training is a loud failure rather
than a pipeline that quietly feeds the model wrong-valued inputs.

**Model IO precedes pipeline planning.** Resolution establishes
all four Mappings first, builds `ModelIOSpec` directly from `ModelMetadata`,
resolved model tunables, `DataSchema` and `CameraMapping`, and only then plans
transforms. State/Action Mapping entries represent real correspondences only;
never expand target-width padding into source-less Mapping entries. Padding and
resize calls consume the source/target shapes in `PlanContext`; never recover an
interface by scanning calls or adding a transform-side `output_*` shape hook.
At inference, keep `model_output_dim` (network output) distinct from
`execution_action_dim` (the command leaving `model_to_robot`).

---

## Extending VLA Factory: a new model

Follow `adapters/pi0.py` (or `act.py`) as the worked example. Steps:

1. **Adapter module** — create `vla_factory/model/adapters/<name>.py` in-tree,
   or publish `name = "package.module:factory"` under the
   `vla_factory.models` entry-point group from an external package.
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
   - Declare facts on the named fields (vision slots, `dim_policy`, image and
     tokenizer requirements, `vector_normalization`, ...), and put only tunable
     upstream hyperparameters in `params`. Do not declare a transform step list:
     the resolver derives it. Every `params` key must be read by the factory or
     by a registered framework consumer, or model construction fails.
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
