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
  → data reader / codec / transforms / sampler → VLADataset / DataLoader
  → VLATrainer (HF Trainer based) → checkpoint + inference_metadata
  → InferenceEngine + platform adapters (simulator / lerobot host)
```

---

## Code structure

- **`vla_factory/cli.py`** — argparse CLI: `train`, `preprocess`, `list`,
  `evaluate`, `infer`, `serve`. Entry: `vlafactory-cli` (installed) or
  `python -m vla_factory` (from source). This is where user commands land.
- **`vla_factory/config/`** — recipe parsing & model defaults.
  - `recipe.py` — `TrainRecipe` and sub-dataclasses (mirror the YAML).
  - `parser.py` — `parse_recipe(path|dict) → TrainRecipe`.
  - `defaults.py` — `load_model_defaults()` + `resolve_recipe()`: deep-merges
    `vla_factory/config/model/<name>.yaml` (baseline profile) under the
    recipe's per-run `model.config` (recipe wins). OmegaConf-based, no Hydra
    runtime. Profiles are external/user-editable/diff-friendly; Hydra's
    `defaults: [model/<name>]` can adopt them later with no file change.
- **`vla_factory/config/model/`** — per-model baseline profiles (`act.yaml`,
  `pi0.yaml`, …). Ships with the package (see `pyproject.toml`
  `package-data`).
- **`vla_factory/model/`**
  - `protocols/` — `ModelMetadata` (frozen descriptor: backend, action head
    type, trainable components), `VLAModel` / `VLAModelPyTorch` /
    `VLAModelJAX` protocols, `Observation`/`ActionSpec`. Framework-agnostic
    contracts that flow between data and model layers.
  - `base_contract.py` — reads a pretrained checkpoint's `config.json` to
    learn its real input contract (camera roles, dims, resolution). Source of
    truth is the checkpoint, never a hard-coded lookup — so new checkpoints
    work without framework changes. Powers `vlafactory-cli list --config`.
  - `registry/` — `@register_vla(metadata)` decorator + `get_entry()` /
    `list_entries()`. Entries auto-discovered from
    `registry/entries/*.py` on first lookup (see "Extending").
- **`vla_factory/model/registry/entries/`** — the actual adapters, one file
  per model: `act.py`, `pi0.py`. **These are the canonical worked examples**
  for adding a new model — diff against them, don't start from scratch.
- **`vla_factory/data/`** — unified data semantics.
  - `formats/` — format readers (`lerobot_v3.py`), registry + `get_reader()`.
  - `codec/` — video decode (`pyav.py`), `resolve_codec()`.
  - `transforms/` — `TransformPipeline` + `TransformRegistry` (`@register`
    steps: `resize_images`, `pad_dimensions`, `image_to_float`,
    `image_layout`, `normalize`, `task_tokenize`, …). YAML-driven.
  - `sampling/sampler.py`, `manifest.py` (`DataSchema`, `FeatureStats`,
    `NormStats`), `dataset.py`, `loader.py` (`create_dataloaders`).
- **`vla_factory/training/`** — `train.py` (orchestration: recipe → trained
  model), `pytorch_trainer.py` (`VLATrainer`, wraps HF `Trainer`),
  `strategies/` (`apply_strategy`: full / freeze / selective; LoRA is WIP).
- **`vla_factory/deploy/`** — `infer.py` (`InferenceEngine`, receding-horizon),
  `adapters.py`, `lerobot_host_adapter.py`, `transport.py`, `zmq_client.py`.
- **`vla_factory/utils/constants.py`** — on-disk artifact layout:
  `inference_metadata/{recipe.yaml,schema.json,norm_stats.json}`, `final/model.pt`.
- **`examples/`** — ready recipes. `reference.yaml` is the fully-annotated
  template (every field documented); `act_lekiwi.yaml`, `pi0.yaml`.
- **`scripts/install.sh`** — uv-based env setup for openpi (see Installing).
- **`test/`** — pytest: `test_act_model.py`, `test_pi0_model.py`,
  `test_data_pipeline.py`, `test_base_contract.py`,
  `test_inference_engine.py`, `test_phase4_engine.py`,
  `test_protocols_registry_config.py`.

---

## How it runs

A `train` invocation goes: `parse_recipe` → `resolve_recipe` (merge model
defaults) → `get_entry(model_name)` → factory builds the wrapped model
from `model.path` (pretrained) or from-scratch → `apply_strategy` freezes
params per `finetuning_strategy` + `ModelMetadata.components` →
`create_dataloaders` (reader + codec + transform pipeline + sampler) →
`VLATrainer` runs HF `Trainer` → saves `final/model.pt` +
`inference_metadata/{recipe,schema,norm_stats}`.

`serve` loads a checkpoint's `inference_metadata` to rebuild recipe/schema/
norm_stats at inference time (no re-fit), builds the `InferenceEngine`, and
exposes it over ZMQ (`transport.py`) to a platform adapter (simulator or
lerobot real robot). `infer`/`evaluate` reuse the same engine on a dataset
sample / per-episode L1.

The transform pipeline is the contract bridge: each model's baseline profile
declares which `TransformRegistry` steps it needs (e.g. pi0 needs
`image_to_float` with `range: [-1,1]` HWC); the data layer runs them. The
adapter never rescales images itself — the pipeline must produce the exact
layout/range the upstream model expects.

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

**Recipe is the single source of intent.** Everything the framework needs —
model name, base checkpoint path, action spec, fine-tuning strategy, data
source, transforms, training params, output dir — lives in one YAML. CLI
overrides (`--steps`, `--batch-size`, `--output-dir`) tweak without editing
the file. `examples/reference.yaml` documents every field.

**Model defaults vs recipe config.** A model ships a baseline profile under
`vla_factory/config/model/<name>.yaml`; the recipe's `model.config` is
deep-merged on top and **recipe wins**. The profile is a starting point, not
a frozen contract. Unknown keys surface as an error from the upstream config
object (no silent typo failures).

**Registry, not if/else.** Each model self-registers with `@register_vla`
inside `entries/<name>.py`; `get_entry(name)` looks it up. Entries are
auto-discovered on first lookup via `pkgutil.iter_modules`. A broken entry
raises `RegistryLoadError` (never masked as "not registered") — so optional
deps must be deferred to *factory-call time* (see `act.py`: lerobot imported
inside the factory, so `list_entries()` works even without the `[act]` extra).

**Checkpoint contract, not hard-coded roles.** `base_contract.py` reads the
base checkpoint's `config.json` to learn which cameras/dims it actually
expects; `camera_mapping` in the recipe is validated against that, not
against a fixed role set. New checkpoints work without framework changes.

**Thin adapters, composition over inheritance.** Each entry holds an
upstream model instance and translates `vla_factory.Observation` ↔ the
upstream's batch format; it does not rewrite the upstream model. (Contrast
RLinf, which *inherits* `PI0Pytorch` to add RL heads for its own use case —
vla_factory only fine-tunes, so a thin wrapper is the correct boundary.)

---

## Extending VLA Factory: a new model

Follow `entries/pi0.py` (or `act.py`) as the worked example. Steps:

1. **Baseline profile** — add `vla_factory/config/model/<name>.yaml`
   declaring the transform steps + defaults the model needs. Register it in
   `pyproject.toml` `package-data` if it should ship with installs.
2. **Entry module** — create `vla_factory/model/registry/entries/<name>.py`:
   - `@register_vla(ModelMetadata(name=..., backend="pytorch",
     action_head_type=..., components={...}))` on a factory
     `(recipe, schema) -> VLAModel`.
   - Defer heavy/optional upstream imports **inside the factory**, not at
     module top, so registry load stays robust (rationale in
     `registry.py:RegistryLoadError`).
   - Wrap the upstream model by composition; translate
     `Observation` ↔ upstream batch format; delegate `forward` /
     `predict_actions`. Do not reimplement architecture.
3. **pyproject extra** — add `[project.optional-dependencies].<name>` for the
   upstream ecosystem deps. If it needs uv (strict pins / patches), wire it
   into `scripts/install.sh` rather than expecting `pip install -e ".[name]"`.
4. **Example recipe** — add `examples/<name>_*.yaml`; list it in the README
   support table.
5. **Tests** — mirror `test/test_<name>_model.py` (load + forward smoke).

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
- Fully-annotated recipe: `examples/reference.yaml`
- Adapting a new model (experience + decision tests): `adapt_new_model` skill
- README (user-facing, multilingual): `README.md` · `README.cn.md`
