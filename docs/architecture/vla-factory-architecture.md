# VLA Factory Architecture

> **Document positioning**: this document is a top-level design describing VLA Factory's target architecture and **may include ideas not yet implemented**; the per-module design documents (linked in each section) always align with the **currently implemented** behavior and are intended for readers to study. The architecture document explains "what to build and why"; the module documents explain "how it works today".

## 0. Overview

### 0.1 Engineering Problems in Embodied AI

A robot policy model is not a single-point task from data preparation to validation; it is a complete workflow: data collection and conversion, training configuration, model adaptation, checkpoint management, offline evaluation, simulation validation, and real-robot validation. Today this pipeline is highly fragmented — data formats (LeRobot/HDF5/RLDS/ROS bags) differ in images, states, actions, episode boundaries, and statistics; each model ecosystem ships its own configuration system and training entry point; and training artifacts' metadata and norm stats follow project-specific conventions. Switching to a different model or dataset means rewriting a whole layer of glue code, and experience barely accumulates across experiments.

The most acute pain point is **post-training fine-tuning**. Pretrained VLA models (e.g. OpenPI, OpenVLA, GR00T) typically have billions of parameters, making full-parameter fine-tuning prohibitively expensive; real-world usage therefore relies on parameter-efficient methods like LoRA/QLoRA. But the training scripts shipped by upstream models support these strategies unevenly — switching fine-tuning methods often means hacking the training loop directly. Worse, fine-tuning must balance "preserving pretrained capability" against "adapting to a new scene or a new robot": new data may come from a different embodiment, a different joint order, or a different camera layout, while the pretrained model has fixed assumptions about input semantics. The slightest mismatch in normalization, camera mapping, or joint order can break what the model has already learned, or even cause catastrophic forgetting. These semantics must then be carried intact to the deployment side, otherwise the fine-tuned checkpoint cannot go online — and today this consistency is almost entirely maintained by hand, which is extremely error-prone. On top of that, post-training data is usually limited (dozens to hundreds of demos), and there is no unified way to diagnose training stability and generalization.

### 0.2 Why Existing Frameworks Fall Short

The open-source community already has many excellent training frameworks, but most of them cannot directly solve VLA's engineering problems.

**Training scripts shipped with each model ecosystem are not directly reusable.** ACT, OpenPI, OpenVLA, GR00T, SmolVLA, etc., each carry their own training entry point, configuration files, and data-loading logic, assuming different data formats and robots, with almost no reusable interface between them.

**Why you cannot just use an LLM fine-tuning framework like Llama Factory.** Llama Factory is a mature LLM fine-tuning framework, but it targets a "text-in → text-out" paradigm, which differs fundamentally from how VLA works:

| Dimension | Llama Factory assumption | VLA Factory reality |
|---|---|---|
| I/O | Pure text token sequences | Multimodal temporal data: images, robot states, action sequences, language instructions |
| Output form | One-shot generated text | Continuous action sequences (action chunks), streamed frame-by-frame to real joints |
| Data carrier | Text corpora | Multimodal temporal data with episode boundaries, video codecs, and normalization stats |
| Embodiment constraints | None | DoF, joint order, control mode, safety bounds, gripper conventions |
| Deployment form | One inference call returns a result | On-device real-time closed-loop control requiring stable frequency and low latency |
| Train-deploy consistency | Weights alone are enough | Must share normalization, camera mapping, and joint order, or the model cannot be deployed |
| Safety boundary | Bad output is at worst a wrong answer | Bad output can damage hardware or injure people; action-legality checks and fallbacks are required |

In other words, VLA adds the **action** modality on top of LLM/VLM and usually runs in an on-device closed loop. The real-time performance, stability, legality, and safety boundary of action output are problems that LLM training frameworks do not cover at all. Directly adopting Llama Factory means either bolting on a thick adaptation layer outside it, or losing train-deploy consistency — in the end you are still rebuilding a framework from scratch.

### 0.3 Positioning and Goals

VLA Factory is a recipe-driven framework for training and deploying robot Vision-Language-Action models. A single YAML recipe describes the model, data, robot, fine-tuning strategy, training parameters, and output directory; the framework then closes the loop from data loading, composition resolution, sample construction, model adaptation, training, and checkpoint artifacts through to an online inference service.

The core positioning of the framework is not to reimplement various VLA or imitation-learning models, but to provide a stable engineering glue layer:

- For users: start training, evaluation, inference, and deployment through one unified recipe.
- For data: convert different data formats into a unified intermediate representation and training samples.
- For models: wrap external ecosystem model implementations through thin adapters.
- For training: reuse the mature capabilities of PyTorch, HuggingFace Trainer, and upstream model libraries.
- For deployment: connect to simulators and real robots through a unified inference engine and platform adapters.

**VLA Factory's goal**: establish stable engineering standards across data, models, training, artifacts, and validation, so that capabilities from different ecosystems can enter the same workflow through clear boundaries. "Unified" here does not mean reshaping every model, data format, or runtime platform into one implementation; it means defining stable description formats and composition rules among the three dimensions — dataset, robot, and VLA model. A new dataset can be combined with existing models, and a new model can be combined with existing data and robots, without writing per-combination adapter code. Quantitatively, N datasets × M models × K robots yield up to `N × M × K` candidate combinations; as long as the N datasets come from F formats, the code assets the framework maintains over the long term approach `F + M + K` rather than `N × M × K` dedicated adapters. `N × M × K` is a candidate space, not a promise that every combination is compatible — the composition resolution layer produces a deterministic result or a structured error for every actual combination. At the ecosystem level, the framework integrates mainstream data formats, model ecosystems, and deployment platforms, and can interoperate with reinforcement-learning and evaluation frameworks such as RLinf in the post-training stage, rather than reimplementing an RL training system internally.

### Table of Contents

- [0. Overview](#0-overview)
- [1. Design Principles](#1-design-principles)
- [2. Global Architecture](#2-global-architecture)
- [3. User Interface](#3-user-interface)
- [4. Core Module Design](#4-core-module-design)
- [5. Dependency Management Strategy](#5-dependency-management-strategy)
- [6. Testing Strategy](#6-testing-strategy)
- [7. Extension and Evolution](#7-extension-and-evolution)

---

## 1. Design Principles

### 1.1 Recipe Driven

A training run should be fully described by a recipe. Model selection, data paths, robot selection, sampling windows, action space, fine-tuning strategy, training steps, and output directory all come from configuration, not from scattered scripts.

The recipe is the user's highest-priority configuration entry point. The top-level fields of the recipe express experiment intent (model/data/robot selection, fine-tuning strategy, training parameters, output); the model's own capabilities and defaults are carried by the model declaration (ModelMetadata) and are not in the recipe; adjustments to the relationships among data/model/robot go in the `overrides` block. This keeps experiment configuration auditable: the user can see every intentional override in one file, rather than tracing behavior through scripts and implicit defaults.

Model-related **facts** (preprocessing semantics, image value range, camera slot layout, dimension policy, ...) are published with the model declaration `ModelMetadata` and cannot be modified in the recipe; the model's **tunable hyperparameters** (depth, inference steps, compile mode, ...) ship their defaults in the same declaration but may be overridden per-run in the recipe's `model.config`. The recipe carries the user's composition selection, composition adjustments, model hyperparameter overrides, and training parameters.

The CLI may provide a few temporary overrides such as `--steps`, `--batch-size`, `--output-dir` for smoke tests or debugging, but the recipe remains the primary contract.

### 1.2 Adaptation Over Reimplementation

VLA Factory does not hold upstream model architecture code. Model capabilities are exposed through registry entries. Each entry is only responsible for:

- Declaring `ModelMetadata`.
- Parsing the recipe and dataset schema.
- Constructing the upstream model object.
- Translating between VLA Factory's `Observation` / action tensors and the upstream model's input/output format.

This reduces subtle behavioral deviations introduced by self-written models and keeps maintenance cost low as upstream ecosystems evolve.

### 1.3 Protocols Do Not Assume Model Structure

The unified model protocol requires only two core capabilities:

- `compute_loss(observation, actions, ...)`
- `predict_actions(observation, ...)`

Parameter access, device transfer, training mode, and similar capabilities are extended by backend. For example, PyTorch models implement `parameters()`, `named_parameters()`, `train()`, `to()`. The framework does not require all models to expose the same internal modules.

### 1.4 Data Contract Decoupled From Models

The data module outputs unified observation/action samples; the model module only consumes the abstracted `Observation` and action tensors. Field paths, video codecs, episode indices, statistics, and vector key ordering from the source data format must not leak into model implementations.

### 1.5 Dependencies Installed On Demand

The core package stays lightweight. Upstream ecosystem dependencies such as ACT, OpenPI, and GR00T should be introduced through optional extras. When a model is not in use, missing dependencies for that model must not affect the registration, training, or deployment of other models.

### 1.6 Composition Resolution Over Implicit Conventions

The relationships among dataset, robot, and VLA model (which camera maps to which model visual slot, how action dimensions are padded, how joint orders are aligned, how gripper direction is flipped) must be derived explicitly by the framework from each dimension's description, not triggered as hidden branches keyed on "model name + dataset name + robot name".

The allowed inputs are data description, model description, robot description, and controlled overrides; writing hidden conditional branches for a specific combination is forbidden.

### 1.7 Determinism and Conservative Failure

Composition resolution must satisfy:

- Same inputs produce the same output.
- Results do not depend on registration order.
- "Equal array length" is never substituted for "semantic match".
- Automatic generation happens only when the mapping is unique.
- Ambiguity requires a controlled override.
- Missing conditions for a high-risk transform is an explicit failure, not a guessed silent execution.

---

## 2. Global Architecture

### 2.1 Overall Architecture Diagram

![VLA Factory overall architecture, generated from ../graph/architecture-text.md](../graph/vla-factory-layered-architecture.en.svg)

Four layers, top to bottom. The diagram shows only the ecosystem each layer plugs into, not internal implementations:

- **User Interface**: currently hosts the YAML Recipe and CLI, with WebUI and Agent entry points available as future peers. Each entry point organizes framework capabilities instead of duplicating training, inference, or composition logic.
- **Finetuning Layer / Inference Layer**: two peer execution engines. The finetuning layer plugs in fine-tuning strategies such as LoRA / PiSSA / GaLore; the inference layer connects to simulation and evaluation environments such as RoboTwin / LIBERO / ManiSkill.
- **Composition Resolution Layer**: built on top of the three unified descriptions, it further composes data, VLA model, and robot into an **embodiment composition** (producing `ResolvedAssembly` on success or `ResolutionError` on failure) shared by the finetuning and inference layers. This layer does not plug into any external ecosystem.
- **Data / VLA Model / Robot**: three dimensions, each with a unified description — a unified data description (`DataSchema`), a unified model description (`ModelMetadata`), and a unified robot description (`RobotProfile`); together they are the framework's "three unifications". Each dimension integrates a concrete ecosystem: LeRobot / RLDS / HDF5 on the data side, GR00T / OpenPI / OpenVLA on the model side, and SO101 / Lekiwi / Franka on the robot side.

Dependencies: the recipe drives the two execution engines; the descriptions of the three dimensions flow into the composition resolution layer; the embodiment composition is then handed to the finetuning and inference layers. The finetuning and inference layers only consume the embodiment composition and no longer derive the relationships among the three dimensions on their own.

### 2.2 Code Directory Structure

The current core code lives under `vla_factory/`. This structure describes only relatively stable directory boundaries and module responsibilities; concrete file names may be added or adjusted as the implementation evolves, and the architecture document does not maintain a file-level inventory.

```text
vla_factory/
├── examples/        # recipe examples and minimal runnable samples
├── docs/            # architecture, usage notes, and design records
├── user_interface/        # user-facing entry points: shared Recipe contract and CLI
│   ├── recipe.py    # TrainRecipe, strict parsing, model-tunable merge
│   └── cli.py       # current CLI; WebUI / Agent may be added alongside it
├── data/            # data reader and intermediate representation
│   ├── data_schema.py # unified data-layer representation and describe_dataset entry
│   ├── reader/      # FormatReader, ReaderRegistry, and format implementations
│   └── codec/       # VideoCodec, CodecRegistry, and decoder implementations
├── assembly/        # composition resolution of dataset × robot × VLA model
│   ├── resolve_assembly.py # public orchestration, ResolvedAssembly, persistence
│   ├── resolve/     # pure resolve_from_facts and resolution rules
│   ├── transform/   # TransformStep / TransformPipeline / TransformRegistry and step implementations
│   └── ...
├── model/           # model abstraction and upstream adapters
│   ├── model_interface.py # ModelMetadata, Observation, and the VLAModel interface
│   ├── registry.py  # ModelRegistry, @register_vla, and external plugin discovery
│   ├── adapters/    # thin bindings for ACT / PI0 / PI05 and other upstream models
│   └── checkpoint_validation.py # optional consistency check for redundant checkpoint facts
├── robot/           # robot embodiment description (RobotProfile) registration and validation
├── training/        # training orchestration: Observation sample construction, dataloader, Trainer, fine-tuning strategies
│   ├── strategies/
│   └── ...
├── inference/       # inference engine, platform adapters, transports, and action execution strategies
│   ├── inference_engine.py # checkpoint-to-ActionChunk inference core
│   ├── execution.py # action-chunk execution policies and PolicyExecutor
│   ├── checkpoint.py # inference metadata and model-weight loading
│   ├── evaluate_dataset.py # single-sample inference and dataset evaluation
│   ├── deploy.py    # public deployment orchestration entry point
│   ├── connectors/  # lightweight connectors imported by remote robot environments and their bootstrap configs
│   ├── platforms/   # adaptation between native platform observation/action and the unified inference interface
│   ├── transports/  # wire protocols and serialization such as ZMQ and length-prefixed JSON RPC
│   └── ...
├── utils/           # shared constants, utilities, and lightweight helpers across modules
│   └── ...
└── test/            # unit tests, contract tests, and integration smoke tests
```

**Dependency direction (top-down, no back-edges):** `data/`, `model/`, `robot/` are leaf layers — `data/` only produces IR such as `DataSchema` / `Episode` / `Frame` / `NormStats`; `model/` holds the VLAModel interface, `Observation`, and `ModelMetadata`; none of them depend upward. `assembly/` reads the three descriptions and produces the embodiment composition; `training/` and `inference/` consume the embodiment composition and each assemble `Observation` samples from IR / platform observations via a TransformPipeline (`data/` does not construct samples). `Observation` lives in `model/model_interface.py` and is depended on by `assembly/`, `training/`, and `inference/`, while the model interface does not depend back on any of them — the graph is acyclic.

---

## 3. User Interface

The user interface is the framework's user-expression layer. Its current entry points are YAML Recipe and CLI; WebUI and Agent user interfaces may be added alongside them later. Recipe is the structured input contract these user interfaces can share, not the name of the whole layer. Every user interface translates user intent into calls to the public assembly, training, and inference capabilities.

The recipe written by the user is the single source of truth for configuration. Model defaults are published with the model declaration (ModelMetadata) and cannot be modified in the recipe; the CLI provides a few temporary overrides.

### 3.1 The Three Zones of a Recipe

The recipe is cleanly divided into three zones with distinct responsibilities:

**① Composition Selection Zone** — only specifies "which dataset, which model, which robot", one or two fields per dimension, without touching the relationships among them:

```yaml
model:
  name: pi05                    # registered model name
  path: lerobot/pi05_base       # pretrained weights path (omit for from-scratch models)
data:
  path: /datasets/aloha_transfer_cube
  format: auto                  # auto-detects LeRobot / HDF5 / RLDS / Zarr
robot:
  name: aloha_vx300s_bimanual   # robot embodiment declaration
```

The model may also be written as `model: act`. When the scalar contains `/`,
the full string is the checkpoint path and its last segment is the default
model name: `model: lerobot/pi0` is equivalent to
`model: {name: pi0, path: lerobot/pi0}`. Use the explicit mapping for exceptions.

**② Composition Adjustment Zone (optional)** — by default, the relationships among the three are derived automatically by the composition resolution layer (Section 4.2) from their descriptions; this zone is filled only when the resolver cannot decide uniquely or the user wants a non-default policy (Section 4.2 calls this a "controlled override"):

```yaml
overrides:                   # optional, empty by default
  camera_mapping:               # model visual slot -> data/robot camera (specified on ambiguity)
    base_0_rgb: front
    left_wrist_0_rgb: wrist
  default_task: "pick up the block"  # language fallback (used when data/deploy has no task)
```

This zone holds only overrides the resolver actually consumes. An adjustment nothing reads is a field a user can set and watch do nothing, so the frequency (`accept_fps_mismatch`) and gripper (`gripper_flip`) knobs are deferred together with their compatibility checks rather than reserved as inert fields.

**③ Training Parameter Zone** — describes "how to train" and is completely independent of the relationships among data, model, and robot:

```yaml
finetuning:
  strategy: lora                # full | lora | freeze | selective
  config:                       # strictly parsed by the selected strategy
    r: 16
    # Bare config defaults: components="all" (LoRA every component),
    # freeze_components=[], target_modules="all-linear" (peft: all Linear/Conv1D).
    # Below overrides components to LoRA just the VLM subtree.
    components: [llm]    # references keys of ModelMetadata.components
training:
  lr: 2.5e-5
  batch_size: 8
  total_steps: 20000
  num_workers: 4
output:
  output_dir: outputs/pi05_aloha
  report_to: tensorboard
```

The relationships among the three — which camera goes into which model visual slot, how action dimensions are padded, how joint orders are aligned, how the gripper is flipped, how normalization is aligned — **do not appear in the recipe by default**; they are derived automatically by the composition resolution layer (Section 4.2) from the three descriptions, and are only written explicitly in the composition adjustment zone when there is ambiguity or a policy choice.

### 3.2 Field Overview

The table below summarizes the main recipe fields by zone (full fields, defaults, and allowed values are in `examples/reference.yaml` and `vla_factory/user_interface/recipe.py`):

| Zone | Block | Main fields | Notes |
|---|---|---|---|
| Composition selection | `model` | `name`, `path` | Model selection; `path` is required for fine-tuning, optional for from-scratch |
| Composition selection | `data` | `path`, `format`, `video_codec` | Dataset path and format; `format: auto` auto-detects |
| Composition selection | `robot` | `name` | Robot embodiment declaration |
| Composition adjustment (optional) | `overrides` | `camera_mapping`, `default_task` | Explicitly specify the three-way relationship when the resolver cannot decide uniquely; cannot rewrite objective facts (shape, checkpoint slots, joint topology, fixed dim caps). Only overrides with a consumer are kept; the rest are deferred with their checks |
| Training params | `finetuning` | `strategy`, `config` | Registered fine-tuning strategy and its strictly validated configuration |
| Training params | `training` | `lr`, `lr_backbone`, `batch_size`, `total_steps`, `gradient_checkpointing`, `num_workers` | Optimizer, scheduling, memory, and data loading |
| Training params | `output` | `output_dir`, `report_to`, `logging_steps`, `save_steps`, `save_total_limit`, `overwrite_output_dir` | Checkpoint, logging, and final weights |

`TrainRecipe` and its sub-dataclasses in `vla_factory/user_interface/recipe.py` define the public YAML shape. `finetuning.config` remains a mapping until the selected `FinetuningStrategy` parses it into that strategy's strict config dataclass, so adding a strategy does not keep expanding `TrainRecipe`.

### 3.3 Configuration Sources and Priority

Configuration merging follows "the closer to this run, the higher the priority":

| Priority | Source | Scope | Notes |
|---|---|---|---|
| 1 | Explicit CLI | Temporary override for this run | Highest priority, for smoke tests, tuning, and ad-hoc output dir changes. |
| 2 | YAML recipe | This experiment's config | The user's main configuration entry point, describing composition selection, adjustment, and training params. |
| 3 | Framework defaults | Fallback | Defaults of `TrainRecipe` and its sub-dataclasses, plus model-builtin defaults (declared via ModelMetadata, immutable). |

The training entry `train()` currently supports CLI overrides `override_steps`, `override_batch_size`, `override_output_dir`.

---

### 3.4 Resolution Workflow and Validation

Once the recipe is written, the composition resolution layer (Section 4.2) derives the relationships among data, model, and robot from it. Facts of different dimensions are determined by their own sources and cannot be governed by a uniform "later-write-wins":

| Field type | Source policy |
|---|---|
| Data attributes and semantics | FormatReader inspects the actual data and produces DataSchema |
| Model static capability | ModelMetadata |
| Checkpoint instance | Selected by recipe `model.path`; optionally checked against ModelMetadata, never an interface-fact source |
| Robot facts | RobotProfile / URDF |
| Three-way relationship | Generated by the resolver; explicitly specified in the recipe's `overrides` block when ambiguous |

The embodiment composition (Section 4.2) must record the source of every final field. Ordinary users do not need to read the DataSchema or RobotProfile field reference first — the first-use flow is:

```text
fill in the three selections
    -> resolve
       ├─ success: show summary, can be handed directly to downstream modules
       └─ failure: show only the relevant fields, candidates, and a minimal override example
```

Only when debugging does `inspect` (Section 3.5) reveal the framework-derived internal facts; error messages follow the principle of progressive disclosure: for example, on a camera-mapping ambiguity they show only the target slot, candidate cameras, and the corresponding override snippet, not the full declarations. The CLI provides `resolve` to resolve and preview a three-way composition, and `inspect` to check actual data, model declarations, and robot declarations. These commands must run without optional model heavy-dependencies installed, without a GPU initialized, and without a robot platform connected.

### 3.5 Dimension Inspection: inspect

`inspect` is the concrete form of the checking capability described in the previous section: it outputs the descriptions of the three dimensions — dataset, model, robot — in structured form, so users can see "what the three things look like in the framework's eyes" before composition resolution. CLI forms:

```bash
vlafactory-cli inspect data  --path <dataset> [--stats]
vlafactory-cli inspect model --name <model> [--path <checkpoint>]
vlafactory-cli inspect robot --name <robot>
vlafactory-cli inspect --config <recipe.yaml>   # all three at once, per the recipe
```

The three dimensions share one output envelope: `{dimension, source, facts}` — human-readable YAML by default, `--json` for tool consumption; key order is deterministic and diffable. Every fact inside `facts` is labeled with its source (`measured` / `inferred` / `undeclared`); for example, `inspect model --path` always reports interface facts from ModelMetadata and lists the checkpoint check result separately as `compatible` / `incompatible` / `unavailable`, never producing a merged view. `inspect --config` emits the three envelopes as one top-level document (a JSON array / YAML list), so `--json` output can be consumed whole by `json.load` / `jq`; a dimension that cannot be read is noted on stderr and skipped.

inspect follows three disciplines:

- **No semantic guessing** — facts that cannot be probed, and cannot be uniquely inferred under the controlled vocabulary, are output as null (`semantic: null (undeclared)`); exposing the gap is preferred over similarity guessing. The gap is exactly what makes the resolver fail conservatively and demand a controlled override.
- **No heavy-dependency activation** — `inspect model` only reads the registry's ModelMetadata and the checkpoint's `config.json` for the optional consistency check, and never calls the model factory; every subcommand runs without GPU, without optional extras, and without a robot connection.
- **No cross-dimension reference resolution** — a data-side `robot_ref` (e.g. lerobot `robot_type`) is output as a plain string; whether it corresponds to a registered RobotProfile is validated by the composition resolution layer.

`--stats` is an explicit cost switch: statistics default to a summary. The data dimension's output strictly follows the `DataSchema` fields and the deterministic inference rules in `data/semantics.py`.

---

## 4. Core Module Design

This chapter splits an experiment into four layers: first describe the three dimensions — data, VLA model, robot (4.1); then the composition resolution layer composes them into an embodiment composition (4.2); finally the finetuning layer (4.3) and the inference layer (4.4) consume it. The composition resolution layer (4.2) in particular is target-oriented and does not imply all capabilities are implemented yet.

### 4.1 Data × Model × Robot

Any embodied-AI experiment is essentially composing three things:

```text
dataset ──┐
          │
VLA model ┼──> resolver ──> embodiment composition
          │
robot ────┘
```

- **Dataset**: the actual content of the training data — which cameras, which state/action fields, what dimensions, what ordering, what fps, what action statistics. It varies with the actual data content.
- **VLA model**: what inputs the model needs — how many visual slots, what image size, whether state/action dimensions are fixed or padded, how long the action horizon is, what normalization is required. The complete interface is described by registry ModelMetadata; checkpoints within one family only change the weight source, not the interface.
- **Robot**: what the robot is physically — how many DoF, what the joints are named, how they are ordered, which control modes are supported, what values represent gripper open/close, where the joint limits are, what the recommended control frequency is. It evolves with robot model and embodiment variant.

Each of the three dimensions has a "description", and the resolver only consumes these descriptions — it does not read raw data, create models, or connect to robot platforms.

#### 4.1.1 Dataset: DataSchema and NormStats

`DataSchema` is a fact snapshot generated by the data reader after inspecting an actual dataset; it describes the fields, dimensions, cameras, temporal information, and action semantics that truly exist in the data. Different datasets of the same format can produce different DataSchemas. Composition resolution cares about the following categories of information:

- Camera names, resolution, layout, color space, and frame semantics;
- state keys, ordering, dimensions, units, and coordinate frames;
- action keys, ordering, dimensions, units, control modes, and rotation representations;
- gripper conventions;
- timestamps, fps, and episode boundaries;
- language fields and default task description;
- the robot identity the data corresponds to;
- the schema source.

`NormStats` is the normalization statistics bound to the actual data content (mean/std, min/max, or quantiles). Together with DataSchema it is read by the reader or computed by the framework, but kept as an independent structure.

The data module parses external datasets into VLA Factory's Canonical IR (`DataSchema` / `Episode` / `Frame` / `NormStats`); video decoding is a replaceable capability used while reading. The training layer persists the resolved schema, norm stats, IO spec, and pipeline plans as part of `ResolvedAssembly` in `inference_metadata/assembly.json`, and deployment reads only that training-time snapshot. **Sample construction** (assembling IR into `Observation` via a transform pipeline) and batching are not in the data layer — they are done in the finetuning layer (4.3).

The data dimension's description **comes entirely from actually reading the dataset**: the Reader probes objective facts (dimensions, resolution, fps, episode boundaries, per-dim names, `robot_type`) and makes deterministic inferences about semantics under a controlled vocabulary (e.g. a camera key uniquely matching `wrist_left`), labeling each fact with its source (measured / inferred / undeclared). Semantics that cannot be probed do not enter the data description, and no dataset-side declaration file is introduced — the gaps are filled on demand by the recipe's controlled overrides (§3.1 zone ②) during composition resolution, or are left to framework-level conventions.

#### 4.1.2 VLA Model: ModelMetadata

The model-dimension description lives in **one declaration published with the model**, `ModelMetadata` (one adapter declaration file per model family). It has two halves, and **the container is the attribute**:

- **Named fields = facts** — interface capability, camera slot layout, input size and image value range, dimension policy, normalization method: everything the composition resolver reads. They are not in the recipe and cannot be modified per-run; changing one would make the embodiment composition disagree with the model that actually runs.
- **`params` = tunables** — only that model's upstream hyperparameters (depth, width, dropout, inference steps, compile mode, ...), each with a default value that the recipe's `model.config` may override. Transform operations are derived from named facts and are not recipe fields.

A model author therefore classifies nothing: framework-level facts have named fields and types, everything else goes in `params`. The `params` key set doubles as the basis for two checks — a `model.config` key the model never declared is an error (with the closest candidates suggested), and a declared key nothing consumes is an error at model construction, which is what keeps "I changed it and nothing happened" from being possible.

If an experiment needs to adjust the relationships among data/model/robot (e.g. camera mapping, language fallback), express it in the recipe's `overrides` block (see Chapter 3) rather than editing the model declaration.

##### ModelMetadata

`ModelMetadata` is the static capability description of a model, reusing and extending existing model metadata to describe the relatively stable interface capabilities and constraints of a model family. It carries two categories of information at once: the interface facts needed for composition resolution, and the model's own capabilities such as backend, trainable components, and fine-tuning abilities (the resolver reads only the former for composition resolution; the latter is kept in the embodiment composition for the training module to access). Specific fields include: model name, backend type, action dim / horizon, action head type, architecture type, training paradigm, trainable-component map, whether prompt is required, image size/range/layout/resize policy, tokenizer requirements, supported fine-tuning methods, and install hint. The key information categories composition resolution depends on:

- visual slots, names, and input shapes;
- state/action dimension policy: fixed, flexible, padded;
- action horizon;
- action representation and control mode;
- rotation, gripper, and unit conventions;
- normalization method and required statistics;
- whether a prompt is required;
- supported input resolutions and dtypes.

The number of model slots does not equal the number of real cameras that must exist. A fixed model slot only means a corresponding key/tensor must be preserved on the model's call boundary. The current design introduces no extra type for missing cameras: any model slot without a real-camera mapping is uniformly padded, with the model input adapter generating the placeholder image and invalid mask the model needs. For example, if the robot or data has only 2 real cameras while the model has 3 fixed slots, the resolver still produces 3 model input slots and plans padding for the third — a mere count mismatch is not deemed incompatible.

##### Optional checkpoint consistency check

A checkpoint `config.json` may repeat input-slot, dimension, and image-shape
information. Before loading weights, the framework may compare those values
with ModelMetadata: an unreadable config skips the optional check, while a
contradiction fails clearly. This is a diagnostic gate, not a second contract;
it cannot override ModelMetadata or alter resolver, ModelIOSpec, Mapping, or
PipelinePlan output. Local directories, weight files, and external repositories
can therefore provide interchangeable checkpoints for one model family.

##### VLAModel Protocol

The unified protocol defines only the minimal methods needed for training and inference:

```python
compute_loss(observation, actions, ...)
predict_actions(observation, **kwargs)
```

PyTorch models additionally implement `parameters()`, `named_parameters()`, `train()`, `to()` for optimizers, freeze strategies, device transfer, and Trainer.

##### Registry

Models are registered via a decorator:

```python
@register_vla(ModelMetadata(name="act", ...))
def load_act(recipe, assembly):
    ...
```

`get_entry(name)` lazily imports `model/adapters/*` on first access, triggering built-in registration. External packages publish models through the `vla_factory.models` Python entry-point group. The registry loader treats an adapter or plugin import failure as a real error, so syntax errors and missing hard dependencies are not disguised as "model not registered". A missing optional dependency should produce a clear error at factory call time. For example, ACT can be registered and listed, but when actually creating it without lerobot installed, the user should be prompted to install the `[act]` extra.

##### Thin Adapter

Each model adapter should remain thin. Taking ACT as an example:

- The upstream `lerobot` holds ACTPolicy and the network structure.
- VLA Factory's wrapper only converts `Observation` into the lerobot batch dict.
- Loss and action-chunk prediction call into the upstream policy.
- Checkpoint loading handles the key-prefix difference between the wrapper and the upstream model.

This boundary requires VLA Factory neither to copy upstream model code into the repo nor to rewrite model details in the adapter.

#### 4.1.3 Robot: RobotProfile

`RobotProfile` describes the robot embodiment and **does not describe which process it connects to or which transport it uses**. It is responsible for:

- robot identity and embodiment variant;
- stable semantic names for sensors and cameras;
- joint names, ordering, units, types, and limits;
- native action representation and supported control modes;
- gripper conventions;
- coordinate frames and URDF references;
- static safety bounds needed for composition resolution;
- recommended control frequency.

The three dimensions share strict schema and provenance recording but differ in lifecycle: the dataset varies with content, the model varies with model family and instance, and the robot varies with embodiment model. The framework only unifies their resolution interface and does not force them to use the same registration and dispatch mechanism.

#### 4.1.4 How to Extend the Three Dimensions

External developers do not need to learn the full composition protocol; they only need to complete the minimal extension entry for one dimension. The framework should provide scaffolding and registration-time validation for each of the three entries — data format, model, and robot. The scaffolding generates only that dimension's adapter, minimal declaration, and contract test, and splits fields into: must-fill (facts the resolver needs that cannot be read from upstream objects), auto-read (obtained from actual data, checkpoint metadata, URDF, or adapter), and optional-supplement (filled only when a specific capability exists).

**Adding a data format**: implement `FormatReader`; produce unified DataSchema, NormStats, and episode/frame from the actual data; validate strictly via DataSchema; add reader-contract and representative composition tests. Adding a new dataset of an existing format usually only requires providing a path, not registering a data instance. A reader must not, to fit some model, rename fields to model-specific names, do model-specific padding, reorder actions per a model's requirement, or inject model-specific normalization.

**Adding a model**: add or extend a ModelMetadata registry entry; declare the model's observation, action, language, normalization, and temporal interface; wrap the upstream implementation with a thin ModelAdapter; add optional checkpoint-config validation as needed; add metadata-contract and representative composition tests. A ModelAdapter must not select cameras by dataset name, adjust output semantics by robot name, guess field correspondences by sorting, or re-execute the resolver's compatibility checks.

**Adding a robot**: add a RobotProfile; reference URDF or other standard embodiment descriptions as needed; declare sensors, joints, control modes, gripper, coordinate frames, and static safety bounds; add profile-contract and representative composition tests. A RobotProfile should support importing determinable fields from URDF, vendor descriptions, or existing adapters; developers only supplement VLA semantics not present in the standard description. The robot's runtime platform, connectors, and transports are out of scope for this section.

---

### 4.2 Composition Resolution Layer

The composition resolution layer resolves the three dimensions into a unified embodiment composition and is the common upstream of the finetuning and inference layers; it is a deterministic, pure-logic layer and also the target architecture's direction of evolution.

#### 4.2.1 Embodiment Composition (ResolvedAssembly)

**The "embodiment composition" is a core concept defined by this framework.** It is the **sole product** of successfully resolving "dataset × robot × VLA model", and corresponds to `ResolvedAssembly` in code.

The reason for defining this concept explicitly is that both the training module and the inference module need to know "exactly which three things are being used this run and how they relate" — and this is precisely what is most error-prone and most often silently assumed by each side. The embodiment composition extracts this knowledge from training code and inference code and fixes it as a non-bypassable handoff object.

##### What the embodiment composition contains

The embodiment composition contains four categories of information, together answering "which descriptions are used, what the final interface is, how fields correspond, and which transforms must run":

```text
embodiment composition ResolvedAssembly
├─ normalized references to the three descriptions
│   ├─ dataset description (DataSchema + NormStats)
│   ├─ VLA model description (ModelMetadata)
│   └─ robot description (RobotProfile)
├─ model IO spec (ModelIOSpec)
│   (the observation / action / language / temporal semantics ultimately used by this composition)
├─ field mappings
│   ├─ CameraMapping  : cameras -> model visual slots
│   ├─ StateMapping   : state fields -> model state vector
│   ├─ ActionMapping  : data actions -> model action vector
│   └─ LanguageMapping: task-text field -> model prompt
└─ declarative descriptions of three Transform Pipelines (TransformPipelinePlan)
    ├─ data_to_model  : data sample -> model training interface
    ├─ robot_to_model : robot real-time observation -> model input
    └─ model_to_robot : model action output -> DataSchema action
```

- **Normalized references to the three descriptions**: downstream no longer needs to query each registry; all facts are in one object. From the downstream perspective they are part of the embodiment composition, not external inputs to be queried again.
- **Model IO spec (`ModelIOSpec`)**: what the model actually takes in and gives out once the descriptions are reconciled — camera keys and per-camera sizes, state width and prompt requirement on the way in; action width and horizon on the way out. It is the fact standard after composition: downstream builds the model against it and reads/writes observation/action by it.
  `cameras` holds the **canonical observation keys the framework uses (data-side names)**, not the model's own vision slots: on pi0 the data side is `front`/`wrist`, the model side is three openpi roles, and `CameraMapping` connects them. `ModelIOSpec` is resolved before pipeline planning from model facts and flexible/native data facts: pi0 camera sizes come from `VisionSlot.resolution`; ACT may select an explicit `model.config.input_image_size`, otherwise it keeps `DataSchema`'s native size. The transform plan consumes this target interface to generate resize/pad calls; step arguments and shape hooks never define the interface in reverse.
  `action_dim` is the **model's output width** (32 for pi0). The inference engine exposes this as `model_output_dim` and separately exposes `execution_action_dim`, the DataSchema action width after `model_to_robot` (8 for pi0).
- **Field mappings**: describe only real field and semantic correspondences; they perform no tensor computation. In particular, `StateMapping` / `ActionMapping` do not manufacture source-less entries for padding. The model target width lives in `ModelIOSpec`; padding count is the target width minus the number of real mapped dimensions, and execution lives in the PipelinePlan.
- **Transform Pipeline declarations**: declarative descriptions telling downstream "which transforms to run, in order, along this path", but containing no instantiated executable objects.

##### What the embodiment composition does not contain

To keep responsibilities clear, the embodiment composition **does not** contain:

- learning rate, batch size, or training steps;
- LoRA, optimizer, or Trainer config;
- checkpoint save layout;
- deployment platform, IP, port, or transport;
- client/server topology;
- any instantiated runtime object (Trainer, DataLoader, PlatformAdapter, model weights, etc.).

These belong to downstream execution config or runtime dependencies and are managed by the training module and the inference module respectively.

##### The embodiment composition is the sole entry point for downstream

The training module and the inference module can access the descriptions and three-way relationships of this composition **only** through the embodiment composition:

```text
embodiment composition + downstream's own config and runtime dependencies
    ├─> training module
    └─> inference module
```

They must not bypass the embodiment composition to independently query the data, model, or robot registries, nor re-derive the three-way relationships from model/dataset/robot names. The two downstream modules may each further parse the embodiment composition into a "training plan" or "inference plan", but there can be only one source of truth.

#### 4.2.2 Resolver

The resolver is the entry point for three-way composition resolution; its public entry is `resolve_assembly()`. It is a **deterministic, pure-logic component**:

- it creates no models;
- it builds no DataLoader;
- it starts no training;
- it loads no deployment platform;
- it modifies no dataset;
- it depends on no GPU;
- it creates no downstream output directory;
- its result is serializable, diffable, and unit-testable.

##### Inputs

- DataSchema generated by the data reader for the actual data;
- NormStats;
- ModelMetadata already in the model registry;
- RobotProfile;
- controlled overrides (explicit user specifications for ambiguous cases);
- resolution rules and the existing TransformRegistry.

Training hyperparameters and deployment session config **do not** enter the resolver.

##### Resolution phases

A single composition resolution executes in these phases:

```text
1. Load            load DataSchema, NormStats, ModelMetadata, RobotProfile
2. Validate        validate the internal structure and provenance of each
3. Check Pairs     check compatibility facts that share an explicit vocabulary
4. Resolve Mappings generate Camera, State, Action, and Language DataSchema-to-model correspondences
5. Build IO Spec   resolve the model interface directly from model tunables/facts, DataSchema, and CameraMapping
6. Plan Pipeline   generate TransformPipelinePlan calls toward that ModelIOSpec
7. Emit            on success emit the embodiment composition; on failure raise a structured ResolutionError
```

The resolver must not create models, DataLoaders, training output directories, or deployment connections before completing all validations.

##### Compatibility checks

Compatibility checks cover:

| Check | Comparison | On mismatch |
|---|---|---|
| State dim | data vs model | flexible/padded convertible; fixed errors |
| Action dim | data vs model | plan a transform if paddable; otherwise error |
| Camera slots | DataSchema cameras vs model slots | unique match → Mapping; unmapped slots follow model policy; ambiguity errors |
| Control mode | explicit controlled vocabulary from data/model/robot | error on an explicit conflict |
| Normalization stats | data stats vs model method | pass if stats satisfy; otherwise error |

Camera compatibility is checked per model slot, not by total camera count. When there is a unique real view, a Mapping is established; when there is no real view, an empty mapping is kept and padding is planned; failure happens only when multiple candidates exist and cannot be decided uniquely.

RobotProfile camera/joint names are not compared with DataSchema names, and robot joint count is not treated as an action-width contract. Without an explicit binding, a name mismatch is absence of evidence, not incompatibility.

##### Transform tiers

Transforms are classified by reliability into three tiers:

**T1: deterministic syntactic or mathematical transforms** — planned automatically when conditions are complete:

- field renaming and unambiguous reordering;
- image layout, dtype, and deterministic resize;
- dimension padding / unpadding;
- normalization / denormalization;
- explicit gripper-convention flip;
- rotation-representation conversion under the same coordinate-frame definition;
- padding of model slots without a real-camera mapping.

**T2: transforms depending on the robot model or runtime conditions** — generated only when conditions are complete and requiring user audit by default:

- FK/IK between joint position and EEF pose/delta;
- resampling across frequencies;
- coordinate-frame conversion depending on extrinsics;
- selection or projection of structured actions for single-arm vs bimanual.

When conditions are insufficient, only a structured "unsupported / extra conditions needed" diagnostic is emitted; T2 is never auto-generated.

**T3: unreliable automatic transforms** — direct failure:

- converting between tokenized actions and an unknown continuous action space;
- camera or coordinate-frame conversion without calibration info;
- joint semantics that cannot be uniquely determined;
- cross-robot-topology action conversion without an explicit projection rule;
- mappings that require task-semantic reasoning to decide.

#### 4.2.3 Mapping

Mapping only expresses stable DataSchema-to-model semantic correspondences and performs no tensor operations. Taking cameras as an example:

```text
model visual slot <- DataSchema camera, or an explicit empty mapping
```

At runtime the platform adapter first converts native camera names to those same DataSchema keys, so a second robot-camera mapping is unnecessary. Explicitly unmapped model slots still receive the placeholder and mask implemented by the model adapter.

Mappings must satisfy:

- every model slot has an explicit source relation or empty mapping;
- a non-empty source must be findable in DataSchema;
- an empty mapping must plan camera-slot padding along the corresponding path;
- automatic mapping uses only deterministic rules;
- a controlled override directly produces the final Mapping;
- semantics are never guessed from dictionary order or string sorting.

#### 4.2.4 Transform Pipeline

The framework reuses the existing Transform system rather than adding another transform abstraction:

| Object | Responsibility |
|---|---|
| `TransformStepCall` | serializable single call: registered name + constructor arguments (`type` / `args`) |
| `TransformPipelinePlan` | ordered list of TransformStepCall produced by the resolver (`calls`) |
| `TransformRegistry` | resolves a step type to an implementation and maintains capability metadata |
| `TransformStep` | instantiated, executable single-step transform |
| `TransformPipeline` | ordered TransformSteps actually run by downstream |

The embodiment composition stores only `TransformPipelinePlan` (declarative). Downstream uses `TransformRegistry` to instantiate an executable `TransformPipeline`. The embodiment composition must not write a `TransformPipeline` containing Python objects and runtime context directly into the resolution result.

##### Three semantic pipelines

The assembly exposes three semantic entries backed by only two actual plans:

**data_to_model**: converts a data sample into the model training interface, including data cameras and state fields to model input slots, image dtype/layout/resize/normalization, state/action normalization, action padding, and task/language field mapping.

**robot_to_model**: the platform adapter first converts a native robot payload to the checkpoint DataSchema interface, then executes this semantic entry. Its calls and `resolved` value equal `data_to_model`; it contains no camera renaming or joint reordering.

**model_to_robot**: unpads and denormalizes model output back to the DataSchema action interface. The platform adapter turns that ordered vector into a platform command and sends it. Assembly currently performs no implicit cross-interface joint, unit, or control-space conversion.

##### Forward and inverse cannot rely on list reversal

Each transform implementation must explicitly state whether it is exactly invertible, approximately invertible, or non-invertible; when an inverse exists, the corresponding implementation must also be explicit — downstream must not guess by name. For example:

- the inverse of pad is unpad;
- the inverse of normalize is denormalize;
- resize usually has no exact inverse;
- safety clamp is irreversible;
- temporal resampling can be lossy.

The resolver plans `data_to_model` and `model_to_robot`, then sets `robot_to_model = data_to_model`. `model_to_robot` must still be generated from each step's explicit inverse; it is not a reversed call list.

##### Rule provenance

Resolution rules and TransformPipelinePlans may depend only on explicit facts from the three dimensions. It is forbidden to trigger branches on hardcoded model names, dataset names, robot names, the current deployment platform, or implementation details of some Trainer. When a specific object does have a unique constraint, that constraint should be lifted into a declaration field of its dimension and consumed by a generic rule.

#### 4.2.5 Failure Handling

A resolution failure must become a structured result before entering downstream, rather than an opaque exception thrown deep inside training or deployment code. The error contract keeps only three stable concepts:

- `code`: a stable machine error code for tests, the CLI, and external tools to classify the problem;
- `path`: the resolution target the error corresponds to, not necessarily the user's original recipe field;
- `params`: the JSON-serializable facts needed to render a message.

`params` is not arbitrary debug context. Each `code` must define its allowed parameter set and be constructed via a dedicated entry point. Check rules must not ad-hoc concatenate error strings or drop in full DataSchemas, model objects, tensors, or other uncontrolled content.

Human-readable messages are not part of the stable error contract. The CLI selects a template from a unified error catalog by `code` and renders it with `params`. For example, a camera-mapping ambiguity needs to show only the target slot, the stably-sorted candidates, and a local override hint. This lets wording be changed independently, supports multilingual output, and prevents tests from depending on full error strings.

The resolver may collect mutually-independent problems and raise a single `ResolutionError`; if a declaration itself is invalid, subsequent checks depending on it stop.

#### 4.2.6 Boundary with the Training and Inference Modules

The **training module** may read: DataSchema and NormStats kept in the embodiment composition, ModelMetadata with its backend/training components/fine-tuning abilities, the model IO spec, the data × model Mapping, and the `data_to_model` TransformPipelinePlan. The training module is itself responsible for training mode, objective function, fine-tuning strategy, backend, Trainer, sampler, DataLoader, batch construction, optimizer, scheduler, distributed execution, and checkpoints/training artifacts. It must not re-derive camera mappings from model names, guess action semantics from array shapes, bypass the embodiment composition to query the registry independently, override joint orders already fixed by the resolver, or silently ignore composition errors.

The **inference module** reads ModelIOSpec, the four data-to-model mappings, and the `robot_to_model` / `model_to_robot` plans. It owns platform adapters/connectors, transports, action-chunk execution, and runtime safety. A platform adapter must explicitly convert native observations/actions to and from the checkpoint DataSchema; inference must not guess a mapping from RobotProfile names.

### 4.3 Finetuning Layer

The finetuning layer is implemented by the `training/` module; its entry point is `train()` in `vla_factory/training/train.py`. Training flow:

```text
parse recipe
    -> resolve recipe + assembly
    -> resolve strategy + strict config
    -> prepare output_dir + save training contract
    -> create model from registry
    -> strategy.prepare_model
    -> create one all-episode training dataloader
    -> VLATrainer.train()
    -> strategy.finalize_model / state_dict
    -> save final/model.pt
```

The entry calls `resolve_assembly()` first. Output directories, models, and DataLoaders are created only after composition succeeds. Training reads descriptions, mappings, IO spec, and pipeline plans from that product instead of re-deriving relationships.

The finetuning layer assembles `Observation` samples from the Canonical IR (`Episode` / `Frame`) produced by the data layer, according to the `data_to_model` TransformPipeline obtained from the embodiment composition, then performs window sampling and batching and hands the result to `VLATrainer`.

#### 4.3.1 Fine-tuning Strategy

The fine-tuning strategy decides which parameters are trainable. It should operate on parameters via `ModelMetadata.components` and `named_parameters()`, not by hardcoding model types. Current core strategies include:

- `full`: full-parameter training.
- `freeze`: freeze specified components.
- `selective`: train only specified components.
- `lora`: for models that support LoRA.

ACT trained from scratch usually uses `full`; pretrained VLA models may use full, freeze, selective, or LoRA.

**LoRA default behavior contract.** A bare `finetuning: {strategy: lora, config: {r, lora_alpha}}` recipe — no `components`, no `freeze_components`, no `target_modules` — gets LoRA on every declared component. The boundary this contract draws: **LoRA only ever lands inside declared component subtrees; linear layers outside them (for pi0: the state/action/time projections) are never touched by peft and stay full-parameter trained** — including under the default `"all"`, which simply runs the same per-subtree path for every declared component. The three fields have defaults that make this the simplest sensible behavior:

- `components` defaults to `"all"` (a string): expands to every key of `ModelMetadata.components` at apply time, i.e. LoRA on every declared subtree (for pi0: both the VLM and the action expert), each wrapped in place by its own `get_peft_model` call. A list (`["llm"]`) restricts LoRA to those subtrees only.
- `freeze_components` defaults to `[]`: subtrees outside `components` keep `requires_grad=True` and are fully fine-tuned. Listing a subtree here freezes it instead, closing the one gap subtree-LoRA could not cover ("action_expert frozen + llm LoRA"). A component must not appear in both `components` and `freeze_components` (validated at config parse for lists, and again after `"all"` expansion at apply time; the freeze itself runs before any peft wrapping so component prefixes match clean parameter names).
- `target_modules` defaults to `"all-linear"`: peft's special string matching every `Linear`/`Conv1D` inside the wrapped scope. It is forwarded to peft verbatim, so a regex string or an explicit list (`["q_proj","v_proj"]`) also works. It is a per-run training decision, not a `ModelMetadata` fact.

This default is parameter-equivalent to openpi's low-mem configs (`gemma_2b_lora` + `gemma_300m_lora` — both VLM and action expert get LoRA) and aligns with llamafactory's `lora_target="all"`. The known limitation: a single `peft_config` is shared across every subtree's wrap, so `r`/`lora_alpha`/`target_modules` are uniform across all wrapped subtrees — per-component different LoRA configs are not expressible until a recipe needs it.

Strategies register through `@register_strategy(name)` and strictly parse their
own `finetuning.config`. `prepare_model()` owns freezing/wrapping, while
`finalize_model()` and `state_dict()` own save-time convergence; neither the
training entry nor checkpoint persistence branches on names such as `lora`.
A method that changes loss, sampling, or the loop is not a strategy and belongs
in a future Training Method layer.

#### 4.3.2 VLATrainer

`VLATrainer` is a thin subclass of the HuggingFace `Trainer`. Its job is to bridge the batch produced by the data pipeline:

```python
{
    "observation": Observation,
    "actions": Tensor,
    "action_is_pad": Tensor | None,
}
```

to:

```python
model.compute_loss(observation, actions, action_is_pad=...)
```

The Trainer ecosystem provides mixed precision, gradient accumulation, checkpointing, logging, and optimizer scheduling. VLA Factory only adds VLA batch adaptation, auxiliary loss logging, and the `lr_backbone` parameter group.

#### 4.3.3 Checkpoint and Final Model

Before training starts, `training/checkpoint.py` writes the metadata needed by inference into `inference_metadata/` under the output directory. Intermediate checkpoints are written by the HF Trainer. After training, the framework additionally writes:

```text
<output_dir>/final/model.pt
```

At inference load time, final weights, root weights, safetensors, or the most recent `checkpoint-*` are searched in this priority order.

### 4.4 Inference Layer

The inference module turns training artifacts into a platform-callable real-time policy service: it rebuilds an inference chain consistent with training from the checkpoint (model + preprocessor/postprocessor), translates each simulator's/real robot's native observation into a unified `ObsDict`, assembles it into an `Observation` via the `robot_to_model` TransformPipeline, runs the model forward, and then restores the normalized action chunk into a platform-executable action command via `model_to_robot` according to the execution strategy. It uses the checkpoint's `inference_metadata/{assembly.json,recipe.yaml}` as the single source of truth; schema, norm stats, IO spec, and pipeline plans all come from the assembly snapshot. It does not rescan the training dataset or re-derive the relationships among data, model, and robot.

The inference layer is organized around these responsibilities:

- The responsibility boundaries of the inference core layer, the platform adaptation layer, and the transport/remote-service layer.
- Core objects `InferenceEngine`, `ObsDict`, platform adapters, `PolicyRunner`, `RemotePolicyModel`, `ZmqPolicyClient`, `LengthPrefixedJsonRpcServer`.
- ObsDict → Observation preprocessing and postprocessing inverse transforms, and the three action-chunk execution strategies: synchronous / temporal_ensembling / receding_horizon.
- In-process and remote service forms (ZMQ and length-prefixed JSON RPC) and the zero-dependency connector.
- How to add new platform adapters, transports, and external connectors.

---

## 5. Dependency Management Strategy

Dependency management follows the "lightweight core, on-demand ecosystem" principle, with a toolchain based on **uv + venv**: each model environment is an independent virtual environment managed by uv, mutually isolated.

### 5.1 Why uv + venv

Core dependencies cover only configuration parsing, the data pipeline, PyTorch training basics, the CLI, and general deployment capability (see `dependencies` in `pyproject.toml`); model-ecosystem dependencies are introduced on demand. The framework uses [uv](https://github.com/astral-sh/uv) to manage versions, virtual environments, and package installation, rather than system Python or conda:

- uv's PubGrub resolver can resolve the strict `==` version pins of upstream ecosystems (especially openpi); these pins, combined with openpi's in-place transformers patch, make a plain `pip install -e ".[pi0]"` fail outright.
- uv natively supports routing torch/torchvision through the PyTorch CUDA wheel index (`--torch-backend`), with no need for hand-written `--find-links` or `PIP_EXTRA_INDEX_URL`.
- uv creates and manages venvs extremely fast; each model environment is isolated, preventing lerobot/openpi dependency conflicts from polluting the global environment.

### 5.2 Environment Setup

Model environments are wrapped by `scripts/install.sh`, the recommended entry point:

```bash
bash scripts/install.sh [venv_dir] [model]
# defaults: venv_dir=.venv, model=pi0
```

The script performs, in order:

- Creates a virtual environment with `uv venv --python 3.12` (default `.venv`) and activates it.
- Auto-selects the torch CUDA wheel backend by the GPU's **compute capability** (not the driver CUDA version): Blackwell (sm_100 and above, e.g. RTX 5090 sm_120) → `cu128`; others (Hopper sm_90 and earlier) → `cu126`. Override with `VLA_TORCH_BACKEND=cu126|cu128`. The reason for checking compute capability rather than driver version is that a Blackwell card reports driver CUDA 12.4 but needs cu128's torch 2.8+ to get sm_100/sm_120 kernels.
- Auto-detects a PyPI mirror (Tsinghua for CN networks, otherwise PyPI); override with `VLA_PYPI_INDEX`.
- Downloads openpi (and lerobot when `VLA_LOCAL_LEROBOT=1`) as a tarball into `.local-deps/` and installs from a local path, avoiding GitHub git-transport instability on weak networks; openpi is pinned to a known-good commit.
- After installing openpi, overlays its `transformers_replace` patch onto site-packages (dtype fixes for SigLIP/PaliGemma/Gemma required by PI0Pytorch).
- Installs vla-factory itself in editable mode (`uv pip install -e .`).

Once done:

```bash
source .venv/bin/activate
vlafactory-cli list
vlafactory-cli train --config examples/pi0_lora.yaml
```

### 5.3 Optional Extras

Model-ecosystem dependencies are declared in `[project.optional-dependencies]` of `pyproject.toml`:

| extra | contents | install |
|---|---|---|
| `act` | lerobot (ACT policy) | `uv pip install -e ".[act]"` directly |
| `pi0` / `pi05` | openpi (pinned commit) | **must go through `scripts/install.sh`** |
| `robotwin` | h5py (RoboTwin native hdf5 data) | `uv pip install -e ".[robotwin]"` directly |
| `all` | all of the above | the pi0/pi05 part still needs install.sh |
| `dev` | pytest, pytest-cov, tensorboard | `uv pip install -e ".[dev]"` |

To emphasize: **pi0 / pi05 cannot be installed with plain pip** — openpi's strict pins and transformers patch must be handled by `install.sh` together with uv.

`ModelMetadata.install_hint` gives a clear message when a dependency is missing, and the CLI `list` command shows registered models and their install hints.

### 5.4 Adapter Dependency Boundary

A model adapter module being importable does not mean the upstream model dependency must already be installed. The recommended practice is:

- The entry top level imports only stable internal VLA Factory modules.
- Upstream model libraries are imported lazily inside the factory.
- A missing optional dependency raises a clear ImportError.
- A genuine entry-import error is reported explicitly by the registry loader.

### 5.5 No Upstream Model Code Copying

VLA Factory does not maintain `vendor/` model implementations. Upstream models should come from a pip extra, an installable package in the user's environment, or a local source dependency pulled by `install.sh` (openpi / lerobot). Any handling of upstream version differences in an adapter should be local, removable, and testable.

---

## 6. Testing Strategy

> TODO: This chapter describes the testing strategy to be filled in later and currently serves as a coverage-target and regression checklist.

Tests should cover the key criteria from configuration parsing through composition resolution, training, inference, and deployment adapters.

### 6.1 Configuration Tests

Configuration tests care about:

- YAML parsing into `TrainRecipe`.
- Defaults matching expectations.
- Stable nested-config structure.
- CLI overrides correctly affecting training parameters.
- A clear compatibility policy between top-level and nested fields.

### 6.2 Composition Resolution Tests

Composition-resolution testing is a new focus; see Section 4.2:

- Input-contract tests for each dimension (required fields, unknown fields, enum vocabularies, dimension and key counts, optional checkpoint/ModelMetadata consistency, RobotProfile/URDF consistency).
- Resolution-rule tests cover each row of the compatibility matrix: direct compatibility, auto-generated Mapping, auto-generated TransformPipelinePlan, warning, error, success after controlled override, and result stability under identical input.
- Failure tests assert `ResolutionError`'s `code`, `path`, and `params` rather than matching the full user-facing text.
- Golden-composition tests save embodiment-composition golden files for a few representative combinations (e.g. LeRobot ACT data × ACT × LeKiwi; RoboTwin data × ACT × simulation robot; LeRobot ALOHA data × PI0/PI0.5 × ALOHA).
- Mapping and Transform-Pipeline tests cover unique field matching, camera-slot ambiguity, padding of unmapped camera slots, state/action key reordering, normalize/denormalize pairing, pad/unpad pairing, gripper flip, rotation conversion, risk and reversibility declarations, and the prohibition on name hardcoding.
- Embodiment-composition serialization round-trips stably, and resolution does not load model heavy-dependencies, GPUs, or deployment runtimes.

### 6.3 Data Pipeline Tests

Data tests care about:

- The reader can read schema, norm stats, and episode info.
- Sample counts, ordering, and time ranges of `SampleWindow` values are correct across all episodes.
- The transform pipeline's normalize, resize, and padding behave correctly.
- `VLADataset` output observation/action shapes match model expectations.
- `collate_fn` handles batch aggregation.

### 6.4 Model Registry and Adapter Tests

Model tests care about:

- The registry can discover model entries.
- Duplicate registration fails.
- Missing optional dependencies produce clear error messages.
- The wrapper implements `compute_loss` and `predict_actions`.
- State-dict save/load round-trips.

### 6.5 Training Smoke Test

Training tests need not run a full experiment but should cover minimal-step training:

- small dataset, small batch, a few steps;
- able to write `inference_metadata`;
- able to write final weights;
- loss logging does not retain the autograd graph.

### 6.6 Inference and Deployment Tests

Inference tests care about:

- `InferenceEngine` can load metadata and weights from a checkpoint.
- Dataset sample inference outputs the correct action shape.
- The postprocessor restores the original action scale.
- `synchronous`, `temporal_ensembling`, and `receding_horizon` strategies behave predictably.
- Simulator and lerobot adapter input/output key mappings are correct.

### 6.7 Regression-Test Principle

Any standard issue that has been fixed should be pinned as a test, especially: state/action key ordering; action-dim padding and inverse cropping; checkpoint path resolution; optional-dependency lazy import; train/deploy transform consistency; resolver stability on three-way relationship derivation.

---

## 7. Extension and Evolution

VLA is currently one of the mainstream routes in embodied AI. It may not be the final model form of embodied intelligence, but it is strongly representative at this stage: academia and industry keep producing new methods around VLA data, models, post-training, and deployment. VLA Factory's evolution goal is therefore not only to be a usable fine-tuning tool but also to explore, along the VLA route, how foundational software should be designed.

VLA Factory is both an engineering framework and a research vehicle. Through its unified recipe, data standards, model adapters, training engine, and deployment engine, it keeps studying:

- how data is collected, cleaned, calibrated, converted, and reused;
- how VLA and imitation-learning models can be fine-tuned, continued, and post-trained at low cost;
- how model-specific tricks can be abstracted into framework capabilities shared by other models;
- how embodied models can be deployed stably, in real time, and safely on-device;
- how training and inference infrastructure can be adapted and optimized on domestic hardware/software stacks.

### 7.1 Horizontal Evolution: Expanding Ecosystem Coverage

Horizontal evolution means expanding VLA Factory's ecosystem coverage. Because the framework is positioned as a glue layer, horizontal expansion focuses on integrating more data formats, model ecosystems, training strategies, and deployment platforms, letting one recipe and deployment interface cover more real scenarios.

Horizontal expansion includes:

- Data formats: from LeRobot to HDF5, RLDS, ROS bags, Zarr, and mixed multi-source sampling.
- Model ecosystems: from ACT to OpenPI, OpenVLA, GR00T, SmolVLA, etc.
- Fine-tuning methods: from full/freeze/selective to LoRA, QLoRA, adapter tuning, and model-specific tuning.
- Deployment platforms: from ZMQ simulation and lerobot real robots to more robot middleware, edge devices, and remote inference services.
- Training and evaluation frameworks: in the post-training stage, integrate with RL and evaluation frameworks such as RLinf, feeding behavior-cloning artifacts into RL or offline evaluation rather than reimplementing RL training inside this framework.
- Runtime environments: from the CUDA ecosystem to domestic stacks such as OpenEuler + Ascend.

Horizontal expansion is engineering-heavy and well suited to AI-coding and loop-engineering acceleration. But it must not come at the cost of architectural boundaries: a new data format stops at `FormatReader`; a new model stops at a registry entry and adapter; a new platform stops at a deploy adapter and transport.

### 7.2 Vertical Evolution: Going Deep Around Real Scenarios

Vertical evolution means going deep along a real need or scenario. The hard part of embodied AI is not "can we integrate some model" but whether the complete loop — data, fine-tuning, post-training, and deployment validation — works stably.

Vertical evolution includes:

- Data pipeline: auto-calibration, cleaning, quality checks, statistics generation, and format conversion of recorded data.
- Fine-tuning pipeline: checkpoint continuation, parameter-efficient fine-tuning across models, cross-dataset transfer, and training-stability diagnosis.
- Post-training pipeline: from behavior cloning to RL, preference optimization, failure-sample mining, and world-model exploration.
- Deployment pipeline: on-device real-time inference, action-chunk strategy, frequency control, abnormal-action detection, and safe fallback.
- Evaluation pipeline: offline metrics, simulation validation, real-robot validation, and deployment-log closed loops.

Vertical evolution emphasizes technical accumulation and requires iterating from real scenarios rather than just interface adaptation. The deployment direction in particular adds the action layer on top of LLM/VLM and often runs in an on-device closed loop, so LLM/VLM deployment facilities cannot be directly reused. The real-time performance, stability, legality, and safety boundary of action output are problems that embodied infrastructure must study on its own.

### 7.3 Standard Abstraction: Distilling Model-Specific Tricks into Framework Capabilities

A key value of VLA Factory is abstracting tricks specific to some model into framework-level capabilities that other models can reuse. A typical example is delta action: if it first appeared only in one model, but the framework abstracts the action transform, normalization, denormalization, and deployment restoration into unified transforms, then other VLA models can also try delta-action fine-tuning on the same data and training standards.

Such abstractions should follow these principles:

- A trick is not hardcoded inside a model adapter; it is distilled into data transforms, training strategies, the action spec, or deployment postprocessors.
- The abstracted capability should be reusable across models, but a model may declare via metadata or recipe whether to enable it.
- Training and deployment must share the same semantics; a trick must not take effect only on the training side.
- Every abstraction must have testable input/output standards, to avoid "looks general but actually serves only one model".

This "unified standard abstraction" is what turns the framework from a glue layer into infrastructure. It lets VLA Factory not just plug in models but also distill new methods into composable, reusable, verifiable foundational modules.

### 7.4 Composition-Resolution Capability Boundary

The composition resolver described in Section 4.2 is now the common entry point for training and inference. Its current boundary follows four rules:

- Automatic planning covers only deterministic T1 conversions backed by sufficient facts, such as camera-slot mapping, resize, layout, normalization, padding/unpadding, and explicit inverses.
- Platform adapters own conversion between platform-native interfaces and the checkpoint DataSchema. Assembly does not infer cross-namespace relations from camera or joint names, so `robot_to_model` currently shares the `data_to_model` plan.
- FK/IK, coordinate-frame conversion, frequency resampling, and cross-robot action projection are T2 capabilities. Add them individually only with a real use case, complete prerequisites, and end-to-end tests; do not predeclare fields or abstractions without consumers.
- Missing or ambiguous information fails conservatively. Implicit defaults, approximate inverses, and legacy-config compatibility layers must not create a second source of truth. Old recipes and old training artifacts without `assembly.json` are unsupported.

Evolution should preserve the one-way dependency: explicit facts → Mapping / ModelIOSpec → PipelinePlan → downstream execution. Any new capability must use the same saved plan in training and deployment rather than living only inside one adapter.

### 7.5 Deployment/Inference Evolution

Deployment inference is an important evolution direction once the unified framework extends to real-robot closed loops, but it is not the core demand of the first stage. The first stage should first stabilize training artifacts, data standards, and the model protocol; on that basis, deployment inference can deepen around the same recipe, schema, norm stats, and transforms.

Key points of this direction include:

- Inference consistency: training and inference share data transforms, normalization statistics, camera/state/action key ordering, and the action spec.
- Real-time control: system optimization around end-to-end latency, control frequency, action-chunk strategy, caching, and asynchronous execution.
- Platform adaptation: connect to more simulators, robot middleware, and real-robot platforms via observation adapters, action adapters, and transports.
- Safety and observability: abnormal-observation detection, action-legality checks, frequency monitoring, log tracing, and fallback strategies.
- Deployment evaluation: distill unified evaluation methods for offline replay, simulation validation, real-robot validation, and deployment-log closed loops.

The boundary of this direction: deployment capability should reuse the unified standards formed during training and must not redefine an independent set of data semantics or model I/O protocols on the deployment side.

### 7.6 Domestic-Compute Evolution

Domestic-compute support is also a follow-up direction, not a precondition for the current framework. VLA Factory's core architecture should first keep the backend, adapter, and optional-dependency boundaries clean, leaving room to later validate the training, inference, and deployment pipelines on domestic stacks such as OpenEuler + Ascend.

Key points include:

- Training-backend adaptation: validate operator support, mixed precision, distributed training, checkpoint format, and performance tuning.
- Upstream-model compatibility: identify implicit CUDA dependencies in ecosystems such as ACT, OpenPI, OpenVLA, and GR00T, and reduce migration cost via adapters or dependency isolation.
- Inference-runtime validation: evaluate end-to-end stability of model loading, data preprocessing, action postprocessing, communication protocols, and the hardware runtime.
- Performance baselines: build comparisons of data loading, training throughput, inference latency, and control frequency between CUDA and domestic environments.
- Engineering know-how: form reusable documentation on environment installation, issue localization, operator substitution, precision differences, and deployment constraints.

The boundary: domestic support should be introduced progressively through backend, adapters, and dependency management, and must not tie core data standards and model protocols to any single hardware or system environment.

### 7.7 Extension-Path Constraints

Whether horizontal or vertical, evolution should obey existing module boundaries:

- New data format: implement a new `FormatReader` that outputs unified schema, norm stats, episode, and frame.
- New model: add a registry entry declaring `ModelMetadata` and wrap the upstream model with a thin adapter.
- New robot: add a RobotProfile declaring embodiment capability and safety constraints; the runtime platform remains the inference module's responsibility.
- New training strategy: select parameters via metadata components or parameter-name rules; do not hardcode model internals.
- New deployment platform: add observation/action adapters and necessary transports; do not modify the `InferenceEngine` core prediction logic.
- New three-way relationship rule: depend only on explicit declarations of the three dimensions; do not branch on object names.

Two constraints should hold throughout evolution: the main path depends only on stable standards, and ecosystem differences stay inside adapters. Horizontal expansion widens ecosystem coverage; vertical evolution deepens technical depth — the two are orthogonal and can proceed in parallel.
