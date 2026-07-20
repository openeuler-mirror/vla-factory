---

name: adapt-new-vla-model
description: >
This skill should be used when integrating a new VLA model into VLA Factory,
including model-specific data extraction, model configuration, training
alignment, and inference adaptation across the framework's four layers.
-----------------------------------------------------------------------

# Adapt a New VLA Model

Use this skill to adapt a new VLA model into VLA Factory through the framework's four-layer architecture:

1. Data Layer
2. Model Abstraction Layer
3. Training Layer
4. Inference Layer

Before coding, inspect the official upstream repository, the document in `docs/modules` and the closest existing VLA Factory model adapter.

---

## 1. Pre-Adaptation Checklist

Identify the following from the upstream repository:

* dataset format and loader;
* preprocessing and postprocessing pipeline;
* model initialization and configuration;
* required observation fields in model's forward;
* training entry point;
* optimizer, loss, and weight update behavior;
* inference entry point;
* action prediction and postprocessing strategy;
* required dependencies and runtime assumptions.

Then summarize:

```text
Model:
Official repository:
Closest existing adapter:
Supported data format:
Training support:
Inference support:
Known unsupported features:
```

---

## 2. Data Layer

Check whether the model's platform data format is already supported by VLA
Factory.

* If supported, reuse the existing dataset adapter.
* If unsupported, use the dataset-adaptation skill instead of implementing data
  loading inside the model adapter.
* Convert data through the framework's intermediate representation before
  constructing `Observation`.
* The model adapter should extract required inputs from `Observation` first.
* If `Observation` is insufficient, check the Data Layer design document before
  extending the intermediate representation or `Observation`.

Output for this layer:

```text
Data format:
Dataset adapter:
Observation fields used:
Missing fields:
Required data-layer extension:
```

---

## 3. Model Abstraction Layer

Create or update:

```text
{model_name}.yaml
```

Put model setup and configurable upstream parameters into this file whenever
reasonable.

Include or map:

* model variant;
* checkpoint path or identifier;
* input image/state configuration;
* action configuration;
* preprocessing configuration;
* normalization configuration;
* tokenizer or language-processing configuration;
* backend-specific runtime options.

If the model needs a preprocessing or configuration concept not supported by the
current framework, check the Model Abstraction Layer design document before
adding an extension.

Output for this layer:

```text
Config file:
Upstream config mapping:
Preprocessing pipeline:
Unsupported upstream options:
Required model-layer extension:
```

---

## 4. Training Layer

Align the model's training process with VLA Factory while preserving upstream
training semantics.

Check and map:

* training entry point;
* dataset input;
* batch construction;
* optimizer;
* scheduler;
* loss function;
* weight update method;
* precision;
* distributed training behavior;
* checkpoint saving and resume behavior;
* logging and metrics.

Prefer invoking or wrapping the upstream training entry point instead of
rewriting the trainer.

Output for this layer:

```text
Training entry point:
Training integration strategy:
Optimizer/loss/update mapping:
Checkpoint mapping:
Metrics mapping:
Training smoke test:
```

---

## 5. Inference Layer

Understand and adapt the model's action inference strategy.

Check and map:

* inference entry point;
* checkpoint loading;
* `Observation` to model input conversion;
* action prediction function;
* action output format;
* action horizon or chunking;
* normalization or denormalization;
* action postprocessing;
* unsupported inference modes.

If inference is not currently supported by the upstream repository, add only a
thin inference adapter when the required behavior can be determined safely.

Output for this layer:

```text
Inference entry point:
Observation-to-input mapping:
Action prediction method:
Action output semantics:
Action postprocessing:
Inference smoke test:
```

---

## 6. Stop Conditions

Stop and report the blocker instead of guessing when:

* upstream behavior is unclear;
* the dataset format is unsupported and requires a separate adapter;
* required information is missing from `Observation`;
* action semantics cannot be mapped safely;
* training update behavior cannot be aligned;
* inference behavior is undocumented;
* a required change conflicts with existing layer design documents;
* the task requires unrelated repository-wide migration.

Report:

```text
Blocker:
Affected layer:
Reason:
Smallest viable next step:
```

---

## 7. Completion Report

Before declaring the model adaptation complete, report:

```text
Model:
Official repository:
Layers changed:
Data format support:
Observation fields used:
Config file:
Training support:
Inference support:
Smoke tests:
Unsupported features:
Known risks:
Files changed:
Validation commands:
Test results:
```

Do not claim full model support if only part of the four-layer adaptation is
implemented.
