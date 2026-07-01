# VLA Factory

> 中文：[README.cn.md](./README.cn.md)

VLA Factory is a **recipe-driven** fine-tuning framework for robot Vision-Language-Action (VLA) models: a single YAML describes the model, data, fine-tuning strategy and training parameters, and the framework runs the full loop — data pipeline → model construction → training → deployment.

---

## Architecture & Key Features

![VLA Factory layered architecture](./docs/architecture/vla-factory-layered-architecture.en.svg)

Architecture overview: VLA Factory's core goal is to unify the data, model, training artifact, and deployment-entry contracts in a VLA fine-tuning workflow. Users describe experiment intent with a recipe; the framework turns external datasets into unified sample semantics, calls upstream model ecosystems through adapters, and produces reusable, verifiable training results.

- **Unified experiment entry point**: describe the model, data, action space, fine-tuning strategy, training parameters, and output location in one recipe, reducing scattered scripts and temporary conventions.
- **Unified data semantics**: convert datasets from different sources into consistent observation/action representations while preserving schema, statistics, and key ordering for training, evaluation, and later reuse.
- **Model ecosystem adaptation**: integrate external model implementations through lightweight adapters. The framework owns the model protocol and training contract, not upstream model architecture code.
- **Reusable training artifacts**: training outputs include the recipe, data schema, normalization statistics, model weights, and related metadata needed for reproduction, evaluation, and later inference validation.

---

## Installation

```bash
# From the repository root
pip install -e ".[act]"      # ACT training (with lerobot>=0.5)
# or
pip install -e ".[all]"      # all model-ecosystem dependencies
pip install -e ".[dev]"      # dev deps: pytest / pytest-cov / tensorboard
```

After installation, a `vlafactory-cli` command is registered (e.g. `vlafactory-cli train --config recipe.yaml`). `vlafactory-cli ...` is equivalent and works without installation / from source.

---

## Quick Start

### 0. List registered models

```bash
vlafactory-cli list
#   act                  backend=pytorch  head=...
```

### 1. Preprocess videos (optional but recommended)

Decode dataset video frames into a `.npy` disk cache to avoid repeated decoding during training:

```bash
vlafactory-cli preprocess --config examples/act_lekiwi_banana.yaml
```

### 2. Train

```bash
vlafactory-cli train --config examples/act_lekiwi_banana.yaml
```

Common overrides (tweak without editing the YAML):

```bash
vlafactory-cli train --config <recipe.yaml> \
    --steps 5 --batch-size 2 --output-dir outputs/smoke_test
```

### 3. Inference / Evaluation

```bash
# One-shot smoke-test inference on a dataset sample
vlafactory-cli infer --checkpoint outputs/act_so101_banana \
    --dataset-index 0 --split val

# Evaluate per-episode L1 loss across a dataset
vlafactory-cli evaluate --checkpoint outputs/act_so101_banana \
    --dataset /path/to/dataset
```

### 4. Start an Inference Service

```bash
# Simulator platform
vlafactory-cli serve --checkpoint outputs/act_so101_banana \
    --platform simulator --strategy receding_horizon

# lerobot real-robot platform
vlafactory-cli serve --checkpoint outputs/act_so101_banana \
    --platform lerobot --remote-ip <robot-ip> --strategy receding_horizon
```

---

## Recipe Configuration

The most complete annotated template is [`examples/reference.yaml`](./examples/reference.yaml), where every field documents its meaning, allowed values, and typical usage. Ready-to-use examples:

| Example | Description |
|---------|-------------|
| `examples/act_lekiwi.yaml` | Train lekiwi from scratch |
| `examples/reference.yaml` | Fully annotated template |

---

**Model default profiles**: each model ships a baseline profile under `vla_factory/config/model/<name>.yaml` (e.g. [`vla_factory/config/model/act.yaml`](./vla_factory/config/model/act.yaml)). The factory loads it as the default and deep-merges the per-run `model.config` from the recipe on top — recipe values win, so the profile is the starting point for an experiment, not a frozen contract. Unknown keys surface as an error from the upstream config object (no silent failures from typos).

---

## Support Roadmap

| Data | Model | Algorithm | Deployment |
|------|-------|-------|------|
| ✅ **LeRobot v2 / v3** | ✅ **ACT** | ✅ **Full-parameter SFT** | ✅ **LeRobot** |
| ⬜ **RLDS** | ⬜ **π₀ / π-FAST / π₀.₅** | ⬜ **LoRA SFT** |  |
| ⬜ **ROS bags** | ⬜ **GR00T** | ⬜ **Selective SFT** | |
| ⬜ **HDF5** | ⬜ **OpenVLA** | | |

---
