"""YAML → TrainRecipe parser.

Supports both flat and nested YAML styles.  Missing sections fall back to
dataclass defaults so a minimal config (just model name + data path) works.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from vla_factory.config.recipe import (
    ActionSpecConfig,
    AugmentationConfig,
    DataConfig,
    DataSourceConfig,
    LoraConfig,
    OutputConfig,
    SamplerConfig,
    SplitConfig,
    TrainRecipe,
)


def parse_recipe(path: str | Path) -> TrainRecipe:
    """Parse a YAML config file into a TrainRecipe."""
    path = Path(path)
    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return _build_recipe(raw)


def parse_recipe_from_string(content: str) -> TrainRecipe:
    """Parse a YAML string into a TrainRecipe (useful for tests)."""
    raw: dict[str, Any] = yaml.safe_load(content) or {}
    return _build_recipe(raw)


# ── Internal ──────────────────────────────────────────────────────


def _build_recipe(raw: dict[str, Any]) -> TrainRecipe:
    # ── Model ──
    model_block = raw.get("model", {})
    model_name = model_block.get("name", "") if isinstance(model_block, dict) else str(model_block)
    if not model_name:
        raise ValueError(
            "model.name is required in the recipe YAML — the `model` entry must "
            "specify a registered model name (e.g. `model: {name: act}`)."
        )
    model_path = model_block.get("path") if isinstance(model_block, dict) else None
    model_config = model_block.get("config", {}) if isinstance(model_block, dict) else {}

    # ── Transform extensions ──
    transforms_block = raw.get("transforms", {})
    transform_imports = []
    if isinstance(transforms_block, dict):
        transform_imports = list(transforms_block.get("imports", []) or [])

    # ── Action spec ──
    action_spec = _pop_dataclass(raw.get("action_spec", {}), ActionSpecConfig)

    # ── Data ──
    data_block = raw.get("data", {})
    data_config = DataConfig(
        source=_pop_dataclass(data_block.get("source", {}), DataSourceConfig),
        sampler=_pop_dataclass(data_block.get("sampler", {}), SamplerConfig),
        split=_pop_dataclass(data_block.get("split", {}), SplitConfig),
    )

    # ── Fine-tuning ──
    ft_block = raw.get("finetuning", {})
    strategy = ft_block.get("strategy", "full")
    lora_block = ft_block.get("lora", {})
    if isinstance(lora_block, dict):
        # Backward-compat aliases: legacy recipes used rank/alpha; promote to the
        # peft-aligned names r/lora_alpha. Both names still accepted.
        lora_block = dict(lora_block)
        if "rank" in lora_block and "r" not in lora_block:
            lora_block["r"] = lora_block.pop("rank")
        if "alpha" in lora_block and "lora_alpha" not in lora_block:
            lora_block["lora_alpha"] = lora_block.pop("alpha")
        lora_config = _pop_dataclass(lora_block, LoraConfig)
    else:
        lora_config = None
    freeze_components = ft_block.get("freeze_components")
    trainable_components = ft_block.get("trainable_components")

    # ── Training ──
    train_block = raw.get("training", {})
    augmentation = _pop_dataclass(
        train_block.get("augmentation", {}), AugmentationConfig
    )

    # ── Output ──
    output_config = _pop_dataclass(raw.get("output", {}), OutputConfig)

    return TrainRecipe(
        model_name=model_name,
        model_path=model_path,
        model_config=model_config,
        transform_imports=transform_imports,
        action_spec=action_spec,
        data=data_config,
        finetuning_strategy=strategy,
        lora_config=lora_config,
        freeze_components=freeze_components,
        trainable_components=trainable_components,
        backend=train_block.get("backend", "pytorch"),
        lr=float(train_block.get("lr", 1e-4)),
        lr_backbone=_optional_float(train_block.get("lr_backbone")),
        batch_size=int(train_block.get("batch_size", 8)),
        total_steps=int(train_block.get("total_steps", 10000)),
        gradient_checkpointing=bool(train_block.get("gradient_checkpointing", False)),
        inference_steps=int(train_block.get("inference_steps", 1)),
        num_workers=int(train_block.get("num_workers", 4)),
        augmentation=augmentation,
        output=output_config,
    )


def _optional_float(v: Any) -> float | None:
    return float(v) if v is not None else None


def _pop_dataclass(data: dict[str, Any], cls: type) -> Any:
    """Construct a dataclass from a dict, ignoring unknown keys."""
    if not isinstance(data, dict):
        return cls()
    # Only pass keys that the dataclass accepts
    valid_fields = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)
