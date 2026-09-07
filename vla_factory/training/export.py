"""Export one training checkpoint as a self-contained inference model."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import torch

from vla_factory.inference.checkpoint import (
    checkpoint_format,
    load_checkpoint_state_dict,
    load_inference_metadata,
    resolve_checkpoint_path,
    validate_delta_load,
)
from vla_factory.model.registry import get_entry
from vla_factory.training.strategies import get_strategy
from vla_factory.utils.constants import (
    INFERENCE_META_DIR,
    MODEL_WEIGHTS_FILE,
    WEIGHTS_META_FILE,
)


def export_checkpoint(checkpoint: str | Path, output_dir: str | Path) -> Path:
    """Merge one checkpoint and write a portable inference directory."""
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Export directory is not empty: {output_dir}")

    assembly, recipe = load_inference_metadata(checkpoint)
    weights = resolve_checkpoint_path(checkpoint)
    weight_format = checkpoint_format(weights)
    is_delta = weight_format == "lora_delta"
    if is_delta and not recipe.model.path:
        raise ValueError("Delta checkpoint requires model.path in its saved recipe")
    if not is_delta:
        recipe = replace(recipe, model=replace(recipe.model, path=None))

    entry = get_entry(recipe.model.name)
    model = entry.factory(recipe=recipe, assembly=assembly)
    strategy = get_strategy(recipe.finetuning.strategy)
    if weight_format in {"lora_delta", "lora_wrapped_full"} or (
        weight_format is None and recipe.finetuning.strategy == "lora"
    ):
        model = strategy.prepare_model(
            model, strategy.parse_config(recipe.finetuning.config), entry.metadata
        )
    result = model.load_state_dict(
        load_checkpoint_state_dict(weights), strict=not is_delta
    )
    if is_delta:
        validate_delta_load(model, result)
    model = strategy.finalize_model(model)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(strategy.state_dict(model), output_dir / MODEL_WEIGHTS_FILE)
    (output_dir / WEIGHTS_META_FILE).write_text('{"format": "bare_full"}\n')
    for parent in (checkpoint, *checkpoint.parents):
        metadata = parent / INFERENCE_META_DIR
        if metadata.is_dir():
            shutil.copytree(metadata, output_dir / INFERENCE_META_DIR)
            break
    else:
        raise FileNotFoundError("No inference_metadata directory found")
    return output_dir
