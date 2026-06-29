"""High-level training orchestration: recipe → trained model.

Wires together:
  1. YAML config parsing (TrainRecipe)
  2. Model creation via registry
  3. Fine-tuning strategy application (freeze/selective/full)
  4. DataLoader creation via data pipeline
  5. TrainingArguments mapping
  6. VLATrainer invocation

Usage::

    from vla_factory.training.train import train
    train("configs/act_aloha.yaml")
    # or
    train(recipe_object)
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Union

import torch
import yaml
from transformers import TrainingArguments

from vla_factory.config.parser import parse_recipe
from vla_factory.config.recipe import TrainRecipe
from vla_factory.data.loader import create_dataloaders
from vla_factory.data.formats import get_reader
from vla_factory.data.manifest import resolve_vector_keys
from vla_factory.training.pytorch_trainer import VLATrainer
from vla_factory.training.strategies import apply_strategy
from vla_factory.utils.constants import (
    INFERENCE_META_DIR, RECIPE_FILE, SCHEMA_FILE, NORM_STATS_FILE,
    FINAL_DIR, MODEL_WEIGHTS_FILE,
)
from vla_factory.model.registry import get_entry

logger = logging.getLogger(__name__)


def train(
    config: Union[str, Path, TrainRecipe],
    *,
    override_steps: int | None = None,
    override_batch_size: int | None = None,
    override_output_dir: str | None = None,
) -> dict:
    """Run a training session from a YAML config or TrainRecipe.

    Parameters
    ----------
    config : str | Path | TrainRecipe
        Path to a YAML recipe file, or an already-parsed TrainRecipe.
    override_steps : int, optional
        Override ``total_steps`` from the recipe (useful for quick tests).
    override_batch_size : int, optional
        Override ``batch_size`` from the recipe.
    override_output_dir : str, optional
        Override ``output_dir`` from the recipe.

    Returns
    -------
    dict
        Training metrics from the final logging callback.
    """
    # 1. Parse recipe
    if isinstance(config, (str, Path)):
        recipe = parse_recipe(config)
    else:
        recipe = config

    if override_steps is not None:
        recipe.total_steps = override_steps
    if override_batch_size is not None:
        recipe.batch_size = override_batch_size
    if override_output_dir is not None:
        recipe.output.output_dir = override_output_dir

    logger.info("Recipe: model=%s, strategy=%s, lr=%s, steps=%d",
                recipe.model_name, recipe.finetuning_strategy,
                recipe.lr, recipe.total_steps)

    # 1.5 Handle output_dir
    output_path = Path(recipe.output.output_dir)
    if recipe.output.overwrite_output_dir and output_path.exists():
        shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. Read dataset schema + norm_stats
    data_cfg = recipe.data
    data_path = Path(data_cfg.source.path)
    reader = get_reader(data_cfg.source.format, path=data_path)
    schema = reader.get_schema(data_path)
    norm_stats = reader.get_norm_stats(data_path)

    # ImageNet image normalisation lives in the Normalize transform
    # (use_imagenet_stats=True default); norm_stats is no longer mutated here.

    logger.info("Dataset schema: cameras=%s, action_dim=%d, state_dim=%d",
                list(schema.cameras), schema.action_dim, schema.state_dim)

    # 2.0 Resolve canonical state/action key order (data/model contract).
    # Writes the resolved keys back onto ``schema`` so they land in
    # inference_metadata/schema.json via ``_save_inference_metadata`` (asdict).
    # This is what lets a checkpoint be served on the real robot without
    # re-sorting motor keys. Lenient: warns (does not raise) if the dataset has
    # no ``names`` — only some platforms (lerobot host) actually need the keys.
    resolved_state_keys, resolved_action_keys = resolve_vector_keys(schema, recipe)
    schema = replace(
        schema,
        state_keys=resolved_state_keys,
        action_keys=resolved_action_keys,
    )
    logger.info(
        "Resolved vector keys — state=%s action=%s",
        list(schema.state_keys), list(schema.action_keys),
    )

    # 2.1 Video frame cache: PyAVCodec.decode_frame() reads from the .npy
    # disk cache (``<video>.frame_cache/``) before falling back to live MP4
    # decoding. Pre-populate it with ``python -m vla_factory preprocess
    # --config ...``; otherwise frames are decoded on first access and cached
    # for subsequent epochs. No action needed here at training time.

    # 2.5 Save inference metadata BEFORE training starts.
    # Any intermediate checkpoint can then be used for inference.
    _save_inference_metadata(output_path, recipe, schema, norm_stats, config)

    # 3. Create model — adapter extracts what it needs from recipe + schema
    entry = get_entry(recipe.model_name)
    metadata = entry.metadata
    model = entry.factory(recipe=recipe, schema=schema)

    # 4. Apply fine-tuning strategy (freeze/selective/full)
    apply_strategy(model, recipe, metadata)

    # 5. Create DataLoaders
    # Use image_size from YAML config (recipe.model_config), not the hardcoded
    # ModelMetadata default — the YAML value is what the user actually intends.
    yaml_image_size = recipe.model_config.get("image_size")
    model_metadata_dict = {
        "action_dim": metadata.action_dim or recipe.action_spec.action_dim,
        "image_size": tuple(yaml_image_size) if yaml_image_size else metadata.image_size,
        # Forward the model entry's declared preprocessor pipeline so the data
        # loader builds it via the registry instead of hardcoding the chain.
        "default_transforms": metadata.default_transforms,
    }
    train_loader, val_loader = create_dataloaders(recipe, model_metadata=model_metadata_dict)

    # 6. Build TrainingArguments
    training_args = _build_training_args(recipe)

    # 7. Create trainer and run
    trainer = VLATrainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset if len(val_loader.dataset) > 0 else None,
        data_collator=train_loader.collate_fn,
    )

    logger.info("Starting training: %d steps, batch_size=%d", recipe.total_steps, recipe.batch_size)
    trainer.train()

    # 8. Save final model
    final_dir = Path(recipe.output.output_dir) / FINAL_DIR
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_dir / MODEL_WEIGHTS_FILE)
    logger.info("Model saved to %s", final_dir)

    # Return final metrics
    return trainer.state.log_history[-1] if trainer.state.log_history else {}


def _build_training_args(recipe: TrainRecipe):
    """Map TrainRecipe → transformers.TrainingArguments."""
    # Resolve report_to: "none" → [], "tensorboard" → ["tensorboard"], etc.
    report_to = _resolve_report_to(recipe.output.report_to)

    args = TrainingArguments(
        output_dir=recipe.output.output_dir,
        # Step-based training (not epoch-based)
        num_train_epochs=1,
        max_steps=recipe.total_steps,
        per_device_train_batch_size=recipe.batch_size,
        # Optimizer
        learning_rate=recipe.lr,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=1e-4,
        # Gradient management
        gradient_accumulation_steps=1,
        max_grad_norm=10.0,
        gradient_checkpointing=recipe.gradient_checkpointing,
        # Logging & saving
        logging_steps=recipe.output.logging_steps,
        save_steps=recipe.output.save_steps,
        save_total_limit=recipe.output.save_total_limit,
        # Eval
        eval_strategy="no",  # MVP: no eval during training
        # DataLoader
        dataloader_drop_last=True,
        dataloader_num_workers=recipe.num_workers,
        # Critical: keep custom columns (Observation, etc.)
        remove_unused_columns=False,
        # Reporting
        report_to=report_to,
        logging_nan_inf_filter=False,
    )

    # Pass lr_backbone to VLATrainer via custom attribute
    args.lr_backbone = recipe.lr_backbone

    return args


def _resolve_report_to(value: str) -> list[str]:
    """Resolve YAML report_to field to a list for TrainingArguments.

    Accepts: "none", "tensorboard", "wandb", or a string like "tensorboard,wandb".
    Validates that the requested backend is actually installed.
    """
    if value == "none" or not value:
        return []

    backends = [b.strip() for b in value.split(",") if b.strip()]
    available = []
    for name in backends:
        try:
            __import__(name)
            available.append(name)
        except ImportError:
            logger.warning(
                "report_to=%r requested but %s is not installed, skipping.",
                name, name,
            )
    return available


def _save_inference_metadata(
    output_path: Path,
    recipe: TrainRecipe,
    schema: object,
    norm_stats: object,
    original_config: str | Path | TrainRecipe,
) -> None:
    """Save recipe, schema, and norm_stats to output_dir for inference.

    These files are immutable during training — written once before training
    starts so that any intermediate checkpoint can be used for inference.

    Saves:
      - ``recipe.yaml``: the original YAML config (if config was a file path)
      - ``schema.json``: dataset feature schema (cameras, dims, etc.)
      - ``norm_stats.json``: normalisation mean/std for state and action
    """
    meta_dir = output_path / INFERENCE_META_DIR
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy original YAML if available
    if isinstance(original_config, (str, Path)):
        src = Path(original_config)
        if src.is_file():
            shutil.copy2(src, meta_dir / RECIPE_FILE)
    else:
        # TrainRecipe object — serialise to YAML
        recipe_dict = asdict(recipe)
        with open(meta_dir / RECIPE_FILE, "w") as f:
            yaml.dump(recipe_dict, f, default_flow_style=False)

    # 2. Schema
    with open(meta_dir / SCHEMA_FILE, "w") as f:
        json.dump(asdict(schema), f, indent=2)

    # 3. Norm stats
    with open(meta_dir / NORM_STATS_FILE, "w") as f:
        json.dump(asdict(norm_stats), f, indent=2)

    logger.info("Inference metadata saved to %s", meta_dir)
