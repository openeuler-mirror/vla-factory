"""Persistence for training contracts and final inference weights."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from vla_factory.assembly import ResolvedAssembly
from vla_factory.user_interface import TrainRecipe
from vla_factory.training.strategies.base import FinetuningStrategy
from vla_factory.utils.constants import (
    ASSEMBLY_FILE,
    FINAL_DIR,
    INFERENCE_META_DIR,
    MODEL_WEIGHTS_FILE,
    RECIPE_FILE,
)


logger = logging.getLogger(__name__)


def save_training_contract(
    output_path: Path,
    recipe: TrainRecipe,
    assembly: ResolvedAssembly,
) -> None:
    """Persist the immutable contract needed to reproduce inference."""
    meta_dir = output_path / INFERENCE_META_DIR
    meta_dir.mkdir(parents=True, exist_ok=True)

    assembly.save(meta_dir / ASSEMBLY_FILE)
    (meta_dir / RECIPE_FILE).write_text(
        yaml.safe_dump(recipe.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    logger.info("Inference metadata saved to %s", meta_dir)


def save_final_model(
    output_path: Path,
    model: nn.Module,
    strategy: FinetuningStrategy,
) -> nn.Module:
    """Finalize strategy-owned wrappers and save one inference state dict.

    Finalization (e.g. LoRA merge) may fail on the user's hardware; a merge
    failure must not lose the whole run, so the un-finalized model is saved
    as a fallback and the failure is surfaced as a warning.
    """
    try:
        finalized = strategy.finalize_model(model)
    except Exception as exc:
        logger.warning(
            "%s finalize_model failed (%s: %s); saving unmerged state_dict.",
            type(strategy).__name__, type(exc).__name__, exc,
        )
        finalized = model
    final_dir = output_path / FINAL_DIR
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(strategy.state_dict(finalized), final_dir / MODEL_WEIGHTS_FILE)
    logger.info("Model saved to %s", final_dir)
    return finalized
