"""Persistence for training contracts and final inference weights."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from vla_factory.assembly import ResolvedAssembly
from vla_factory.user_interface import TrainRecipe
from vla_factory.utils.constants import (
    ASSEMBLY_FILE,
    INFERENCE_META_DIR,
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
