"""Public training orchestration: recipe to saved fine-tuned model.

Start here to follow the training lifecycle. Data materialization, Trainer
adaptation, strategy internals, and persistence live in their respective
modules; this entry only orders those operations.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import replace
from pathlib import Path

from vla_factory.assembly import resolve_assembly
from vla_factory.model.model_interface import ModelMetadata
from vla_factory.model.registry import get_entry
from vla_factory.recipe import TrainRecipe, merge_model_config, parse_recipe
from vla_factory.training.checkpoint import (
    save_final_model,
    save_training_contract,
)
from vla_factory.training.dataloader import create_dataloader
from vla_factory.training.strategies import get_strategy
from vla_factory.training.trainer import VLATrainer, build_training_args


logger = logging.getLogger(__name__)


def train(
    config: str | Path | TrainRecipe,
    *,
    override_steps: int | None = None,
    override_batch_size: int | None = None,
    override_output_dir: str | None = None,
) -> dict:
    """Resolve, train, and persist one recipe-driven VLA model."""
    recipe = _prepare_recipe(
        config,
        override_steps=override_steps,
        override_batch_size=override_batch_size,
        override_output_dir=override_output_dir,
    )
    logger.info(
        "Recipe: model=%s, strategy=%s, lr=%s, steps=%d",
        recipe.model.name,
        recipe.finetuning.strategy,
        recipe.training.lr,
        recipe.training.total_steps,
    )

    # Resolve every data/model/robot relationship before creating or deleting
    # training output. Downstream code executes this saved composition.
    assembly = resolve_assembly(recipe)
    entry = get_entry(recipe.model.name)
    metadata = entry.metadata

    strategy = get_strategy(recipe.finetuning.strategy)
    strategy_config = strategy.parse_config(recipe.finetuning.config)
    _validate_training_request(recipe, metadata)

    output_path = _prepare_output_directory(recipe)
    save_training_contract(output_path, recipe, assembly)

    model = entry.factory(recipe=recipe, assembly=assembly)
    model = strategy.prepare_model(model, strategy_config, metadata)
    train_loader = create_dataloader(recipe, assembly)

    trainer = VLATrainer(
        model=model,
        args=build_training_args(recipe),
        train_dataset=train_loader.dataset,
        data_collator=train_loader.collate_fn,
    )
    logger.info(
        "Starting training: %d steps, batch_size=%d",
        recipe.training.total_steps,
        recipe.training.batch_size,
    )
    trainer.train()
    save_final_model(output_path, model, strategy)
    return trainer.state.log_history[-1] if trainer.state.log_history else {}


def _prepare_recipe(
    config: str | Path | TrainRecipe,
    *,
    override_steps: int | None,
    override_batch_size: int | None,
    override_output_dir: str | None,
) -> TrainRecipe:
    recipe = parse_recipe(config) if isinstance(config, (str, Path)) else config
    training = recipe.training
    if override_steps is not None:
        training = replace(training, total_steps=override_steps)
    if override_batch_size is not None:
        training = replace(training, batch_size=override_batch_size)
    if training is not recipe.training:
        recipe = replace(recipe, training=training)
    if override_output_dir is not None:
        recipe = replace(
            recipe,
            output=replace(recipe.output, output_dir=override_output_dir),
        )
    return merge_model_config(recipe)


def _validate_training_request(
    recipe: TrainRecipe,
    metadata: ModelMetadata,
) -> None:
    if metadata.training_paradigm == "pretrained_finetune" and not recipe.model.path:
        raise ValueError(
            f"Model {recipe.model.name!r} is finetune-only "
            "(training_paradigm=pretrained_finetune): model.path must point to "
            "a base checkpoint. From-scratch training is not supported for "
            "this model."
        )


def _prepare_output_directory(recipe: TrainRecipe) -> Path:
    output_path = Path(recipe.output.output_dir)
    if recipe.output.overwrite_output_dir and output_path.exists():
        shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
