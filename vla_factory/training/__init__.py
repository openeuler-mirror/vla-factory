"""Public entry point for VLA Factory supervised training."""

from pathlib import Path
from typing import Any

from vla_factory.recipe import TrainRecipe


def train(
    config: str | Path | TrainRecipe,
    *,
    override_steps: int | None = None,
    override_batch_size: int | None = None,
    override_output_dir: str | None = None,
) -> dict[str, Any]:
    """Run training without importing the heavy Trainer stack eagerly."""
    from vla_factory.training.train import train as run_training

    return run_training(
        config,
        override_steps=override_steps,
        override_batch_size=override_batch_size,
        override_output_dir=override_output_dir,
    )

__all__ = ["train"]
