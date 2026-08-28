"""Load the metadata and model weights required for inference."""

from __future__ import annotations

import re
from pathlib import Path

import torch

from vla_factory.assembly import ResolvedAssembly
from vla_factory.user_interface import TrainRecipe, parse_recipe
from vla_factory.utils.constants import (
    ASSEMBLY_FILE,
    FINAL_DIR,
    INFERENCE_META_DIR,
    MODEL_WEIGHTS_FILE,
    RECIPE_FILE,
)


def load_inference_metadata(
    checkpoint_path: str | Path,
) -> tuple[ResolvedAssembly, TrainRecipe]:
    """Load the saved assembly and resolved recipe for a checkpoint.

    Training writes both files under ``inference_metadata/`` before the first
    training step. Inference deliberately does not reconstruct a missing
    assembly from the currently installed model declaration because that may no
    longer describe the interface the checkpoint was trained against.
    """
    search_dir = Path(checkpoint_path)
    for _ in range(3):
        metadata_dir = search_dir / INFERENCE_META_DIR
        if metadata_dir.is_dir():
            break
        search_dir = search_dir.parent
    else:
        raise FileNotFoundError(
            f"No {INFERENCE_META_DIR}/ found under {checkpoint_path}. "
            "Train a model first to generate it."
        )

    assembly_file = metadata_dir / ASSEMBLY_FILE
    if not assembly_file.exists():
        raise FileNotFoundError(
            f"{metadata_dir} has no {ASSEMBLY_FILE}. This checkpoint cannot be "
            "served because its resolved execution contract is missing. "
            "Retrain with the current version."
        )

    recipe_file = metadata_dir / RECIPE_FILE
    if not recipe_file.exists():
        raise FileNotFoundError(
            f"{metadata_dir} has no {RECIPE_FILE}; checkpoint metadata is incomplete."
        )

    return ResolvedAssembly.load(assembly_file), parse_recipe(recipe_file)


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve a checkpoint directory, run directory, or file to its weights."""
    path = Path(path)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    candidates = [
        path / FINAL_DIR / MODEL_WEIGHTS_FILE,
        path / MODEL_WEIGHTS_FILE,
        path / "pytorch_model.bin",
        path / "model.safetensors",
    ]
    for checkpoint_dir in sorted(
        path.glob("checkpoint-*"), key=_checkpoint_sort_key, reverse=True
    ):
        candidates.extend(
            [
                checkpoint_dir / MODEL_WEIGHTS_FILE,
                checkpoint_dir / "pytorch_model.bin",
                checkpoint_dir / "model.safetensors",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No model weights found under {path}. Expected "
        f"{FINAL_DIR}/{MODEL_WEIGHTS_FILE}, {MODEL_WEIGHTS_FILE}, "
        "or Trainer checkpoint weights."
    )


def load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load PyTorch or safetensors weights into a state dictionary."""
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "This checkpoint is model.safetensors, but safetensors is not "
                "installed. Install it with: pip install safetensors"
            ) from exc
        return load_file(str(path), device="cpu")

    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint did not contain a state_dict: {path}")
    return state


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    step = int(match.group(1)) if match else -1
    return step, path.name


__all__ = [
    "load_checkpoint_state_dict",
    "load_inference_metadata",
    "resolve_checkpoint_path",
]
