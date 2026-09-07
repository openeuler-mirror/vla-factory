"""Load the metadata and model weights required for inference."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn as nn

from vla_factory.assembly import ResolvedAssembly
from vla_factory.user_interface import TrainRecipe, parse_recipe
from vla_factory.utils.constants import (
    ASSEMBLY_FILE,
    INFERENCE_META_DIR,
    MODEL_WEIGHTS_FILE,
    RECIPE_FILE,
    WEIGHTS_META_FILE,
)

_WEIGHT_FORMATS = {"bare_full", "lora_wrapped_full", "lora_delta"}


def checkpoint_format(weights: str | Path) -> str | None:
    """Return the declared weight shape; ``None`` is legacy inference."""
    directory = Path(weights).parent
    marker = directory / WEIGHTS_META_FILE
    if marker.exists():
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))["format"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid {marker}: {exc}") from exc
        if value not in _WEIGHT_FORMATS:
            raise ValueError(f"Invalid weight format in {marker}: {value!r}")
        return value
    return None


def validate_delta_load(model: nn.Module, result) -> None:
    """A delta may omit frozen parameters, never trainable state or buffers."""
    if result.unexpected_keys:
        raise ValueError(f"Delta checkpoint has unknown parameters: {result.unexpected_keys}")
    parameters = dict(model.named_parameters())
    invalid = [
        key for key in result.missing_keys
        if key not in parameters or parameters[key].requires_grad
    ]
    if invalid:
        raise ValueError(f"Delta checkpoint is missing trainable state: {invalid}")


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
        path / MODEL_WEIGHTS_FILE,
        path / "pytorch_model.bin",
        path / "model.safetensors",
    ]
    for checkpoint_dir in sorted(
        path.glob("checkpoint-*"), key=_checkpoint_sort_key, reverse=True
    ):
        if _is_complete_checkpoint(checkpoint_dir):
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
        f"{MODEL_WEIGHTS_FILE}, "
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


def _is_complete_checkpoint(path: Path) -> bool:
    return (path / "trainer_state.json").is_file() and any(
        (path / name).is_file()
        for name in (MODEL_WEIGHTS_FILE, "pytorch_model.bin", "model.safetensors")
    )


__all__ = [
    "load_checkpoint_state_dict",
    "checkpoint_format",
    "load_inference_metadata",
    "resolve_checkpoint_path",
    "validate_delta_load",
]
