"""Optional checkpoint-to-ModelMetadata consistency checks.

``ModelMetadata`` is the model family's only interface contract.  A checkpoint
may carry a ``config.json`` with redundant shape information; this module can
compare that information with the declaration, but it never returns facts that
the resolver or model factory may use to change the declared interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vla_factory.model.interfaces.model import ModelMetadata


class CheckpointCompatibilityError(ValueError):
    """A checkpoint self-report contradicts its model-family declaration."""

    def __init__(self, model_path: str, issues: list[str]) -> None:
        self.model_path = model_path
        self.issues = tuple(issues)
        super().__init__(
            f"Checkpoint {model_path!r} is incompatible with ModelMetadata: "
            + "; ".join(issues)
        )


def load_checkpoint_config(model_path: str) -> dict[str, Any]:
    """Load ``config.json`` from a directory, weight file, JSON file, or HF repo.

    A local weight file uses a sibling ``config.json`` when present.  Supporting
    that layout keeps the check usable for independently downloaded weights
    without imposing a framework-specific checkpoint directory structure.
    """
    path = Path(model_path)
    if path.is_dir():
        config_path = path / "config.json"
    elif path.is_file():
        config_path = path if path.suffix.lower() == ".json" else path.parent / "config.json"
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                f"Cannot inspect {model_path!r}: it is not a local path and "
                "`huggingface_hub` is not installed."
            ) from exc
        config_path = Path(hf_hub_download(repo_id=model_path, filename="config.json"))

    if not config_path.is_file():
        raise FileNotFoundError(f"No config.json found for checkpoint {model_path!r}")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint config {str(config_path)!r} must contain a JSON object")
    return config


def extract_checkpoint_observations(config: dict[str, Any]) -> dict[str, Any]:
    """Extract only interface-shaped values that a known config can report.

    The plain dictionary is diagnostic input, not a second contract type.  An
    absent key means the checkpoint format cannot confirm that metadata fact.
    """
    camera_roles: dict[str, tuple[int, ...]] | None = None
    state_dim: int | None = None
    inputs = config.get("input_features")
    if isinstance(inputs, dict):
        for key, feature in inputs.items():
            if not isinstance(feature, dict):
                continue
            shape = tuple(int(v) for v in (feature.get("shape") or ()))
            feature_type = str(feature.get("type") or "").upper()
            if str(key).startswith("observation.images.") and feature_type == "VISUAL":
                if camera_roles is None:
                    camera_roles = {}
                camera_roles[str(key)[len("observation.images."):]] = shape
            elif feature_type == "STATE" or key == "observation.state":
                state_dim = shape[0] if shape else None

    action_dim: int | None = None
    outputs = config.get("output_features") or {}
    action = outputs.get("action") if isinstance(outputs, dict) else None
    if isinstance(action, dict):
        shape = action.get("shape") or ()
        action_dim = int(shape[0]) if shape else None

    resolution = config.get("image_resolution")
    image_resolution = (
        tuple(int(v) for v in resolution)
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2
        else None
    )
    return {
        "camera_roles": camera_roles,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action_dim": config.get("max_action_dim"),
        "image_resolution": image_resolution,
    }


def checkpoint_compatibility_issues(
    metadata: ModelMetadata, observations: dict[str, Any]
) -> list[str]:
    """Return contradictions; missing checkpoint facts are intentionally ignored."""
    issues: list[str] = []
    action_dim = observations.get("action_dim")
    declared_action_dim = int(metadata.action_dim or metadata.dim_policy_max or 0)
    if action_dim is not None and declared_action_dim and int(action_dim) != declared_action_dim:
        issues.append(
            f"action_dim={action_dim} (metadata declares {declared_action_dim})"
        )

    checkpoint_max = observations.get("max_action_dim")
    if (
        checkpoint_max is not None
        and metadata.dim_policy_max is not None
        and int(checkpoint_max) != int(metadata.dim_policy_max)
    ):
        issues.append(
            f"max_action_dim={checkpoint_max} "
            f"(metadata declares {metadata.dim_policy_max})"
        )

    state_dim = observations.get("state_dim")
    if state_dim is not None and metadata.dim_policy_max is not None:
        limit = int(metadata.dim_policy_max)
        invalid = False
        if metadata.dim_policy == "fixed":
            invalid = int(state_dim) != limit
        elif metadata.dim_policy == "padded_to_max":
            invalid = int(state_dim) > limit
        if invalid:
            issues.append(
                f"state_dim={state_dim} violates metadata "
                f"dim_policy={metadata.dim_policy!r}, dim_policy_max={limit}"
            )

    roles = observations.get("camera_roles")
    if roles is not None and metadata.vision_slots:
        declared = {slot.name: slot for slot in metadata.vision_slots}
        extra = sorted(set(roles) - set(declared))
        missing = sorted(slot.name for slot in metadata.vision_slots if slot.required and slot.name not in roles)
        if extra:
            issues.append(f"camera roles not declared by metadata: {extra}")
        if missing:
            issues.append(f"required metadata camera roles missing from checkpoint: {missing}")
        global_resolution = observations.get("image_resolution")
        for role in sorted(set(roles) & set(declared)):
            shape = tuple(roles[role])
            slot = declared[role]
            if shape and shape[0] != slot.channels:
                issues.append(
                    f"camera {role!r} has {shape[0]} channels "
                    f"(metadata declares {slot.channels})"
                )
            observed_resolution = tuple(shape[-2:]) if len(shape) >= 2 else global_resolution
            if slot.resolution and observed_resolution and observed_resolution != slot.resolution:
                issues.append(
                    f"camera {role!r} resolution={observed_resolution} "
                    f"(metadata declares {slot.resolution})"
                )
    return issues


def validate_checkpoint_if_available(
    model_path: str | None, metadata: ModelMetadata,
) -> dict[str, Any]:
    """Run the optional checkpoint check through one stable entry point.

    Returns a small diagnostic envelope with status ``not_configured``,
    ``unavailable``, or ``compatible``. Unreadable and unsupported checkpoint
    configs are optional and therefore become ``unavailable`` here; a readable
    config that contradicts ``ModelMetadata`` raises
    :class:`CheckpointCompatibilityError`.
    """
    if not model_path:
        return {"status": "not_configured"}
    try:
        config = load_checkpoint_config(model_path)
        observations = extract_checkpoint_observations(config)
        issues = checkpoint_compatibility_issues(metadata, observations)
    except Exception as exc:
        return {"status": "unavailable", "detail": str(exc)}
    if issues:
        raise CheckpointCompatibilityError(model_path, issues)
    return {"status": "compatible", "observed": observations}
