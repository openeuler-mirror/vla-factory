"""Public assembly entry point, resolved contract, and persistence.

Start here to follow the composition flow:

1. :func:`resolve_assembly` gathers model, data, checkpoint, and robot facts;
2. ``assembly.resolve`` applies the pure mapping, IO, and pipeline rules;
3. :class:`ResolvedAssembly` is the serializable contract consumed by training
   and deployment, and owns its JSON lifecycle.

Keeping those boundaries in one public module makes the entry and result easy
to discover. The rule implementations remain in ``assembly/resolve/`` so they
can be tested directly without dataset, registry, or checkpoint I/O.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any

from vla_factory.data.data_schema import DataSchema, NormStats, describe_dataset
from vla_factory.model.checkpoint_validation import validate_checkpoint_if_available
from vla_factory.model.model_interface import ModelMetadata
from vla_factory.model.registry import list_entries
from vla_factory.user_interface import TrainRecipe
from vla_factory.robot import RobotProfile, get_robot_profile, list_robot_profiles

from .transform.plan import TransformPipelinePlan

logger = logging.getLogger(__name__)


# ── Resolved contract ──────────────────────────────────────────────


class InvalidAssemblyError(ValueError):
    """A saved assembly is incomplete or internally inconsistent."""


class ModelInterfaceMismatch(ValueError):
    """Installed model metadata no longer matches the saved model interface."""


class MappingSource(str, Enum):
    """How a mapping relationship was selected."""

    INFERRED = "inferred"
    OVERRIDE = "override"


@dataclass(frozen=True)
class FieldMapping:
    """A collection of real field correspondences; no tensor computation."""

    entries: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [
                {
                    key: value.value if isinstance(value, MappingSource) else value
                    for key, value in entry.items()
                }
                for entry in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FieldMapping":
        if not isinstance(value, dict) or "entries" not in value:
            raise InvalidAssemblyError("mapping must be an object containing 'entries'")
        entries = value["entries"]
        if not isinstance(entries, list) or not all(
            isinstance(item, dict) for item in entries
        ):
            raise InvalidAssemblyError("mapping 'entries' must be a list of objects")
        restored = []
        for entry in entries:
            item = dict(entry)
            if "source" not in item:
                raise InvalidAssemblyError("mapping entry is missing 'source'")
            try:
                item["source"] = MappingSource(item["source"])
            except ValueError as exc:
                raise InvalidAssemblyError(
                    f"mapping has invalid source {item['source']!r}"
                ) from exc
            restored.append(item)
        return cls(entries=tuple(restored))


# Semantic aliases keep annotations and imports readable without four identical
# wrapper implementations.
CameraMapping = FieldMapping
StateMapping = FieldMapping
ActionMapping = FieldMapping
LanguageMapping = FieldMapping


@dataclass(frozen=True)
class ModelIOSpec:
    """The model-facing tensor interface resolved before transform planning."""

    action_dim: int = 0
    action_horizon: int = 0
    n_obs_steps: int = 1
    state_dim: int = 0
    cameras: tuple[str, ...] = ()
    camera_shapes: dict[str, tuple[int, int]] = field(default_factory=dict)
    requires_language: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "n_obs_steps": self.n_obs_steps,
            "state_dim": self.state_dim,
            "cameras": list(self.cameras),
            "camera_shapes": {
                key: list(shape) for key, shape in self.camera_shapes.items()
            },
            "requires_language": self.requires_language,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelIOSpec":
        required = {
            "action_dim", "action_horizon", "n_obs_steps", "state_dim",
            "cameras", "camera_shapes", "requires_language",
        }
        if not isinstance(value, dict):
            raise InvalidAssemblyError("model_io_spec must be an object")
        missing = sorted(required - set(value))
        if missing:
            raise InvalidAssemblyError(f"model_io_spec is missing fields {missing}")
        return cls(
            action_dim=int(value["action_dim"]),
            action_horizon=int(value["action_horizon"]),
            n_obs_steps=int(value["n_obs_steps"]),
            state_dim=int(value["state_dim"]),
            cameras=tuple(value["cameras"]),
            camera_shapes={
                key: (int(shape[0]), int(shape[1]))
                for key, shape in value["camera_shapes"].items()
            },
            requires_language=bool(value["requires_language"]),
        )


@dataclass(frozen=True)
class ResolvedAssembly:
    """A complete composition contract ready for training or inference."""

    schema_ref: dict[str, Any]
    norm_stats_ref: dict[str, Any]
    metadata_ref: dict[str, Any]
    model_io_spec: ModelIOSpec
    camera_mapping: FieldMapping
    state_mapping: FieldMapping
    action_mapping: FieldMapping
    language_mapping: FieldMapping
    data_to_model: TransformPipelinePlan
    robot_to_model: TransformPipelinePlan
    model_to_robot: TransformPipelinePlan
    robot_ref: dict[str, Any] | None = None
    overrides_ref: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def schema(self) -> DataSchema:
        return DataSchema.from_dict(self.schema_ref)

    @cached_property
    def norm_stats(self) -> NormStats:
        return NormStats.from_dict(self.norm_stats_ref)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_ref": dict(self.schema_ref),
            "norm_stats_ref": dict(self.norm_stats_ref),
            "metadata_ref": dict(self.metadata_ref),
            "robot_ref": (
                dict(self.robot_ref) if self.robot_ref is not None else None
            ),
            "overrides_ref": dict(self.overrides_ref),
            "model_io_spec": self.model_io_spec.to_dict(),
            "camera_mapping": self.camera_mapping.to_dict(),
            "state_mapping": self.state_mapping.to_dict(),
            "action_mapping": self.action_mapping.to_dict(),
            "language_mapping": self.language_mapping.to_dict(),
            "data_to_model": self.data_to_model.to_dict(),
            "robot_to_model": self.robot_to_model.to_dict(),
            "model_to_robot": self.model_to_robot.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResolvedAssembly":
        required_objects = (
            "schema_ref", "norm_stats_ref", "metadata_ref", "model_io_spec",
            "camera_mapping", "state_mapping", "action_mapping", "language_mapping",
            "data_to_model", "robot_to_model", "model_to_robot",
        )
        if not isinstance(value, dict):
            raise InvalidAssemblyError("assembly must be a JSON object")
        missing = [key for key in required_objects if key not in value]
        if missing:
            raise InvalidAssemblyError(f"assembly is missing fields {missing}")
        for key in ("schema_ref", "norm_stats_ref", "metadata_ref"):
            if not isinstance(value[key], dict) or not value[key]:
                raise InvalidAssemblyError(f"'{key}' must be a non-empty object")

        try:
            assembly = cls(
                schema_ref=dict(value["schema_ref"]),
                norm_stats_ref=dict(value["norm_stats_ref"]),
                metadata_ref=dict(value["metadata_ref"]),
                robot_ref=(
                    dict(value["robot_ref"])
                    if value.get("robot_ref") is not None else None
                ),
                overrides_ref=dict(value.get("overrides_ref") or {}),
                model_io_spec=ModelIOSpec.from_dict(value["model_io_spec"]),
                camera_mapping=FieldMapping.from_dict(value["camera_mapping"]),
                state_mapping=FieldMapping.from_dict(value["state_mapping"]),
                action_mapping=FieldMapping.from_dict(value["action_mapping"]),
                language_mapping=FieldMapping.from_dict(value["language_mapping"]),
                data_to_model=TransformPipelinePlan.from_dict(value["data_to_model"]),
                robot_to_model=TransformPipelinePlan.from_dict(value["robot_to_model"]),
                model_to_robot=TransformPipelinePlan.from_dict(value["model_to_robot"]),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            if isinstance(exc, InvalidAssemblyError):
                raise
            raise InvalidAssemblyError(f"invalid assembly: {exc}") from exc
        if assembly.robot_to_model != assembly.data_to_model:
            raise InvalidAssemblyError(
                "'robot_to_model' must equal 'data_to_model'; platform adapters "
                "emit the checkpoint DataSchema interface"
            )
        return assembly

    def save(self, path: str | Path) -> None:
        """Write the assembly itself as JSON, without a version envelope."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ResolvedAssembly":
        """Load and strictly validate a saved assembly."""
        path = Path(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InvalidAssemblyError(f"{path} is not valid JSON: {exc}") from exc
        try:
            return cls.from_dict(value)
        except InvalidAssemblyError as exc:
            raise InvalidAssemblyError(f"{path}: {exc}") from exc

    def check_model_compatibility(self, metadata: ModelMetadata) -> None:
        """Reject a model declaration that differs from the trained interface."""
        stored = {
            key: self.metadata_ref.get(key, "<absent>")
            for key in metadata.interface_fields()
        }
        current = metadata.interface_dict()
        drifted = {
            key: (stored[key], current[key])
            for key in current
            if stored[key] != current[key]
        }
        if not drifted:
            return
        detail = "; ".join(
            f"{key}: trained with {old!r}, now declared {new!r}"
            for key, (old, new) in sorted(drifted.items())
        )
        raise ModelInterfaceMismatch(
            f"Model {metadata.name!r} no longer declares the trained interface: "
            f"{detail}"
        )


# The pure resolver returns the result types above, while the public recipe
# entry below calls it. Keep this documented cycle at the boundary: importing
# after the result definitions lets both modules use normal module-level imports
# without hiding lightweight dependencies inside functions.
from .resolve.core import resolve_from_facts
from .resolve.errors import MISSING_INPUT, UNKNOWN_MODEL, UNKNOWN_ROBOT, make_error


# ── Recipe orchestration ───────────────────────────────────────────


def _model_metadata_for(recipe: TrainRecipe) -> ModelMetadata:
    """Return the registry declaration without constructing the model."""
    entries = list_entries()
    metadata = entries.get(recipe.model.name)
    if metadata is None:
        raise make_error(
            UNKNOWN_MODEL, "model.name",
            model_name=recipe.model.name, known=sorted(entries),
        )
    return metadata


def _robot_profile(recipe: TrainRecipe) -> RobotProfile | None:
    if not recipe.robot.name:
        return None
    try:
        return get_robot_profile(recipe.robot.name)
    except Exception as exc:
        raise make_error(
            UNKNOWN_ROBOT, "robot.name",
            robot_name=recipe.robot.name, known=list_robot_profiles(),
        ) from exc


def _read_descriptions(recipe: TrainRecipe) -> tuple[DataSchema, NormStats]:
    """Read one dataset's schema and statistics as an inseparable pair."""
    path = recipe.data.path
    if not path:
        raise make_error(
            MISSING_INPUT, "data.path",
            field="data.path",
            detail="a dataset path is required to resolve the composition",
        )
    try:
        schema, norm_stats = describe_dataset(path, recipe.data.format)
    except Exception as exc:
        raise make_error(
            MISSING_INPUT, "data.path",
            field="data.path", detail=f"{path}: {exc}",
        ) from exc
    return schema, norm_stats


def resolve_assembly(recipe: TrainRecipe) -> ResolvedAssembly:
    """Gather a resolved recipe's descriptions and build its assembly.

    This is the single I/O orchestration entry. It queries registries, reads the
    dataset metadata, and optionally checks redundant checkpoint facts before
    calling the pure :func:`assembly.resolve.resolve_from_facts` implementation.
    The recipe must already have passed ``merge_model_config()`` so model tunable
    defaults are present in ``model.config``.
    """
    metadata = _model_metadata_for(recipe)

    if recipe.model.path:
        check = validate_checkpoint_if_available(recipe.model.path, metadata)
        if check["status"] == "unavailable":
            logger.info(
                "Checkpoint compatibility check skipped: %s", check["detail"],
            )

    robot_profile = _robot_profile(recipe)
    schema, norm_stats = _read_descriptions(recipe)

    return resolve_from_facts(
        schema=schema,
        norm_stats=norm_stats,
        metadata=metadata,
        robot_profile=robot_profile,
        overrides=recipe.overrides,
        model_config=recipe.model.config,
        # Checkpoint selection is only a task-tokenizer address fallback.
        model_path=recipe.model.path,
    )


__all__ = [
    "resolve_assembly",
    "ResolvedAssembly",
    "ModelIOSpec",
    "FieldMapping",
    "CameraMapping",
    "StateMapping",
    "ActionMapping",
    "LanguageMapping",
    "MappingSource",
    "InvalidAssemblyError",
    "ModelInterfaceMismatch",
]
