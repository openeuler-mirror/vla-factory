"""``resolve_from_recipe`` — the single recipe → ``ResolvedAssembly`` entry point.

This is an **orchestration adapter, not a pure function**: it queries the model
registry, reads the dataset's meta files, may open a checkpoint's ``config.json``
for the optional consistency check, and looks up a robot profile.
``resolve_assembly()`` stays pure behind it (architecture §4.2.2).

Everything that turns "a recipe" into "the three descriptions" lives here, once:
training, inference and ``vlafactory-cli resolve`` all come through this
function, so none of them can drift into its own way of assembling the inputs.

Order matters. The checkpoint consistency check runs *before* any downstream
side effect (``train()`` creates its output directory only after this returns),
so an incompatible checkpoint can no longer be reported after the previous run's
output directory has already been wiped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vla_factory.data.formats import get_reader
from vla_factory.data.manifest import DataSchema, NormStats
from vla_factory.model.checkpoint_validation import validate_checkpoint_if_available
from vla_factory.model.interfaces.model import ModelMetadata
from vla_factory.model.registry import list_entries
from vla_factory.recipe.recipe import TrainRecipe
from vla_factory.robot import get_robot_profile, list_robot_profiles
from vla_factory.robot.profile import RobotProfile

from .resolver import ResolvedAssembly, resolve_assembly
from .resolver.errors import MISSING_INPUT, UNKNOWN_MODEL, UNKNOWN_ROBOT, make_error

logger = logging.getLogger(__name__)


# Recipe ``assembly:`` fields → resolver override keys. One home for the mapping;
# the resolver rejects any key it has no consumer for (``CONSUMED_OVERRIDES``).
_OVERRIDE_FIELDS = (
    "camera_mapping",
    "accept_fps_mismatch",
    "gripper_flip",
    "default_task",
)


def assembly_overrides(recipe: TrainRecipe) -> dict[str, Any]:
    """The controlled overrides a recipe actually set (unset fields are absent)."""
    return {
        name: value
        for name in _OVERRIDE_FIELDS
        if (value := getattr(recipe.assembly, name, None)) is not None
    }


def model_metadata_for(recipe: TrainRecipe) -> ModelMetadata:
    """The registry's declaration for the recipe's model.

    Registry lookup only — no factory call, so this stays usable without the
    model's optional extra installed.
    """
    entries = list_entries()
    metadata = entries.get(recipe.model_name)
    if metadata is None:
        raise make_error(
            UNKNOWN_MODEL, "model.name",
            model_name=recipe.model_name, known=sorted(entries),
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
    """Read the dataset's DataSchema + NormStats (meta files only).

    The two are read together on purpose: a schema from one source and
    statistics from another would describe a dataset that never existed.
    """
    path = recipe.data.path
    if not path:
        raise make_error(
            MISSING_INPUT, "data.path",
            field_name="data.path",
            detail="a dataset path is required to resolve the composition",
        )
    try:
        reader = get_reader(recipe.data.format, path=Path(path))
        schema = reader.get_schema(Path(path))
        norm_stats = reader.get_norm_stats(Path(path))
    except Exception as exc:
        raise make_error(
            MISSING_INPUT, "data.path",
            field_name="data.path", detail=f"{path}: {exc}",
        ) from exc
    return schema, norm_stats


def resolve_from_recipe(recipe: TrainRecipe) -> ResolvedAssembly:
    """Resolve a recipe into its ``ResolvedAssembly``.

    The recipe must already be resolved (``resolve_recipe()``): the model's
    declared tunables have to sit under the per-run ``model.config`` before the
    transform declaration and the action horizon are read from it.

    Notice there is no way to inject the descriptions: since a checkpoint's
    assembly is now saved whole (``assembly.json``), nothing needs to re-resolve
    from half-read state. If a future caller ever does, schema and norm_stats
    must be injected **as a pair** — one from a checkpoint and the other from a
    live dataset would silently describe a dataset that never existed.

    Raises
    ------
    ResolutionError
        Unknown model or robot, unreadable dataset, or any failure of the
        combination itself (structured ``code`` / ``path`` / ``params``).
    CheckpointCompatibilityError
        The checkpoint's own config contradicts the model declaration.
    ValueError
        A registry entry declaring something impossible (see ``resolve_assembly``).
    """
    metadata = model_metadata_for(recipe)

    if recipe.model_path:
        check = validate_checkpoint_if_available(recipe.model_path, metadata)
        if check["status"] == "unavailable":
            logger.info(
                "Checkpoint compatibility check skipped: %s", check["detail"],
            )

    robot_profile = _robot_profile(recipe)
    schema, norm_stats = _read_descriptions(recipe)

    return resolve_assembly(
        schema=schema,
        norm_stats=norm_stats,
        metadata=metadata,
        robot_profile=robot_profile,
        overrides=assembly_overrides(recipe) or None,
        # Tunables after resolve_recipe() merged the model's declared defaults
        # under this run's model.config — the transform step list and the
        # from-scratch action horizon are read from here, not from the
        # declaration alone, so a per-run override is honoured.
        model_config=recipe.model_config,
        # Checkpoint selection: only the task_tokenize tokenizer fallback.
        model_path=recipe.model_path,
    )
