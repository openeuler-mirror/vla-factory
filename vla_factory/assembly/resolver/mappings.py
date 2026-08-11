"""Resolve Mapping: the five field Mappings (architecture §4.2.3).

A Mapping states a stable semantic correspondence and performs no tensor math.
Every candidate question is delegated to :mod:`.matching`, the same module Check
Pairs asks, so a mapping is never derivable where the check failed or vice versa.
"""

from __future__ import annotations

from typing import Any

from vla_factory.data.manifest import ActionDim, DataSchema, StateDim
from vla_factory.model.interfaces.model import ModelMetadata
from vla_factory.robot.profile import RobotProfile

from .errors import CAMERA_MAPPING_INVALID, make_error
from .matching import camera_candidates, data_camera_candidates, embed_joints
from .types import (
    ActionMapping,
    CameraMapping,
    JointMapping,
    LanguageMapping,
    StateMapping,
)


def validate_camera_override(
    override_map: dict[str, Any], schema: DataSchema, metadata: ModelMetadata,
) -> None:
    """Both halves of every override entry must exist.

    Without this a typo'd slot or camera silently degrades to slot padding —
    the model trains on a placeholder image and nothing says so (§1.7).
    """
    known_slots = sorted(s.name for s in metadata.vision_slots)
    known_cameras = sorted(schema.cameras)
    for slot_name, camera in sorted(override_map.items()):
        path = f"assembly.camera_mapping.{slot_name}"
        # A model with no declared slots (ACT) has no slot vocabulary to check
        # against — only the camera half is verifiable there.
        if known_slots and slot_name not in known_slots:
            raise make_error(
                CAMERA_MAPPING_INVALID, path,
                field="slot", requested=str(slot_name), known=known_slots,
            )
        if camera not in known_cameras:
            raise make_error(
                CAMERA_MAPPING_INVALID, path,
                field="camera", requested=str(camera), known=known_cameras,
            )


def resolve_camera_mapping(
    schema: DataSchema, metadata: ModelMetadata, overrides: dict[str, Any],
) -> CameraMapping:
    """Camera → model visual slot.

    Only the training source is derived here; the realtime (robot) source
    arrives with ``robot_to_model``.
    """
    override_map = dict(overrides.get("camera_mapping") or {})
    if override_map:
        validate_camera_override(override_map, schema, metadata)

    if not metadata.vision_slots:
        # The model declares no fixed slots (ACT): its visual inputs *are* the
        # dataset cameras in schema order — ``entries/act.py`` builds
        # input_features from ``schema.cameras``. Identity, nothing to infer.
        return CameraMapping(
            entries=tuple(
                {"model_slot": cam, "data_source": cam, "source": "inferred"}
                for cam in schema.cameras
            ),
            resolved=True,
        )

    # A supplied override is the *complete* slot→camera statement, not a patch
    # on top of inference: a slot the user left out is deliberately unmapped.
    # This is also what the only consumer does today — ``entries/pi0.py`` reads
    # ``camera_mapping.get(role)`` and hands unlisted roles a placeholder image
    # + zero mask (``examples/pi0_lora.yaml`` documents exactly that intent for
    # ``right_wrist_0_rgb``). Inferring a source for the leftovers would make
    # the assembly claim a camera the model never receives.
    candidates = [] if override_map else data_camera_candidates(schema)

    entries: list[dict[str, Any]] = []
    for slot in metadata.vision_slots:
        if slot.name in override_map:
            entries.append({
                "model_slot": slot.name,
                "data_source": override_map[slot.name],
                "source": "override",
            })
            continue
        hits = camera_candidates(slot, candidates)
        if len(hits) == 1:
            entries.append({
                "model_slot": slot.name, "data_source": hits[0], "source": "inferred",
            })
            continue
        # Zero candidates — either nothing matched, or an override took over and
        # left this slot out. (Two-or-more, and a required slot under an "error"
        # policy, already raised in Check Pairs.) The slot survives, fed by
        # padding.
        entries.append({
            "model_slot": slot.name, "data_source": None, "source": "padding",
        })
    return CameraMapping(entries=tuple(entries), resolved=True)


def _vector_entries(
    dims: tuple[StateDim, ...] | tuple[ActionDim, ...],
    data_width: int, model_width: int, with_mode: bool,
) -> tuple[dict[str, Any], ...]:
    """One entry per model vector slot: which dataset dim feeds it, or padding.

    The dataset's dims occupy the leading slots in their own order (that is what
    the sample builder concatenates today); anything beyond the dataset's width
    is a padded slot with no source.
    """
    mapped = min(data_width, len(dims))
    entries: list[dict[str, Any]] = []
    for index in range(model_width):
        dim = dims[index] if index < mapped else None
        entry: dict[str, Any] = {
            "model_index": index,
            "data_dim_index": index if dim is not None else None,
            "data_name": dim.name if dim is not None else None,
            "padded": dim is None,
        }
        if with_mode:
            entry["mode"] = getattr(dim, "mode", None) if dim is not None else None
        entries.append(entry)
    return tuple(entries)


def resolve_state_mapping(schema: DataSchema, model_width: int) -> StateMapping:
    return StateMapping(
        entries=_vector_entries(schema.state_dims, schema.state_dim, model_width,
                                with_mode=False),
        resolved=True,
    )


def resolve_action_mapping(schema: DataSchema, model_width: int) -> ActionMapping:
    return ActionMapping(
        entries=_vector_entries(schema.action_dims, schema.action_dim, model_width,
                                with_mode=True),
        resolved=True,
    )


def resolve_language_mapping(
    schema: DataSchema, metadata: ModelMetadata, overrides: dict[str, Any],
) -> LanguageMapping:
    """Task text → model prompt.

    Never a hard failure: the fallback chain mirrors ``task_tokenize``'s runtime
    behaviour (sample task > default_task > empty prompt) rather than the
    matrix's "error when no fallback" row.
    """
    if not metadata.requires_prompt:
        # Nothing to map (ACT takes no prompt). Resolved, deliberately empty.
        return LanguageMapping(entries=(), resolved=True)
    default_task = overrides.get("default_task")
    data_field = schema.instruction_task_field or None
    if data_field:
        source, fallback = "inferred", (default_task or None)
    elif default_task:
        source, fallback = "override", default_task
    else:
        source, fallback = "undeclared", "empty_prompt"
    return LanguageMapping(
        entries=({
            "model_field": "tokenized_prompt",
            "data_field": data_field,
            "template": metadata.language_template,
            "fallback": fallback,
            "source": source,
        },),
        resolved=True,
    )


def resolve_joint_mapping(
    schema: DataSchema, robot_profile: RobotProfile | None,
) -> JointMapping:
    """Canonical joint order → robot-native joint names.

    Built from the *action* dims: this mapping's consumer is the joint reorder
    on the ``model_to_robot`` path, i.e. the command vector. Needs a robot; with
    none declared (every example recipe today) it stays unresolved rather than
    guessing a body.
    """
    if robot_profile is None:
        return JointMapping()
    names = [d.name for d in schema.action_dims if d.name is not None]
    if not names:
        return JointMapping()
    pairs, unmatched, duplicates = embed_joints(names, robot_profile.joints.names)
    if unmatched or duplicates:  # pragma: no cover - Check Pairs raised already
        return JointMapping()
    return JointMapping(
        entries=tuple(
            {"canonical_index": i, "data_name": name, "robot_joint_name": joint}
            for i, (name, joint) in enumerate(pairs)
        ),
        resolved=True,
    )
