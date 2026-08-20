"""Resolve the four DataSchema-to-model field mappings.

A mapping states a stable semantic correspondence and performs no tensor math.
Each resolver owns both validation and construction of that correspondence, so
there is no separate matching layer that can disagree with it.
"""

from __future__ import annotations

from typing import Any

from vla_factory.data.data_schema import ActionDim, DataSchema, StateDim
from vla_factory.model.model_interface import ModelMetadata, VisionSlot
from .errors import (
    CAMERA_MAPPING_INVALID,
    CAMERA_SLOT_AMBIGUOUS,
    CAMERA_SLOT_UNRESOLVED,
    make_error,
)
from ..resolve_assembly import (
    ActionMapping,
    CameraMapping,
    LanguageMapping,
    MappingSource,
    StateMapping,
)


def camera_semantic_satisfies(semantic: str, accepts: tuple[str, ...]) -> bool:
    return semantic in accepts or (
        "third_person" in accepts and semantic.startswith("third_person_")
    )


def camera_candidates(
    slot: VisionSlot, candidates: list[tuple[str, str]],
) -> list[str]:
    return sorted(
        name for name, semantic in candidates
        if camera_semantic_satisfies(semantic, slot.semantic_accepts)
    )


def data_camera_candidates(schema: DataSchema) -> list[tuple[str, str]]:
    return [
        (entry.key, entry.semantic)
        for entry in schema.cameras_entries
        if entry.semantic
    ]


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
    schema: DataSchema,
    metadata: ModelMetadata,
    camera_override: dict[str, str] | None = None,
) -> CameraMapping:
    """DataSchema camera → model visual slot."""
    override_map = dict(camera_override or {})
    if override_map:
        validate_camera_override(override_map, schema, metadata)

    if not metadata.vision_slots:
        # The model declares no fixed slots (ACT): its visual inputs *are* the
        # dataset cameras in schema order — ``adapters/act.py`` builds
        # input_features from ``schema.cameras``. Identity, nothing to infer.
        return CameraMapping(
            entries=tuple(
                {
                    "model_slot": cam,
                    "data_source": cam,
                    "source": MappingSource.INFERRED,
                }
                for cam in schema.cameras
            ),
        )

    # A supplied override is the *complete* slot→camera statement, not a patch
    # on top of inference: a slot the user left out is deliberately unmapped.
    # This is also what the OpenPI adapter does — ``adapters/openpi.py`` reads
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
                "source": MappingSource.OVERRIDE,
            })
            continue
        hits = camera_candidates(slot, candidates)
        if len(hits) > 1:
            raise make_error(
                CAMERA_SLOT_AMBIGUOUS, f"model.vision_slots.data.{slot.name}",
                slot_name=slot.name, candidates=hits,
            )
        if len(hits) == 1:
            entries.append({
                "model_slot": slot.name,
                "data_source": hits[0],
                "source": MappingSource.INFERRED,
            })
            continue
        # An override is a complete declaration, so an omitted slot is an
        # intentional placeholder. In inferred mode, however, a required slot
        # with an error policy must not silently become padding.
        if (
            not override_map
            and slot.required
            and metadata.missing_slot_policy == "error"
        ):
            raise make_error(
                CAMERA_SLOT_UNRESOLVED,
                f"model.vision_slots.data.{slot.name}",
                slot_name=slot.name,
                missing_slot_policy=metadata.missing_slot_policy,
            )
        entries.append({
            "model_slot": slot.name,
            "data_source": None,
            "source": MappingSource.INFERRED,
        })
    return CameraMapping(entries=tuple(entries))


def _vector_entries(
    dims: tuple[StateDim, ...] | tuple[ActionDim, ...],
    with_mode: bool,
) -> tuple[dict[str, Any], ...]:
    """The real dataset-dimension correspondences, in canonical order.

    Padding is not a correspondence: it has no source. Model target width
    belongs to ``ModelIOSpec`` and the padding operation belongs to the pipeline
    plan, so this Mapping contains no synthetic padding-only entries.
    """
    entries: list[dict[str, Any]] = []
    for index, dim in enumerate(dims):
        entry: dict[str, Any] = {
            "model_index": index,
            "data_dim_index": index,
            "data_name": dim.name,
            "source": MappingSource.INFERRED,
        }
        if with_mode:
            entry["mode"] = getattr(dim, "mode", None)
        entries.append(entry)
    return tuple(entries)


def resolve_state_mapping(schema: DataSchema) -> StateMapping:
    return StateMapping(
        entries=_vector_entries(schema.state_dims, with_mode=False),
    )


def resolve_action_mapping(schema: DataSchema) -> ActionMapping:
    return ActionMapping(
        entries=_vector_entries(schema.action_dims, with_mode=True),
    )


def resolve_language_mapping(
    schema: DataSchema,
    metadata: ModelMetadata,
    default_task: str | None = None,
) -> LanguageMapping:
    """Task text → model prompt.

    Runtime fallback policy is compiled into ``task_tokenize``; this mapping
    records only whether its input relationship came from data or an override.
    """
    if not metadata.requires_prompt:
        # Nothing to map (ACT takes no prompt). Resolved, deliberately empty.
        return LanguageMapping(entries=())
    data_field = schema.instruction_task_field or None
    if data_field:
        source = MappingSource.INFERRED
    elif default_task:
        source = MappingSource.OVERRIDE
    else:
        source = MappingSource.INFERRED
    return LanguageMapping(
        entries=({
            "model_field": "tokenized_prompt",
            "data_field": data_field,
            "template": metadata.language_template,
            "source": source,
        },),
    )
