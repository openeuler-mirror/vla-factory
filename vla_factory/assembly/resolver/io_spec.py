"""Build the model runtime interface from description facts.

``ModelIOSpec`` is an input to pipeline planning, not an observation inferred
from a planned pipeline.  Model facts define the target interface; the dataset
schema supplies flexible/native dimensions; transform steps only reconcile the
two.
"""

from __future__ import annotations

from typing import Any

from vla_factory.data.manifest import DataSchema
from vla_factory.model.interfaces.model import ModelMetadata

from .types import CameraMapping, ModelIOSpec


def _action_horizon(
    metadata: ModelMetadata, model_config: dict[str, Any] | None,
) -> int:
    """Resolve the model's output chunk length from its one legal source."""
    declared = int(metadata.action_horizon or 0)
    tunables = model_config if model_config is not None else (metadata.params or {})
    tunable = int((tunables or {}).get("action_horizon") or 0)

    if declared and tunable:
        raise ValueError(
            f"Model {metadata.name!r} declares action_horizon twice: "
            f"ModelMetadata.action_horizon={declared} and "
            f"params['action_horizon']={tunable}. Declare exactly one."
        )
    if metadata.training_paradigm == "pretrained_finetune" and tunable:
        raise ValueError(
            f"Model {metadata.name!r} is pretrained_finetune but declares "
            "action_horizon as a tunable in params. Move it to "
            "ModelMetadata.action_horizon."
        )
    if metadata.training_paradigm == "from_scratch" and declared:
        raise ValueError(
            f"Model {metadata.name!r} is from_scratch but declares "
            "ModelMetadata.action_horizon. Move it to params['action_horizon']."
        )
    horizon = declared or tunable
    if not horizon:
        raise ValueError(
            f"Model {metadata.name!r} declares no action horizon. Set "
            "ModelMetadata.action_horizon (pretrained_finetune) or "
            "params['action_horizon'] (from_scratch)."
        )
    return horizon


def _model_vector_widths(
    schema: DataSchema, metadata: ModelMetadata,
) -> tuple[int, int]:
    """Return the state-input and action-output widths the model declares.

    Flexible models build their projections around the dataset widths.  Fixed
    and padded families use ``dim_policy_max`` for proprioception and their
    explicit ``action_dim`` (falling back to the same limit) for actions.
    Compatibility checks have already rejected data wider than these targets.
    """
    if metadata.dim_policy == "flexible":
        if metadata.action_dim:
            raise ValueError(
                f"Model {metadata.name!r} declares dim_policy='flexible' and "
                f"a fixed action_dim={metadata.action_dim}. A flexible model "
                "must leave action_dim=0 so DataSchema supplies the width."
            )
        return int(schema.state_dim), int(schema.action_dim)

    limit = int(metadata.dim_policy_max or 0)
    if not limit:
        raise ValueError(
            f"Model {metadata.name!r} declares dim_policy="
            f"{metadata.dim_policy!r} but no dim_policy_max. A capping policy "
            "without a cap leaves the widths undefined — state would quietly "
            "fall back to the dataset's width while the model expects a fixed "
            "one, and no padding would be planned. Set dim_policy_max, or "
            "declare dim_policy='flexible'."
        )
    action_dim = int(metadata.action_dim or limit)
    return limit, action_dim


def _configured_image_size(
    metadata: ModelMetadata, model_config: dict[str, Any] | None,
) -> tuple[int, int] | None:
    """Read a from-scratch model's explicit per-run input image size."""
    tunables = model_config if model_config is not None else (metadata.params or {})
    value = (tunables or {}).get("input_image_size")
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("model.config.input_image_size must be [height, width].")
    size = (int(value[0]), int(value[1]))
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("model.config.input_image_size values must be positive.")
    return size


def _camera_shapes(
    schema: DataSchema,
    metadata: ModelMetadata,
    model_config: dict[str, Any] | None,
    camera_mapping: CameraMapping,
) -> dict[str, tuple[int, int]]:
    """Resolve canonical-camera sizes at the model boundary.

    A fixed-slot model gets the resolution of the slot each canonical camera
    feeds.  When every declared slot has the same resolution (pi0/pi05), that
    global model requirement applies to every input image processed by the
    common resize step.  A flexible model may declare ``input_image_size`` as a
    tunable; otherwise its model input stays at the dataset's native size.
    """
    native = {
        camera.key: (int(camera.resolution[0]), int(camera.resolution[1]))
        for camera in schema.cameras_entries
        if camera.resolution and len(camera.resolution) == 2
    }
    configured = _configured_image_size(metadata, model_config)
    if configured is not None and metadata.vision_slots:
        raise ValueError(
            f"Model {metadata.name!r} has fixed vision-slot resolutions; "
            "input_image_size cannot override them."
        )
    if configured is not None:
        return {camera: configured for camera in schema.cameras}

    slot_sizes = {
        slot.name: tuple(map(int, slot.resolution))
        for slot in metadata.vision_slots
        if slot.resolution is not None
    }
    unique_slot_sizes = set(slot_sizes.values())
    if len(unique_slot_sizes) == 1:
        target = next(iter(unique_slot_sizes))
        return {camera: target for camera in schema.cameras}

    shapes = dict(native)
    for entry in camera_mapping.entries:
        camera = entry.get("data_source")
        size = slot_sizes.get(str(entry.get("model_slot")))
        if camera and size is not None:
            shapes[str(camera)] = size
    return shapes


def resolve_model_io_spec(
    schema: DataSchema,
    metadata: ModelMetadata,
    model_config: dict[str, Any] | None,
    camera_mapping: CameraMapping,
) -> ModelIOSpec:
    """Resolve the model-facing tensor interface before planning transforms."""
    state_dim, action_dim = _model_vector_widths(schema, metadata)
    return ModelIOSpec(
        action_dim=action_dim,
        action_horizon=_action_horizon(metadata, model_config),
        state_dim=state_dim,
        cameras=tuple(schema.cameras),
        camera_shapes=_camera_shapes(
            schema, metadata, model_config, camera_mapping,
        ),
        requires_language=bool(metadata.requires_prompt),
    )
