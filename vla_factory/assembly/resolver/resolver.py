"""``resolve_assembly`` — the deterministic composition-resolution entry point.

Phase 0 (architecture §7.4) implements three stages only:

* **Load**        — require the three core inputs (DataSchema, NormStats,
                    ModelMetadata); attach optional BaseContract / RobotProfile.
* **Materialize** — merge ModelMetadata with BaseContract, failing on a
                    capability-boundary conflict.
* **Validate**    — check each description's internal structure (e.g. a
                    RobotProfile's joint/limit consistency).

The five field Mappings and the three ``TransformPipelineSpec`` are emitted as
empty placeholders (``resolved=False``); concrete derivation lands in later
phases. The function is pure: it creates no model, no DataLoader, no output
directory and uses no GPU, and its result is fully serializable.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from vla_factory.data.manifest import DataSchema, NormStats
from vla_factory.model.base_contract import BaseContract
from vla_factory.model.interfaces.model import ModelMetadata
from vla_factory.robot.profile import RobotProfile

from .errors import (
    INVALID_DESCRIPTION,
    MISSING_INPUT,
    METADATA_CONTRACT_CONFLICT,
    ResolutionError,
    make_error,
)
from .types import (
    CanonicalInterface,
    ResolvedAssembly,
)


# ── helpers ───────────────────────────────────────────────────────


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclass/tuple structures into JSON-friendly
    plain ``dict`` / ``list`` / scalar values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


# ── stages ────────────────────────────────────────────────────────


def _load(
    schema: DataSchema | None,
    norm_stats: NormStats | None,
    metadata: ModelMetadata | None,
) -> tuple[DataSchema, NormStats, ModelMetadata]:
    """Stage 1 — require the three core descriptions."""
    if schema is None:
        raise make_error(MISSING_INPUT, "schema", field_name="schema", detail="DataSchema is required")
    if not isinstance(schema, DataSchema):
        raise make_error(
            INVALID_DESCRIPTION, "schema",
            field_name="schema", value=type(schema).__name__,
            detail="schema must be a DataSchema instance",
        )
    if norm_stats is None:
        raise make_error(
            MISSING_INPUT, "norm_stats", field_name="norm_stats", detail="NormStats is required"
        )
    if not isinstance(norm_stats, NormStats):
        raise make_error(
            INVALID_DESCRIPTION, "norm_stats",
            field_name="norm_stats", value=type(norm_stats).__name__,
            detail="norm_stats must be a NormStats instance",
        )
    if metadata is None:
        raise make_error(
            MISSING_INPUT, "model", field_name="metadata", detail="ModelMetadata is required"
        )
    if not isinstance(metadata, ModelMetadata):
        raise make_error(
            INVALID_DESCRIPTION, "model",
            field_name="metadata", value=type(metadata).__name__,
            detail="metadata must be a ModelMetadata instance",
        )
    return schema, norm_stats, metadata


def _materialize(
    metadata: ModelMetadata,
    base_contract: BaseContract | None,
) -> dict[str, Any]:
    """Stage 2 — merge ModelMetadata with BaseContract facts.

    The checkpoint can self-state some instance facts (camera roles, real
    action/state dims, image resolution). Those refine the metadata but must
    stay within the model family's capability boundary — a checkpoint declaring
    *more* action dims than the model supports is a hard conflict.

    Returns the merged, JSON-friendly fact dict used to build the canonical
    interface. Records the source of the action-dim fact.
    """
    facts: dict[str, Any] = {
        "name": metadata.name,
        "backend": metadata.backend,
        "action_dim": metadata.action_dim,
        "action_horizon": metadata.action_horizon,
        "action_head_type": metadata.action_head_type,
        "requires_prompt": metadata.requires_prompt,
    }
    if base_contract is not None:
        # Capability-boundary check on action_dim. ModelMetadata.action_dim is
        # the model family's internal max (0 == "from data, no fixed cap", e.g.
        # ACT trained from scratch). A checkpoint may declare fewer dims
        # (padding covers the rest) but never more.
        if (
            metadata.action_dim > 0
            and base_contract.action_dim is not None
            and base_contract.action_dim > metadata.action_dim
        ):
            raise make_error(
                METADATA_CONTRACT_CONFLICT,
                "model.action_dim",
                field_name="action_dim",
                metadata_value=metadata.action_dim,
                contract_value=base_contract.action_dim,
            )
        if base_contract.action_dim is not None:
            facts["action_dim"] = base_contract.action_dim
            facts["action_dim_source"] = "base_contract"
        if base_contract.state_dim is not None:
            facts["state_dim_from_contract"] = base_contract.state_dim
    return facts


def _validate(robot_profile: RobotProfile | None) -> None:
    """Stage 3 — validate each description's internal structure."""
    if robot_profile is not None:
        if not isinstance(robot_profile, RobotProfile):
            raise make_error(
                INVALID_DESCRIPTION, "robot",
                field_name="robot_profile", value=type(robot_profile).__name__,
                detail="robot_profile must be a RobotProfile instance",
            )
        try:
            robot_profile.validate()
        except ValueError as e:
            raise make_error(
                INVALID_DESCRIPTION, f"robot({robot_profile.name})",
                field_name="robot_profile", value=None, detail=str(e),
            ) from e


# ── public entry point ────────────────────────────────────────────


def resolve_assembly(
    schema: DataSchema | None,
    norm_stats: NormStats | None,
    metadata: ModelMetadata,
    *,
    base_contract: BaseContract | None = None,
    robot_profile: RobotProfile | None = None,
    overrides: dict[str, Any] | None = None,
) -> ResolvedAssembly:
    """Resolve a ``data × model × robot`` combination into a ``ResolvedAssembly``.

    Deterministic pure logic (architecture §4.2.2): no model construction, no
    DataLoader, no training, no deploy platform, no GPU, no output directory.

    Parameters
    ----------
    schema, norm_stats, metadata
        Core descriptions (required). ``schema`` / ``norm_stats`` come from the
        data reader; ``metadata`` comes from the model registry.
    base_contract, robot_profile
        Optional instance / body descriptions.
    overrides
        Controlled overrides (e.g. ``camera_mapping``, ``accept_fps_mismatch``,
        ``gripper_flip``, ``default_task``). Phase 0 stores them on the
        resulting ``ResolvedAssembly`` (``overrides_ref``) so the source of each
        adjusted field is recorded (architecture §3.4); concrete consumption
        arrives with Mapping derivation.

    Returns
    -------
    ResolvedAssembly
        The serialized combination. Raises :class:`ResolutionError` (structured)
        on any failure.
    """
    overrides_ref = dict(overrides or {})

    # 1. Load
    schema, norm_stats, metadata = _load(schema, norm_stats, metadata)

    # 2. Materialize
    merged = _materialize(metadata, base_contract)

    # 3. Validate (only the optional descriptions need extra checks here; the
    #    core descriptions were type-checked in Load).
    _validate(robot_profile)

    # Canonical interface — the post-combination fact standard. Action dim comes
    # from the merged facts (checkpoint refines metadata); state dim prefers a
    # contract fact, then the data schema; cameras/language come from the data
    # schema and model requirement.
    action_dim = int(merged.get("action_dim", 0)) or schema.action_dim
    state_dim = merged.get("state_dim_from_contract")
    if state_dim is None:
        state_dim = schema.state_dim
    canonical = CanonicalInterface(
        action_dim=action_dim,
        action_horizon=metadata.action_horizon,
        state_dim=int(state_dim),
        cameras=tuple(schema.cameras),
        requires_language=bool(metadata.requires_prompt),
    )

    return ResolvedAssembly(
        schema_ref=_to_jsonable(schema),
        norm_stats_ref=_to_jsonable(norm_stats),
        metadata_ref=_to_jsonable(metadata),
        contract_ref=_to_jsonable(base_contract) if base_contract is not None else None,
        robot_ref=robot_profile.to_dict() if robot_profile is not None else None,
        overrides_ref=overrides_ref,
        canonical_interface=canonical,
        # Phase-0 placeholders: the five Mappings and three Pipelines are
        # intentionally left unresolved (derived in later phases).
    )
