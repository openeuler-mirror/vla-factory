"""``resolve_assembly`` — the deterministic composition-resolution entry point.

Phase 0 (architecture §7.4) implements three stages; phase 2 adds a fourth:

* **Load**        — require the three core inputs (DataSchema, NormStats,
                    ModelMetadata); attach optional BaseContract / RobotProfile.
* **Materialize** — merge ModelMetadata with BaseContract, failing on a
                    capability-boundary conflict.
* **Validate**    — check each description's internal structure (e.g. a
                    RobotProfile's joint/limit consistency).
* **Check Pairs** — pairwise/triple compatibility checks (architecture §4.2.2),
                    phase-2 scope only: the six matrix rows §7.4 names by name
                    (dimension ×2, camera, control mode, stats, field order
                    ×2 sub-checks). See ``_check_pairs`` and
                    ``docs/plans/phase2-resolution-diagnostics.cn.md`` for the
                    five rows deliberately left out (language, gripper,
                    rotation, frequency, safety) and why.

The five field Mappings and the three ``TransformPipelineSpec`` are emitted as
empty placeholders (``resolved=False``); concrete derivation lands in later
phases. The function is pure: it creates no model, no DataLoader, no output
directory and uses no GPU, and its result is fully serializable.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from vla_factory.data.manifest import ActionDim, DataSchema, NormStats, StateDim
from vla_factory.data.semantics import infer_camera_semantic, strip_known_suffix
from vla_factory.model.base_contract import BaseContract
from vla_factory.model.interfaces.model import ModelMetadata, VisionSlot
from vla_factory.robot.profile import RobotProfile
from vla_factory.utils.vocabulary import CONTROL_MODES

from .errors import (
    ACTION_DIM_INCOMPATIBLE,
    CAMERA_SLOT_AMBIGUOUS,
    CAMERA_SLOT_UNRESOLVED,
    CONTROL_MODE_INCOMPATIBLE,
    INVALID_DESCRIPTION,
    JOINT_ORDER_AMBIGUOUS,
    JOINT_ORDER_MISMATCH,
    MISSING_INPUT,
    METADATA_CONTRACT_CONFLICT,
    NORM_STATS_INSUFFICIENT,
    STATE_DIM_INCOMPATIBLE,
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
    stay within the model family's capability boundary:

    - a checkpoint declaring *more* action dims than the model supports is a
      hard conflict;
    - a checkpoint's measured camera slots must fall within the model's declared
      ``vision_slots`` (a role the model family does not expose is a conflict).

    Returns the merged, JSON-friendly fact dict used to build the canonical
    interface. Each instance-refined fact records its source
    (``metadata`` / ``base_contract``).
    """
    facts: dict[str, Any] = {
        "name": metadata.name,
        "backend": metadata.backend,
        "action_dim": metadata.action_dim,
        "action_dim_source": "metadata",
        "action_horizon": metadata.action_horizon,
        "action_horizon_source": "metadata",
        "action_head_type": metadata.action_head_type,
        "requires_prompt": metadata.requires_prompt,
        # New interface facts (model-module §4.3) — declared by metadata;
        # BaseContract may refine a few of them below.
        "dim_policy": metadata.dim_policy,
        "dim_policy_max": metadata.dim_policy_max,
        "vector_normalization": metadata.vector_normalization,
        "expected_hz": metadata.expected_hz,
        "vision_slot_names": tuple(s.name for s in metadata.vision_slots),
    }
    if base_contract is not None:
        # ── action_dim capability boundary ──
        # ModelMetadata.action_dim is the family's internal max (0 == "from
        # data, no fixed cap", e.g. ACT from scratch). A checkpoint may declare
        # fewer dims (padding covers the rest) but never more.
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

        # ── vision-slot capability boundary ──
        # When the model family declares fixed slots, every slot the checkpoint
        # actually exposes must be among them (a checkpoint cannot invent a slot
        # the family does not have). Empty declared slots (e.g. ACT, follows the
        # data) skips this check.
        if metadata.vision_slots:
            declared = {s.name for s in metadata.vision_slots}
            extra = [r for r in base_contract.camera_role_names if r not in declared]
            if extra:
                raise make_error(
                    METADATA_CONTRACT_CONFLICT,
                    "model.vision_slots",
                    field_name="vision_slots",
                    metadata_value=sorted(declared),
                    contract_value=base_contract.camera_role_names,
                )
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


# ── Check Pairs (stage 4, architecture §7.4 phase 2) ────────────────
#
# Six matrix rows only (state dim, action dim, camera slots, control mode,
# norm stats, joint order ×2) — the subset §7.4 names explicitly for phase 2.
# language / gripper / rotation / frequency / safety are deliberately left for
# a later phase (docs/plans/phase2-resolution-diagnostics.cn.md records why).
#
# Each check raises via ``make_error`` on the first problem it finds, matching
# the sequential, raise-on-first-failure style already used by Load/
# Materialize/Validate above — phase 2 does not introduce a "collect every
# independent problem, then report them together" convention (that would be a
# second error-reporting style living alongside the first one, for a benefit
# none of these six checks currently need: a user re-runs ``resolve`` after
# fixing the first problem, same as they already do for the first three
# stages).


def _model_dim_limit(dim_policy: str, dim_policy_max: int | None) -> tuple[int, str] | None:
    """Return ``(limit, source)`` if ``dim_policy`` caps the dimension.

    ``fixed`` and ``padded_to_max`` both cap; ``flexible`` (or a missing
    ``dim_policy_max``) means "no declared limit" — ``None``.
    """
    if dim_policy_max is None or dim_policy == "flexible":
        return None
    return dim_policy_max, "metadata.dim_policy_max"


def _check_state_dim(schema: DataSchema, metadata: ModelMetadata, merged: dict[str, Any]) -> None:
    """Matrix row 1 — data vs model."""
    data_dim = schema.state_dim
    if data_dim == 0:
        return
    # A BaseContract-reported state dim is the checkpoint's own true pad
    # target (base_contract.py's state_dim), so it is authoritative over the
    # family's dim_policy_max when present.
    contract_limit = merged.get("state_dim_from_contract")
    if contract_limit is not None:
        if data_dim > int(contract_limit):
            raise make_error(
                STATE_DIM_INCOMPATIBLE, "schema.state_dim",
                data_dim=data_dim, limit=int(contract_limit),
                limit_source="base_contract.state_dim",
            )
        return
    limit_info = _model_dim_limit(metadata.dim_policy, metadata.dim_policy_max)
    if limit_info is None:
        return
    limit, source = limit_info
    exact = metadata.dim_policy == "fixed"
    if (exact and data_dim != limit) or (not exact and data_dim > limit):
        raise make_error(
            STATE_DIM_INCOMPATIBLE, "schema.state_dim",
            data_dim=data_dim, limit=limit, limit_source=source,
        )


def _check_action_dim(
    schema: DataSchema, metadata: ModelMetadata, merged: dict[str, Any],
    robot_profile: RobotProfile | None,
) -> None:
    """Matrix row 2 — data vs model vs robot."""
    data_dim = schema.action_dim
    if data_dim == 0:
        return
    if metadata.dim_policy != "flexible":
        # merged["action_dim"] already prefers a BaseContract refinement over
        # the family declaration (Materialize, stage 2) — reuse it so a
        # checkpoint with a tighter real pad target than its family's max is
        # honoured, and so this check agrees with what canonical_interface
        # ends up reporting.
        limit = int(merged.get("action_dim") or 0)
        if limit > 0:
            exact = metadata.dim_policy == "fixed"
            if (exact and data_dim != limit) or (not exact and data_dim > limit):
                raise make_error(
                    ACTION_DIM_INCOMPATIBLE, "schema.action_dim",
                    data_dim=data_dim, limit=limit,
                    limit_source=merged.get("action_dim_source", "metadata"),
                )
    # Robot side is a coarse necessary condition only: the dataset's action
    # width cannot exceed how many joints the robot physically has. The
    # precise per-name relationship is the joint-order check's job — a robot
    # legitimately having MORE joints than the dataset records (e.g. LeKiwi's
    # mobile base, absent from arm-only training data) is not an error here.
    if robot_profile is not None and robot_profile.native_action_type in CONTROL_MODES:
        robot_limit = len(robot_profile.joints.names)
        if data_dim > robot_limit:
            raise make_error(
                ACTION_DIM_INCOMPATIBLE, "schema.action_dim",
                data_dim=data_dim, limit=robot_limit, limit_source="robot.joints",
            )


def _camera_semantic_satisfies(semantic: str, accepts: tuple[str, ...]) -> bool:
    """A data/robot camera satisfies a slot if its semantic is directly
    accepted, or is a specific third-person view and the slot accepts the
    ``third_person`` generalization (architecture §4.1.2 / vocabulary.py)."""
    if semantic in accepts:
        return True
    return semantic.startswith("third_person") and "third_person" in accepts


def _check_camera_slots_against(
    path_prefix: str,
    vision_slots: tuple[VisionSlot, ...],
    candidates: list[tuple[str, str]],
    missing_slot_policy: str,
) -> None:
    """One side (data or robot) of matrix row 3, checked independently.

    ``candidates`` is ``[(camera_key, inferred_semantic), ...]`` for real
    cameras with a resolved semantic — a camera whose semantic could not be
    inferred (e.g. RoboTwin's ``left_camera`` / ``right_camera``, see
    ``docs/plans/phase2-resolution-diagnostics.cn.md``) simply is not a
    candidate; it neither satisfies nor conflicts with anything.
    """
    for slot in vision_slots:
        hits = [key for key, sem in candidates if _camera_semantic_satisfies(sem, slot.semantic_accepts)]
        if len(hits) > 1:
            raise make_error(
                CAMERA_SLOT_AMBIGUOUS, f"{path_prefix}.{slot.name}",
                slot_name=slot.name, candidates=sorted(hits),
            )
        if not hits and slot.required and missing_slot_policy == "error":
            raise make_error(
                CAMERA_SLOT_UNRESOLVED, f"{path_prefix}.{slot.name}",
                slot_name=slot.name, missing_slot_policy=missing_slot_policy,
            )


def _check_camera_slots(
    schema: DataSchema, metadata: ModelMetadata, robot_profile: RobotProfile | None,
) -> None:
    """Matrix row 3 — data/robot cameras vs model slots.

    Data and robot are checked independently (they feed different pipelines —
    ``data_to_model`` vs ``robot_to_model``), not pooled into one candidate
    set. A model with no declared ``vision_slots`` (e.g. ACT — vision follows
    the dataset) has nothing to check; the loop below is then trivially empty.
    """
    data_candidates = [
        (e.key, e.semantic) for e in schema.cameras_entries if e.semantic
    ]
    _check_camera_slots_against(
        "model.vision_slots.data", metadata.vision_slots,
        data_candidates, metadata.missing_slot_policy,
    )
    if robot_profile is not None:
        robot_candidates = [
            (cam, sem) for cam in robot_profile.cameras
            if (sem := infer_camera_semantic(cam)) is not None
        ]
        _check_camera_slots_against(
            "model.vision_slots.robot", metadata.vision_slots,
            robot_candidates, metadata.missing_slot_policy,
        )


def _check_control_mode(
    schema: DataSchema, metadata: ModelMetadata, robot_profile: RobotProfile | None,
) -> None:
    """Matrix row 5 — data/model/robot.

    A per-dim ``mode`` of ``None`` (undeclared) is not itself a problem here —
    data-module §8.3: the ``data_to_model`` path allows it; only a downstream
    ``model_to_robot`` plan would need every dim resolved, and that planning
    is phase 3's job, not this check's.
    """
    data_modes = {d.mode for d in schema.action_dims if d.mode is not None}
    if not data_modes:
        return
    model_modes = set(metadata.control_mode_pref)
    robot_modes = set(robot_profile.control_modes) if robot_profile is not None else set()
    unsupported = set()
    if model_modes:
        unsupported |= data_modes - model_modes
    if robot_modes:
        unsupported |= data_modes - robot_modes
    if unsupported:
        raise make_error(
            CONTROL_MODE_INCOMPATIBLE, "schema.action.mode",
            data_modes=sorted(data_modes), model_modes=sorted(model_modes),
            robot_modes=sorted(robot_modes) if robot_profile is not None else None,
        )


def _norm_stats_missing_fields(stats: Any | None, method: str) -> list[str]:
    if stats is None:
        return ["mean", "std"] if method == "mean_std" else \
               ["q01", "q99"] if method == "quantile" else \
               ["min", "max"] if method == "min_max" else []
    missing: list[str] = []
    if method == "mean_std":
        if not stats.mean: missing.append("mean")
        if not stats.std: missing.append("std")
    elif method == "quantile":
        if not stats.q01: missing.append("q01")
        if not stats.q99: missing.append("q99")
    elif method == "min_max":
        if not stats.min: missing.append("min")
        if not stats.max: missing.append("max")
    return missing


def _check_norm_stats(schema: DataSchema, norm_stats: NormStats, metadata: ModelMetadata) -> None:
    """Matrix row 8 — data stats vs model method."""
    method = metadata.vector_normalization
    if method is None:
        return
    if schema.state_dim > 0:
        missing = _norm_stats_missing_fields(norm_stats.state, method)
        if missing:
            raise make_error(
                NORM_STATS_INSUFFICIENT, "norm_stats.state",
                field="state", method=method, missing=missing,
            )
    if schema.action_dim > 0:
        missing = _norm_stats_missing_fields(norm_stats.action, method)
        if missing:
            raise make_error(
                NORM_STATS_INSUFFICIENT, "norm_stats.action",
                field="action", method=method, missing=missing,
            )


def _check_joint_order(
    field: str, dims: tuple[StateDim, ...] | tuple[ActionDim, ...],
    robot_profile: RobotProfile | None,
) -> None:
    """Matrix row 10 — data keys vs robot joints (decision D4).

    Names are compared after stripping a known suffix (``.pos``/``.vel``/
    ``.delta`` — data-module §8.3 keeps the suffix, robot joint names don't
    carry one; ``data/semantics.py:strip_known_suffix`` is the single source
    for that table). This is a *subset* embedding, not set equality: the
    robot is allowed to have joints the dataset never recorded (e.g. LeKiwi's
    mobile base absent from arm-only training data).
    """
    if robot_profile is None:
        return
    names = [d.name for d in dims if d.name is not None]
    if not names:
        return
    stripped = [strip_known_suffix(n) for n in names]
    robot_names = set(robot_profile.joints.names)
    unmatched = sorted({orig for orig, s in zip(names, stripped) if s not in robot_names})
    if unmatched:
        raise make_error(
            JOINT_ORDER_MISMATCH, f"schema.{field}.joint_order",
            field=field, unmatched_names=unmatched,
            robot_joint_names=list(robot_profile.joints.names),
        )
    if len(set(stripped)) != len(stripped):
        duplicates = sorted({s for s in stripped if stripped.count(s) > 1})
        raise make_error(
            JOINT_ORDER_AMBIGUOUS, f"schema.{field}.joint_order",
            field=field, duplicate_names=duplicates,
        )


def _check_pairs(
    schema: DataSchema,
    norm_stats: NormStats,
    metadata: ModelMetadata,
    merged: dict[str, Any],
    robot_profile: RobotProfile | None,
) -> None:
    """Stage 4 — run the six phase-2 compatibility checks in order."""
    _check_state_dim(schema, metadata, merged)
    _check_action_dim(schema, metadata, merged, robot_profile)
    _check_camera_slots(schema, metadata, robot_profile)
    _check_control_mode(schema, metadata, robot_profile)
    _check_norm_stats(schema, norm_stats, metadata)
    _check_joint_order("state", schema.state_dims, robot_profile)
    _check_joint_order("action", schema.action_dims, robot_profile)


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

    # 4. Check Pairs (phase 2, architecture §7.4) — six matrix rows only.
    _check_pairs(schema, norm_stats, metadata, merged, robot_profile)

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
        schema_ref=schema.to_dict(),
        norm_stats_ref=_to_jsonable(norm_stats),
        metadata_ref=_to_jsonable(metadata),
        contract_ref=_to_jsonable(base_contract) if base_contract is not None else None,
        robot_ref=robot_profile.to_dict() if robot_profile is not None else None,
        overrides_ref=overrides_ref,
        canonical_interface=canonical,
        # Phase-0 placeholders: the five Mappings and three Pipelines are
        # intentionally left unresolved (derived in later phases).
    )
