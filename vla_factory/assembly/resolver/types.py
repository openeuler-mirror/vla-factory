"""Serializable data structures produced by the composition resolver.

``TransformStepCall`` / ``TransformPipelinePlan`` describe a pipeline as data;
the five ``Mapping`` types describe the field correspondences; and
``ResolvedAssembly`` bundles those with the ``ModelIOSpec`` and
normalized references to the three input descriptions.

Why plain data instead of ready-to-run pipelines
------------------------------------------------
A ``TransformStepCall`` is exactly what its name says — one call to a
registered transform: a step *name* plus the *arguments* to build it with. It
deliberately does not hold the step class (that lives in ``TransformRegistry``,
which maps name → ``type[TransformStep]``) and it does not hold a built
``TransformStep``, because the plan has to cross a process boundary: inference
rebuilds the pipeline in a *different process* from a checkpoint's
``inference_metadata/`` (see ``inference/infer.py``), long after the training
process that resolved it is gone. Live objects — numpy stats, a HF tokenizer,
cv2 handles — cannot make that trip, and the resolver must also stay runnable
without the optional model extras installed (architecture §4.2.2).

So this is not an extra layer over "just build the pipeline": the serialized
name+args form already exists today as ``model.config.transforms.inputs`` in the
saved recipe. What the resolver adds is that the facts are already baked in —
pad target, normalize method, and the skip decisions each step's ``from_config``
would otherwise re-derive — so the consuming side executes instead of deriving
the same thing a second time (architecture §4.2.6).

The two names mirror the executable pair they are built into::

    TransformStepCall      --instantiate-->  TransformStep
    TransformPipelinePlan  --instantiate-->  TransformPipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Transform pipeline (planned calls, not yet instantiated) ──────


@dataclass(frozen=True)
class TransformStepCall:
    """One call to a registered transform: its name plus the arguments to
    build it with (serializable).

    Attributes
    ----------
    type : str
        Registered transform name (resolved by ``TransformRegistry`` downstream).
    args : dict
        Constructor arguments (JSON-serializable). Named ``args``, not
        ``config``, because these are resolved values — ``model.config`` is the
        recipe's per-run tunable block, and reusing that word here would suggest
        the same overridability.
    """

    type: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "args": dict(self.args)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TransformStepCall":
        return cls(type=d.get("type", ""), args=dict(d.get("args") or {}))


@dataclass(frozen=True)
class TransformPipelinePlan:
    """An ordered, planned transform pipeline (not yet instantiated).

    Attributes
    ----------
    calls : tuple[TransformStepCall, ...]
        Calls to make, in order.
    resolved : bool
        Whether the resolver produced a plan for this path. ``False`` means it
        did not: no step list was declared, or the path is not derivable yet
        (``robot_to_model``). A resolved plan lists every call that runs, with
        every argument the resolver can determine — the one it cannot is
        ``task_tokenize``'s tokenizer repo when a declaration omits it, whose
        only source is the recipe's ``model.path``.
    """

    calls: tuple[TransformStepCall, ...] = ()
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": [c.to_dict() for c in self.calls],
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TransformPipelinePlan":
        return cls(
            calls=tuple(TransformStepCall.from_dict(c) for c in (d.get("calls") or [])),
            resolved=bool(d.get("resolved", False)),
        )


# ── Field mappings (semantic correspondence only; no tensor math) ──


@dataclass(frozen=True)
class CameraMapping:
    """Camera → model visual-slot correspondence.

    ``entries`` is a tuple of ``{model_slot, data_source, robot_source}`` dicts;
    a ``None`` source means that slot is left unmapped (placeholder padding).
    """

    entries: tuple[dict[str, Any], ...] = ()
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [dict(e) for e in self.entries], "resolved": self.resolved}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CameraMapping":
        return cls(
            entries=tuple(dict(e) for e in (d.get("entries") or [])),
            resolved=bool(d.get("resolved", False)),
        )


@dataclass(frozen=True)
class StateMapping:
    """Data/robot state field → model state-vector correspondence."""

    entries: tuple[dict[str, Any], ...] = ()
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [dict(e) for e in self.entries], "resolved": self.resolved}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StateMapping":
        return cls(
            entries=tuple(dict(e) for e in (d.get("entries") or [])),
            resolved=bool(d.get("resolved", False)),
        )


@dataclass(frozen=True)
class ActionMapping:
    """Dimension/semantic relation between data / model-output / robot actions."""

    entries: tuple[dict[str, Any], ...] = ()
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [dict(e) for e in self.entries], "resolved": self.resolved}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActionMapping":
        return cls(
            entries=tuple(dict(e) for e in (d.get("entries") or [])),
            resolved=bool(d.get("resolved", False)),
        )


@dataclass(frozen=True)
class LanguageMapping:
    """Task-text field → model prompt correspondence."""

    entries: tuple[dict[str, Any], ...] = ()
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [dict(e) for e in self.entries], "resolved": self.resolved}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LanguageMapping":
        return cls(
            entries=tuple(dict(e) for e in (d.get("entries") or [])),
            resolved=bool(d.get("resolved", False)),
        )


@dataclass(frozen=True)
class JointMapping:
    """Canonical joint order → robot-native joint-name correspondence."""

    entries: tuple[dict[str, Any], ...] = ()
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [dict(e) for e in self.entries], "resolved": self.resolved}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JointMapping":
        return cls(
            entries=tuple(dict(e) for e in (d.get("entries") or [])),
            resolved=bool(d.get("resolved", False)),
        )


# ── Model IO spec (the post-composition fact standard) ────────────


@dataclass(frozen=True)
class ModelIOSpec:
    """What the model takes in and gives out, once the three descriptions have
    been reconciled — the post-composition fact standard.

    Inputs are the camera keys, the state width and whether a prompt is
    required; outputs are the action width and horizon. Downstream builds the
    model against this and feeds it tensors shaped by it.

    Widths are folded through the planned pipeline, so they are the shapes that
    really arrive, not the ones any single description hoped for.
    """

    action_dim: int = 0
    action_horizon: int = 0
    state_dim: int = 0
    cameras: tuple[str, ...] = ()
    requires_language: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "state_dim": self.state_dim,
            "cameras": list(self.cameras),
            "requires_language": self.requires_language,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelIOSpec":
        return cls(
            action_dim=int(d.get("action_dim", 0)),
            action_horizon=int(d.get("action_horizon", 0)),
            state_dim=int(d.get("state_dim", 0)),
            cameras=tuple(d.get("cameras") or ()),
            requires_language=bool(d.get("requires_language", False)),
        )


# ── ResolvedAssembly (the resolver's single output) ───────────────


@dataclass(frozen=True)
class ResolvedAssembly:
    """The single product of a successful composition resolution.

    Holds normalized references to the three descriptions, the model IO spec,
    the five field mappings and the three declared transform pipelines. Fully serializable via :meth:`to_dict` / :meth:`from_dict`.
    """

    # ── Three-description refs (serializable snapshots) ──
    schema_ref: dict[str, Any] = field(default_factory=dict)
    norm_stats_ref: dict[str, Any] = field(default_factory=dict)
    metadata_ref: dict[str, Any] = field(default_factory=dict)
    robot_ref: dict[str, Any] | None = None
    # Controlled overrides actually applied (architecture §3.4 — record every
    # field's source). Empty when none were supplied.
    overrides_ref: dict[str, Any] = field(default_factory=dict)

    # ── Model IO spec ──
    model_io_spec: ModelIOSpec = field(default_factory=ModelIOSpec)

    # ── Field mappings ──
    camera_mapping: CameraMapping = field(default_factory=CameraMapping)
    state_mapping: StateMapping = field(default_factory=StateMapping)
    action_mapping: ActionMapping = field(default_factory=ActionMapping)
    language_mapping: LanguageMapping = field(default_factory=LanguageMapping)
    joint_mapping: JointMapping = field(default_factory=JointMapping)

    # ── Three declared pipelines (``robot_to_model`` not derivable yet) ──
    data_to_model: TransformPipelinePlan = field(default_factory=TransformPipelinePlan)
    robot_to_model: TransformPipelinePlan = field(default_factory=TransformPipelinePlan)
    model_to_robot: TransformPipelinePlan = field(default_factory=TransformPipelinePlan)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_ref": dict(self.schema_ref),
            "norm_stats_ref": dict(self.norm_stats_ref),
            "metadata_ref": dict(self.metadata_ref),
            "robot_ref": dict(self.robot_ref) if self.robot_ref is not None else None,
            "overrides_ref": dict(self.overrides_ref),
            "model_io_spec": self.model_io_spec.to_dict(),
            "camera_mapping": self.camera_mapping.to_dict(),
            "state_mapping": self.state_mapping.to_dict(),
            "action_mapping": self.action_mapping.to_dict(),
            "language_mapping": self.language_mapping.to_dict(),
            "joint_mapping": self.joint_mapping.to_dict(),
            "data_to_model": self.data_to_model.to_dict(),
            "robot_to_model": self.robot_to_model.to_dict(),
            "model_to_robot": self.model_to_robot.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResolvedAssembly":
        robot_ref = d.get("robot_ref")
        return cls(
            schema_ref=dict(d.get("schema_ref") or {}),
            norm_stats_ref=dict(d.get("norm_stats_ref") or {}),
            metadata_ref=dict(d.get("metadata_ref") or {}),
            robot_ref=dict(robot_ref) if robot_ref is not None else None,
            overrides_ref=dict(d.get("overrides_ref") or {}),
            model_io_spec=ModelIOSpec.from_dict(
                d.get("model_io_spec") or {}
            ),
            camera_mapping=CameraMapping.from_dict(d.get("camera_mapping") or {}),
            state_mapping=StateMapping.from_dict(d.get("state_mapping") or {}),
            action_mapping=ActionMapping.from_dict(d.get("action_mapping") or {}),
            language_mapping=LanguageMapping.from_dict(d.get("language_mapping") or {}),
            joint_mapping=JointMapping.from_dict(d.get("joint_mapping") or {}),
            data_to_model=TransformPipelinePlan.from_dict(d.get("data_to_model") or {}),
            robot_to_model=TransformPipelinePlan.from_dict(d.get("robot_to_model") or {}),
            model_to_robot=TransformPipelinePlan.from_dict(d.get("model_to_robot") or {}),
        )
