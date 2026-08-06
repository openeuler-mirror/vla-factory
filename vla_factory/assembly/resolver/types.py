"""Serializable data structures produced by the composition resolver.

Phase 0 (architecture §7.4) fixes the *terms and data structures*: the
``TransformStepSpec`` / ``TransformPipelineSpec`` shape, the five field
``Mapping`` types, the ``CanonicalInterface`` and the ``ResolvedAssembly`` that
bundles them. The resolver currently fills only the three-description refs, the
canonical interface and a Materialize-stage merge — the five Mappings and the
three ``TransformPipelineSpec`` are emitted as empty placeholders (``resolved``
flag set to ``False``). Their shape is stable so later phases can populate them
without changing downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Transform pipeline (declarative spec) ─────────────────────────


@dataclass(frozen=True)
class TransformStepSpec:
    """A single declared transform step (serializable).

    Attributes
    ----------
    type : str
        Registered transform name (resolved by ``TransformRegistry`` downstream).
    config : dict
        Step configuration (JSON-serializable).
    """

    type: str
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "config": dict(self.config)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TransformStepSpec":
        return cls(type=d.get("type", ""), config=dict(d.get("config") or {}))


@dataclass(frozen=True)
class TransformPipelineSpec:
    """An ordered, declared transform pipeline (not yet instantiated).

    Attributes
    ----------
    steps : tuple[TransformStepSpec, ...]
        Steps to apply, in order.
    risk : str
        Reliability class of the pipeline: ``none`` | ``lossy`` | ``irreversible``.
    reversible : bool
        Whether every step has a precise inverse implemented (architecture
        §4.2.4 — pipelines never rely on "reverse the list").
    resolved : bool
        ``False`` means the resolver has not yet planned concrete steps for this
        path (phase-0 placeholder).
    """

    steps: tuple[TransformStepSpec, ...] = ()
    risk: str = "none"
    reversible: bool = True
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "risk": self.risk,
            "reversible": self.reversible,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TransformPipelineSpec":
        return cls(
            steps=tuple(TransformStepSpec.from_dict(s) for s in (d.get("steps") or [])),
            risk=d.get("risk", "none"),
            reversible=bool(d.get("reversible", True)),
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


# ── Canonical interface (the post-composition fact standard) ──────


@dataclass(frozen=True)
class CanonicalInterface:
    """The final observation/action/language/temporal semantics the three
    descriptions agree on. Downstream layers read/write against this interface.

    Phase 0 derives it from the merged model description + data schema facts
    that are unambiguous today; richer derivation lands in later phases.
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
    def from_dict(cls, d: dict[str, Any]) -> "CanonicalInterface":
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

    Holds normalized references to the three descriptions, the canonical
    interface, the five field mappings and the three declared transform
    pipelines. Fully serializable via :meth:`to_dict` / :meth:`from_dict`.
    """

    # ── Three-description refs (serializable snapshots) ──
    schema_ref: dict[str, Any] = field(default_factory=dict)
    norm_stats_ref: dict[str, Any] = field(default_factory=dict)
    metadata_ref: dict[str, Any] = field(default_factory=dict)
    contract_ref: dict[str, Any] | None = None
    robot_ref: dict[str, Any] | None = None
    # Controlled overrides actually applied (architecture §3.4 — record every
    # field's source). Empty when none were supplied.
    overrides_ref: dict[str, Any] = field(default_factory=dict)

    # ── Canonical interface ──
    canonical_interface: CanonicalInterface = field(default_factory=CanonicalInterface)

    # ── Field mappings (placeholders in phase 0) ──
    camera_mapping: CameraMapping = field(default_factory=CameraMapping)
    state_mapping: StateMapping = field(default_factory=StateMapping)
    action_mapping: ActionMapping = field(default_factory=ActionMapping)
    language_mapping: LanguageMapping = field(default_factory=LanguageMapping)
    joint_mapping: JointMapping = field(default_factory=JointMapping)

    # ── Three declared pipelines (placeholders in phase 0) ──
    data_to_model: TransformPipelineSpec = field(default_factory=TransformPipelineSpec)
    robot_to_model: TransformPipelineSpec = field(default_factory=TransformPipelineSpec)
    model_to_robot: TransformPipelineSpec = field(default_factory=TransformPipelineSpec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_ref": dict(self.schema_ref),
            "norm_stats_ref": dict(self.norm_stats_ref),
            "metadata_ref": dict(self.metadata_ref),
            "contract_ref": dict(self.contract_ref) if self.contract_ref is not None else None,
            "robot_ref": dict(self.robot_ref) if self.robot_ref is not None else None,
            "overrides_ref": dict(self.overrides_ref),
            "canonical_interface": self.canonical_interface.to_dict(),
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
        contract_ref = d.get("contract_ref")
        robot_ref = d.get("robot_ref")
        return cls(
            schema_ref=dict(d.get("schema_ref") or {}),
            norm_stats_ref=dict(d.get("norm_stats_ref") or {}),
            metadata_ref=dict(d.get("metadata_ref") or {}),
            contract_ref=dict(contract_ref) if contract_ref is not None else None,
            robot_ref=dict(robot_ref) if robot_ref is not None else None,
            overrides_ref=dict(d.get("overrides_ref") or {}),
            canonical_interface=CanonicalInterface.from_dict(
                d.get("canonical_interface") or {}
            ),
            camera_mapping=CameraMapping.from_dict(d.get("camera_mapping") or {}),
            state_mapping=StateMapping.from_dict(d.get("state_mapping") or {}),
            action_mapping=ActionMapping.from_dict(d.get("action_mapping") or {}),
            language_mapping=LanguageMapping.from_dict(d.get("language_mapping") or {}),
            joint_mapping=JointMapping.from_dict(d.get("joint_mapping") or {}),
            data_to_model=TransformPipelineSpec.from_dict(d.get("data_to_model") or {}),
            robot_to_model=TransformPipelineSpec.from_dict(d.get("robot_to_model") or {}),
            model_to_robot=TransformPipelineSpec.from_dict(d.get("model_to_robot") or {}),
        )
