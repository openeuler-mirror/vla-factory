"""``RobotProfile`` and the per-joint / gripper description dataclasses.

Fields follow architecture §4.1.3 (RobotProfile) and ``robot-module.cn.md``.
All structures are frozen, validated, and serializable so a
``ResolvedAssembly`` can round-trip them without holding live Python objects.
File discovery and YAML loading belong to :mod:`vla_factory.robot.registry`;
this module contains only the robot description itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vla_factory.utils.vocabulary import CONTROL_MODES, is_control_mode


@dataclass(frozen=True)
class JointGroup:
    """Canonical description of one ordered group of joints (e.g. the arm).

    Attributes
    ----------
    names : tuple[str, ...]
        Ordered robot-native joint names used by the platform. They are not
        implicitly matched to DataSchema dimension names. Required and
        non-empty.
    units : str
        Joint unit. Typically ``"radian"`` for revolute joints or ``"meter"``
        for prismatic joints.
    types : tuple[str, ...]
        Per-joint type from the controlled vocabulary ``revolute`` /
        ``prismatic`` / ``continuous`` / ``fixed``. Empty means "unspecified".
    limits_low / limits_high : tuple[float, ...]
        Per-joint position limits, in ``units``. Empty means "unspecified".
        When non-empty, length must equal ``len(names)``.
    """

    names: tuple[str, ...] = ()
    units: str = "radian"
    types: tuple[str, ...] = ()
    limits_low: tuple[float, ...] = ()
    limits_high: tuple[float, ...] = ()

    def validate(self, where: str = "joints") -> None:
        if not self.names:
            raise ValueError(f"{where}.names must be a non-empty list")
        if any(not name for name in self.names):
            raise ValueError(f"{where}.names entries must be non-empty strings")
        duplicates = sorted({name for name in self.names if self.names.count(name) > 1})
        if duplicates:
            raise ValueError(f"{where}.names contains duplicates: {duplicates}")
        for label, seq in (("types", self.types),
                           ("limits_low", self.limits_low),
                           ("limits_high", self.limits_high)):
            if seq and len(seq) != len(self.names):
                raise ValueError(
                    f"{where}.{label} has {len(seq)} entries but joints has "
                    f"{len(self.names)} — they must match or be empty."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "units": self.units,
            "types": list(self.types),
            "limits_low": list(self.limits_low),
            "limits_high": list(self.limits_high),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JointGroup":
        return cls(
            names=tuple(d.get("names") or ()),
            units=d.get("units", "radian"),
            types=tuple(d.get("types") or ()),
            limits_low=tuple(d.get("limits_low") or ()),
            limits_high=tuple(d.get("limits_high") or ()),
        )


@dataclass(frozen=True)
class GripperConvention:
    """How the gripper DoF is encoded.

    Attributes
    ----------
    open_value / close_value : float
        The action/command value that corresponds to a fully open / fully
        closed gripper. The composition resolver uses these to decide whether a
        ``gripper_flip`` transform might be required (e.g. data says 1=open
        but the model expects 1=close). Such a relationship requires an
        explicit binding; the resolver does not infer it from this profile.
    joint_index : int | None
        Position of the gripper DoF within the joint vector (0-based), when the
        gripper is one of the arm joints rather than a separate actuator.
    """

    open_value: float = 0.0
    close_value: float = 1.0
    joint_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "open_value": self.open_value,
            "close_value": self.close_value,
            "joint_index": self.joint_index,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GripperConvention":
        return cls(
            open_value=float(d.get("open_value", 0.0)),
            close_value=float(d.get("close_value", 1.0)),
            joint_index=d.get("joint_index"),
        )


# Control-mode vocabulary is shared across data/model/robot dimensions and
# defined once in ``vla_factory.utils.vocabulary`` (architecture §4.5). The
# first version is joint-space only: joint_pos / joint_delta / joint_vel.
# EEF modes (eef_pos / eef_delta / se3) enter together with EEF model support.


@dataclass(frozen=True)
class RobotProfile:
    """Static description of a robot body.

    See module docstring for scope. All fields are static body facts; nothing
    here references a runtime process, transport or platform session.
    """

    # ── Identity ──
    name: str

    # ── Sensors ──
    cameras: tuple[str, ...] = ()    # stable semantic camera names

    # ── Joints / action ──
    joints: JointGroup = field(default_factory=JointGroup)
    gripper: GripperConvention = field(default_factory=GripperConvention)
    control_modes: tuple[str, ...] = ()        # supported control modes
    native_action_type: str = "joint_pos"      # canonical action representation

    # ── Frames / model ──
    coordinate_frame: str = "base_link"
    urdf_ref: str = ""

    # ── Safety / timing ──
    safety_bounds_low: tuple[float, ...] = ()
    safety_bounds_high: tuple[float, ...] = ()
    recommended_control_hz: int = 30

    def validate(self) -> None:
        """Check required fields and internal consistency.

        Raises ``ValueError`` with a field-specific message on any problem.
        Used by the registry on load and by the resolver's Validate stage.
        """
        if not self.name:
            raise ValueError("robot.name must be a non-empty string")
        self.joints.validate(where=f"robot({self.name}).joints")
        duplicate_cameras = sorted({c for c in self.cameras if self.cameras.count(c) > 1})
        if duplicate_cameras:
            raise ValueError(
                f"robot({self.name}).cameras contains duplicates: {duplicate_cameras}"
            )
        if any(not camera for camera in self.cameras):
            raise ValueError(
                f"robot({self.name}).cameras entries must be non-empty strings"
            )
        if bool(self.safety_bounds_low) != bool(self.safety_bounds_high):
            raise ValueError(
                f"robot({self.name}).safety_bounds_low/high must both be set or empty"
            )
        if self.safety_bounds_low:
            joint_count = len(self.joints.names)
            if (
                len(self.safety_bounds_low) != joint_count
                or len(self.safety_bounds_high) != joint_count
            ):
                raise ValueError(
                    f"robot({self.name}).safety_bounds_low/high must each have "
                    f"{joint_count} entries"
                )
        if not is_control_mode(self.native_action_type):
            raise ValueError(
                f"robot({self.name}).native_action_type "
                f"{self.native_action_type!r} is not in the controlled "
                f"vocabulary {list(CONTROL_MODES)}"
            )
        for m in self.control_modes:
            if not is_control_mode(m):
                raise ValueError(
                    f"robot({self.name}).control_modes entry {m!r} is not in "
                    f"the controlled vocabulary {list(CONTROL_MODES)}"
                )
        duplicate_modes = sorted(
            {mode for mode in self.control_modes if self.control_modes.count(mode) > 1}
        )
        if duplicate_modes:
            raise ValueError(
                f"robot({self.name}).control_modes contains duplicates: "
                f"{duplicate_modes}"
            )
        if self.recommended_control_hz <= 0:
            raise ValueError(
                f"robot({self.name}).recommended_control_hz must be positive"
            )

    # ── Serialization ──

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cameras": list(self.cameras),
            "joints": self.joints.to_dict(),
            "gripper": self.gripper.to_dict(),
            "control_modes": list(self.control_modes),
            "native_action_type": self.native_action_type,
            "coordinate_frame": self.coordinate_frame,
            "urdf_ref": self.urdf_ref,
            "safety_bounds_low": list(self.safety_bounds_low),
            "safety_bounds_high": list(self.safety_bounds_high),
            "recommended_control_hz": self.recommended_control_hz,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RobotProfile":
        joints = JointGroup.from_dict(d.get("joints") or {})
        gripper = GripperConvention.from_dict(d.get("gripper") or {})
        profile = cls(
            name=d.get("name", ""),
            cameras=tuple(d.get("cameras") or ()),
            joints=joints,
            gripper=gripper,
            control_modes=tuple(d.get("control_modes") or ()),
            native_action_type=d.get("native_action_type", "joint_pos"),
            coordinate_frame=d.get("coordinate_frame", "base_link"),
            urdf_ref=d.get("urdf_ref", ""),
            safety_bounds_low=tuple(d.get("safety_bounds_low") or ()),
            safety_bounds_high=tuple(d.get("safety_bounds_high") or ()),
            recommended_control_hz=int(d.get("recommended_control_hz", 30)),
        )
        profile.validate()
        return profile
