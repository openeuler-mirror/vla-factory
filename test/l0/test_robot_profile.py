"""Tests for the robot profile registry and validation."""

from __future__ import annotations

import pytest

from vla_factory.robot import (
    GripperConvention,
    JointGroup,
    RobotProfile,
    get_robot_profile,
    list_robot_profiles,
    load_robot_profile,
)


def test_list_and_load_bundled_profile():
    names = list_robot_profiles()
    assert "lekiwi" in names
    assert len(names) >= 1

    profile = get_robot_profile("lekiwi")
    assert profile.name == "lekiwi"
    # 9-DoF: 3 base + 6 arm (last is the gripper).
    assert len(profile.joints.names) == 9
    assert profile.cameras == ("front", "wrist")
    assert profile.gripper.joint_index == 8
    assert profile.native_action_type == "joint_pos"
    assert profile.recommended_control_hz > 0


def test_profile_round_trip():
    profile = get_robot_profile("lekiwi")
    restored = RobotProfile.from_dict(profile.to_dict())
    assert restored == profile


def test_unknown_profile_raises():
    with pytest.raises(FileNotFoundError):
        get_robot_profile("does_not_exist")


def test_load_profile_from_path_uses_validated_deserialization(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("name: custom\njoints:\n  names: [joint_0]\n")

    profile = load_robot_profile(path)

    assert profile.name == "custom"
    assert profile.joints.names == ("joint_0",)


def test_robotwin_bimanual_profile():
    p = get_robot_profile("robotwin")
    # 14-DoF bimanual: left_arm(6) + left_gripper + right_arm(6) + right_gripper.
    assert len(p.joints.names) == 14
    assert p.joints.names[6] == "left_gripper"
    assert p.joints.names[13] == "right_gripper"
    assert p.native_action_type == "joint_pos"
    assert p.to_dict() == RobotProfile.from_dict(p.to_dict()).to_dict()


def test_empty_joint_names_invalid():
    bad = RobotProfile(name="x", joints=JointGroup(names=()))
    with pytest.raises(ValueError):
        bad.validate()


def test_duplicate_joint_and_camera_names_invalid():
    with pytest.raises(ValueError, match="joints.names contains duplicates"):
        RobotProfile(
            name="x", joints=JointGroup(names=("a", "a")),
        ).validate()
    with pytest.raises(ValueError, match="cameras contains duplicates"):
        RobotProfile(
            name="x", joints=JointGroup(names=("a",)),
            cameras=("front", "front"),
        ).validate()


def test_incomplete_or_wrong_width_safety_bounds_invalid():
    with pytest.raises(ValueError, match="must both be set"):
        RobotProfile(
            name="x", joints=JointGroup(names=("a",)),
            safety_bounds_low=(-1.0,),
        ).validate()
    with pytest.raises(ValueError, match="must each have 2 entries"):
        RobotProfile(
            name="x", joints=JointGroup(names=("a", "b")),
            safety_bounds_low=(-1.0,), safety_bounds_high=(1.0,),
        ).validate()


def test_joint_length_mismatch_invalid():
    bad = RobotProfile(
        name="x",
        joints=JointGroup(names=("a", "b"), types=("revolute",)),
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_bad_control_mode_invalid():
    bad = RobotProfile(
        name="x",
        joints=JointGroup(names=("a",)),
        native_action_type="not_a_real_mode",
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_from_dict_validates():
    raw = {
        "name": "tester",
        "joints": {"names": ["a", "b"], "units": "radian"},
        "cameras": ["cam0"],
        "gripper": {"open_value": 0.0, "close_value": 1.0, "joint_index": 1},
    }
    profile = RobotProfile.from_dict(raw)
    assert profile.joints.names == ("a", "b")
    assert profile.gripper == GripperConvention(joint_index=1)


def test_from_dict_rejects_invalid():
    with pytest.raises(ValueError):
        RobotProfile.from_dict({"name": "", "joints": {"names": ["a"]}})


@pytest.mark.parametrize(
    "raw",
    [
        # A scalar string would be split into single-character entries by
        # tuple() and silently pass every downstream validation.
        {"name": "x", "joints": {"names": "shoulder_pan"}, "cameras": ["front"]},
        {"name": "x", "joints": {"names": ["a"]}, "cameras": "front"},
        {"name": "x", "joints": {"names": ["a"], "types": "revolute"}},
        {"name": "x", "joints": {"names": ["a"]}, "control_modes": "joint_pos"},
    ],
)
def test_from_dict_rejects_scalar_string_collections(raw):
    with pytest.raises(TypeError, match="must be a list of strings"):
        RobotProfile.from_dict(raw)


def test_get_robot_profile_rejects_path_components():
    for name in ("../evil", "../../etc/passwd", "/abs/path/evil", "a/b", ".."):
        with pytest.raises(ValueError, match="bare profile stem"):
            get_robot_profile(name)
