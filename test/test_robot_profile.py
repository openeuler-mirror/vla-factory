"""Tests for the robot profile registry and validation."""

from __future__ import annotations

import pytest

from vla_factory.robot import get_robot_profile, list_robot_profiles
from vla_factory.robot.profile import (
    GripperConvention,
    JointGroup,
    RobotProfile,
    profile_from_dict,
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


def test_profile_from_dict_validates():
    raw = {
        "name": "tester",
        "joints": {"names": ["a", "b"], "units": "radian"},
        "cameras": ["cam0"],
        "gripper": {"open_value": 0.0, "close_value": 1.0, "joint_index": 1},
    }
    profile = profile_from_dict(raw)
    assert profile.joints.names == ("a", "b")
    assert profile.gripper == GripperConvention(joint_index=1)


def test_profile_from_dict_rejects_invalid():
    with pytest.raises(ValueError):
        profile_from_dict({"name": "", "joints": {"names": ["a"]}})
