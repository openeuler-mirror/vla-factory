"""Tests for the shared cross-dimension vocabularies (WP0)."""

from __future__ import annotations

import pytest

from vla_factory.utils import vocabulary as vocab
from vla_factory.utils.vocabulary import (
    ACTION_HEADS,
    CAMERA_SEMANTICS,
    CONTROL_MODES,
    DATA_SOURCES,
    MODEL_SOURCES,
    is_action_head,
    is_control_mode,
)
from vla_factory.robot.profile import RobotProfile, JointGroup


def test_control_modes_joint_space_only():
    # D4: first version is joint-space only; EEF modes + tokenized removed,
    # delta_joint renamed to joint_delta.
    assert CONTROL_MODES == ("joint_pos", "joint_delta", "joint_vel")
    assert "delta_joint" not in CONTROL_MODES   # renamed
    assert "tokenized" not in CONTROL_MODES      # it's an action_head, not a mode
    for removed in ("delta_eef", "se3"):
        assert removed not in CONTROL_MODES     # EEF group deferred


def test_camera_semantics_and_action_heads_present():
    assert "wrist" in CAMERA_SEMANTICS
    assert "third_person" in CAMERA_SEMANTICS  # generalization
    assert ACTION_HEADS == ("flow_matching", "diffusion", "autoregressive", "regression")
    assert is_action_head("flow_matching") and not is_action_head("tokenized")


def test_source_annotation_vocabularies():
    assert DATA_SOURCES == frozenset({"measured", "inferred", "undeclared"})
    assert MODEL_SOURCES == frozenset({"metadata", "base_contract"})


def test_robot_profile_uses_shared_control_mode_vocabulary():
    # Single-place definition: robot/profile must not re-declare the vocabulary.
    # It must reject exactly what the shared vocabulary rejects.
    import vla_factory.robot.profile as profile_mod

    assert not hasattr(profile_mod, "_CONTROL_MODES"), (
        "robot/profile.py must not keep a local _CONTROL_MODES; import from "
        "vla_factory.utils.vocabulary instead."
    )
    assert profile_mod.CONTROL_MODES is vocab.CONTROL_MODES


@pytest.mark.parametrize("mode", ["joint_pos", "joint_delta", "joint_vel"])
def test_profile_accepts_valid_control_modes(mode):
    RobotProfile(
        name="t",
        joints=JointGroup(names=("a",)),
        native_action_type=mode,
        control_modes=(mode,),
    ).validate()


@pytest.mark.parametrize("mode", ["delta_joint", "delta_eef", "se3", "tokenized", "nonsense"])
def test_profile_rejects_invalid_control_modes(mode):
    with pytest.raises(ValueError):
        RobotProfile(
            name="t",
            joints=JointGroup(names=("a",)),
            native_action_type=mode,
        ).validate()
