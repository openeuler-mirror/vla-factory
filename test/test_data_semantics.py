"""Tests for the deterministic data-side inference rules (WP1, §8.5)."""

from __future__ import annotations

from vla_factory.data.semantics import infer_action_mode, infer_camera_semantic


class TestInferCameraSemantic:
    def test_unique_matches(self):
        assert infer_camera_semantic("front") == "third_person_front"
        assert infer_camera_semantic("observation.images.front") == "third_person_front"
        assert infer_camera_semantic("wrist") == "wrist"
        assert infer_camera_semantic("cam_left_wrist") == "wrist_left"
        assert infer_camera_semantic("cam_right_wrist") == "wrist_right"
        assert infer_camera_semantic("top") == "third_person_top"
        assert infer_camera_semantic("overhead_camera") == "third_person_top"
        assert infer_camera_semantic("head_camera") == "third_person_front"

    def test_zero_candidates_is_undeclared(self):
        # "cam_0" / "cam_1" carry no view evidence → None (needs controlled override).
        assert infer_camera_semantic("cam_0") is None
        assert infer_camera_semantic("observation.images.0") is None

    def test_multiple_candidates_is_undeclared(self):
        # Two competing third-person roles (neither excludes the other) → None.
        assert infer_camera_semantic("top_side") is None
        assert infer_camera_semantic("high_side") is None

    def test_wrist_takes_precedence_over_incidental_words(self):
        # A key that names a wrist view plus an incidental word still resolves
        # deterministically to the wrist role (third-person roles exclude wrist).
        assert infer_camera_semantic("wrist_top") == "wrist"

    def test_more_specific_directional_wrist_wins(self):
        assert infer_camera_semantic("left_wrist_top") == "wrist_left"

    def test_equal_priority_directional_roles_remain_ambiguous(self):
        assert infer_camera_semantic("left_right_wrist") is None

    def test_explicit_view_outranks_generic_front_wording(self):
        assert infer_camera_semantic("head_top") == "third_person_top"
        assert infer_camera_semantic("front_side") == "third_person_side"


class TestInferActionMode:
    def test_suffix_matches(self):
        assert infer_action_mode("shoulder_pan.pos") == "joint_pos"
        assert infer_action_mode("gripper.vel") == "joint_vel"
        assert infer_action_mode("elbow.delta") == "joint_delta"

    def test_no_evidence_is_undeclared(self):
        # Container-format placeholder names ("dim_0") carry no source evidence.
        assert infer_action_mode("dim_0") is None
        assert infer_action_mode("action_3") is None

    def test_empty_name(self):
        assert infer_action_mode("") is None
        assert infer_action_mode(None) is None  # type: ignore[arg-type]
