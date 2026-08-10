"""Tests for the entry-table DataSchema: round-trip and derived compat props."""

from __future__ import annotations

from helpers import make_schema

from vla_factory.data.manifest import DataSchema


def test_to_dict_from_dict_round_trip():
    schema = make_schema(
        state_dim=3, action_dim=2,
        cameras=("front", "wrist"),
        image_sizes={"front": (480, 640), "wrist": (480, 640)},
        state_keys=("s0", "s1", "s2"),
        action_keys=("a0", "a1"),
        fps=30, has_language=True, total_episodes=5, total_frames=900,
        robot_type="so101_follower",
    )
    restored = DataSchema.from_dict(schema.to_dict())
    assert restored == schema


def test_derived_properties_match_legacy_fields():
    schema = make_schema(
        state_dim=6, action_dim=8,
        cameras=("front", "wrist"),
        image_sizes={"front": (480, 640), "wrist": (480, 640)},
        state_keys=("s0", "s1", "s2", "s3", "s4", "s5"),
        action_keys=tuple(f"a{i}" for i in range(8)),
        fps=20, has_language=True, total_episodes=3, total_frames=400,
        robot_type="lekiwi",
    )
    assert schema.state_dim == 6
    assert schema.action_dim == 8
    assert schema.cameras == ("front", "wrist")
    assert schema.image_sizes == {"front": (480, 640), "wrist": (480, 640)}
    assert schema.fps == 20
    assert schema.has_language is True
    assert schema.total_episodes == 3
    assert schema.total_frames == 400
    assert schema.robot_type == "lekiwi"
    assert schema.state_keys == tuple(f"s{i}" for i in range(6))
    assert schema.action_keys == tuple(f"a{i}" for i in range(8))


def test_unknown_robot_type_is_undeclared():
    schema = make_schema(state_dim=1, robot_type="unknown")
    assert schema.robot_ref is None
    assert schema.robot_type == "unknown"  # derived compat prop

