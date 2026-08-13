"""Tests for the phase-0 composition resolver (``resolve_assembly``).

These cover the stable contract: determinism, serializable round-trip, and structured
``ResolutionError`` (asserting ``code`` / ``path`` / ``params`` — never the
full user-facing message).
"""

from __future__ import annotations
from helpers import make_schema

import pytest

from vla_factory.assembly.resolver import (
    INVALID_DESCRIPTION,
    MISSING_INPUT,
    ResolutionError,
    UNSUPPORTED_OVERRIDE,
    resolve_assembly,
)
from vla_factory.data.manifest import DataSchema, NormStats
from vla_factory.model.interfaces.model import ModelMetadata
from vla_factory.robot import get_robot_profile


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def schema() -> DataSchema:
    # Per-dim names are the LeKiwi motor keys, so this schema composes with the
    # ``robot`` fixture below: the joint-order check embeds the dataset's dims
    # into the robot's joint list, and names that match nothing would fail it.
    return make_schema(
        state_dim=6,
        action_dim=8,
        cameras=("front", "wrist"),
        fps=30,
        has_language=True,
        state_keys=("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"),
        action_keys=("base_x", "base_y", "base_z",
                     "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                     "wrist_flex.pos", "wrist_roll.pos"),
    )


@pytest.fixture
def metadata() -> ModelMetadata:
    return ModelMetadata(
        name="act",
        action_horizon=100,
        requires_prompt=False,
    )


@pytest.fixture
def norm_stats() -> NormStats:
    return NormStats()


@pytest.fixture
def robot():
    return get_robot_profile("lekiwi")


# ── Determinism + round-trip ──────────────────────────────────────


def test_same_inputs_same_output(schema, norm_stats, metadata, robot):
    a = resolve_assembly(schema, norm_stats, metadata, robot_profile=robot)
    b = resolve_assembly(schema, norm_stats, metadata, robot_profile=robot)
    assert a.to_dict() == b.to_dict()


def test_to_dict_from_dict_round_trip(schema, norm_stats, metadata, robot):
    assembly = resolve_assembly(
        schema, norm_stats, metadata, robot_profile=robot,
        overrides={"default_task": "pick"},
    )
    restored = type(assembly).from_dict(assembly.to_dict())
    assert restored == assembly


def test_overrides_are_recorded(schema, norm_stats, metadata):
    # Architecture §3.4: the assembly records the controlled overrides applied
    # (the source of each adjusted field), rather than silently dropping them.
    overrides = {"default_task": "pick"}
    assembly = resolve_assembly(schema, norm_stats, metadata, overrides=overrides)
    assert assembly.overrides_ref == overrides
    # No overrides → empty ref, and it still round-trips.
    plain = resolve_assembly(schema, norm_stats, metadata)
    assert plain.overrides_ref == {}
    assert type(plain).from_dict(plain.to_dict()) == plain


def test_model_io_spec_derived_from_facts(schema, norm_stats, metadata):
    assembly = resolve_assembly(schema, norm_stats, metadata)
    ci = assembly.model_io_spec
    assert ci.cameras == schema.cameras
    assert ci.action_dim == schema.action_dim
    assert ci.action_horizon == metadata.action_horizon
    assert ci.state_dim == schema.state_dim
    assert ci.requires_language is metadata.requires_prompt


def test_serialized_assembly_has_metadata_as_only_model_interface_source(
    schema, norm_stats, metadata,
):
    serialized = resolve_assembly(schema, norm_stats, metadata).to_dict()
    assert serialized["metadata_ref"]["action_dim"] == 0
    assert "contract_ref" not in serialized


def test_flexible_model_cannot_also_declare_a_fixed_action_width(
    schema, norm_stats,
):
    metadata = ModelMetadata(name="broken", action_dim=7, action_horizon=1)
    with pytest.raises(ValueError, match="dim_policy='flexible'.*action_dim=7"):
        resolve_assembly(schema, norm_stats, metadata)


# ── Missing input ─────────────────────────────────────────────────


def test_missing_schema(norm_stats, metadata):
    with pytest.raises(ResolutionError) as exc_info:
        resolve_assembly(None, norm_stats, metadata)
    err = exc_info.value
    assert err.code == MISSING_INPUT
    assert err.path == "schema"
    assert err.params["field"] == "schema"


def test_missing_norm_stats(schema, metadata):
    with pytest.raises(ResolutionError) as exc_info:
        resolve_assembly(schema, None, metadata)
    assert exc_info.value.code == MISSING_INPUT
    assert exc_info.value.path == "norm_stats"


def test_missing_metadata(schema, norm_stats):
    with pytest.raises(ResolutionError) as exc_info:
        resolve_assembly(schema, norm_stats, None)
    assert exc_info.value.code == MISSING_INPUT
    assert exc_info.value.path == "model"


# ── Invalid description ───────────────────────────────────────────


def test_wrong_schema_type(norm_stats, metadata):
    with pytest.raises(ResolutionError) as exc_info:
        resolve_assembly({"not": "a schema"}, norm_stats, metadata)
    err = exc_info.value
    assert err.code == INVALID_DESCRIPTION
    assert err.path == "schema"
    assert err.params["field"] == "schema"


def test_invalid_robot_profile(schema, norm_stats, metadata):
    from vla_factory.robot.profile import RobotProfile, JointGroup

    bad = RobotProfile(name="bad", joints=JointGroup(names=()))
    with pytest.raises(ResolutionError) as exc_info:
        resolve_assembly(schema, norm_stats, metadata, robot_profile=bad)
    err = exc_info.value
    assert err.code == INVALID_DESCRIPTION
    assert err.path.startswith("robot")


def test_override_without_a_consumer_is_rejected(schema, norm_stats, metadata):
    """``gripper_flip`` is a documented recipe field that no stage reads yet.
    Dropping it silently would let a user believe a gripper decision was made
    on their behalf."""
    with pytest.raises(ResolutionError) as exc_info:
        resolve_assembly(schema, norm_stats, metadata,
                         overrides={"gripper_flip": True})
    err = exc_info.value.to_dict()
    assert err["code"] == UNSUPPORTED_OVERRIDE
    assert err["path"] == "assembly"
    assert err["params"] == {
        "keys": ["gripper_flip"],
        "supported": ["camera_mapping", "default_task"],
    }


def test_every_assembly_override_is_accounted_for():
    """Drift guard: every AssemblyConfig field must be consumed by the resolver.

    A controlled override nothing reads is a field a user can set and watch do
    nothing — which is why the two that had no consumer (frequency, gripper) were
    removed rather than left documented and inert. Add a field in the same commit
    that starts reading it.
    """
    from dataclasses import fields

    from vla_factory.assembly.resolver.resolver import CONSUMED_OVERRIDES
    from vla_factory.recipe.recipe import AssemblyConfig

    declared = {f.name for f in fields(AssemblyConfig)}
    assert declared == set(CONSUMED_OVERRIDES)


# ── What stays unresolved after phase 3 ───────────────────────────


def test_robot_side_stays_unresolved_without_a_robot(schema, norm_stats, metadata):
    """No robot declared (every example recipe today) → the two robot-side
    products are absent, not wrong. JointMapping needs a body to map onto, and
    ``robot_to_model`` needs the joint-reorder step that does not exist yet.
    """
    assembly = resolve_assembly(schema, norm_stats, metadata)
    assert assembly.joint_mapping.resolved is False
    assert assembly.joint_mapping.entries == ()
    assert assembly.robot_to_model.resolved is False
    assert assembly.robot_to_model.calls == ()


def test_pipelines_unresolved_when_the_model_declares_no_transforms(
    schema, norm_stats, metadata,
):
    """This ``metadata`` fixture carries no ``params["transforms"]``, so there
    is no step list to compile — the plan stays unresolved rather than being
    invented from the facts."""
    assembly = resolve_assembly(schema, norm_stats, metadata)
    assert assembly.data_to_model.resolved is False
    assert assembly.model_to_robot.resolved is False
