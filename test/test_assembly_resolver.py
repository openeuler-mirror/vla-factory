"""Tests for the phase-0 composition resolver (``resolve_assembly``).

These cover the stable contract: determinism, serializable round-trip, the
ModelMetadata × BaseContract capability-boundary conflict, and structured
``ResolutionError`` (asserting ``code`` / ``path`` / ``params`` — never the
full user-facing message).
"""

from __future__ import annotations

import pytest

from vla_factory.assembly.resolver import (
    INVALID_DESCRIPTION,
    METADATA_CONTRACT_CONFLICT,
    MISSING_INPUT,
    ResolutionError,
    resolve_assembly,
)
from vla_factory.data.manifest import DataSchema, NormStats
from vla_factory.model.base_contract import BaseContract
from vla_factory.model.interfaces.model import ModelMetadata
from vla_factory.robot import get_robot_profile


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def schema() -> DataSchema:
    return DataSchema(
        state_dim=6,
        action_dim=8,
        cameras=("front", "wrist"),
        fps=30,
        has_language=True,
    )


@pytest.fixture
def metadata() -> ModelMetadata:
    return ModelMetadata(
        name="act",
        action_dim=7,
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
    overrides = {"default_task": "pick", "accept_fps_mismatch": True}
    assembly = resolve_assembly(schema, norm_stats, metadata, overrides=overrides)
    assert assembly.overrides_ref == overrides
    # No overrides → empty ref, and it still round-trips.
    plain = resolve_assembly(schema, norm_stats, metadata)
    assert plain.overrides_ref == {}
    assert type(plain).from_dict(plain.to_dict()) == plain


def test_canonical_interface_derived_from_facts(schema, norm_stats, metadata):
    assembly = resolve_assembly(schema, norm_stats, metadata)
    ci = assembly.canonical_interface
    assert ci.cameras == schema.cameras
    assert ci.action_dim == metadata.action_dim
    assert ci.action_horizon == metadata.action_horizon
    assert ci.state_dim == schema.state_dim
    assert ci.requires_language is metadata.requires_prompt


def test_base_contract_refines_action_dim(schema, norm_stats, metadata):
    # Contract declares fewer dims than the model supports → allowed (padding).
    contract = BaseContract(action_dim=6)
    assembly = resolve_assembly(schema, norm_stats, metadata, base_contract=contract)
    assert assembly.canonical_interface.action_dim == 6
    assert assembly.contract_ref is not None


# ── Conflict ──────────────────────────────────────────────────────


def test_metadata_contract_action_dim_conflict(schema, norm_stats, metadata):
    # Contract declares MORE action dims than the model family supports.
    contract = BaseContract(action_dim=14)
    with pytest.raises(ResolutionError) as exc_info:
        resolve_assembly(schema, norm_stats, metadata, base_contract=contract)
    err = exc_info.value
    assert err.code == METADATA_CONTRACT_CONFLICT
    assert err.path == "model.action_dim"
    assert err.params["field"] == "action_dim"
    assert err.params["metadata_value"] == 7
    assert err.params["contract_value"] == 14


def test_no_conflict_when_metadata_action_dim_zero(schema, norm_stats):
    # ACT-like from-scratch model: metadata.action_dim == 0 means "from data".
    meta = ModelMetadata(name="act", action_dim=0)
    contract = BaseContract(action_dim=8)
    # Must not raise — no capability boundary to breach.
    assembly = resolve_assembly(schema, norm_stats, meta, base_contract=contract)
    assert assembly.canonical_interface.action_dim == 8


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


# ── Phase-0 placeholders ──────────────────────────────────────────


def test_mappings_and_pipelines_are_phase0_placeholders(schema, norm_stats, metadata):
    assembly = resolve_assembly(schema, norm_stats, metadata)
    for mapping in (
        assembly.camera_mapping,
        assembly.state_mapping,
        assembly.action_mapping,
        assembly.language_mapping,
        assembly.joint_mapping,
    ):
        assert mapping.resolved is False
        assert mapping.entries == ()
    for pipeline in (
        assembly.data_to_model,
        assembly.robot_to_model,
        assembly.model_to_robot,
    ):
        assert pipeline.resolved is False
        assert pipeline.steps == ()
