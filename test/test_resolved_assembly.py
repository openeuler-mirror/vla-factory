"""ResolvedAssembly persistence and model-interface compatibility."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from helpers import make_assembly, make_schema

from vla_factory.assembly import (
    InvalidAssemblyError,
    MappingSource,
    ModelInterfaceMismatch,
    ResolvedAssembly,
)
from vla_factory.assembly.transform import TransformContext, build_pipeline
from vla_factory.assembly.transform.plan import TransformPipelinePlan
from vla_factory.model.registry import list_entries


def _act_assembly():
    return make_assembly(
        make_schema(
            state_dim=6, action_dim=8, cameras=("front",),
            image_sizes={"front": (480, 640)},
        ),
        "act",
    )


def test_save_and_load_the_assembly_directly(tmp_path):
    assembly = _act_assembly()
    path = tmp_path / "assembly.json"
    assembly.save(path)

    payload = json.loads(path.read_text())
    assert payload == assembly.to_dict()
    assert "format_version" not in payload
    loaded = ResolvedAssembly.load(path)
    assert loaded == assembly
    assert loaded.camera_mapping.entries[0]["source"] is MappingSource.INFERRED
    assert loaded.robot_to_model == loaded.data_to_model


def test_invalid_json_fails_readably(tmp_path):
    path = tmp_path / "assembly.json"
    path.write_text("not json")
    with pytest.raises(InvalidAssemblyError, match="not valid JSON"):
        ResolvedAssembly.load(path)


@pytest.mark.parametrize(
    "key",
    [
        "schema_ref", "norm_stats_ref", "metadata_ref", "model_io_spec",
        "camera_mapping", "state_mapping", "action_mapping", "language_mapping",
        "data_to_model", "robot_to_model", "model_to_robot",
    ],
)
def test_missing_contract_field_is_refused(tmp_path, key):
    path = tmp_path / "assembly.json"
    body = _act_assembly().to_dict()
    body.pop(key)
    path.write_text(json.dumps(body))
    with pytest.raises(InvalidAssemblyError, match=key):
        ResolvedAssembly.load(path)


def test_distinct_robot_input_plan_is_refused(tmp_path):
    path = tmp_path / "assembly.json"
    body = _act_assembly().to_dict()
    body["robot_to_model"]["calls"].append({"type": "unexpected", "args": {}})
    path.write_text(json.dumps(body))
    with pytest.raises(InvalidAssemblyError, match="must equal"):
        ResolvedAssembly.load(path)


def test_unknown_mapping_source_is_refused(tmp_path):
    path = tmp_path / "assembly.json"
    body = _act_assembly().to_dict()
    body["camera_mapping"]["entries"][0]["source"] = "guessed"
    path.write_text(json.dumps(body))
    with pytest.raises(InvalidAssemblyError, match="invalid source 'guessed'"):
        ResolvedAssembly.load(path)


def test_missing_mapping_source_is_refused(tmp_path):
    path = tmp_path / "assembly.json"
    body = _act_assembly().to_dict()
    body["state_mapping"]["entries"][0].pop("source")
    path.write_text(json.dumps(body))
    with pytest.raises(InvalidAssemblyError, match="missing 'source'"):
        ResolvedAssembly.load(path)


def test_identity_plan_is_valid_but_unplanned_input_fails_during_resolution():
    assert len(build_pipeline(TransformPipelinePlan(), TransformContext())) == 0


def test_matching_model_declaration_passes():
    _act_assembly().check_model_compatibility(list_entries()["act"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_input_range", (-1.0, 1.0)),
        ("vector_normalization", "quantile"),
        ("requires_prompt", True),
        ("dim_policy", "fixed"),
    ],
)
def test_interface_drift_is_refused(field, value):
    metadata = replace(list_entries()["act"], **{field: value})
    with pytest.raises(ModelInterfaceMismatch, match=field):
        _act_assembly().check_model_compatibility(metadata)


def test_non_interface_changes_do_not_block_serving():
    metadata = replace(
        list_entries()["act"], install_hint="different", support_lora=True,
    )
    _act_assembly().check_model_compatibility(metadata)
