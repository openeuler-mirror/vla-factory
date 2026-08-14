"""Tests for optional checkpoint consistency validation."""

from __future__ import annotations

import json

import pytest

from vla_factory.model.checkpoint_validation import (
    CheckpointCompatibilityError,
    checkpoint_compatibility_issues,
    extract_checkpoint_observations,
    load_checkpoint_config,
    validate_checkpoint_if_available,
)
from vla_factory.model.model_interface import ModelMetadata, VisionSlot


PI0_CONFIG = {
    "type": "pi0",
    "input_features": {
        "observation.images.base_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.left_wrist_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.state": {"type": "STATE", "shape": [32]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [32]}},
    "max_action_dim": 32,
    "image_resolution": [224, 224],
}


def _metadata() -> ModelMetadata:
    return ModelMetadata(
        name="pi0",
        action_dim=32,
        dim_policy="padded_to_max",
        dim_policy_max=32,
        vision_slots=(
            VisionSlot(name="base_0_rgb", resolution=(224, 224)),
            VisionSlot(name="left_wrist_0_rgb", resolution=(224, 224)),
        ),
    )


def test_loads_directory_and_weight_sibling_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(PI0_CONFIG))
    weights_path = tmp_path / "model.safetensors"
    weights_path.write_bytes(b"weights")
    assert load_checkpoint_config(str(tmp_path)) == PI0_CONFIG
    assert load_checkpoint_config(str(weights_path)) == PI0_CONFIG


def test_extracts_plain_diagnostic_observations():
    observations = extract_checkpoint_observations(PI0_CONFIG)
    assert observations["camera_roles"]["base_0_rgb"] == (3, 224, 224)
    assert observations["state_dim"] == 32
    assert observations["action_dim"] == 32
    assert observations["max_action_dim"] == 32


def test_compatible_checkpoint_does_not_change_metadata(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(PI0_CONFIG))
    metadata = _metadata()
    before = metadata
    result = validate_checkpoint_if_available(str(tmp_path), metadata)
    assert result["status"] == "compatible"
    assert result["observed"]["action_dim"] == 32
    assert metadata is before
    assert metadata.action_dim == 32


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        (lambda config: config["output_features"]["action"].update(shape=[16]), "action_dim=16"),
        (lambda config: config.update(max_action_dim=16), "max_action_dim=16"),
        (
            lambda config: config["input_features"].update(
                {"observation.images.unknown": {"type": "VISUAL", "shape": [3, 224, 224]}}
            ),
            "not declared",
        ),
        (
            lambda config: config["input_features"]["observation.images.base_0_rgb"].update(
                shape=[3, 256, 256]
            ),
            "resolution",
        ),
    ],
)
def test_reports_checkpoint_contradictions(mutation, fragment):
    config = json.loads(json.dumps(PI0_CONFIG))
    mutation(config)
    issues = checkpoint_compatibility_issues(_metadata(), extract_checkpoint_observations(config))
    assert any(fragment in issue for issue in issues)


def test_validate_raises_dedicated_error_for_mismatch(tmp_path):
    config = json.loads(json.dumps(PI0_CONFIG))
    config["output_features"]["action"]["shape"] = [16]
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(CheckpointCompatibilityError) as exc:
        validate_checkpoint_if_available(str(tmp_path), _metadata())
    assert "action_dim=16" in str(exc.value)


def test_unrecognized_camera_convention_is_not_treated_as_an_empty_set():
    config = json.loads(json.dumps(PI0_CONFIG))
    config["input_features"] = {
        "camera.front": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.state": {"type": "STATE", "shape": [32]},
    }
    observations = extract_checkpoint_observations(config)
    assert observations["camera_roles"] is None
    issues = checkpoint_compatibility_issues(_metadata(), observations)
    assert issues == []


def test_recognized_camera_roles_can_confirm_required_slots_are_missing():
    config = json.loads(json.dumps(PI0_CONFIG))
    del config["input_features"]["observation.images.left_wrist_0_rgb"]
    issues = checkpoint_compatibility_issues(_metadata(), extract_checkpoint_observations(config))
    assert any("required metadata camera roles missing" in issue for issue in issues)


def test_absent_camera_facts_do_not_fail_optional_check():
    config = {"output_features": {"action": {"shape": [32]}}}
    assert checkpoint_compatibility_issues(
        _metadata(), extract_checkpoint_observations(config)
    ) == []


def test_optional_entry_reports_not_configured_and_unavailable(tmp_path):
    assert validate_checkpoint_if_available(None, _metadata()) == {
        "status": "not_configured",
    }
    result = validate_checkpoint_if_available(str(tmp_path), _metadata())
    assert result["status"] == "unavailable"
    assert "config.json" in result["detail"]

    (tmp_path / "config.json").write_text('{"max_action_dim": "not-an-int"}')
    malformed = validate_checkpoint_if_available(str(tmp_path), _metadata())
    assert malformed["status"] == "unavailable"
