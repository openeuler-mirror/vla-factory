"""Model interface facts, built-in declarations, and registry extensions."""

from __future__ import annotations

from dataclasses import fields
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
from vla_factory.model.registry import (
    ModelEntry,
    ModelRegistry,
    RegistryLoadError,
    get_entry,
    list_entries,
)
from vla_factory.utils.vocabulary import CONTROL_MODES


def _plugin_factory(recipe, assembly):
    return "plugin-model"


@pytest.fixture
def isolated_model_registry():
    """Restore global registry state after an external-plugin test."""
    ModelRegistry._ensure_builtins_loaded()
    entries = dict(ModelRegistry._entries)
    plugins = set(ModelRegistry._plugins_loaded)
    builtins_loaded = ModelRegistry._builtins_loaded
    builtin_error = ModelRegistry._builtin_error
    yield
    ModelRegistry._entries.clear()
    ModelRegistry._entries.update(entries)
    ModelRegistry._plugins_loaded.clear()
    ModelRegistry._plugins_loaded.update(plugins)
    ModelRegistry._builtins_loaded = builtins_loaded
    ModelRegistry._builtin_error = builtin_error


def test_every_model_metadata_field_is_classified():
    """A new field must deliberately enter or stay out of interface checks."""
    non_interface_fields = {
        "backend",
        "action_head_type",
        "training_paradigm",
        "components",
        "support_lora",
        "support_full",
        "support_freeze",
        "install_hint",
        "params",
    }
    interface_fields = set(ModelMetadata.INTERFACE_FIELDS)
    declared_fields = {item.name for item in fields(ModelMetadata)}

    assert interface_fields.isdisjoint(non_interface_fields)
    assert interface_fields | non_interface_fields == declared_fields


def test_three_models_registered():
    entries = list_entries()
    assert set(entries) >= {"act", "pi0", "pi05"}


def test_act_contract_fields():
    act = list_entries()["act"]
    # ACT trains its own projection from scratch → flexible dim, vision from data.
    assert act.dim_policy == "flexible"
    assert act.vision_slots == ()
    assert act.image_input_range == (0.0, 1.0)
    assert act.image_normalize_mode == "imagenet"
    assert act.vector_normalization == "mean_std"
    assert act.requires_prompt is False
    assert act.control_mode_pref == ("joint_pos",)


def test_pi0_contract_fields():
    pi0 = list_entries()["pi0"]
    assert pi0.dim_policy == "padded_to_max"
    assert pi0.dim_policy_max == 32
    assert pi0.image_input_range == (-1.0, 1.0)
    assert pi0.image_normalize_mode is None       # SigLIP [-1,1], no ImageNet step
    assert pi0.vector_normalization == "mean_std"
    assert pi0.expected_hz == 50
    assert pi0.language_template == "{task}"
    # 3 fixed visual slots at 224×224.
    assert len(pi0.vision_slots) == 3
    assert all(s.resolution == (224, 224) for s in pi0.vision_slots)


def test_pi05_uses_quantile_normalization():
    pi05 = list_entries()["pi05"]
    # Only difference from pi0 in the contract: quantile vector normalization.
    assert pi05.vector_normalization == "quantile"
    assert pi05.dim_policy == "padded_to_max"
    assert len(pi05.vision_slots) == 3


def test_control_mode_pref_uses_shared_vocabulary():
    for name, meta in list_entries().items():
        for mode in meta.control_mode_pref:
            assert mode in CONTROL_MODES, f"{name}.control_mode_pref has non-vocab {mode!r}"


def test_external_model_entry_is_discovered_by_name(
    monkeypatch, isolated_model_registry
):
    from vla_factory.model import registry

    class EntryPoint:
        name = "_plugin-model"

        @staticmethod
        def load():
            return ModelEntry(ModelMetadata(name="_plugin-model"), _plugin_factory)

    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda *, group: (EntryPoint(),)
        if group == ModelRegistry.ENTRY_POINT_GROUP else (),
    )

    assert get_entry("_plugin-model").factory(None, None) == "plugin-model"


def test_plugin_name_must_match_its_metadata(
    monkeypatch, isolated_model_registry
):
    from vla_factory.model import registry

    class EntryPoint:
        name = "_plugin-model"

        @staticmethod
        def load():
            return ModelEntry(ModelMetadata(name="different-name"), _plugin_factory)

    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda *, group: (EntryPoint(),)
        if group == ModelRegistry.ENTRY_POINT_GROUP else (),
    )

    with pytest.raises(RegistryLoadError, match="metadata"):
        get_entry("_plugin-model")


PI0_CHECKPOINT_CONFIG = {
    "type": "pi0",
    "input_features": {
        "observation.images.base_0_rgb": {
            "type": "VISUAL", "shape": [3, 224, 224],
        },
        "observation.images.left_wrist_0_rgb": {
            "type": "VISUAL", "shape": [3, 224, 224],
        },
        "observation.state": {"type": "STATE", "shape": [32]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [32]}},
    "max_action_dim": 32,
    "image_resolution": [224, 224],
}


def _checkpoint_metadata() -> ModelMetadata:
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


def test_checkpoint_config_is_found_beside_weights(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(PI0_CHECKPOINT_CONFIG))
    weights_path = tmp_path / "model.safetensors"
    weights_path.write_bytes(b"weights")

    assert load_checkpoint_config(str(tmp_path)) == PI0_CHECKPOINT_CONFIG
    assert load_checkpoint_config(str(weights_path)) == PI0_CHECKPOINT_CONFIG


def test_checkpoint_observations_are_plain_diagnostics():
    observations = extract_checkpoint_observations(PI0_CHECKPOINT_CONFIG)
    assert observations["camera_roles"]["base_0_rgb"] == (3, 224, 224)
    assert observations["state_dim"] == 32
    assert observations["action_dim"] == 32
    assert observations["max_action_dim"] == 32


def test_compatible_checkpoint_does_not_change_metadata(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(PI0_CHECKPOINT_CONFIG))
    metadata = _checkpoint_metadata()

    result = validate_checkpoint_if_available(str(tmp_path), metadata)

    assert result["status"] == "compatible"
    assert result["observed"]["action_dim"] == 32
    assert metadata.action_dim == 32


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        (lambda config: config["output_features"]["action"].update(shape=[16]), "action_dim=16"),
        (lambda config: config.update(max_action_dim=16), "max_action_dim=16"),
        (
            lambda config: config["input_features"].update({
                "observation.images.unknown": {
                    "type": "VISUAL", "shape": [3, 224, 224],
                }
            }),
            "not declared",
        ),
        (
            lambda config: config["input_features"][
                "observation.images.base_0_rgb"
            ].update(shape=[3, 256, 256]),
            "resolution",
        ),
    ],
)
def test_checkpoint_contradictions_are_reported(mutation, fragment):
    config = json.loads(json.dumps(PI0_CHECKPOINT_CONFIG))
    mutation(config)

    issues = checkpoint_compatibility_issues(
        _checkpoint_metadata(), extract_checkpoint_observations(config)
    )

    assert any(fragment in issue for issue in issues)


def test_checkpoint_mismatch_raises_a_dedicated_error(tmp_path):
    config = json.loads(json.dumps(PI0_CHECKPOINT_CONFIG))
    config["output_features"]["action"]["shape"] = [16]
    (tmp_path / "config.json").write_text(json.dumps(config))

    with pytest.raises(CheckpointCompatibilityError, match="action_dim=16"):
        validate_checkpoint_if_available(str(tmp_path), _checkpoint_metadata())


def test_unknown_camera_convention_is_not_an_empty_camera_set():
    config = json.loads(json.dumps(PI0_CHECKPOINT_CONFIG))
    config["input_features"] = {
        "camera.front": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.state": {"type": "STATE", "shape": [32]},
    }
    observations = extract_checkpoint_observations(config)

    assert observations["camera_roles"] is None
    assert checkpoint_compatibility_issues(
        _checkpoint_metadata(), observations
    ) == []


def test_recognized_checkpoint_can_confirm_a_missing_camera_role():
    config = json.loads(json.dumps(PI0_CHECKPOINT_CONFIG))
    del config["input_features"]["observation.images.left_wrist_0_rgb"]

    issues = checkpoint_compatibility_issues(
        _checkpoint_metadata(), extract_checkpoint_observations(config)
    )

    assert any("required metadata camera roles missing" in issue for issue in issues)


def test_absent_checkpoint_camera_facts_do_not_fail_optional_check():
    config = {"output_features": {"action": {"shape": [32]}}}
    assert checkpoint_compatibility_issues(
        _checkpoint_metadata(), extract_checkpoint_observations(config)
    ) == []


def test_optional_checkpoint_check_reports_unavailable_states(tmp_path):
    assert validate_checkpoint_if_available(None, _checkpoint_metadata()) == {
        "status": "not_configured",
    }
    unavailable = validate_checkpoint_if_available(str(tmp_path), _checkpoint_metadata())
    assert unavailable["status"] == "unavailable"
    assert "config.json" in unavailable["detail"]

    (tmp_path / "config.json").write_text('{"max_action_dim": "not-an-int"}')
    malformed = validate_checkpoint_if_available(str(tmp_path), _checkpoint_metadata())
    assert malformed["status"] == "unavailable"
