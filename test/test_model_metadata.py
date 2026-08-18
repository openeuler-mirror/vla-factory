"""Contract tests for the three registered models' ModelMetadata (§4.3)."""

from __future__ import annotations

from dataclasses import fields

from vla_factory.model.model_interface import ModelMetadata
from vla_factory.model.registry import list_entries
from vla_factory.utils.vocabulary import CONTROL_MODES


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
