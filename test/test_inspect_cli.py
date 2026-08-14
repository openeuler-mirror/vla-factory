"""Golden-structure tests for the `inspect` CLI (WP6, architecture §3.5)."""

from __future__ import annotations

import json
import logging
import yaml

from helpers import make_schema
from vla_factory import cli as cli_module
from vla_factory.recipe import ModelConfig, TrainRecipe

DATA_PATH = "test/data/lerobot_train_data_3_episodes"


def _emit(dimension):
    """Helper: capture one inspect envelope as a dict (YAML round-trip)."""
    return dimension  # placeholder for readability


def test_inspect_data_envelope(capsys):
    cli_module._inspect_data(DATA_PATH, "lerobot-v3", with_stats=False, as_json=False)
    out = capsys.readouterr().out
    doc = yaml.safe_load(out)
    assert doc["dimension"] == "data"
    assert "facts" in doc and "schema" in doc["facts"]
    cams = doc["facts"]["schema"]["observation"]["cameras"]
    # Per-fact source labels present; front uniquely inferred.
    by_key = {c["key"]: c for c in cams}
    assert by_key["front"]["semantic"] == "third_person_front"
    assert by_key["front"]["semantic_source"] == "inferred"
    # Summary stats block present (not full values) by default.
    assert "norm_stats_summary" in doc["facts"]


def test_inspect_model_envelope(capsys):
    cli_module._inspect_model("act", path=None, as_json=True)
    out = capsys.readouterr().out
    import json
    doc = json.loads(out)
    assert doc["dimension"] == "model"
    assert doc["source"] == "metadata"
    meta = doc["facts"]["metadata"]
    assert meta["name"] == "act"
    # New interface-contract fields are surfaced (model-module §4.3).
    assert meta["dim_policy"] == "flexible"
    assert meta["vector_normalization"] == "mean_std"


def test_inspect_model_keeps_checkpoint_observations_separate(tmp_path, capsys):
    config = {
        "input_features": {
            f"observation.images.{role}": {"type": "VISUAL", "shape": [3, 224, 224]}
            for role in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        "output_features": {"action": {"shape": [32]}},
        "max_action_dim": 32,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    cli_module._inspect_model("pi0", path=str(tmp_path), as_json=True)
    doc = json.loads(capsys.readouterr().out)
    assert doc["source"] == "metadata"
    assert doc["facts"]["metadata"]["action_dim"] == 32
    assert doc["facts"]["checkpoint_check"]["status"] == "compatible"
    assert doc["facts"]["checkpoint_check"]["observed"]["action_dim"] == 32


def test_model_report_uses_resolver_camera_validation_for_dynamic_slots():
    """A model with no declared slots (ACT) has no slot vocabulary to check the
    override against, so only the camera half is validated — the report must not
    invent an error for a slot name it cannot know."""
    from vla_factory.recipe import AssemblyOverrides

    recipe = TrainRecipe(
        model=ModelConfig(name="act"),
        overrides=AssemblyOverrides(camera_mapping={"dynamic_slot": "front"}),
    )
    report = cli_module._describe_model_config(
        recipe, make_schema(cameras=("front", "wrist")),
    )
    assert "ERROR:" not in report
    assert "dynamic_slot" in report


def test_inspect_robot_envelope(capsys):
    cli_module._inspect_robot("lekiwi", as_json=False)
    out = capsys.readouterr().out
    doc = yaml.safe_load(out)
    assert doc["dimension"] == "robot"
    assert doc["source"] == "declared"
    assert doc["facts"]["name"] == "lekiwi"
    assert len(doc["facts"]["joints"]["names"]) == 9


def test_inspect_data_key_order_is_stable(capsys):
    """Deterministic key order → diffable golden output (architecture §3.5)."""
    cli_module._inspect_data(DATA_PATH, "lerobot-v3", with_stats=False, as_json=False)
    first = capsys.readouterr().out
    cli_module._inspect_data(DATA_PATH, "lerobot-v3", with_stats=False, as_json=False)
    second = capsys.readouterr().out
    assert first == second
