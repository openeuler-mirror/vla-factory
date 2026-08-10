"""Golden-structure tests for the `inspect` CLI (WP6, architecture §3.5)."""

from __future__ import annotations

import yaml

from vla_factory.recipe import cli as cli_module

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
