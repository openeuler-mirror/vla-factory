"""Tests for vla_factory.model.base_contract.

Covers config.json (lerobot/openpi pytorch format) parsing and camera_mapping
validation — the foundation of `vlafactory-cli list --config`.

Runnable both via pytest and directly: `python test/test_base_contract.py`.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_factory.model.base_contract import (  # noqa: E402
    check_camera_mapping,
    load_base_contract,
)

# Mirrors lerobot/pi0_base config.json structure.
PI0_BASE_CONFIG = {
    "type": "pi0",
    "input_features": {
        "observation.images.base_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.left_wrist_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.right_wrist_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.state": {"type": "STATE", "shape": [32]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [32]}},
    "max_action_dim": 32,
    "image_resolution": [224, 224],
}


def _write_config(cfg=None) -> str:
    d = tempfile.mkdtemp()
    Path(d, "config.json").write_text(json.dumps(cfg if cfg is not None else PI0_BASE_CONFIG))
    return d


def test_load_base_contract_parses_lerobot_format():
    c = load_base_contract(_write_config())
    assert c is not None
    assert c.camera_role_names == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    assert c.camera_roles["base_0_rgb"] == (3, 224, 224)
    assert c.state_dim == 32
    assert c.action_dim == 32
    assert c.max_action_dim == 32
    assert c.image_resolution == (224, 224)
    assert c.model_type == "pi0"


def test_load_base_contract_none_path_returns_none():
    assert load_base_contract(None) is None
    assert load_base_contract("") is None


def test_mapping_valid_with_placeholder():
    c = load_base_contract(_write_config())
    r = check_camera_mapping(
        {"base_0_rgb": "front", "left_wrist_0_rgb": "side"}, c, ["front", "side"]
    )
    assert ("base_0_rgb", "front") in r["mapped"]
    assert ("left_wrist_0_rgb", "side") in r["mapped"]
    assert r["empty"] == ["right_wrist_0_rgb"]
    assert r["errors"] == []
    assert r["unused"] == []


def test_mapping_illegal_role():
    c = load_base_contract(_write_config())
    r = check_camera_mapping({"base_0_rgb": "front", "bogus_role": "side"}, c, ["front", "side"])
    assert any("bogus_role" in e for e in r["errors"])


def test_mapping_missing_dataset_camera():
    c = load_base_contract(_write_config())
    r = check_camera_mapping({"base_0_rgb": "front", "left_wrist_0_rgb": "ghost"}, c, ["front"])
    assert any("ghost" in e for e in r["errors"])


def test_mapping_unused_dataset_camera():
    c = load_base_contract(_write_config())
    r = check_camera_mapping({"base_0_rgb": "front"}, c, ["front", "extra"])
    assert r["unused"] == ["extra"]


def test_mapping_no_contract_warns():
    r = check_camera_mapping({"base_0_rgb": "front"}, None, ["front"])
    assert r["errors"] == []
    assert r["warnings"]  # "model.path 未设置" nudge


if __name__ == "__main__":
    passed, failed = 0, 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"✓ {name}")
                passed += 1
            except AssertionError as e:
                print(f"✗ {name}: {e}")
                failed += 1
    print(f"\n{'✅ ALL PASS' if not failed else '❌ HAS FAILURES'} ({passed} passed, {failed} failed)")
    sys.exit(1 if failed else 0)
