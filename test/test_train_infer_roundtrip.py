"""train → inference_metadata → InferenceEngine, on the real test dataset.

This is the only test that constructs a real ``InferenceEngine`` from a real
checkpoint, and it exists because that contract broke silently: after the
DataSchema entry-table refactor, ``InferenceEngine.__init__`` still tried to
``replace(schema, state_keys=...)`` — fields that had become derived properties
— so *every* checkpoint failed to load while the whole suite stayed green.

It also pins the action-fact routing at both ends: the recipe deliberately
disagrees with the dataset about ``action_dim``, and the dataset must win in the
trained head, in the saved metadata and in the engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

DATASET_PATH = _project_root / "test/data" / "lerobot_train_data_3_episodes"
DATASET_ACTION_DIM = 8      # what the dataset actually carries
RECIPE_ACTION_DIM = 6       # deliberately wrong, to prove the dataset wins
ACTION_HORIZON = 4


def _lerobot_available() -> bool:
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _lerobot_available(), reason="lerobot not installed"),
    pytest.mark.skipif(not DATASET_PATH.exists(), reason="test dataset not found"),
]


def _recipe(output_dir: Path):
    from vla_factory.recipe.parser import parse_recipe_from_string

    return parse_recipe_from_string(f"""
model:
  name: act
  config:
    dim_model: 64
    n_heads: 1
    dim_feedforward: 64
    n_encoder_layers: 1
    n_decoder_layers: 1
    n_vae_encoder_layers: 1
action_spec:
  action_dim: {RECIPE_ACTION_DIM}
  action_horizon: {ACTION_HORIZON}
data:
  source:
    path: {DATASET_PATH}
    format: lerobot-v3
  sampler:
    type: sliding_window
    n_obs_steps: 1
    action_horizon: {ACTION_HORIZON}
  split:
    strategy: episode
    train_ratio: 0.9
    seed: 42
finetuning:
  strategy: full
training:
  batch_size: 1
  total_steps: 1
  num_workers: 0
  lr: 1.0e-5
output:
  output_dir: {output_dir}
  overwrite_output_dir: true
  report_to: none
  save_steps: 1000
""")


def test_train_then_infer_roundtrip(tmp_path):
    from vla_factory.training.train import train
    from vla_factory.inference.infer import infer_from_dataset_sample

    output_dir = tmp_path / "run"
    train(_recipe(output_dir))

    meta_dir = output_dir / "inference_metadata"
    assert (meta_dir / "recipe.yaml").exists()
    assert (meta_dir / "schema.json").exists()
    assert (meta_dir / "norm_stats.json").exists()

    # Drive it from the saved recipe alone, the way `vlafactory-cli infer`
    # does when no --config is given: the checkpoint must be self-describing.
    result = infer_from_dataset_sample(
        config=meta_dir / "recipe.yaml", checkpoint=output_dir,
        dataset_index=0, split="val", device="cpu",
    )

    # The dataset owns action_dim, so the head is 8 wide despite the recipe's 6;
    # ACT is from_scratch, so the recipe owns the horizon.
    assert result["action_shape"] == (ACTION_HORIZON, DATASET_ACTION_DIM)
    assert result["target_shape"] == (ACTION_HORIZON, DATASET_ACTION_DIM)


def test_saved_recipe_is_resolved_and_self_contained(tmp_path):
    """The saved recipe carries the merged declaration, not the sparse input.

    Inference reads it without re-resolving, which is what keeps a checkpoint
    from silently drifting when a model's declared defaults change later.
    """
    import yaml

    from vla_factory.training.train import train

    output_dir = tmp_path / "run"
    train(_recipe(output_dir))

    saved = yaml.safe_load((output_dir / "inference_metadata" / "recipe.yaml").read_text())
    model_config = saved["model"]["config"]

    # Declared defaults the recipe never mentioned are present.
    assert model_config["kl_weight"] == 10.0
    assert model_config["num_inference_steps"] == 1
    assert [s["type"] for s in model_config["transforms"]["inputs"]] == [
        "image_to_float", "image_layout", "image_normalize",
        "normalize_vector", "pad_dimensions",
    ]
    # Recipe overrides still win.
    assert model_config["dim_model"] == 64
