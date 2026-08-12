"""train → inference_metadata → InferenceEngine, on the real test dataset.

This is the only test that constructs a real ``InferenceEngine`` from a real
checkpoint, and it exists because that contract broke silently: after the
DataSchema entry-table refactor, ``InferenceEngine.__init__`` still tried to
``replace(schema, state_keys=...)`` — fields that had become derived properties
— so *every* checkpoint failed to load while the whole suite stayed green.

It also pins where the shapes come from at both ends: the recipe deliberately
disagrees with the dataset about ``action_dim`` (a deprecated field nothing
reads any more), and the composition's direct answer for flexible ACT — the
dataset width — must be what the head is built with, what the saved assembly
states, and what the engine serves.
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
    assert (meta_dir / "assembly.json").exists()
    assert (meta_dir / "recipe.yaml").exists()
    assert (meta_dir / "schema.json").exists()
    assert (meta_dir / "norm_stats.json").exists()

    # Drive it from the saved recipe alone, the way `vlafactory-cli infer`
    # does when no --config is given: the checkpoint must be self-describing.
    result = infer_from_dataset_sample(
        config=meta_dir / "recipe.yaml", checkpoint=output_dir,
        dataset_index=0, split="val", device="cpu",
    )

    # The resolved execution width is 8 (the dataset's), despite the recipe's 6;
    # ACT is from_scratch, so its declared/overridden tunable owns the horizon.
    assert result["action_shape"] == (ACTION_HORIZON, DATASET_ACTION_DIM)
    assert result["target_shape"] == (ACTION_HORIZON, DATASET_ACTION_DIM)


def test_engine_serves_the_saved_assembly(tmp_path):
    """The engine executes ``assembly.json`` and refuses a checkpoint without it.

    A checkpoint from before the artifact existed cannot be served by
    re-resolving here: that would resolve against the model declaration
    installed *now*, and a drifted image range or normalization method loads its
    weights perfectly and simply behaves wrongly.
    """
    from vla_factory.assembly.artifact import load_assembly_artifact
    from vla_factory.inference.infer import InferenceEngine
    from vla_factory.training.train import train

    output_dir = tmp_path / "run"
    train(_recipe(output_dir))
    assembly_file = output_dir / "inference_metadata" / "assembly.json"

    engine = InferenceEngine(checkpoint_path=output_dir, device="cpu")
    saved = load_assembly_artifact(assembly_file)
    assert (
        engine.execution_action_dim
        == saved.model_io_spec.action_dim
        == DATASET_ACTION_DIM
    )
    assert engine.action_horizon == saved.model_io_spec.action_horizon == ACTION_HORIZON
    assert engine.camera_keys == tuple(saved.model_io_spec.cameras)
    # Preprocessor and postprocessor are the two planned pipelines, executed —
    # the reverse one is not the forward list reversed.
    assert len(engine.preprocessor) == len(saved.data_to_model.calls)
    assert len(engine.postprocessor) == len(saved.model_to_robot.calls)

    assembly_file.unlink()
    with pytest.raises(FileNotFoundError, match="assembly.json"):
        InferenceEngine(checkpoint_path=output_dir, device="cpu")


def test_camera_keys_cannot_be_renamed_at_deploy_time(tmp_path):
    """There is no camera-name override.

    Renaming them here would leave the camera mapping pointing at names the
    observation no longer has: ACT would raise on the missing key, and pi0 would
    quietly feed every slot its placeholder image and keep predicting, blind.
    """
    import inspect

    from vla_factory.inference.infer import InferenceEngine

    assert "camera_names" not in inspect.signature(InferenceEngine.__init__).parameters


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
        "image_to_float", "image_layout", "resize_images", "image_normalize",
        "normalize_vector", "pad_dimensions",
    ]
    # Recipe overrides still win.
    assert model_config["dim_model"] == 64
