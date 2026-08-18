"""``resolve_assembly`` — the one place a recipe becomes a composition.

Training, inference and ``vlafactory-cli resolve`` all enter here, so what it
gathers (and in which order) is the contract these tests pin down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vla_factory.assembly import resolve_assembly
from vla_factory.assembly.resolve import (
    MISSING_INPUT, UNKNOWN_MODEL, UNKNOWN_ROBOT, ResolutionError,
)
from vla_factory.user_interface import merge_model_config, parse_recipe_from_string

DATASET_PATH = _project_root / "test/data" / "lerobot_train_data_3_episodes"


def _recipe(body: str):
    return merge_model_config(parse_recipe_from_string(body))


def _act_recipe(extra: str = ""):
    return _recipe(f"""
model:
  name: act
data:
  path: {DATASET_PATH}
  format: lerobot-v3
{extra}
""")


def test_model_name_shorthand_is_kept():
    recipe = parse_recipe_from_string("model: act")
    assert recipe.model.name == "act"
    assert recipe.model.path is None


def test_model_path_shorthand_uses_the_last_segment_as_name():
    recipe = parse_recipe_from_string("model: lerobot/pi0")
    assert recipe.model.name == "pi0"
    assert recipe.model.path == "lerobot/pi0"


def test_model_path_shorthand_rejects_a_trailing_separator():
    with pytest.raises(ValueError, match="must not end"):
        parse_recipe_from_string("model: lerobot/pi0/")


def test_explicit_model_mapping_is_the_escape_hatch():
    recipe = parse_recipe_from_string("""
model:
  name: local-pi0
  path: /checkpoints/pi0
""")
    assert recipe.model.name == "local-pi0"
    assert recipe.model.path == "/checkpoints/pi0"


@pytest.mark.parametrize(
    "body,path",
    [
        ("model: act\nunknown: true", "recipe"),
        ("model: act\ndata:\n  split: {}", "data"),
        ("model: act\nrobot:\n  port: 1234", "robot"),
        ("model: act\ntraining:\n  epochs: 2", "training"),
        ("model: act\noutput:\n  log_dir: logs", "output"),
    ],
)
def test_closed_recipe_blocks_reject_unknown_fields(body, path):
    with pytest.raises(ValueError, match=f"Unknown {path} field"):
        parse_recipe_from_string(body)


def test_legacy_assembly_block_is_not_accepted():
    with pytest.raises(ValueError, match="Unknown recipe field.*assembly"):
        parse_recipe_from_string("model: act\nassembly: {}")


requires_dataset = pytest.mark.skipif(
    not DATASET_PATH.exists(), reason="test dataset not found"
)


@requires_dataset
def test_resolves_a_real_recipe():
    assembly = resolve_assembly(_act_recipe())
    assert assembly.model_io_spec.action_dim == 8
    assert assembly.model_io_spec.action_horizon == 100   # ACT's declared default
    # The descriptions travel inside the assembly, typed on demand.
    assert assembly.schema.cameras == ("front", "wrist")
    assert assembly.norm_stats.action is not None


@requires_dataset
def test_overrides_are_collected_from_the_overrides_block():
    recipe = _act_recipe("""
overrides:
  default_task: "pick up the block"
""")
    assembly = resolve_assembly(recipe)
    assert assembly.overrides_ref == {"default_task": "pick up the block"}


def test_unknown_override_is_rejected_by_the_parser():
    with pytest.raises(ValueError, match="Unknown overrides field.*gripper_flip"):
        parse_recipe_from_string("""
model:
  name: act
overrides:
  gripper_flip: true
""")


def test_unknown_model_is_structured():
    recipe = _recipe("model:\n  name: nosuchmodel\n")
    with pytest.raises(ResolutionError) as exc:
        resolve_assembly(recipe)
    err = exc.value.to_dict()
    assert err["code"] == UNKNOWN_MODEL
    assert err["path"] == "model.name"


@requires_dataset
def test_unknown_robot_is_structured():
    recipe = _act_recipe("robot:\n  name: nosuchrobot\n")
    with pytest.raises(ResolutionError) as exc:
        resolve_assembly(recipe)
    assert exc.value.to_dict()["code"] == UNKNOWN_ROBOT


def test_missing_dataset_path_is_structured():
    with pytest.raises(ResolutionError) as exc:
        resolve_assembly(_recipe("model:\n  name: act\n"))
    err = exc.value.to_dict()
    assert err["code"] == MISSING_INPUT
    assert err["path"] == "data.path"


def test_unreadable_dataset_keeps_the_reason():
    """The structured error carries why the read failed — a bare "schema is
    required" would send the user looking in the wrong place."""
    recipe = _recipe("""
model:
  name: act
data:
  path: /nonexistent/dataset
  format: lerobot-v3
""")
    with pytest.raises(ResolutionError) as exc:
        resolve_assembly(recipe)
    err = exc.value.to_dict()
    assert err["code"] == MISSING_INPUT
    assert "/nonexistent/dataset" in err["params"]["detail"]


@requires_dataset
def test_a_contradicting_checkpoint_is_reported_here():
    """The optional checkpoint check lives inside this entry point, so it runs
    before any downstream side effect (see the next test)."""
    import json

    from vla_factory.model.checkpoint_validation import CheckpointCompatibilityError

    checkpoint = _project_root / "test" / "data"      # any dir; config is written below
    tmp = checkpoint / "_tmp_ckpt_for_test"
    tmp.mkdir(exist_ok=True)
    try:
        (tmp / "config.json").write_text(json.dumps({"max_action_dim": 7}))
        recipe = _recipe(f"""
model:
  name: pi0
  path: {tmp}
data:
  path: {DATASET_PATH}
  format: lerobot-v3
""")
        with pytest.raises(CheckpointCompatibilityError, match="max_action_dim"):
            resolve_assembly(recipe)
    finally:
        (tmp / "config.json").unlink(missing_ok=True)
        tmp.rmdir()


@requires_dataset
def test_failed_resolution_leaves_the_output_directory_untouched(tmp_path):
    """``train()`` resolves before it wipes anything.

    The old order reported an unresolvable composition *after* deleting the
    previous run's output directory, which is the worst possible moment to
    learn the run was never going to start.
    """
    from vla_factory.training.train import train

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "previous.txt").write_text("an earlier run's artifacts")

    recipe = _act_recipe("robot:\n  name: nosuchrobot\n")
    recipe.output.output_dir = str(output_dir)
    recipe.output.overwrite_output_dir = True

    with pytest.raises(ResolutionError):
        train(recipe)
    assert (output_dir / "previous.txt").exists()


@requires_dataset
@pytest.mark.parametrize(
    "recipe_body,message",
    [
        # A finetune-only model with no base checkpoint.
        ("""
model:
  name: pi0
data:
  path: {dataset}
  format: lerobot-v3
overrides:
  camera_mapping:
    base_0_rgb: front
""", "finetune-only"),
        # Transform operations are not a per-run config surface.
        ("""
model:
  name: act
  config:
    transforms:
      inputs: []
data:
  path: {dataset}
  format: lerobot-v3
""", "cannot be overridden"),
    ],
)
def test_every_refusal_lands_before_the_output_directory_is_touched(
    tmp_path, recipe_body, message,
):
    """Not only composition failures: anything that means "this run cannot
    start" has to be decided before the previous run's artifacts are deleted."""
    from vla_factory.training.train import train

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "previous.txt").write_text("an earlier run's artifacts")

    recipe = parse_recipe_from_string(recipe_body.format(dataset=DATASET_PATH))
    recipe.output.output_dir = str(output_dir)
    recipe.output.overwrite_output_dir = True

    with pytest.raises(ValueError, match=message):
        train(recipe)
    assert (output_dir / "previous.txt").exists()


@requires_dataset
def test_an_incomplete_dataset_description_fails_in_validate():
    """One name per dimension is a description fact, so the resolver refuses it
    — before a model, a dataloader or an output directory exists."""
    from dataclasses import replace

    from vla_factory.assembly.resolve import INVALID_DESCRIPTION, resolve_from_facts
    from vla_factory.data.data_schema import StateDim
    from vla_factory.model.registry import list_entries

    recipe = _act_recipe()
    good = resolve_assembly(recipe)
    nameless = replace(
        good.schema,
        state_dims=tuple(replace(d, name=None) for d in good.schema.state_dims),
    )
    with pytest.raises(ResolutionError) as exc:
        resolve_from_facts(
            nameless, good.norm_stats, list_entries()["act"],
            model_config=recipe.model.config,
        )
    assert exc.value.to_dict()["code"] == INVALID_DESCRIPTION
