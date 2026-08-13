"""Where the model's temporal contract comes from, and the mutual exclusion
that keeps the chunk length in one place.

A pretrained model's chunk length is a family fact (``ModelMetadata``); a
from-scratch model's is the user's choice (``params`` → ``model.config``). The
recipe side is already guarded by the tunable allow-list, but nothing stops a
registry entry from declaring both or neither — which would leave two answers,
or none, for a value the whole pipeline is shaped by.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import make_norm_stats, make_schema

from vla_factory.assembly.resolver import resolve_assembly
from vla_factory.model.interfaces.model import ModelMetadata
from vla_factory.recipe.defaults import resolve_recipe
from vla_factory.recipe.parser import parse_recipe_from_string


def _resolve(metadata: ModelMetadata, model_config: dict | None = None):
    schema = make_schema(state_dim=6, action_dim=6, cameras=("front",))
    return resolve_assembly(
        schema,
        make_norm_stats(state_dim=6, action_dim=6),
        metadata,
        model_config=model_config,
    )


def _finetune(**kwargs) -> ModelMetadata:
    return ModelMetadata(
        name="stub", training_paradigm="pretrained_finetune",
        requires_prompt=False, vector_normalization="mean_std", **kwargs,
    )


def _from_scratch(**kwargs) -> ModelMetadata:
    return ModelMetadata(
        name="stub", training_paradigm="from_scratch",
        requires_prompt=False, vector_normalization="mean_std", **kwargs,
    )


def test_finetune_horizon_comes_from_the_named_fact():
    assembly = _resolve(_finetune(action_horizon=50))
    assert assembly.model_io_spec.action_horizon == 50


def test_from_scratch_horizon_comes_from_the_tunable():
    metadata = _from_scratch(params={"action_horizon": 100})
    assert _resolve(metadata).model_io_spec.action_horizon == 100
    # And a per-run override of that tunable wins.
    assert _resolve(metadata, {"action_horizon": 25}).model_io_spec.action_horizon == 25


def test_declaring_both_is_a_broken_entry():
    metadata = _finetune(action_horizon=50, params={"action_horizon": 100})
    with pytest.raises(ValueError, match="twice"):
        _resolve(metadata)


def test_finetune_may_not_make_the_horizon_tunable():
    metadata = _finetune(params={"action_horizon": 100})
    with pytest.raises(ValueError, match="pretrained"):
        _resolve(metadata)


def test_from_scratch_may_not_hardcode_the_horizon():
    metadata = _from_scratch(action_horizon=50)
    with pytest.raises(ValueError, match="from_scratch"):
        _resolve(metadata)


def test_declaring_neither_is_a_broken_entry():
    with pytest.raises(ValueError, match="no action horizon"):
        _resolve(_finetune())


def test_a_recipe_that_says_nothing_gets_the_model_default():
    """A recipe that names no chunk length gets the model's own declared one —
    there is no framework-wide default, because a chunk length belongs to a
    model and not to the framework."""
    resolved = resolve_recipe(parse_recipe_from_string("model:\n  name: act\n"))
    assert resolved.model_config["action_horizon"] == 100


# ── The other half of the temporal contract ───────────────────────


def test_observation_window_comes_from_the_model_declaration():
    """``n_obs_steps`` is the model's ``history_frames``, not a recipe field.

    A sample must carry exactly the frames the model consumes; a second field
    for it could only ever agree or be wrong.
    """
    assembly = _resolve(_finetune(action_horizon=50, history_frames=3))
    assert assembly.model_io_spec.n_obs_steps == 3


def test_the_default_window_is_a_single_frame():
    assembly = _resolve(_finetune(action_horizon=50))
    assert assembly.model_io_spec.n_obs_steps == 1


def test_a_model_declaring_no_observation_frame_is_a_broken_entry():
    with pytest.raises(ValueError, match="at least one"):
        _resolve(_finetune(action_horizon=50, history_frames=0))
