"""L0 tests for the GR00T platform adapter.

``platforms/groot.py`` had no coverage before Issue #7. It is small, but it is
a boundary: it converts a plain dict from the GR00T runtime into the framework's
``ObsDict`` and hands back a raw array. A wrong dtype or a silently dropped
language field here reaches the model as bad input with nothing to signal it.
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_factory.inference.execution import ActionChunk, ActionCommand
from vla_factory.inference.inference_engine import ObsDict
from vla_factory.inference.platforms.groot import GROOTAdapter

ACTION_DIM = 4


class _FakePolicy:
    """PolicyExecutor stand-in: records the ObsDict, returns a fixed command."""

    def __init__(self, command=None) -> None:
        self.command = command if command is not None else ActionCommand(
            np.arange(ACTION_DIM, dtype=np.float32).reshape(1, ACTION_DIM)
        )
        self.last_obs: ObsDict | None = None
        self.reset_count = 0

    def predict(self, obs):
        self.last_obs = obs
        return self.command

    def reset(self) -> None:
        self.reset_count += 1


def _obs_dict() -> dict:
    return {
        "video": {"front": np.zeros((4, 4, 3), dtype=np.uint8)},
        "state": [0.1, 0.2, 0.3],
        "language": "pick up the cube",
    }


def test_get_action_returns_the_command_values():
    policy = _FakePolicy()
    adapter = GROOTAdapter(policy)

    actions = adapter.get_action(_obs_dict())

    np.testing.assert_array_equal(actions, policy.command.values)


def test_observation_fields_are_converted_and_forwarded():
    """state must arrive as float32 and language must survive the hand-off."""
    policy = _FakePolicy()
    GROOTAdapter(policy).get_action(_obs_dict())

    obs = policy.last_obs
    assert isinstance(obs, ObsDict)
    assert obs.state.dtype == np.float32
    np.testing.assert_allclose(obs.state, [0.1, 0.2, 0.3], rtol=1e-6)
    assert obs.video["front"].shape == (4, 4, 3)
    assert obs.language == "pick up the cube"


def test_missing_optional_fields_are_tolerated():
    """GR00T may send video only; state/language stay None rather than crashing."""
    policy = _FakePolicy()
    GROOTAdapter(policy).get_action({"video": {}})

    assert policy.last_obs.state is None
    assert policy.last_obs.language is None
    assert policy.last_obs.video == {}


def test_non_command_return_is_rejected():
    """An ActionChunk is not an ActionCommand — returning one un-executed would
    hand GR00T a whole horizon where it expects the selected steps."""
    chunk = ActionChunk(np.zeros((8, ACTION_DIM), dtype=np.float32))
    adapter = GROOTAdapter(_FakePolicy(command=chunk))

    with pytest.raises(TypeError, match="must return ActionCommand"):
        adapter.get_action(_obs_dict())


def test_reset_delegates_to_the_policy():
    policy = _FakePolicy()
    adapter = GROOTAdapter(policy)

    adapter.reset()

    assert policy.reset_count == 1


def test_tag_is_stored_for_future_embodiment_routing():
    adapter = GROOTAdapter(_FakePolicy(), tag="franka")
    assert adapter.tag == "franka"
    assert GROOTAdapter(_FakePolicy()).tag is None
