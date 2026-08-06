"""PolicyRunner orchestration tests.

Exercises the client-shaped deployment loop with an injected fake transport
client and a fake engine: adapter wiring, action-adapter application, reset
control messages and loop termination. No ZMQ sockets involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_factory.inference.infer import ActionCommand
from vla_factory.inference.platforms.simulator import SimulatorAdapter
from vla_factory.inference.policy_runtime import PolicyClientTransport, PolicyRunner

CAMERAS = ("front",)
STATE_DIM = 3
ACTION_DIM = 4


class _FakeEngine:
    """Minimal InferenceEngine stand-in recording predict/reset calls."""

    def __init__(self) -> None:
        self.camera_keys = CAMERAS
        self.predict_calls = []
        self.reset_count = 0

    def predict(self, obsdict) -> np.ndarray:
        self.predict_calls.append(obsdict)
        return ActionCommand(np.arange(ACTION_DIM, dtype=np.float32)[None, :])

    def reset(self) -> None:
        self.reset_count += 1


class _FakeClient:
    """Scripted client transport; exhausting the script interrupts the loop."""

    def __init__(self, script) -> None:
        self.sent = []
        self.closed = False
        self._script = iter(script)

    def wait_for_connection(self) -> None:
        pass

    def recv_observation(self):
        item = next(self._script, KeyboardInterrupt)
        if item is KeyboardInterrupt:
            raise KeyboardInterrupt
        return item

    def send_action(self, action) -> None:
        self.sent.append(action)

    def close(self) -> None:
        self.closed = True


def _obs() -> dict:
    return {
        "observation.images.front": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.state": [0.1] * STATE_DIM,
    }


def _run(script, *, action_adapter=None, task=""):
    engine = _FakeEngine()
    client = _FakeClient(script)
    runner = PolicyRunner(
        engine,
        SimulatorAdapter(CAMERAS),
        action_adapter,
        task=task,
        max_loop_freq_hz=10000.0,
    )
    runner.run(client)
    return engine, client


def test_fake_client_satisfies_transport_protocol():
    assert isinstance(_FakeClient([]), PolicyClientTransport)


def test_predict_and_send():
    engine, client = _run([_obs()])

    assert len(engine.predict_calls) == 1
    np.testing.assert_allclose(
        engine.predict_calls[0].state, np.full(STATE_DIM, 0.1, dtype=np.float32)
    )
    assert len(client.sent) == 1
    np.testing.assert_array_equal(
        client.sent[0], np.arange(ACTION_DIM, dtype=np.float32)[None, :]
    )
    assert client.closed  # transport released on exit


def test_action_adapter_applied():
    adapter_calls = []

    def action_adapter(action):
        adapter_calls.append(action)
        return {"motor_0": float(action[0])}

    _, client = _run([_obs()], action_adapter=action_adapter)

    assert len(adapter_calls) == 1
    assert client.sent == [{"motor_0": 0.0}]


def test_reset_control_message():
    script = [{"__control__": "reset", "episode_index": 2}, _obs()]
    engine, client = _run(script)

    assert engine.reset_count == 1
    assert len(engine.predict_calls) == 1  # reset message is not predicted on
    assert len(client.sent) == 1


def test_none_observation_skipped():
    engine, client = _run([None, _obs()])

    assert len(engine.predict_calls) == 1
    assert len(client.sent) == 1


def test_task_forwarded_to_adapter():
    engine, _ = _run([_obs()], task="pick up the block")

    assert engine.predict_calls[0].language == "pick up the block"


def test_nonpositive_loop_freq_rejected():
    with pytest.raises(ValueError, match="max_loop_freq_hz"):
        PolicyRunner(
            _FakeEngine(), SimulatorAdapter(CAMERAS), max_loop_freq_hz=0.0
        )
