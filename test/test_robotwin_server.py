"""RoboTwin connector and model-server protocol tests.

Stands up the generic RPC transport with a fake engine and drives it over a
socket using RoboTwin's exact wire protocol. Verifies transport framing,
remote-policy dispatch, platform adaptation and compatibility exports. No
RoboTwin install required.
"""

from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vla_factory.deploy import robotwin_connector
from vla_factory.deploy.robotwin_adapter import RobotwinObsAdapter
from vla_factory.deploy.robotwin_server import (
    RobotwinEngineModel,
    RobotwinModelServer,
    json_to_numpy,
    numpy_to_json,
)

CAMERAS = ("front_camera", "head_camera", "left_camera", "right_camera")
STATE_DIM = 14
ACTION_DIM = 14
HORIZON = 20


class _FakeEngine:
    """Minimal InferenceEngine stand-in recording the ObsDict it receives."""

    def __init__(self) -> None:
        self.camera_keys = CAMERAS
        self.action_dim = ACTION_DIM
        self.action_horizon = HORIZON
        self.schema = SimpleNamespace(state_dim=STATE_DIM)
        self.last_obs = None
        self.reset_count = 0

    def predict(self, obsdict) -> np.ndarray:
        self.last_obs = obsdict
        return (
            np.arange(HORIZON * ACTION_DIM, dtype=np.float32)
            .reshape(HORIZON, ACTION_DIM)
        )

    def reset(self) -> None:
        self.reset_count += 1


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _MockRobotwinClient:
    """Reimplements RoboTwin ModelClient framing to exercise the server."""

    def __init__(self, host: str, port: int, tries: int = 100) -> None:
        last: Exception | None = None
        for _ in range(tries):
            try:
                self.sock = socket.create_connection((host, port), timeout=5)
                return
            except OSError as exc:  # server not listening yet
                last = exc
                time.sleep(0.05)
        raise last  # type: ignore[misc]

    def call(self, func_name: str, obs=None):
        body = numpy_to_json({"cmd": func_name, "obs": obs}).encode("utf-8")
        self.sock.sendall(len(body).to_bytes(4, "big"))
        self.sock.sendall(body)
        header = self.sock.recv(4)
        size = int.from_bytes(header, "big")
        chunks, received = [], 0
        while received < size:
            c = self.sock.recv(min(size - received, 4096))
            chunks.append(c)
            received += len(c)
        resp = json_to_numpy(b"".join(chunks).decode("utf-8"))
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp["res"]

    def close(self) -> None:
        self.sock.close()


def _make_obs() -> dict:
    obs = {cam: np.full((8, 8, 3), 7, dtype=np.uint8) for cam in CAMERAS}
    obs["qpos"] = np.arange(STATE_DIM, dtype=np.float32)
    return obs


def _make_native_obs() -> dict:
    return {
        "observation": {
            cam: {
                "rgb": np.full((8, 8, 3), 7, dtype=np.uint8),
                "intrinsic_cv": np.eye(3, dtype=np.float32),
            }
            for cam in CAMERAS
        },
        "joint_action": {
            "left_arm": np.arange(6, dtype=np.float32),
            "left_gripper": np.float32(6),
            "right_arm": np.arange(7, 13, dtype=np.float32),
            "right_gripper": np.float32(13),
            "vector": np.arange(STATE_DIM, dtype=np.float32),
        },
        "endpose": {},
        "pointcloud": [],
    }


@pytest.fixture
def server_and_engine():
    engine = _FakeEngine()
    adapter = RobotwinObsAdapter(camera_keys=CAMERAS, state_dim=STATE_DIM)
    model = RobotwinEngineModel(engine, adapter, task="pick up the block")
    port = _free_port()
    server = RobotwinModelServer(model, host="127.0.0.1", port=port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield server, engine, port
    server.stop()
    t.join(timeout=2)


def test_get_action_roundtrip_and_obs_parsing(server_and_engine):
    server, engine, port = server_and_engine
    client = _MockRobotwinClient("127.0.0.1", port)
    try:
        actions = client.call("get_action", _make_obs())
    finally:
        client.close()

    # action chunk shape survives the wire round-trip
    assert isinstance(actions, np.ndarray)
    assert actions.shape == (HORIZON, ACTION_DIM)
    np.testing.assert_array_equal(actions[0], np.arange(ACTION_DIM))

    # obs was adapted to an ObsDict with all cameras + 14-D state + task
    obsdict = engine.last_obs
    assert set(obsdict.video.keys()) == set(CAMERAS)
    for cam in CAMERAS:
        assert obsdict.video[cam].shape == (8, 8, 3)
        assert obsdict.video[cam].dtype == np.uint8
    assert obsdict.state.shape == (STATE_DIM,)
    np.testing.assert_array_equal(obsdict.state, np.arange(STATE_DIM))
    assert obsdict.language == "pick up the block"


def test_native_observation_roundtrip_and_instruction(server_and_engine):
    _, engine, port = server_and_engine
    request = {
        "robotwin_observation": _make_native_obs(),
        "instruction": "hit the block with the hammer",
        "step": 12,
    }
    client = _MockRobotwinClient("127.0.0.1", port)
    try:
        actions = client.call("get_action", request)
    finally:
        client.close()

    assert actions.shape == (HORIZON, ACTION_DIM)
    obsdict = engine.last_obs
    assert tuple(obsdict.video) == CAMERAS
    np.testing.assert_array_equal(obsdict.state, np.arange(STATE_DIM))
    assert obsdict.language == "hit the block with the hammer"


def test_native_observation_named_joint_fallback():
    native = _make_native_obs()
    del native["joint_action"]["vector"]
    adapter = RobotwinObsAdapter(camera_keys=CAMERAS, state_dim=STATE_DIM)

    obsdict = adapter({"robotwin_observation": native})

    np.testing.assert_array_equal(obsdict.state, np.arange(STATE_DIM))


def test_lightweight_connector_forwards_raw_observation_and_executes_chunk():
    raw_observation = _make_native_obs()

    class _TaskEnv:
        take_action_cnt = 17

        def __init__(self):
            self.executed = []

        def get_instruction(self):
            return "hit the block"

        def take_action(self, action, action_type):
            self.executed.append((action, action_type))

        def get_obs(self):
            return raw_observation

    class _Client:
        def __init__(self):
            self.call_args = None

        def call(self, func_name=None, obs=None):
            self.call_args = (func_name, obs)
            return np.ones((2, ACTION_DIM), dtype=np.float32)

    env = _TaskEnv()
    client = _Client()
    returned = robotwin_connector.eval(env, client, raw_observation)

    func_name, payload = client.call_args
    assert func_name == "get_action"
    assert payload["robotwin_observation"] is raw_observation
    assert payload["instruction"] == "hit the block"
    assert payload["step"] == 17
    assert len(env.executed) == 2
    assert all(action_type == "qpos" for _, action_type in env.executed)
    assert returned is raw_observation


def test_update_obs_then_get_action_uses_cache(server_and_engine):
    server, engine, port = server_and_engine
    client = _MockRobotwinClient("127.0.0.1", port)
    try:
        assert client.call("update_obs", _make_obs()) is None
        actions = client.call("get_action")  # no obs → cached
    finally:
        client.close()
    assert actions.shape == (HORIZON, ACTION_DIM)
    assert engine.last_obs is not None


def test_reset_model(server_and_engine):
    server, engine, port = server_and_engine
    client = _MockRobotwinClient("127.0.0.1", port)
    try:
        assert client.call("reset_model") is None
    finally:
        client.close()
    assert engine.reset_count == 1


def test_unknown_cmd_returns_error(server_and_engine):
    server, engine, port = server_and_engine
    client = _MockRobotwinClient("127.0.0.1", port)
    try:
        with pytest.raises(RuntimeError, match="No model method named"):
            client.call("no_such_method", _make_obs())
    finally:
        client.close()


def test_n_action_steps_truncation():
    engine = _FakeEngine()
    adapter = RobotwinObsAdapter(camera_keys=CAMERAS, state_dim=STATE_DIM)
    model = RobotwinEngineModel(engine, adapter, n_action_steps=5)
    port = _free_port()
    server = RobotwinModelServer(model, host="127.0.0.1", port=port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        client = _MockRobotwinClient("127.0.0.1", port)
        actions = client.call("get_action", _make_obs())
        client.close()
        assert actions.shape == (5, ACTION_DIM)
    finally:
        server.stop()
        t.join(timeout=2)


def test_numpy_codec_roundtrip():
    arr = np.random.RandomState(0).randn(3, 4).astype(np.float32)
    out = json_to_numpy(numpy_to_json({"x": arr}))["x"]
    np.testing.assert_array_equal(out, arr)
    assert out.dtype == np.float32
