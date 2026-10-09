"""SimplerEnv contract without requiring SAPIEN / simpler_env installs.

The mocks mirror the real upstream API surface (simpler_env.make, the
ManiSkill obs dict layout, get_language_instruction, the 7-D action space)
so a contract change upstream moves these tests, unlike a mock of an
invented API.
"""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from vla_factory.inference.connectors import simplerenv as connector
from vla_factory.inference.platforms.simplerenv import SimplerEnvAdapter


_FRAME = np.full((240, 320, 3), 7, dtype=np.uint8)
_EEF_POS = np.arange(8, dtype=np.float32) / 8.0


def _env_obs(camera="overhead_camera"):
    return {
        "image": {camera: {"rgb": _FRAME, "depth": np.zeros((240, 320))}},
        "agent": {"eef_pos": _EEF_POS, "qpos": np.zeros(9)},
    }


def _request(camera="overhead_camera"):
    return {
        "simplerenv_observation": {
            f"image.{camera}": _FRAME,
            "state": _EEF_POS,
        },
        "instruction": "pick up the coke can",
        "step": 0,
    }


# ── Adapter ────────────────────────────────────────────────────────────


def test_adapter_uses_single_simplerenv_camera_and_eef_state():
    result = SimplerEnvAdapter(("overhead_camera",), 8)(_request())

    assert tuple(result.video) == ("overhead_camera",)
    np.testing.assert_array_equal(result.video["overhead_camera"], _FRAME)
    np.testing.assert_array_equal(result.state, _EEF_POS)
    assert result.language == "pick up the coke can"


def test_adapter_pairs_single_checkpoint_camera_positionally():
    # Checkpoint trained as "agentview"; env forwards "3rd_view_camera".
    result = SimplerEnvAdapter(("agentview",), 8)(_request("3rd_view_camera"))

    assert tuple(result.video) == ("agentview",)
    np.testing.assert_array_equal(result.video["agentview"], _FRAME)


def test_adapter_rejects_multi_camera_checkpoint():
    with pytest.raises(ValueError, match="single third-person camera"):
        SimplerEnvAdapter(("agentview", "wrist"), 8)(_request())


def test_adapter_rejects_state_dim_mismatch():
    with pytest.raises(ValueError, match="state_dim"):
        SimplerEnvAdapter(("overhead_camera",), 9)(_request())


def test_adapter_rejects_missing_camera():
    observation = _request()["simplerenv_observation"]
    observation.pop("image.overhead_camera")

    with pytest.raises(KeyError, match="overhead_camera"):
        SimplerEnvAdapter(("overhead_camera",), 8)(
            {"simplerenv_observation": observation}
        )


def test_adapter_rejects_non_hwc_image():
    observation = _request()["simplerenv_observation"]
    observation["image.overhead_camera"] = np.zeros((240, 320), dtype=np.uint8)

    with pytest.raises(ValueError, match="HWC"):
        SimplerEnvAdapter(("overhead_camera",), 8)(
            {"simplerenv_observation": observation}
        )


# ── Connector helpers ──────────────────────────────────────────────────


def test_embodiment_camera_inference():
    assert (
        connector._embodiment_camera("google_robot_pick_coke_can", None)
        == "overhead_camera"
    )
    assert (
        connector._embodiment_camera("widowx_put_eggplant_in_basket", None)
        == "3rd_view_camera"
    )
    assert (
        connector._embodiment_camera("anything", ("3rd_view_camera",))
        == "3rd_view_camera"
    )
    with pytest.raises(ValueError, match="embodiment camera"):
        connector._embodiment_camera("unknown_task", None)
    with pytest.raises(ValueError, match="single third-person camera"):
        connector._embodiment_camera("x", ("a", "b"))


def test_obs_dict_readers():
    obs = _env_obs()
    np.testing.assert_array_equal(connector._get_image(obs, "overhead_camera"), _FRAME)
    np.testing.assert_array_equal(connector._get_state(obs), _EEF_POS)

    with pytest.raises(KeyError, match="3rd_view_camera"):
        connector._get_image(obs, "3rd_view_camera")
    with pytest.raises(KeyError, match="eef_pos"):
        connector._get_state({"agent": {}})


def test_action_passthrough_and_expansion():
    gripper = connector._GripperPostProcessor("raw")

    seven = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)
    np.testing.assert_allclose(
        connector._to_simplerenv_action(seven, gripper), seven
    )

    four = np.array([0.1, -0.2, 0.3, 2.0], dtype=np.float32)
    converted = connector._to_simplerenv_action(four, gripper)
    assert converted.shape == (7,)
    np.testing.assert_allclose(converted, [0.1, -0.2, 0.3, 0, 0, 0, 1.0])

    with pytest.raises(ValueError, match="7-D action"):
        connector._to_simplerenv_action(np.zeros(6, dtype=np.float32), gripper)


def test_gripper_post_processor_modes():
    widowx = connector._GripperPostProcessor("widowx")
    out = widowx.process(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]))
    assert out[-1] == 1.0
    out = widowx.process(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4]))
    assert out[-1] == -1.0

    google = connector._GripperPostProcessor("google", sticky_steps=3)
    out = google.process(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    assert out[-1] == 1.0  # open stays open
    out = google.process(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]))
    assert out[-1] == -1.0  # close latches…
    out = google.process(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    assert out[-1] == -1.0  # …through the sticky window
    google.reset()
    out = google.process(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    assert out[-1] == 1.0

    with pytest.raises(ValueError, match="gripper mode"):
        connector._GripperPostProcessor("bogus")


def test_success_flag_branches():
    class Env:
        success_once = False

    env = Env()
    assert connector._success_flag({"success": True}, False, env)
    assert connector._success_flag({}, True, env)  # terminated
    env2 = Env()
    env2.success_once = True
    assert connector._success_flag({}, False, env2)
    assert not connector._success_flag({}, False, env)  # neither
    assert not connector._success_flag({}, False, env)  # truncated alone


# ── Benchmark loop ─────────────────────────────────────────────────────


class _FakeEnv:
    """Mirror the real SimplerEnv surface used by the connector."""

    def __init__(self, terminate_on_step=1):
        self.terminate_on_step = terminate_on_step
        self.steps: list[np.ndarray] = []

    def reset(self, seed=None):
        return _env_obs(), {"success": False}

    def step(self, action):
        self.steps.append(action)
        done = len(self.steps) >= self.terminate_on_step
        return _env_obs(), 1.0, done, False, {"success": done}

    def close(self):
        pass

    class _Unwrapped:
        @staticmethod
        def get_language_instruction():
            return "pick up the coke can"

    unwrapped = _Unwrapped()


class _Model:
    calls: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def call(self, func_name, obs=None):
        _Model.calls.append((func_name, obs))
        if func_name == "get_action":
            return np.tile(
                np.array([[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.9]], dtype=np.float32),
                (3, 1),
            )
        return None


def _install_fake_simpler_env(monkeypatch, env):
    module = ModuleType("simpler_env")
    module.make = lambda task: env
    monkeypatch.setitem(sys.modules, "simpler_env", module)
    monkeypatch.setattr(connector, "_ensure_simulator", lambda: None)


def test_benchmark_loop_uses_real_contract(monkeypatch):
    env = _FakeEnv(terminate_on_step=2)
    _install_fake_simpler_env(monkeypatch, env)
    _Model.calls.clear()

    report = connector.run_benchmark(
        model=_Model(),
        tasks=["google_robot_pick_coke_can"],
        trials=2,
        max_steps=5,
    )

    get_action_calls = [c for c in _Model.calls if c[0] == "get_action"]
    reset_calls = [c for c in _Model.calls if c[0] == "reset_model"]
    assert len(reset_calls) == 2
    assert len(get_action_calls) >= 2
    request = get_action_calls[0][1]
    assert set(request["simplerenv_observation"]) == {
        "image.overhead_camera", "state",
    }
    np.testing.assert_array_equal(
        request["simplerenv_observation"]["state"], _EEF_POS
    )
    assert request["instruction"] == "pick up the coke can"
    assert all(a.shape == (7,) for a in env.steps)

    task_report = report["tasks"]["google_robot_pick_coke_can"]
    assert task_report["successes"] == 2
    assert task_report["success_rate"] == 1.0
    assert report["benchmark"] == "simplerenv"


def test_benchmark_instruction_override(monkeypatch):
    env = _FakeEnv(terminate_on_step=1)
    _install_fake_simpler_env(monkeypatch, env)
    _Model.calls.clear()

    connector.run_benchmark(
        model=_Model(),
        tasks=["google_robot_pick_coke_can"],
        trials=1,
        max_steps=2,
        instruction="custom instruction",
    )

    request = next(c[1] for c in _Model.calls if c[0] == "get_action")
    assert request["instruction"] == "custom instruction"


def test_benchmark_records_env_creation_error(monkeypatch):
    class _Boom:
        @staticmethod
        def make(task):
            raise RuntimeError("unknown task id")

    module = ModuleType("simpler_env")
    module.make = _Boom.make
    monkeypatch.setitem(sys.modules, "simpler_env", module)
    monkeypatch.setattr(connector, "_ensure_simulator", lambda: None)

    report = connector.run_benchmark(
        model=_Model(),
        tasks=["google_robot_not_a_real_task"],
        trials=3,
    )

    task_report = report["tasks"]["google_robot_not_a_real_task"]
    assert "error" in task_report
    assert task_report["trials"] == 0
