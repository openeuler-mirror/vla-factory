"""SimplerEnv contract without requiring SAPIEN / simpler_env installs."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from vla_factory.inference.connectors import simplerenv as connector
from vla_factory.inference.platforms.simplerenv import SimplerEnvAdapter


_CAMERAS = ("agentview", "robot0_eye_in_hand")


def _observation():
    return {
        "image.agentview": np.zeros((8, 8, 3), dtype=np.uint8),
        "image.robot0_eye_in_hand": np.ones((8, 8, 3), dtype=np.uint8),
        # get_robot_state: gripper open amount + eef pos + eef quat = 8-D.
        "state": np.arange(8, dtype=np.float32) / 8.0,
    }


def test_adapter_maps_images_state_and_instruction():
    result = SimplerEnvAdapter(_CAMERAS, 8)(
        {"simplerenv_observation": _observation(), "instruction": "pick coke can"}
    )

    assert tuple(result.video) == _CAMERAS
    np.testing.assert_array_equal(
        result.video["robot0_eye_in_hand"], np.ones((8, 8, 3), dtype=np.uint8)
    )
    np.testing.assert_array_equal(result.state, np.arange(8, dtype=np.float32) / 8.0)
    assert result.language == "pick coke can"


def test_adapter_resolves_wrist_alias():
    observation = _observation()
    observation["image.wrist"] = observation.pop("image.robot0_eye_in_hand")

    result = SimplerEnvAdapter(("agentview", "wrist"), 8)(
        {"simplerenv_observation": observation}
    )

    assert tuple(result.video) == ("agentview", "wrist")


def test_adapter_rejects_state_dim_mismatch():
    with pytest.raises(ValueError, match="state_dim"):
        SimplerEnvAdapter(_CAMERAS, 9)({"simplerenv_observation": _observation()})


def test_adapter_rejects_missing_camera():
    observation = _observation()
    observation.pop("image.agentview")

    with pytest.raises(KeyError, match="agentview"):
        SimplerEnvAdapter(_CAMERAS, 8)({"simplerenv_observation": observation})


def test_adapter_rejects_non_hwc_image():
    observation = _observation()
    observation["image.agentview"] = np.zeros((8, 8), dtype=np.uint8)

    with pytest.raises(ValueError, match="HWC"):
        SimplerEnvAdapter(_CAMERAS, 8)({"simplerenv_observation": observation})


def test_action_conversion_clips_and_validates_dim():
    action = connector._to_simplerenv_action(
        np.array([2.0, -0.5, 0.0, 1.5], dtype=np.float32), action_dim=4
    )

    np.testing.assert_allclose(action, [1.0, -0.5, 0.0, 1.0])

    with pytest.raises(ValueError, match="4-D action"):
        connector._to_simplerenv_action(np.zeros(7, dtype=np.float32), action_dim=4)


def test_benchmark_loop_and_success_report(monkeypatch):
    actions_taken: list[np.ndarray] = []

    class Env:
        def reset(self, seed):
            return {}, {}

        def get_image(self, size, camera):
            return np.zeros((*size, 3), dtype=np.uint8)

        def get_robot_state(self):
            return np.arange(8, dtype=np.float32)

        def step(self, action):
            actions_taken.append(action)
            return {}, 1.0, True, False, {}

        def close(self):
            pass

    class Box:
        shape = (4,)

    Env.action_space = Box()

    env = Env()
    env_builder = ModuleType("simpler_env.utils.env.env_builder")
    env_builder.make_sim_env = lambda task: env
    simpler_env = ModuleType("simpler_env")
    simpler_env_utils = ModuleType("simpler_env.utils")
    simpler_env_utils_env = ModuleType("simpler_env.utils.env")
    monkeypatch.setitem(sys.modules, "simpler_env", simpler_env)
    monkeypatch.setitem(sys.modules, "simpler_env.utils", simpler_env_utils)
    monkeypatch.setitem(sys.modules, "simpler_env.utils.env", simpler_env_utils_env)
    monkeypatch.setitem(
        sys.modules, "simpler_env.utils.env.env_builder", env_builder
    )
    monkeypatch.setattr(connector, "_ensure_simulator", lambda: None)

    class Model:
        calls: list[tuple] = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def call(self, func_name, obs=None):
            Model.calls.append((func_name, obs))
            if func_name == "get_action":
                return np.tile(
                    np.array([[0.1, -0.2, 0.3, 0.9]], dtype=np.float32), (3, 1)
                )
            return None

    report = connector.run_benchmark(
        model=Model(),
        tasks=["google_robot_pick_coke_can"],
        trials=2,
        max_steps=3,
    )

    reset_calls = [c for c in Model.calls if c[0] == "reset_model"]
    action_calls = [c for c in Model.calls if c[0] == "get_action"]
    assert len(reset_calls) == 2
    assert len(action_calls) == 2
    request = action_calls[0][1]
    assert set(request["simplerenv_observation"]) == {
        "image.agentview", "image.robot0_eye_in_hand", "state",
    }
    assert request["instruction"] == "google_robot_pick_coke_can"
    assert actions_taken and actions_taken[0].shape == (4,)

    task_report = report["tasks"]["google_robot_pick_coke_can"]
    assert task_report["successes"] == 2
    assert task_report["trials"] == 2
    assert task_report["success_rate"] == 1.0
    assert report["overall_success_rate"] == 1.0
    assert report["benchmark"] == "simplerenv"


def test_max_steps_resolution_uses_task_table():
    assert connector._resolve_max_steps("google_robot_pick_coke_can", None) == 280
    assert connector._resolve_max_steps("unknown_task", None) == (
        connector.DEFAULT_MAX_STEPS
    )
    assert connector._resolve_max_steps("google_robot_pick_coke_can", 7) == 7
