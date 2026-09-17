"""RoboCasa gym contract without requiring Mujoco or RoboCasa installs."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

from vla_factory.inference.connectors import robocasa as connector
from vla_factory.inference.platforms.robocasa import RoboCasaAdapter


_CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


def _observation():
    return {
        **{f"video.{camera}": np.zeros((8, 8, 3), dtype=np.uint8) for camera in _CAMERAS},
        "state.base_position": np.arange(3, dtype=np.float32),
        "state.base_rotation": np.arange(3, 7, dtype=np.float32),
        "state.end_effector_position_relative": np.arange(7, 10, dtype=np.float32),
        "state.end_effector_rotation_relative": np.arange(10, 14, dtype=np.float32),
        "state.gripper_qpos": np.arange(14, 16, dtype=np.float32),
        "annotation.human.task_description": "put the mug away",
    }


def test_adapter_uses_robo_casa_gym_keys_and_dataset_order():
    result = RoboCasaAdapter(_CAMERAS, 16)({"robocasa_observation": _observation()})

    assert tuple(result.video) == _CAMERAS
    np.testing.assert_array_equal(result.state, np.arange(16, dtype=np.float32))
    assert result.language == "put the mug away"


def test_action_conversion_uses_robo_casa_dict_contract():
    action = connector._to_robocasa_action(np.arange(12, dtype=np.float32))

    assert set(action) == {
        "action.end_effector_position", "action.end_effector_rotation",
        "action.gripper_close", "action.base_motion", "action.control_mode",
    }
    np.testing.assert_array_equal(action["action.base_motion"], [0, 1, 2, 3])
    np.testing.assert_array_equal(action["action.end_effector_position"], [5, 6, 7])
    np.testing.assert_array_equal(action["action.gripper_close"], [11])


def test_benchmark_uses_instruction_action_dict_and_success(monkeypatch):
    class Env:
        actions = []

        def reset(self, seed):
            return _observation(), {"success": False}

        def step(self, action):
            self.actions.append(action)
            return _observation(), 1.0, False, False, {"success": True}

        def close(self):
            pass

    env = Env()
    gym = ModuleType("gymnasium")
    gym.make = lambda *args, **kwargs: env
    registry_utils = ModuleType("robocasa.utils.dataset_registry_utils")
    registry_utils.get_task_horizon = lambda task: 3
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    monkeypatch.setitem(sys.modules, "robocasa", ModuleType("robocasa"))
    monkeypatch.setitem(sys.modules, "robocasa.utils", ModuleType("robocasa.utils"))
    monkeypatch.setitem(sys.modules, "robocasa.utils.dataset_registry_utils", registry_utils)
    monkeypatch.setattr(connector, "_ensure_simulator", lambda: None)

    class Model:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def call(self, name, observation=None):
            return np.arange(12, dtype=np.float32)[None, :] if name == "get_action" else None

    report = connector.run_benchmark(model=Model(), tasks=["PickPlaceCounterToCabinet"], trials=1)
    assert report["total_successes"] == 1
    assert len(env.actions) == 1
    assert env.actions[0]["action.gripper_close"].shape == (1,)
