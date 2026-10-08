"""SimplerEnv observation adapter.

Converts the observation dict produced by the SimplerEnv connector
(``vla_factory.inference.connectors.simplerenv``) into VLA Factory's
:class:`ObsDict`. This is the embodiment/platform boundary for the
``simplerenv`` deploy platform: the connector forwards the images and robot
state it read from the SimplerEnv environment and the InferenceEngine
consumes :class:`ObsDict`.

The wire contract between the connector and this adapter is deliberately
narrow and numpy-shaped::

    {
        "simplerenv_observation": {
            "image.<camera>": uint8 HWC RGB,   # one per checkpoint camera
            "state": float32[D],               # env.get_robot_state()
        },
        "instruction": str,                     # task instruction
        "step": int,
    }

The camera set is a data/model contract: images are returned for exactly
``camera_keys`` (the checkpoint's trained cameras), each mapped back to its
SimplerEnv source camera. A missing camera or a state-dim mismatch raises a
clear error rather than silently degrading.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vla_factory.inference.inference_engine import ObsDict

logger = logging.getLogger(__name__)


# Checkpoint camera name → SimplerEnv camera name. SimplerEnv (ManiSkill)
# exposes "agentview" and "robot0_eye_in_hand"; checkpoints commonly store
# the shorter "wrist" alias.
_CAMERA_ALIASES: dict[str, str] = {
    "wrist": "robot0_eye_in_hand",
    "robot0_wrist": "robot0_eye_in_hand",
    "eye_in_hand": "robot0_eye_in_hand",
}


def _simplerenv_camera_key(checkpoint_key: str) -> str:
    """Map a checkpoint camera feature key to its SimplerEnv camera name."""
    name = checkpoint_key
    for prefix in ("observation.images.", "images.", "image."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return _CAMERA_ALIASES.get(name, name)


class SimplerEnvAdapter:
    """Convert a connector-wrapped SimplerEnv observation into ``ObsDict``.

    Parameters
    ----------
    camera_keys : tuple[str, ...]
        Camera names the checkpoint was trained on (from the saved schema).
        Each is mapped back to a SimplerEnv camera (``agentview`` /
        ``robot0_eye_in_hand``) and must arrive as an HWC uint8 RGB array.
    state_dim : int
        Expected proprioception width; asserted against the assembled state
        (SimplerEnv's ``get_robot_state`` yields 8-D per arm: gripper open
        amount + end-effector pos/quat).
    """

    def __init__(
        self,
        camera_keys: tuple[str, ...],
        state_dim: int,
    ) -> None:
        self._camera_keys = tuple(camera_keys)
        self._state_dim = state_dim
        logger.info(
            "SimplerEnvAdapter — cameras: %s, state_dim: %d",
            list(self._camera_keys), state_dim,
        )

    def __call__(self, observation: dict[str, Any], task: str = "") -> ObsDict:
        if not isinstance(observation, dict):
            raise TypeError(
                "SimplerEnv observation must be a dict, got "
                f"{type(observation).__name__}."
            )

        if "simplerenv_observation" not in observation:
            raise KeyError(
                "SimplerEnv connector request must contain "
                f"'simplerenv_observation'. Available: {list(observation.keys())}"
            )
        request = observation
        env_obs = request["simplerenv_observation"]
        if not isinstance(env_obs, dict):
            raise TypeError(
                "'simplerenv_observation' must be a dict, got "
                f"{type(env_obs).__name__}."
            )
        language = (
            request.get("instruction")
            or env_obs.get("instruction")
            or task
            or None
        )

        video: dict[str, np.ndarray] = {}
        for cam in self._camera_keys:
            env_key = _simplerenv_camera_key(cam)
            raw = env_obs.get(f"image.{env_key}")
            if raw is None:
                # Tolerate bare camera names for custom wrappers.
                raw = (
                    env_obs.get(env_key)
                    or env_obs.get(cam)
                    or env_obs.get(f"image.{cam}")
                )
            if raw is None:
                raise KeyError(
                    f"Camera '{cam}' (SimplerEnv key 'image.{env_key}') not "
                    "found in observation. The runtime task must expose every "
                    "camera stored in the checkpoint schema. Available: "
                    f"{sorted(env_obs.keys())}"
                )
            arr = np.asarray(raw)
            if arr.ndim != 3:
                raise ValueError(
                    f"Camera '{cam}' must be HWC (3D), got shape {arr.shape}. "
                    "Send raw uint8 RGB; the model's transform pipeline handles "
                    "float/CHW/normalisation."
                )
            video[cam] = arr

        state = env_obs.get("state")
        if state is None:
            raise KeyError(
                "SimplerEnv observation must contain 'state' "
                "(env.get_robot_state())."
            )
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self._state_dim,):
            raise ValueError(
                f"SimplerEnv robot state has width {state.shape[-1]}, but the "
                f"checkpoint was trained with state_dim={self._state_dim}. "
                "Check the embodiment used at deploy vs train time "
                "(SimplerEnv's get_robot_state yields 8-D per arm: gripper "
                "open amount + end-effector pos/quat)."
            )

        return ObsDict(video=video, state=state, language=language)
