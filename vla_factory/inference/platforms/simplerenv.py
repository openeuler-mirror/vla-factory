"""SimplerEnv observation adapter.

Converts the observation dict produced by the SimplerEnv connector
(``vla_factory.inference.connectors.simplerenv``) into VLA Factory's
:class:`ObsDict`. This is the embodiment/platform boundary for the
``simplerenv`` deploy platform: the connector forwards the camera frame and
end-effector state it read from the SimplerEnv observation dict and the
InferenceEngine consumes :class:`ObsDict`.

The wire contract between the connector and this adapter is deliberately
narrow and numpy-shaped::

    {
        "simplerenv_observation": {
            "image.<camera>": uint8 HWC RGB,   # SimplerEnv camera
            "state": float32[8],               # obs["agent"]["eef_pos"]
        },
        "instruction": str,                     # get_language_instruction()
        "step": int,
    }

SimplerEnv (ManiSkill2_real2sim) exposes a single third-person camera per
embodiment — ``overhead_camera`` for Google-robot tasks and
``3rd_view_camera`` for WidowX tasks. Checkpoints trained on more cameras
than that cannot be evaluated on SimplerEnv; the adapter reports this
rather than silently degrading. The robot state is
``obs["agent"]["eef_pos"]``: 8-D = translation (3) + quaternion (4, wxyz) +
gripper (1).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vla_factory.inference.inference_engine import ObsDict

logger = logging.getLogger(__name__)


# Checkpoint camera name → SimplerEnv camera name. Both embodiments have
# exactly one third-person camera; common checkpoint aliases map onto it.
_CAMERA_ALIASES: dict[str, str] = {
    "overhead": "overhead_camera",
    "agentview": "overhead_camera",
    "agent_view": "overhead_camera",
    "third_person": "3rd_view_camera",
    "third_person_front": "3rd_view_camera",
    "front": "3rd_view_camera",
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
        SimplerEnv provides one camera per embodiment, so at most one
        checkpoint camera is expected; it is matched to the forwarded
        SimplerEnv camera by name (with aliasing) or, failing that, by
        position when both sides are single-camera.
    state_dim : int
        Expected proprioception width; asserted against the forwarded state
        (SimplerEnv's ``eef_pos`` is 8-D: translation 3 + quaternion 4 +
        gripper 1).
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

        env_cameras = [
            key[len("image."):]
            for key in env_obs
            if isinstance(key, str) and key.startswith("image.")
        ]
        if env_cameras and len(self._camera_keys) > len(env_cameras):
            raise ValueError(
                f"The checkpoint was trained on {len(self._camera_keys)} "
                f"cameras {list(self._camera_keys)}, but SimplerEnv exposed "
                f"{len(env_cameras)} ({env_cameras}). SimplerEnv provides a "
                "single third-person camera per embodiment "
                "(overhead_camera / 3rd_view_camera) and no wrist camera, so "
                "multi-camera checkpoints cannot be evaluated on it."
            )

        video: dict[str, np.ndarray] = {}
        for cam in self._camera_keys:
            env_key = _simplerenv_camera_key(cam)
            raw = env_obs.get(f"image.{env_key}")
            if raw is None and len(env_cameras) == 1:
                # Positional pairing for single-camera checkpoints whose
                # trained camera name does not match the SimplerEnv camera.
                raw = env_obs.get(f"image.{env_cameras[0]}")
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
                "(obs['agent']['eef_pos'])."
            )
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self._state_dim,):
            raise ValueError(
                f"SimplerEnv robot state has width {state.shape[-1]}, but the "
                f"checkpoint was trained with state_dim={self._state_dim}. "
                "Check the embodiment used at deploy vs train time "
                "(SimplerEnv's obs['agent']['eef_pos'] is 8-D: translation 3 "
                "+ quaternion 4 + gripper 1)."
            )

        return ObsDict(video=video, state=state, language=language)
