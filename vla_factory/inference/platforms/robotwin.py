"""RoboTwin native observation adapter.

Converts the observation dict produced by a RoboTwin ``deploy_policy.encode_obs``
into VLA Factory's :class:`ObsDict`. This is the embodiment/platform boundary for
the ``robotwin`` deploy platform: the connector forwards RoboTwin's native
observation and the InferenceEngine consumes :class:`ObsDict`.

The camera set is a data/model contract: images are read for exactly
``camera_keys`` (the checkpoint's trained cameras), in that set, so each maps to
the camera it was trained on. A missing camera or a state-dim mismatch raises a
clear error rather than silently degrading.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vla_factory.inference.inference_engine import ObsDict

logger = logging.getLogger(__name__)


class RoboTwinAdapter:
    """Convert a connector-wrapped native RoboTwin observation into ``ObsDict``.

    Parameters
    ----------
    camera_keys : tuple[str, ...]
        Camera names the checkpoint was trained on (from the saved schema).
        Each must be present in the observation dict as an HWC uint8 RGB array.
    state_dim : int
        Expected qpos length; asserted against the incoming ``qpos`` vector.
    """

    def __init__(
        self,
        camera_keys: tuple[str, ...],
        state_dim: int,
    ) -> None:
        self._camera_keys = tuple(camera_keys)
        self._state_dim = state_dim
        logger.info(
            "RoboTwinAdapter — cameras: %s, state_dim: %d",
            list(self._camera_keys),
            state_dim,
        )

    def __call__(self, observation: dict[str, Any], task: str = "") -> ObsDict:
        if not isinstance(observation, dict):
            raise TypeError(
                "RoboTwin observation must be a dict, got "
                f"{type(observation).__name__}."
            )

        if "robotwin_observation" not in observation:
            raise KeyError(
                "RoboTwin connector request must contain 'robotwin_observation'. "
                f"Available: {list(observation.keys())}"
            )
        request = observation
        observation = request["robotwin_observation"]
        if not isinstance(observation, dict):
            raise TypeError(
                "'robotwin_observation' must be a dict, got "
                f"{type(observation).__name__}."
            )
        language = request.get("instruction") or task or None

        camera_group = observation.get("observation")
        if not isinstance(camera_group, dict):
            raise TypeError(
                "RoboTwin native 'observation' camera group must be a dict, got "
                f"{type(camera_group).__name__}."
            )

        video: dict[str, np.ndarray] = {}
        for cam in self._camera_keys:
            camera_value = camera_group.get(cam)
            raw = camera_value.get("rgb") if isinstance(camera_value, dict) else None
            if raw is None:
                raise KeyError(
                    f"Camera '{cam}' not found in RoboTwin observation. "
                    "The runtime task configuration must expose every camera "
                    "stored in the checkpoint schema. "
                    f"Available: {list(camera_group.keys())}"
                )
            arr = np.asarray(raw)
            if arr.ndim != 3:
                raise ValueError(
                    f"Camera '{cam}' must be HWC (3D), got shape {arr.shape}. "
                    "Send raw uint8 RGB; the model's transform pipeline handles "
                    "float/CHW/normalisation."
                )
            video[cam] = arr

        qpos = self._native_qpos(observation)
        if qpos is None:
            raise KeyError(
                "Joint vector not found in RoboTwin observation. Expected "
                "'joint_action.vector' or its four named components. "
                f"Available: {list(observation.keys())}"
            )
        state = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if state.shape[0] != self._state_dim:
            raise ValueError(
                f"qpos length {state.shape[0]} != model state_dim "
                f"{self._state_dim}. Check the embodiment / camera-config used "
                "at deploy vs train time."
            )

        return ObsDict(video=video, state=state, language=language)

    @staticmethod
    def _native_qpos(observation: dict[str, Any]) -> Any:
        joint_action = observation.get("joint_action")
        if not isinstance(joint_action, dict):
            return None
        if joint_action.get("vector") is not None:
            return joint_action["vector"]

        order = ("left_arm", "left_gripper", "right_arm", "right_gripper")
        if any(joint_action.get(name) is None for name in order):
            return None
        parts = [
            np.asarray(joint_action[name], dtype=np.float32).reshape(-1)
            for name in order
        ]
        return np.concatenate(parts)
