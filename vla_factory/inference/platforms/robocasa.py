"""RoboCasa gymnasium observation adapter.

Converts the observation dict produced by a RoboCasa ``gym.make`` environment
into VLA Factory's :class:`ObsDict`. This is the embodiment/platform boundary
for the ``robocasa`` deploy platform: the connector forwards RoboCasa's native
gym observation and the InferenceEngine consumes :class:`ObsDict`.

RoboCasa's gym wrapper exposes cameras as ``video.robot0_<cam>`` arrays and
state as named ``state.*`` vectors. The adapter assembles them in the same
raw order declared by RoboCasa365's ``modality.json``.

The camera set is a data/model contract: images are read for exactly
``camera_keys`` (the checkpoint's trained cameras), each mapped back to its gym
source. A missing camera or a state-dim mismatch raises a clear error rather
than silently degrading.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from vla_factory.inference.inference_engine import ObsDict

logger = logging.getLogger(__name__)


# RoboCasa365's stored state order: base pose, EEF pose, gripper.
_DEFAULT_STATE_KEYS: tuple[tuple[str, int | None], ...] = (
    ("state.base_position", 3),
    ("state.base_rotation", 4),
    ("state.end_effector_position_relative", 3),
    ("state.end_effector_rotation_relative", 4),
    ("state.gripper_qpos", 2),
)


def _gym_camera_key(checkpoint_key: str) -> str:
    """Map a checkpoint camera feature key to its gym observation key.

    ``robot0_eye_in_hand`` → ``video.robot0_eye_in_hand``. The reader stores
    bare camera names in the checkpoint schema; tolerate legacy prefixed keys.
    """
    name = checkpoint_key
    for prefix in ("observation.images.", "images."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name if name.startswith("video.") else f"video.{name}"


class RoboCasaAdapter:
    """Convert a connector-wrapped RoboCasa gym observation into ``ObsDict``.

    Parameters
    ----------
    camera_keys : tuple[str, ...]
        Camera names the checkpoint was trained on (from the saved schema).
        Each is mapped back to a ``video.robot0_<cam>`` key in the gym obs and
        must be present as an HWC uint8 RGB array.
    state_dim : int
        Expected proprioception width; asserted against the assembled state.
    state_keys : sequence[tuple[str, int | None]] | None
        Gym obs keys to concatenate into the state vector, in order, with an
        optional expected width per key (``None`` = take whatever length the
        gym exposes). Defaults to the RoboCasa365 16-D Panda assembly.
    """

    def __init__(
        self,
        camera_keys: tuple[str, ...],
        state_dim: int,
        state_keys: Sequence[tuple[str, int | None]] | None = None,
    ) -> None:
        self._camera_keys = tuple(camera_keys)
        self._state_dim = state_dim
        self._state_keys = (
            tuple(state_keys) if state_keys is not None else _DEFAULT_STATE_KEYS
        )
        logger.info(
            "RoboCasaAdapter — cameras: %s, state_dim: %d, state_keys: %s",
            list(self._camera_keys), state_dim,
            [k for k, _ in self._state_keys],
        )

    def __call__(self, observation: dict[str, Any], task: str = "") -> ObsDict:
        if not isinstance(observation, dict):
            raise TypeError(
                "RoboCasa observation must be a dict, got "
                f"{type(observation).__name__}."
            )

        if "robocasa_observation" not in observation:
            raise KeyError(
                "RoboCasa connector request must contain 'robocasa_observation'. "
                f"Available: {list(observation.keys())}"
            )
        request = observation
        gym_obs = request["robocasa_observation"]
        if not isinstance(gym_obs, dict):
            raise TypeError(
                "'robocasa_observation' must be a dict, got "
                f"{type(gym_obs).__name__}."
            )
        language = (
            request.get("instruction")
            or gym_obs.get("annotation.human.task_description")
            or task
            or None
        )

        video: dict[str, np.ndarray] = {}
        for cam in self._camera_keys:
            gym_key = _gym_camera_key(cam)
            raw = gym_obs.get(gym_key)
            if raw is None:
                # Tolerate raw checkpoint keys for custom wrappers.
                raw = gym_obs.get(cam) or gym_obs.get(cam.split(".")[-1])
            if raw is None:
                raise KeyError(
                    f"Camera '{cam}' (gym key '{gym_key}') not found in "
                    "RoboCasa observation. The runtime task must expose every "
                    "camera stored in the checkpoint schema. "
                    f"Available: {sorted(gym_obs.keys())}"
                )
            arr = np.asarray(raw)
            if arr.ndim != 3:
                raise ValueError(
                    f"Camera '{cam}' must be HWC (3D), got shape {arr.shape}. "
                    "Send raw uint8 RGB; the model's transform pipeline handles "
                    "float/CHW/normalisation."
                )
            video[cam] = arr

        state = self._assemble_state(gym_obs)
        if state.shape[0] != self._state_dim:
            raise ValueError(
                f"Assembled state length {state.shape[0]} != model state_dim "
                f"{self._state_dim}. Check the embodiment / state assembly used "
                "at deploy vs train time (RoboCasa365 stores 16-D: base pose, "
                "EEF pose, gripper)."
            )

        return ObsDict(video=video, state=state, language=language)

    def _assemble_state(self, gym_obs: dict[str, Any]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for key, expected in self._state_keys:
            value = gym_obs.get(key)
            if value is None:
                raise KeyError(
                    f"State key '{key}' not found in RoboCasa observation. "
                    "The checkpoint's proprioception was trained on a vector "
                    f"assembled from { [k for k, _ in self._state_keys] }; "
                    f"available: {sorted(gym_obs.keys())}"
                )
            vec = np.asarray(value, dtype=np.float32).reshape(-1)
            if expected is not None and vec.shape[0] != expected:
                raise ValueError(
                    f"State key '{key}' has {vec.shape[0]} values, expected "
                    f"{expected}. The gym observation layout does not match "
                    "the checkpoint's state assembly."
                )
            parts.append(vec)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
