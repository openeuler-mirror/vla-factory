"""lerobot host wire-format observation/action adapter.

Adapts between the **lerobot host wire format** (a flat JSON dict of per-motor
state scalars + base64-JPEG camera frames, exchanged over ZMQ) and VLA
Factory's :class:`InferenceEngine` input/output format. This is a *transport*
concern — it is embodiment-agnostic and carries no knowledge of any specific
robot. It does not sort or invent motor keys: the dimension→key order is a
data/model contract that the caller supplies (resolved by
``data.manifest.resolve_vector_keys`` from the dataset ``names``). The adapter
simply reads/writes those exact keys in the given order, so each dimension
maps to the motor it was trained on.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import cv2
import numpy as np

from vla_factory.deploy.infer import ObsDict

logger = logging.getLogger(__name__)


class LerobotHostObsAdapter:
    """Convert a lerobot host observation dict → :class:`ObsDict`.

    The state vector is reassembled by reading exactly ``state_keys`` from the
    host dict **in order** — no sorting, no auto-detection. A missing key
    raises a clear ``KeyError`` rather than silently misordering dimensions.

    Parameters
    ----------
    camera_keys : tuple[str, ...]
        Camera key names (from the checkpoint schema). Each must be present in
        the host dict as a base64 JPEG string or an ndarray.
    state_keys : tuple[str, ...]
        Ordered per-dimension state key names. Read in this exact order to
        build the state vector; length must equal ``state_dim``.
    state_dim : int
        Expected state dimension. Asserted to match ``len(state_keys)``.
    """

    def __init__(
        self,
        camera_keys: tuple[str, ...],
        state_keys: tuple[str, ...],
        state_dim: int,
    ) -> None:
        self._camera_keys = tuple(camera_keys)
        self._state_keys = tuple(state_keys)
        assert len(self._state_keys) == state_dim, (
            f"state_keys count ({len(self._state_keys)}) does not match "
            f"state_dim ({state_dim}); got keys {list(self._state_keys)}"
        )
        logger.info(
            "LerobotHostObsAdapter — state keys: %s, cameras: %s",
            list(self._state_keys), list(self._camera_keys),
        )

    def __call__(self, observation: dict[str, Any], task: str = "") -> ObsDict:
        # Decode camera images (exact keys from the schema — no guessing).
        video: dict[str, np.ndarray] = {}
        for cam in self._camera_keys:
            raw = observation.get(cam)
            if raw is None:
                raise KeyError(
                    f"Camera '{cam}' not found in observation. "
                    f"Available: {list(observation.keys())}"
                )
            if isinstance(raw, str):
                video[cam] = _decode_base64_jpeg(raw)
            elif isinstance(raw, np.ndarray):
                video[cam] = raw
            else:
                raise TypeError(
                    f"Unexpected type for camera '{cam}': {type(raw).__name__}"
                )

        # Build the state vector from exactly state_keys, in order.
        state_parts: list[float] = []
        for key in self._state_keys:
            if key not in observation:
                raise KeyError(
                    f"State motor key '{key}' not found in observation. "
                    f"Available: {list(observation.keys())}"
                )
            state_parts.append(float(observation[key]))
        state = np.array(state_parts, dtype=np.float32)

        return ObsDict(video=video, state=state, language=task or None)


class LerobotHostActionAdapter:
    """Convert a VLA Factory action array → lerobot host action dict.

    Maps ``action[i] → action_keys[i]`` — exactly, in order. No
    auto-detection, no ``ready()``/``learn_keys()``: the key mapping is a
    contract supplied at construction.

    Parameters
    ----------
    action_dim : int
        Number of action dimensions the model outputs.
    action_keys : tuple[str, ...]
        Ordered per-dimension action key names; ``action[i]`` is sent to
        ``action_keys[i]``. Length must equal ``action_dim``.
    """

    def __init__(
        self,
        action_dim: int,
        action_keys: tuple[str, ...],
    ) -> None:
        self.action_dim = action_dim
        self._action_keys = tuple(action_keys)
        assert len(self._action_keys) == action_dim, (
            f"action_keys count ({len(self._action_keys)}) does not match "
            f"action_dim ({action_dim}); got keys {list(self._action_keys)}"
        )
        logger.info(
            "LerobotHostActionAdapter — action mapping: %s",
            list(self._action_keys),
        )

    def __call__(self, action: np.ndarray) -> dict[str, float]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        return {key: float(val) for key, val in zip(self._action_keys, action)}


def _decode_base64_jpeg(b64_str: str) -> np.ndarray:
    """Decode a base64 JPEG string to HWC uint8 RGB ndarray."""
    jpg_data = base64.b64decode(b64_str)
    np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to JPEG-decode observation image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
