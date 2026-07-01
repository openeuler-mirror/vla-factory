"""Platform adapters for InferenceEngine (§12.6, §12.7).

Thin wrappers that adapt InferenceEngine's ``predict(ObsDict)`` interface
to various robot control platform APIs.
"""

from __future__ import annotations

import numpy as np
import torch

from vla_factory.deploy.infer import InferenceEngine, ObsDict


class ReplayPolicy:
    """Replay recorded episode data without model inference (§12.7).

    From GR00T's ReplayPolicy design.  Used for validating environment
    integration without wasting GPU inference time:

    1. Deploy new embodiment → use ReplayPolicy to verify obs→action format
    2. Confirm robot can execute recorded action sequence
    3. Then switch to real model inference

    Parameters
    ----------
    episode_data : list[dict]
        Each dict must have an ``"action"`` key with an ndarray value.
    """

    def __init__(self, episode_data: list[dict]) -> None:
        self.data = episode_data
        self._index = 0

    def predict(self, obs: ObsDict) -> np.ndarray:
        """Ignore input, return recorded action at current step."""
        if self._index >= len(self.data):
            raise StopIteration("Episode replay exhausted")
        action = np.asarray(self.data[self._index]["action"], dtype=np.float32)
        self._index += 1
        return action

    def reset(self) -> None:
        self._index = 0


class LeRobotAdapter:
    """Adapt InferenceEngine to LeRobot's ``policy.predict_action()`` interface.

    LeRobot control loop::

        action = policy.predict_action(obs_tensor_dict)
        env.step(action)

    After adaptation::

        engine = InferenceEngine(...)
        adapter = LeRobotAdapter(engine)
        action = adapter.predict_action(obs_tensor_dict)
    """

    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine

    def predict_action(self, obs_tensor_dict: dict) -> torch.Tensor:
        video: dict[str, np.ndarray] = {}
        for key, value in obs_tensor_dict.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if "image" in k or "images" in k:
                        arr = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
                        video[k] = arr
            elif "image" in key or "images" in key:
                arr = value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)
                video[key] = arr

        state_val = obs_tensor_dict.get("observation.state")
        state = None
        if state_val is not None:
            state = state_val.cpu().numpy() if hasattr(state_val, "cpu") else np.asarray(state_val)

        language = obs_tensor_dict.get("language_instruction")
        if language is not None and hasattr(language, "item"):
            language = language.item()

        obs = ObsDict(video=video, state=state, language=language)
        actions = self.engine.predict(obs)
        return torch.from_numpy(actions)

    def reset(self) -> None:
        self.engine.reset()


class GROOTAdapter:
    """Adapt InferenceEngine to GR00T's ``BasePolicy.get_action()`` interface.

    GR00T control loop::

        action = policy.get_action(obs_dict)
        robot.step(action)

    Supports embodiment tagging for multi-embodiment servers.
    """

    def __init__(self, engine: InferenceEngine, tag: str | None = None) -> None:
        self.engine = engine
        self.tag = tag

    def get_action(self, obs_dict: dict) -> np.ndarray:
        video = obs_dict.get("video", {})
        state = obs_dict.get("state")
        language = obs_dict.get("language")

        obs = ObsDict(
            video={k: np.asarray(v) for k, v in video.items()},
            state=np.asarray(state, dtype=np.float32) if state is not None else None,
            language=language,
        )
        return self.engine.predict(obs)

    def reset(self) -> None:
        self.engine.reset()
