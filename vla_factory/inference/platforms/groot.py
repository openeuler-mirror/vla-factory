"""GR00T platform adapter."""

from __future__ import annotations

import numpy as np

from vla_factory.inference.infer import ActionCommand, ObsDict, PolicyExecutor


class GROOTAdapter:
    """Expose an inference engine through GR00T's ``get_action`` interface.

    Only the method signature is adapted. ``tag`` is stored for future
    embodiment routing / schema mapping, which is not implemented yet.
    """

    def __init__(self, policy: PolicyExecutor, tag: str | None = None) -> None:
        self.policy = policy
        self.tag = tag

    def get_action(self, obs_dict: dict) -> np.ndarray:
        video = obs_dict.get("video", {})
        state = obs_dict.get("state")
        language = obs_dict.get("language")
        obs = ObsDict(
            video={key: np.asarray(value) for key, value in video.items()},
            state=np.asarray(state, dtype=np.float32) if state is not None else None,
            language=language,
        )
        command = self.policy.predict(obs)
        if not isinstance(command, ActionCommand):
            raise TypeError(
                "PolicyExecutor.predict() must return ActionCommand, got "
                f"{type(command).__name__}."
            )
        return command.values

    def reset(self) -> None:
        self.policy.reset()
