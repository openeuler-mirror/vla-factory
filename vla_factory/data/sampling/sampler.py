"""Sliding-window sampler for episode -> training-sample indexing.

Generates one ``SampleLocator`` per frame — every frame is a training sample.
Episode tail frames that don't have enough future frames for a full
``action_horizon`` are still included; the dataset layer handles repeat-last
padding and generates an ``action_is_pad`` mask so the loss function can
ignore padded positions.

Example
-------
414-frame episode, n_obs_steps=1, action_horizon=100:

    414 locators: start_frame_index = 0, 1, 2, ..., 413
    The last 99 locators have padded action horizons.
"""

from __future__ import annotations

from ..manifest import SampleLocator


class SlidingWindowSampler:
    """Generate one ``SampleLocator`` per frame in an episode.

    Parameters
    ----------
    n_obs_steps : int
        Number of observation frames per sample (typically 1).
    action_horizon : int
        Number of future action frames per sample.
    """

    def __init__(
        self,
        n_obs_steps: int = 1,
        action_horizon: int = 100,
    ) -> None:
        self.n_obs_steps = n_obs_steps
        self.action_horizon = action_horizon

    def sample_episode(
        self, episode_index: int, episode_length: int
    ) -> list[SampleLocator]:
        """Return locators for every training sample in one episode.

        Parameters
        ----------
        episode_index : int
            Zero-based episode identifier.
        episode_length : int
            Number of frames in the episode.

        Returns
        -------
        list[SampleLocator]
            One locator per valid observation position.
        """
        if episode_length < self.n_obs_steps:
            return []

        return [
            SampleLocator(
                episode_index=episode_index,
                start_frame_index=pos,
                n_obs_steps=self.n_obs_steps,
                action_horizon=self.action_horizon,
            )
            for pos in range(episode_length - self.n_obs_steps + 1)
        ]
