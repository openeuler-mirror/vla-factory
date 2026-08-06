"""Sample-manifest construction (training-side).

``build_manifest()`` assembles a ``DatasetManifest`` (train/val split + sample
locators) by sliding a window over each episode via the sampler. This is sample
construction, so it lives in the training layer — ``data/`` only owns the
read-only IR (``DataSchema`` / ``Episode`` / ``Frame`` / ``NormStats`` and the
``DatasetManifest`` / ``SampleLocator`` data classes) and must not depend on the
sampler or any other sample-building logic.
"""

from __future__ import annotations

import random

from vla_factory.data.manifest import (
    DataSchema,
    DatasetManifest,
    NormStats,
    SampleLocator,
)
from vla_factory.training.sampling.sampler import SlidingWindowSampler


def build_manifest(
    *,
    schema: DataSchema,
    norm_stats: NormStats,
    episode_ranges: dict[int, tuple[int, int]],
    episode_lengths: dict[int, int],
    n_obs_steps: int,
    action_horizon: int,
    split_strategy: str = "episode",
    train_ratio: float = 0.9,
    seed: int = 42,
) -> DatasetManifest:
    """Build a complete ``DatasetManifest``.

    Parameters
    ----------
    schema, norm_stats, episode_ranges
        Retrieved from the data source.
    episode_lengths
        Mapping ``episode_index -> num_frames``.
    n_obs_steps, action_horizon
        Sampler hyper-parameters (from ``SamplerConfig``).
    train_ratio, seed
        Episode-level train/val split.
    """
    # Validate split strategy — only "episode" is currently implemented.
    if split_strategy not in ("episode",):
        raise ValueError(
            f"split strategy '{split_strategy}' is not yet implemented. "
            f"Currently only 'episode' is supported."
        )

    sampler = SlidingWindowSampler(
        n_obs_steps=n_obs_steps,
        action_horizon=action_horizon,
    )

    # 1. Build all locators
    all_locators: list[SampleLocator] = []
    for ep_idx in sorted(episode_lengths.keys()):
        ep_len = episode_lengths[ep_idx]
        locators = sampler.sample_episode(ep_idx, ep_len)
        all_locators.extend(locators)

    # 2. Split episodes into train / val
    all_ep_indices = sorted(episode_lengths.keys())
    rng = random.Random(seed)
    rng.shuffle(all_ep_indices)
    n_train = max(1, int(len(all_ep_indices) * train_ratio))
    train_eps = set(all_ep_indices[:n_train])
    val_eps = set(all_ep_indices[n_train:])

    # 3. Map locator positions to split
    train_indices: list[int] = []
    val_indices: list[int] = []
    for i, loc in enumerate(all_locators):
        if loc.episode_index in train_eps:
            train_indices.append(i)
        elif loc.episode_index in val_eps:
            val_indices.append(i)

    manifest = DatasetManifest(
        locators=all_locators,
        schema=schema,
        norm_stats=norm_stats,
        episode_ranges=episode_ranges,
        splits={
            "train": sorted(train_eps),
            "val": sorted(val_eps),
        },
        train_indices=train_indices,
        val_indices=val_indices,
    )
    return manifest
