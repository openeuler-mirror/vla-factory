"""Core data structures for the VLA-Factory data pipeline.

These dataclasses describe *what* a training sample looks like and *where*
to find the raw data, without depending on any specific storage format.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from vla_factory.config.recipe import TrainRecipe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleLocator:
    """Pointer to a single training sample within an episode.

    Together with ``episode_ranges`` this fully determines which frames
    to load from the raw dataset.

    Attributes
    ----------
    episode_index : int
        Zero-based episode identifier.
    start_frame_index : int
        Frame index *within* the episode where the observation window starts.
    n_obs_steps : int
        Number of consecutive observation frames (usually 1 for ACT).
    action_horizon : int
        Number of future action frames to include.
    """

    episode_index: int
    start_frame_index: int
    n_obs_steps: int = 1
    action_horizon: int = 100


@dataclass(frozen=True)
class DataSchema:
    """Static description of the dataset's feature space.

    Built once at the start of a training run from ``meta/info.json``.
    """

    state_dim: int = 0
    action_dim: int = 0
    cameras: tuple[str, ...] = ()
    image_sizes: dict[str, tuple[int, int]] = field(default_factory=dict)
    fps: int = 30
    has_language: bool = False
    total_episodes: int = 0
    total_frames: int = 0
    robot_type: str = "unknown"
    # Per-dimension key names from dataset features["action/state"]["names"]
    state_keys: tuple[str, ...] = ()
    action_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeatureStats:
    """Per-feature normalisation statistics."""

    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    min: list[float] = field(default_factory=list)
    max: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class NormStats:
    """Normalisation statistics for state, action, and image features."""

    state: FeatureStats | None = None
    action: FeatureStats | None = None
    images: dict[str, FeatureStats] | None = None
    method: str = "zscore"


@dataclass
class DatasetManifest:
    """Complete index of all training samples.

    Built by ``build_manifest()`` from the episode list and sampler output.
    """

    locators: list[SampleLocator] = field(default_factory=list)
    schema: DataSchema = field(default_factory=DataSchema)
    norm_stats: NormStats = field(default_factory=NormStats)
    # episode_index -> (start_frame, end_frame)  inclusive
    episode_ranges: dict[int, tuple[int, int]] = field(default_factory=dict)
    # "train" | "val" -> list of episode indices
    splits: dict[str, list[int]] = field(default_factory=dict)

    # Convenience indices built from ``locators`` + ``splits``. Public (no leading
    # underscore): callers like VLADataset need the raw indices, not only the
    # locator views exposed by train_locators / val_locators.
    train_indices: list[int] = field(default_factory=list, repr=False)
    val_indices: list[int] = field(default_factory=list, repr=False)

    @property
    def train_locators(self) -> list[SampleLocator]:
        """Locators belonging to training episodes."""
        return [self.locators[i] for i in self.train_indices]

    @property
    def val_locators(self) -> list[SampleLocator]:
        """Locators belonging to validation episodes."""
        return [self.locators[i] for i in self.val_indices]


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
    from .sampling.sampler import SlidingWindowSampler

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


# ── State/action key resolution ──────────────────────────────────


def resolve_vector_keys(
    schema: DataSchema,
    recipe: TrainRecipe,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Best-effort resolution of the canonical per-dimension key order.

    The dimension→key mapping is a *data/model contract*: it must come from
    the dataset, never invented by sorting the host's motor keys (which would
    scramble every dimension — e.g. the shoulder command driving the elbow).

    Resolution precedence (applied independently to state and action):

    1. ``schema.state_keys`` / ``schema.action_keys`` — populated from the
       dataset feature ``names`` by ``LeRobotV3Reader.get_schema``.
    2. Re-read the dataset ``info.json`` live via
       ``get_reader(...).get_schema(...)`` using ``recipe.data.source.path``.
       Auto-heals a stale checkpoint ``schema.json`` whose ``names`` were
       absent at train time but present now.

    This is deliberately **lenient**: if neither source yields keys, it logs
    a warning and returns empty tuples rather than raising. The per-dimension
    order is only actually needed by platforms that reassemble vectors from
    per-motor keys (e.g. the lerobot host adapter); other platforms — and
    datasets that simply don't carry ``names`` — should not be blocked by a
    hard contract that doesn't apply to them. A platform that *does* need the
    keys will fail clearly at its own adapter construction if they're missing.

    Returns ``(state_keys, action_keys)`` as ordered tuples (possibly empty).
    """
    # ── Step 2: best-effort live re-read of the dataset schema ──────
    live_schema: DataSchema | None = None
    source_path = recipe.data.source.path
    if source_path and Path(source_path).exists():
        try:
            from .formats import get_reader

            reader = get_reader(recipe.data.source.format, path=Path(source_path))
            live_schema = reader.get_schema(Path(source_path))
        except Exception as exc:  # noqa: BLE001 — degrade gracefully, don't crash
            logger.warning(
                "Could not re-read dataset schema from %s for key resolution "
                "(%s: %s).",
                source_path, type(exc).__name__, exc,
            )

    live_state = live_schema.state_keys if live_schema else ()
    live_action = live_schema.action_keys if live_schema else ()

    state_keys = schema.state_keys or live_state
    action_keys = schema.action_keys or live_action

    state_keys = _validate_keys("state", state_keys, schema.state_dim)
    action_keys = _validate_keys("action", action_keys, schema.action_dim)

    return state_keys, action_keys


def _validate_keys(which: str, keys: tuple[str, ...], dim: int) -> tuple[str, ...]:
    """Check a resolved key list against the schema dimension.

    ``dim == 0`` means the vector is absent (e.g. a stateless policy); empty
    keys are correct and returned as-is. Missing keys (``dim > 0`` but no
    keys resolved) or a count mismatch are **warnings, not errors**: empty
    keys are returned so callers that don't need them are unaffected, while a
    platform that does need them (e.g. lerobot host) will surface a clear
    failure at its adapter.
    """
    if dim == 0:
        return ()

    if not keys:
        logger.warning(
            "Could not determine the canonical %s key order: the dataset has no "
            "feature `names`. This is only required by platforms that reassemble "
            "vectors from per-motor keys (e.g. lerobot host); other platforms "
            "ignore it. Returning empty %s_keys.",
            which, which,
        )
        return ()

    if len(keys) != dim:
        logger.warning(
            "%s_keys length mismatch: resolved %d keys %s but schema.%s_dim=%d. "
            "Returning empty %s_keys to avoid mis-ordering dimensions.",
            which, len(keys), list(keys), which, dim, which,
        )
        return ()

    return keys
