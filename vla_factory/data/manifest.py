"""Core data structures for the VLA-Factory data pipeline.

These dataclasses describe *what* a training sample looks like and *where*
to find the raw data, without depending on any specific storage format.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    """Per-feature normalisation statistics.

    ``q01``/``q99`` are the 1st/99th percentiles used by quantile
    normalisation (pi05). lerobot v3 ships them in ``meta/stats.json``; they
    stay empty for datasets/stats sources that do not provide quantiles.
    """

    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    min: list[float] = field(default_factory=list)
    max: list[float] = field(default_factory=list)
    q01: list[float] = field(default_factory=list)
    q99: list[float] = field(default_factory=list)


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


# ── State/action key resolution ──────────────────────────────────


def resolve_vector_keys(
    schema: DataSchema,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate the canonical per-dimension key order carried by the schema.

    The dimension→key mapping is a *data/model contract*: it must come from
    the dataset feature ``names`` (resolved into the schema by the format
    reader at training time and saved to ``inference_metadata/schema.json``),
    never invented by sorting the host's motor keys (which would scramble
    every dimension — e.g. the shoulder command driving the elbow). At
    inference time the checkpoint schema is the sole source; the training
    dataset is never re-read.

    For every non-empty vector, the schema must contain exactly one key per
    dimension. Missing or mismatched keys make the checkpoint metadata
    incomplete and fail immediately. ``dim == 0`` is the only case where an
    empty key tuple is valid.

    Returns ``(state_keys, action_keys)`` as ordered tuples.
    """
    state_keys = _validate_keys("state", schema.state_keys, schema.state_dim)
    action_keys = _validate_keys("action", schema.action_keys, schema.action_dim)
    return state_keys, action_keys


def _validate_keys(which: str, keys: tuple[str, ...], dim: int) -> tuple[str, ...]:
    """Check a resolved key list against the schema dimension.

    ``dim == 0`` means the vector is absent (e.g. a stateless policy), so an
    empty key tuple is valid. For ``dim > 0``, missing keys or a count mismatch
    violate the self-contained schema contract and are errors.
    """
    if dim == 0:
        return ()

    if not keys:
        raise ValueError(
            f"schema.{which}_keys is empty, but schema.{which}_dim={dim}. "
            "The dataset reader must provide exactly one canonical key per "
            "vector dimension before inference metadata is saved."
        )

    if len(keys) != dim:
        raise ValueError(
            f"schema.{which}_keys has {len(keys)} entries {list(keys)}, but "
            f"schema.{which}_dim={dim}. Expected exactly one canonical key per "
            "vector dimension."
        )

    return keys
