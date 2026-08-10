"""Core data structures for the VLA-Factory data pipeline.

These dataclasses describe *what* a training sample looks like and *where*
to find the raw data, without depending on any specific storage format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
class CameraEntry:
    """One camera stream (data-module §8.3 ``observation.cameras[]``).

    ``resolution`` / ``fps`` / ``encoding`` are directly probed (measured).
    ``semantic`` is the only inferred field: the unique deterministic match of
    the camera key against the ``CAMERA_SEMANTICS`` vocabulary, or ``None`` when
    no unique match exists (undeclared → the resolver asks for a controlled
    override). ``semantic_source`` records which of the three it is.
    """

    key: str
    resolution: tuple[int, int] | None = None
    fps: int | None = None
    encoding: str | None = None
    semantic: str | None = None
    semantic_source: str = "undeclared"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "resolution": list(self.resolution) if self.resolution is not None else None,
            "fps": self.fps,
            "encoding": self.encoding,
            "semantic": self.semantic,
            "semantic_source": self.semantic_source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CameraEntry":
        res = d.get("resolution")
        return cls(
            key=d.get("key", ""),
            resolution=tuple(res) if res else None,
            fps=d.get("fps"),
            encoding=d.get("encoding"),
            semantic=d.get("semantic"),
            semantic_source=d.get("semantic_source", "undeclared"),
        )


@dataclass(frozen=True)
class StateDim:
    """One proprioceptive vector dimension (data-module §8.3 ``state.dims[]``).

    ``name`` keeps the raw dataset suffix unstripped (e.g. ``shoulder_pan.pos``);
    ``source_field`` is the dataset field this dim was read from (lerobot: the
    whole ``observation.state`` vector; RoboTwin: one of the concatenated
    ``/joint_action/*`` segments).
    """

    name: str | None
    source_field: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "source_field": self.source_field}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StateDim":
        return cls(name=d.get("name"), source_field=d.get("source_field", ""))


@dataclass(frozen=True)
class ActionDim:
    """One action vector dimension (data-module §8.3 ``action.dims[]``).

    Like ``StateDim`` plus a per-dim ``mode`` (``joint_pos`` / ``joint_delta`` /
    ``joint_vel``) — there is intentionally no global control_mode; heterogeneous
    bodies mix modes per dim. ``mode`` is measured when the format spec binds it
    (RoboTwin ``/joint_action/*`` = qpos target), inferred from the name suffix
    for container formats (lerobot ``.pos`` / ``.vel``), or ``None`` when
    neither evidence exists (undeclared).
    """

    name: str | None
    source_field: str = ""
    mode: str | None = None
    mode_source: str = "undeclared"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_field": self.source_field,
            "mode": self.mode,
            "mode_source": self.mode_source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActionDim":
        return cls(
            name=d.get("name"),
            source_field=d.get("source_field", ""),
            mode=d.get("mode"),
            mode_source=d.get("mode_source", "undeclared"),
        )


@dataclass(frozen=True)
class DataSchema:
    """Static description of the dataset's feature space (data-module §8.3).

    Canonical storage is the entry-table form (``cameras`` / ``state.dims`` /
    ``action.dims`` as per-entry records with source annotations). The legacy
    flat fields (``state_dim`` / ``action_dim`` / ``cameras`` / ``image_sizes``
    / …) are exposed as **read-only derived properties** so the in-tree
    consumers need no changes this phase (decision D2); the compat layer is
    removed in phase 4 once downstream reads ``ResolvedAssembly`` instead.

    Serialized via :meth:`to_dict` / :meth:`from_dict` (entry-table form only —
    no legacy flat-format compatibility).
    """

    # ── identity ──
    identity_name: str = ""
    source_format: str = ""
    episodes: int = 0
    total_frames: int = 0

    # ── robot reference (string preserved; resolution is the resolver's job) ──
    robot_ref: str | None = None

    # ── entry tables ──
    cameras_entries: tuple[CameraEntry, ...] = ()
    state_dims: tuple[StateDim, ...] = ()
    action_dims: tuple[ActionDim, ...] = ()
    action_frequency_hz: int | None = None

    # ── temporal ──
    temporal_fps: int = 30

    # ── instruction ──
    instruction_task_field: str | None = None
    instruction_granularity: str | None = None

    # ── Legacy flat API (read-only derived; D2 compat layer) ──

    @property
    def state_dim(self) -> int:
        return len(self.state_dims)

    @property
    def action_dim(self) -> int:
        return len(self.action_dims)

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(e.key for e in self.cameras_entries)

    @property
    def image_sizes(self) -> dict[str, tuple[int, int]]:
        return {e.key: e.resolution for e in self.cameras_entries if e.resolution}

    @property
    def fps(self) -> int:
        return self.temporal_fps

    @property
    def has_language(self) -> bool:
        return self.instruction_task_field is not None

    @property
    def total_episodes(self) -> int:
        return self.episodes

    @property
    def robot_type(self) -> str:
        return self.robot_ref if self.robot_ref is not None else "unknown"

    @property
    def state_keys(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.state_dims if d.name is not None)

    @property
    def action_keys(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.action_dims if d.name is not None)

    # ── Serialization ──

    def to_dict(self) -> dict[str, Any]:
        instruction = (
            None
            if self.instruction_task_field is None
            else {
                "task_field": self.instruction_task_field,
                "granularity": self.instruction_granularity,
            }
        )
        return {
            "identity": {
                "name": self.identity_name,
                "source_format": self.source_format,
                "episodes": self.episodes,
                "total_frames": self.total_frames,
            },
            "robot_ref": {"name": self.robot_ref} if self.robot_ref is not None else None,
            "observation": {
                "cameras": [e.to_dict() for e in self.cameras_entries],
            },
            "state": {"dims": [d.to_dict() for d in self.state_dims]},
            "action": {
                "dims": [d.to_dict() for d in self.action_dims],
                "frequency_hz": self.action_frequency_hz,
            },
            "temporal": {"fps": self.temporal_fps},
            "instruction": instruction,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataSchema":
        """Rebuild from the entry-table structure produced by :meth:`to_dict`."""
        obs = d.get("observation") or {}
        action = d.get("action") or {}
        instruction = d.get("instruction")
        identity = d.get("identity") or {}
        return cls(
            identity_name=identity.get("name", ""),
            source_format=identity.get("source_format", ""),
            episodes=int(identity.get("episodes", 0)),
            total_frames=int(identity.get("total_frames", 0)),
            robot_ref=(d.get("robot_ref") or {}).get("name"),
            cameras_entries=tuple(
                CameraEntry.from_dict(e) for e in obs.get("cameras", [])
            ),
            state_dims=tuple(StateDim.from_dict(x) for x in (d.get("state") or {}).get("dims", [])),
            action_dims=tuple(ActionDim.from_dict(x) for x in action.get("dims", [])),
            action_frequency_hz=action.get("frequency_hz"),
            temporal_fps=int((d.get("temporal") or {}).get("fps", 30)),
            instruction_task_field=(
                instruction.get("task_field") if isinstance(instruction, dict) else None
            ),
            instruction_granularity=(
                instruction.get("granularity") if isinstance(instruction, dict) else None
            ),
        )


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

    With the per-entry dim table the "one key per dim" contract is structural;
    this collapses to a presence check: every dim of a non-empty vector must
    carry a canonical ``name``. Returns ``(state_keys, action_keys)``.
    """
    state_keys = _resolve_dim_names("state", schema.state_dims)
    action_keys = _resolve_dim_names("action", schema.action_dims)
    return state_keys, action_keys


def _resolve_dim_names(
    which: str, dims: tuple[StateDim | ActionDim, ...]
) -> tuple[str, ...]:
    if not dims:
        return ()
    names: list[str] = []
    for d in dims:
        if not d.name:
            raise ValueError(
                f"schema {which} vector has {len(dims)} dims but one is missing "
                "its canonical name. The dataset reader must provide exactly one "
                "canonical key per vector dimension before inference metadata is saved."
            )
        names.append(d.name)
    return tuple(names)
