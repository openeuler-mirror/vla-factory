"""Unified data-layer schema, runtime records, and description entry point.

Here, "data schema" names the whole format-neutral representation owned by the
data layer, not only the :class:`DataSchema` dataclass.  It includes static
dataset facts, normalisation statistics, runtime episode/frame records, and the
entry point that describes an external dataset.  Storage-specific knowledge
lives behind :mod:`vla_factory.data.reader`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from numpy.typing import NDArray


@dataclass(frozen=True)
class VideoRef:
    """Location of one encoded image frame, without decoding behaviour."""

    video_path: Path
    frame_index: int
    height: int
    width: int
    channels: int = 3
    stream: str | None = None


@dataclass
class Frame:
    """One format-neutral dataset timestep."""

    index: int
    images: dict[str, VideoRef]
    state: NDArray | None
    action: NDArray | None
    timestamp: float | None = None
    is_first: bool = False
    is_last: bool = False
    language: str | None = None


@dataclass
class Episode:
    """An episode whose frames are materialised lazily and cached on demand."""

    episode_id: str
    episode_index: int
    num_frames: int
    _frame_loader: object | None = None
    _frames_cache: list[Frame] | None = field(default=None, repr=False)

    def frames(self) -> Iterator[Frame]:
        """Iterate frames, reusing an existing materialised cache."""
        if self._frames_cache is not None:
            yield from self._frames_cache
            return
        if self._frame_loader is not None:
            yield from self._frame_loader()

    def load_frames(self) -> list[Frame]:
        """Materialise all frames once and return the cached list."""
        if self._frames_cache is not None:
            return self._frames_cache
        if self._frame_loader is None:
            return []
        self._frames_cache = list(self._frame_loader())
        return self._frames_cache


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
    / …) are exposed as **read-only derived properties** so consumers share one
    derivation of common widths and keys instead of rebuilding them.

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

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": list(self.mean),
            "std": list(self.std),
            "min": list(self.min),
            "max": list(self.max),
            "q01": list(self.q01),
            "q99": list(self.q99),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "FeatureStats | None":
        if d is None:
            return None
        return cls(
            mean=list(d.get("mean") or []),
            std=list(d.get("std") or []),
            min=list(d.get("min") or []),
            max=list(d.get("max") or []),
            q01=list(d.get("q01") or []),
            q99=list(d.get("q99") or []),
        )


@dataclass(frozen=True)
class NormStats:
    """Normalisation statistics for state, action, and image features."""

    state: FeatureStats | None = None
    action: FeatureStats | None = None
    images: dict[str, FeatureStats] | None = None
    method: str = "zscore"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict() if self.state is not None else None,
            "action": self.action.to_dict() if self.action is not None else None,
            "images": (
                {key: value.to_dict() for key, value in self.images.items()}
                if self.images is not None else None
            ),
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "NormStats":
        """Rebuild statistics from their serialized form.

        The counterpart of ``dataclasses.asdict``, which is how they are written
        into a checkpoint's ``inference_metadata/`` — the deploy process rebuilds
        them from there and never re-reads the training dataset.
        """
        d = d or {}
        images_raw = d.get("images")
        images = (
            {k: FeatureStats.from_dict(v) for k, v in images_raw.items()}
            if isinstance(images_raw, dict) else None
        )
        return cls(
            state=FeatureStats.from_dict(d.get("state")),
            action=FeatureStats.from_dict(d.get("action")),
            images=images,
            method=d.get("method", "zscore"),
        )


def describe_dataset(
    path: str | Path,
    format_name: str = "auto",
) -> tuple[DataSchema, NormStats]:
    """Read one dataset's schema and normalisation statistics as a pair."""
    # Reader implementations construct the result types defined above, so the
    # registry import stays at the orchestration boundary to avoid a cycle.
    from .reader import get_reader

    dataset_path = Path(path)
    reader = get_reader(format_name, path=dataset_path)
    return reader.get_schema(dataset_path), reader.get_norm_stats(dataset_path)


# ── State/action key resolution ──────────────────────────────────


def resolve_vector_keys(
    schema: DataSchema,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate the canonical per-dimension key order carried by the schema.

    The dimension→key mapping is a *data/model contract*: it must come from
    the dataset feature ``names`` (resolved into the schema by the format
    reader at training time and saved inside ``assembly.json.schema_ref``),
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
