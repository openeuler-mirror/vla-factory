"""LeRobot v3 format reader — reads parquet + locates video files.

Handles:
  1. ``meta/info.json`` → DataSchema
  2. ``meta/stats.json`` → NormStats
  3. ``data/*.parquet`` → episode lengths / ranges
  4. ``videos/{cam_key}/chunk-*/episode_*.mp4`` or ``file-*.mp4`` → Episode with
     VideoRef (dataset global ``index`` mapped to the correct video file + its
     within-file offset via per-file cumulative global ranges)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..data_schema import (
    ActionDim,
    CameraEntry,
    DataSchema,
    Episode,
    FeatureStats,
    Frame,
    NormStats,
    StateDim,
    VideoRef,
)
from ..semantics import (
    DATA_INFERRED,
    DATA_MEASURED,
    DATA_UNDECLARED,
    infer_action_mode,
    infer_camera_semantic,
)
from ..codec.base import VideoCodec
from .registry import ReaderRegistry

logger = logging.getLogger(__name__)


def _camera_size(spec: dict[str, Any], shape: list[Any]) -> tuple[int, int] | None:
    """Best-effort ``(H, W)`` for a camera feature.

    LeRobot v3 video features are typically 4D ``[frames, H, W, C]`` while plain
    image features are 3D ``[H, W, C]``. Prefer explicit ``video_info`` metadata;
    otherwise read H/W from the correct axes according to the dimension count so
    a 4D shape like ``[1, 480, 640, 3]`` resolves to ``(480, 640)`` rather than
    the previous buggy ``(1, 480)``.
    """
    video_info = spec.get("video_info") or spec.get("info") or {}
    height = video_info.get("video.height")
    width = video_info.get("video.width")
    if height is not None and width is not None:
        return (int(height), int(width))

    def _pair(a: Any, b: Any) -> tuple[int, int] | None:
        return (int(a), int(b)) if isinstance(a, int) and isinstance(b, int) else None

    if len(shape) == 3:
        return _pair(shape[0], shape[1])
    if len(shape) == 4:
        return _pair(shape[1], shape[2])
    return None


def _load_tasks(path: Path) -> dict[int, str]:
    """Load ``{task_index: task_text}`` from ``meta/tasks.parquet`` or ``tasks.jsonl``.

    LeRobot v3 stores the task table separately; each frame carries a
    ``task_index`` column that indexes into it. Returns ``{}`` when no task
    file is present (non-language-conditioned datasets).
    """
    pq_tasks = path / "meta" / "tasks.parquet"
    if pq_tasks.exists():
        df = pq.read_table(pq_tasks).to_pandas()
        if "task_index" in df.columns and "task" in df.columns:
            return {int(r["task_index"]): str(r["task"]) for _, r in df.iterrows()}
        return {}
    jsonl_tasks = path / "meta" / "tasks.jsonl"
    if jsonl_tasks.exists():
        out: dict[int, str] = {}
        for line in jsonl_tasks.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[int(obj["task_index"])] = str(obj["task"])
        return out
    return {}



@dataclass(frozen=True)
class _VideoFileSpan:
    """One multi-episode video file and the dataset-global range it covers."""

    video_path: Path
    global_start: int  # inclusive
    global_end: int  # inclusive (global_start + frame_count - 1)
    frame_count: int


def _video_frame_count(video_path: Path) -> int:
    """Return a video file's frame count, preferring container metadata.

    ``av`` exposes ``stream.frames`` for most muxers (O(1) header read). When the
    metadata reports 0/None, fall back to an exact decode-count; that path is slow
    but runs once per file (cached upstream by the reader).
    """
    container = av.open(str(video_path))
    try:
        streams = container.streams.video
        if not streams:
            raise ValueError(f"No video stream in {video_path}")
        n = streams[0].frames
        if n is not None and n > 0:
            return int(n)
        return sum(1 for _ in container.decode(streams[0]))
    finally:
        container.close()


def _build_video_spans(dataset_path: Path, cam_key: str) -> list[_VideoFileSpan]:
    """Map each multi-episode video file to its dataset-global index range.

    ``file-NNN.mp4`` files are sorted by path so filename order equals global
    order; each file's frame count extends a running global cursor. Per-episode
    files (``episode_*.mp4``) are excluded — they have no global-range meaning and
    are handled by the per-episode path.
    """
    videos_dir = dataset_path / "videos" / cam_key
    if not videos_dir.exists():
        return []
    mp4s = sorted(
        p for p in videos_dir.rglob("*.mp4") if not p.stem.startswith("episode_")
    )
    spans: list[_VideoFileSpan] = []
    cursor = 0
    for mp4 in mp4s:
        n = _video_frame_count(mp4)
        if n <= 0:
            raise ValueError(f"Video file has 0 decodable frames: {mp4}")
        spans.append(_VideoFileSpan(mp4, cursor, cursor + n - 1, n))
        cursor += n
    return spans


def _span_error(g: int, cam_key: str, spans: list[_VideoFileSpan]) -> ValueError:
    """Build an actionable error when a global index matches no video span."""
    details = ", ".join(
        f"{s.video_path.name}[{s.global_start}-{s.global_end}]" for s in spans
    )
    return ValueError(
        f"Global frame index {g} (camera '{cam_key}') is not covered by any video "
        f"file; available spans: {details}. Likely a parquet/video frame-count "
        "mismatch or a missing/corrupt video file."
    )


@ReaderRegistry.register("lerobot-v3", aliases=("lerobot_v3",))
class LeRobotV3Reader:
    """Read LeRobot v3 datasets (parquet + MP4)."""

    def __init__(self) -> None:
        # tasks is a dataset-level static table; cache it per dataset path so
        # per-episode reads don't re-parse meta/tasks.* (N+1 I/O otherwise).
        self._tasks_cache: dict[Path, dict[int, str]] = {}
        # Video file→global-range spans are dataset-level too; cache per
        # (dataset_path, cam_key) so per-episode reads don't re-open every mp4.
        self._video_spans_cache: dict[tuple[Path, str], list[_VideoFileSpan]] = {}

    def can_read(self, path: Path) -> bool:
        """Check for ``meta/info.json`` with ``codebase_version >= 3.0``."""
        info_path = path / "meta" / "info.json"
        if not info_path.exists():
            return False
        try:
            with open(info_path) as f:
                info = json.load(f)
            version = info.get("codebase_version", "2.0")
            # Accept both "v3.0" and "3.0" formats
            v = version.lstrip("v")
            return v >= "3.0"
        except (json.JSONDecodeError, ValueError, IOError, OSError):
            return False

    # ── Schema & stats (from meta/ JSON files) ────────────────────

    def get_schema(self, path: Path) -> DataSchema:
        """Infer ``DataSchema`` from ``meta/info.json``."""
        info = _load_json(path / "meta" / "info.json")
        features: dict[str, Any] = info.get("features", {})

        state_dims: list[StateDim] = []
        action_dims: list[ActionDim] = []
        cameras: list[CameraEntry] = []

        for key, spec in features.items():
            dtype = spec.get("dtype", "")
            shape = spec.get("shape", [])
            names = spec.get("names")
            video_info = spec.get("video_info") or spec.get("info") or {}

            if key == "action":
                dim = shape[0] if shape else 0
                name_list = list(names) if isinstance(names, list) else []
                for i in range(dim):
                    nm = name_list[i] if i < len(name_list) else None
                    mode = infer_action_mode(nm) if nm else None
                    action_dims.append(ActionDim(
                        name=nm,
                        source_field="action",
                        mode=mode,
                        mode_source=DATA_INFERRED if mode else DATA_UNDECLARED,
                    ))
            elif "state" in key.lower() and dtype != "video":
                dim = shape[0] if shape else 0
                name_list = list(names) if isinstance(names, list) else []
                for i in range(dim):
                    nm = name_list[i] if i < len(name_list) else None
                    state_dims.append(StateDim(name=nm, source_field=key))
            elif dtype == "video" or ("image" in key.lower() and len(shape) == 3):
                cam_name = key.split(".")[-1]
                size = _camera_size(spec, shape)
                semantic = infer_camera_semantic(cam_name)
                cameras.append(CameraEntry(
                    key=cam_name,
                    resolution=size,
                    encoding=video_info.get("video.codec") or None,
                    semantic=semantic,
                    semantic_source=DATA_INFERRED if semantic else DATA_UNDECLARED,
                ))

        has_language = (
            (path / "meta" / "tasks.parquet").exists()
            or (path / "meta" / "tasks.jsonl").exists()
        )
        robot_type = info.get("robot_type", "unknown")

        return DataSchema(
            identity_name=path.name,
            source_format="lerobot_v3",
            episodes=info.get("total_episodes", 0),
            total_frames=info.get("total_frames", 0),
            robot_ref=None if robot_type in (None, "", "unknown") else str(robot_type),
            cameras_entries=tuple(cameras),
            state_dims=tuple(state_dims),
            action_dims=tuple(action_dims),
            action_frequency_hz=None,
            temporal_fps=info.get("fps", 30),
            instruction_task_field="task" if has_language else None,
            instruction_granularity=None,
        )

    def get_norm_stats(self, path: Path) -> NormStats:
        """Load normalisation statistics from ``meta/stats.json``."""
        stats_path = path / "meta" / "stats.json"
        if not stats_path.exists():
            return NormStats()

        raw = _load_json(stats_path)

        def _flatten(values: Any) -> list[float]:
            if isinstance(values, (int, float)):
                return [float(values)]
            if isinstance(values, list):
                result: list[float] = []
                for v in values:
                    result.extend(_flatten(v))
                return result
            return []

        state_stats = None
        action_stats = None
        images_stats: dict[str, FeatureStats] = {}

        for key, val in raw.items():
            stats = FeatureStats(
                mean=_flatten(val.get("mean", [])),
                std=_flatten(val.get("std", [])),
                min=_flatten(val.get("min", [])),
                max=_flatten(val.get("max", [])),
                # 1st/99th percentiles for quantile normalisation (pi05);
                # lerobot v3 writes them to stats.json, older stats may omit.
                q01=_flatten(val.get("q01", [])),
                q99=_flatten(val.get("q99", [])),
            )

            if "state" in key.lower() and "image" not in key.lower():
                state_stats = stats
            elif key == "action":
                action_stats = stats
            elif key.startswith("observation.images."):
                cam_name = key.split(".")[-1]
                images_stats[cam_name] = stats

        return NormStats(
            state=state_stats,
            action=action_stats,
            images=images_stats or None,
            method="zscore",
        )

    # ── Episode-level queries (from parquet) ──────────────────────

    def get_episode_lengths(self, path: Path) -> dict[int, int]:
        """Return ``{episode_index: num_frames}``."""
        lengths: dict[int, int] = {}
        for pq_file in sorted((path / "data").rglob("*.parquet")):
            table = pq.read_table(pq_file)
            df = table.to_pandas()
            if "episode_index" not in df.columns:
                continue
            for ep_idx in df["episode_index"].unique():
                count = int((df["episode_index"] == ep_idx).sum())
                lengths[ep_idx] = lengths.get(ep_idx, 0) + count
        return lengths

    def get_episode_ranges(self, path: Path) -> dict[int, tuple[int, int]]:
        """Return ``{ep_idx: (global_start, global_end)}`` (inclusive)."""
        data_dir = path / "data"
        if not data_dir.exists():
            return {}

        ranges: dict[int, tuple[int, int]] = {}
        for pq_file in sorted(data_dir.rglob("*.parquet")):
            table = pq.read_table(pq_file)
            df = table.to_pandas()
            if "episode_index" not in df.columns or "index" not in df.columns:
                continue
            for ep_idx in df["episode_index"].unique():
                ep_df = df[df["episode_index"] == ep_idx]
                start = int(ep_df["index"].min())
                end = int(ep_df["index"].max())
                if ep_idx not in ranges:
                    ranges[ep_idx] = (start, end)
                else:
                    prev_start, prev_end = ranges[ep_idx]
                    ranges[ep_idx] = (min(prev_start, start), max(prev_end, end))
        return ranges

    # ── Episode reading (parquet + video) ─────────────────────────

    def _video_spans(self, dataset_path: Path, cam_key: str) -> list[_VideoFileSpan]:
        """Cached per-(dataset, camera) multi-episode video global ranges."""
        key = (dataset_path, cam_key)
        spans = self._video_spans_cache.get(key)
        if spans is None:
            spans = _build_video_spans(dataset_path, cam_key)
            self._video_spans_cache[key] = spans
        return spans

    def _make_video_resolver(
        self,
        dataset_path: Path,
        cam_key: str,
        ep_idx: int,
        ep_global_start: int,
        ep_global_end: int,
    ) -> Callable[[Any], tuple[Path, int]] | None:
        """Build a ``row -> (video_path, within_file_index)`` resolver for one episode.

        Returns ``None`` when the camera has no video files. Per-episode videos
        (``episode_{ep_idx:06d}.mp4``) map via ``frame_index``. Multi-episode
        files (``file-*.mp4``) resolve to the file whose global range contains
        the episode's first frame; every frame in the episode then maps via
        ``index - file_global_start`` (codecs treat ``VideoRef.frame_index`` as a
        0-based within-file ordinal). An episode must fit entirely inside one
        multi-episode video file; if it crosses a file boundary we fail fast
        rather than silently decoding black/wrong frames.
        """
        videos_dir = dataset_path / "videos" / cam_key
        if not videos_dir.exists():
            return None

        # Pattern 1: per-episode video file → within-file index is frame_index.
        per_episode = None
        for mp4 in sorted(videos_dir.rglob(f"episode_{ep_idx:06d}.mp4")):
            per_episode = mp4
            break
        if per_episode is not None:
            def _per_episode(row: Any) -> tuple[Path, int]:
                return per_episode, int(row["frame_index"])

            return _per_episode

        # Patterns 2 & 3: multi-episode / flat files → cumulative global ranges.
        spans = self._video_spans(dataset_path, cam_key)
        if not spans:
            return None
        chosen = None
        for span in spans:
            if span.global_start <= ep_global_start <= span.global_end:
                chosen = span
                break
        if chosen is None or ep_global_end > chosen.global_end:
            raise _span_error(ep_global_start, cam_key, spans)

        def _multi_episode(row: Any) -> tuple[Path, int]:
            return chosen.video_path, int(row["index"]) - chosen.global_start

        return _multi_episode

    def read_episode(
        self, path: Path, episode_index: int, codec: VideoCodec
    ) -> Episode:
        """Read a single episode, building Frame objects with VideoRef."""
        # Load info for camera metadata
        info = _load_json(path / "meta" / "info.json")
        features: dict[str, Any] = info.get("features", {})

        # Identify camera keys and their video metadata
        camera_keys: dict[str, dict[str, Any]] = {}
        for key, spec in features.items():
            if spec.get("dtype") == "video":
                shape = spec.get("shape", [0, 0, 3])
                video_info = spec.get("video_info", spec.get("info", {}))
                camera_keys[key] = {
                    "shape": shape,
                    "height": video_info.get("video.height", shape[0]),
                    "width": video_info.get("video.width", shape[1]),
                    "channels": video_info.get("video.channels", shape[2] if len(shape) > 2 else 3),
                }

        # Read parquet data for this episode
        ep_rows = None
        for pq_file in sorted((path / "data").rglob("*.parquet")):
            table = pq.read_table(pq_file)
            df = table.to_pandas()
            if "episode_index" not in df.columns:
                continue
            mask = df["episode_index"] == episode_index
            if mask.any():
                chunk = df[mask].sort_values("frame_index")
                if ep_rows is None:
                    ep_rows = chunk
                else:
                    ep_rows = _concat_rows(ep_rows, chunk)

        if ep_rows is None or len(ep_rows) == 0:
            raise KeyError(f"Episode {episode_index} not found in {path}")

        num_frames = len(ep_rows)
        rows = ep_rows

        # Language: map each frame's task_index → task text (meta/tasks.*).
        tasks = self._tasks_cache.get(path)
        if tasks is None:
            tasks = _load_tasks(path)
            self._tasks_cache[path] = tasks

        # Per camera, a row → (video_path, within-file index) resolver.
        ep_idx = int(rows["episode_index"].iloc[0])
        ep_global_start = int(rows["index"].iloc[0])
        ep_global_end = int(rows["index"].iloc[-1])
        video_resolvers: dict[str, Callable[[Any], tuple[Path, int]]] = {}
        for cam_key in camera_keys:
            resolver = self._make_video_resolver(
                path, cam_key, ep_idx, ep_global_start, ep_global_end
            )
            if resolver is not None:
                video_resolvers[cam_key] = resolver

        # Build frame loader closure
        def frame_loader() -> Iterator[Frame]:
            for _, row in rows.iterrows():
                images: dict[str, VideoRef] = {}
                for cam_key, meta in camera_keys.items():
                    cam_name = cam_key.split(".")[-1]
                    resolver = video_resolvers.get(cam_key)
                    if resolver is None:
                        continue
                    vpath, vid_frame_idx = resolver(row)
                    images[cam_name] = VideoRef(
                        video_path=vpath,
                        frame_index=vid_frame_idx,
                        height=meta["height"],
                        width=meta["width"],
                        channels=meta["channels"],
                    )

                state = None
                if "observation.state" in row.index:
                    s = row["observation.state"]
                    state = np.array(s, dtype=np.float32) if s is not None else None

                action = None
                if "action" in row.index:
                    a = row["action"]
                    action = np.array(a, dtype=np.float32) if a is not None else None

                timestamp = None
                if "timestamp" in row.index:
                    timestamp = float(row["timestamp"])

                frame_idx = int(row.get("frame_index", row.get("index", 0)))

                language = None
                if "task_index" in row.index:
                    ti = row["task_index"]
                    if ti is not None and not (isinstance(ti, float) and np.isnan(ti)):
                        language = tasks.get(int(ti))

                yield Frame(
                    index=int(row["index"]),
                    images=images,
                    state=state,
                    action=action,
                    timestamp=timestamp,
                    is_first=(frame_idx == 0),
                    is_last=(frame_idx == num_frames - 1),
                    language=language,
                )

        return Episode(
            episode_id=f"episode_{episode_index:06d}",
            episode_index=episode_index,
            num_frames=num_frames,
            _frame_loader=frame_loader,
        )


# ── Module-level helpers ─────────────────────────────────────────


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _concat_rows(a: Any, b: Any) -> Any:
    return pd.concat([a, b], ignore_index=True)
