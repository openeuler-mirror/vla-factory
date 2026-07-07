"""LeRobot v3 format reader — reads parquet + locates video files.

Handles:
  1. ``meta/info.json`` → DataSchema
  2. ``meta/stats.json`` → NormStats
  3. ``data/*.parquet`` → episode lengths / ranges
  4. ``videos/{cam_key}/chunk-*/episode_*.mp4`` or ``file-*.mp4`` → Episode with VideoRef
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..manifest import DataSchema, FeatureStats, NormStats
from .base import Episode, Frame, VideoRef
from ..codec.base import VideoCodec

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


class LeRobotV3Reader:
    """Read LeRobot v3 datasets (parquet + MP4)."""

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

        state_dim = 0
        action_dim = 0
        cameras: list[str] = []
        image_sizes: dict[str, tuple[int, int]] = {}
        state_keys: list[str] = []
        action_keys: list[str] = []

        for key, spec in features.items():
            dtype = spec.get("dtype", "")
            shape = spec.get("shape", [])
            names = spec.get("names")
            if key == "action":
                action_dim = shape[0] if shape else 0
                if isinstance(names, list):
                    action_keys = names
            elif "state" in key.lower() and dtype != "video":
                state_dim = shape[0] if shape else 0
                if isinstance(names, list):
                    state_keys = names
            elif dtype == "video" or ("image" in key.lower() and len(shape) == 3):
                cam_name = key.split(".")[-1]
                cameras.append(cam_name)
                size = _camera_size(spec, shape)
                if size is not None:
                    image_sizes[cam_name] = size

        # Check language
        has_language = (
            (path / "meta" / "tasks.parquet").exists()
            or (path / "meta" / "tasks.jsonl").exists()
        )

        return DataSchema(
            state_dim=state_dim,
            action_dim=action_dim,
            cameras=tuple(cameras),
            image_sizes=image_sizes,
            fps=info.get("fps", 30),
            has_language=has_language,
            total_episodes=info.get("total_episodes", 0),
            total_frames=info.get("total_frames", 0),
            robot_type=info.get("robot_type", "unknown"),
            state_keys=tuple(state_keys),
            action_keys=tuple(action_keys),
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
            flat_mean = _flatten(val.get("mean", []))
            flat_std = _flatten(val.get("std", []))
            flat_min = _flatten(val.get("min", []))
            flat_max = _flatten(val.get("max", []))

            if "state" in key.lower() and "image" not in key.lower():
                state_stats = FeatureStats(
                    mean=flat_mean, std=flat_std, min=flat_min, max=flat_max
                )
            elif key == "action":
                action_stats = FeatureStats(
                    mean=flat_mean, std=flat_std, min=flat_min, max=flat_max
                )
            elif key.startswith("observation.images."):
                cam_name = key.split(".")[-1]
                images_stats[cam_name] = FeatureStats(
                    mean=flat_mean, std=flat_std, min=flat_min, max=flat_max
                )

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

        # Build video path map: cam_key → video_path
        video_paths = _resolve_video_paths(path, camera_keys, rows)

        # Compute global frame offset for each video file
        # (for multi-episode videos, frame_index in video = global `index`)
        global_offset = int(rows["index"].iloc[0])

        # Build frame loader closure
        def frame_loader() -> Iterator[Frame]:
            for _, row in rows.iterrows():
                images: dict[str, VideoRef] = {}
                for cam_key, meta in camera_keys.items():
                    cam_name = cam_key.split(".")[-1]
                    vpath = video_paths.get(cam_key)
                    if vpath is None:
                        continue
                    # For multi-episode video: use global `index` as video frame index
                    # For per-episode video: use `frame_index` as video frame index
                    vid_frame_idx = _compute_video_frame_index(
                        row, vpath, path, cam_key
                    )
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

                yield Frame(
                    index=int(row["index"]),
                    images=images,
                    state=state,
                    action=action,
                    timestamp=timestamp,
                    is_first=(frame_idx == 0),
                    is_last=(frame_idx == num_frames - 1),
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


def _resolve_video_paths(
    dataset_path: Path,
    camera_keys: dict[str, dict[str, Any]],
    rows: Any,
) -> dict[str, Path]:
    """Resolve video file paths for each camera key.

    Tries these patterns in order:
      1. ``videos/{cam_key}/chunk-{chunk}/episode_{ep_idx:06d}.mp4`` (per-episode)
      2. ``videos/{cam_key}/chunk-{chunk}/{any_file}.mp4`` (multi-episode)
      3. ``videos/{cam_key}/{any_file}.mp4`` (flat structure)
    """
    result: dict[str, Path] = {}
    ep_idx = int(rows["episode_index"].iloc[0])

    for cam_key in camera_keys:
        videos_dir = dataset_path / "videos" / cam_key
        if not videos_dir.exists():
            continue

        # Pattern 1: per-episode video file
        for mp4 in sorted(videos_dir.rglob(f"episode_{ep_idx:06d}.mp4")):
            result[cam_key] = mp4
            break

        if cam_key in result:
            continue

        # Pattern 2 & 3: any mp4 file (multi-episode or flat)
        mp4_files = sorted(videos_dir.rglob("*.mp4"))
        if mp4_files:
            result[cam_key] = mp4_files[0]

    return result


def _compute_video_frame_index(
    row: Any,
    video_path: Path,
    dataset_path: Path,
    cam_key: str,
) -> int:
    """Compute the frame index within the video file.

    For per-episode videos: use ``frame_index``.
    For multi-episode videos: use ``index`` (global position).
    """
    # Check if the video filename contains "episode_XXXXXX"
    name = video_path.stem
    if name.startswith("episode_"):
        return int(row["frame_index"])
    # Multi-episode video: use global index
    return int(row["index"])
