"""Read RLDS episodes from TFDS-style TFRecord directories.

The first access scans record offsets and step counts into a validated
sidecar index. Episode reads then seek directly to one ``tf.train.Example``;
embedded JPEG/PNG frames remain lazy until the image codec needs them.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ...data_schema import (
    ActionDim,
    CameraEntry,
    DataSchema,
    Episode,
    Frame,
    ImageStorage,
    ImageRef,
    NormStats,
    StateDim,
)
from ...semantics import (
    DATA_INFERRED,
    DATA_UNDECLARED,
    infer_camera_semantic,
)
from ..registry import ReaderRegistry
from . import tfrecord
from .episode import (
    EpisodeEntry,
    context_language,
    episode_steps,
    flag_values,
    step_bytes,
    step_language,
    vector_rows,
)
from .layout import dataset_name, locate_root, shard_files
from .spec import CameraSpec, RldsSpec, load_rlds_spec
from .stats import StreamingStats, norm_stats_from_dlimp
from .tf_example import parse_example

logger = logging.getLogger(__name__)

_INDEX_FILENAME = ".vla_factory_rlds_index.json"
_INDEX_VERSION = 2


@dataclass
class _ScanResult:
    """The episode table (and scan-derived stats) for one dataset root."""

    root: Path
    shards: tuple[Path, ...]
    episodes: tuple[EpisodeEntry, ...]
    stats: NormStats | None = None


@ReaderRegistry.register("rlds", aliases=("rlds_tfrecord",))
class RldsReader:
    """Read RLDS/TFDS datasets (TFRecord shards of episode examples)."""

    def __init__(self) -> None:
        self._spec_cache: dict[Path, RldsSpec] = {}
        self._index_cache: dict[Path, _ScanResult] = {}
        self._camera_hw_cache: dict[tuple[Path, str], tuple[int, int]] = {}

    # ── Detection & root location ──────────────────────────────────

    def can_read(self, path: Path) -> bool:
        """True when ``path`` resolves to exactly one RLDS dataset directory."""
        try:
            root = locate_root(path)
            return self._load_spec(root) is not None
        except (FileNotFoundError, ValueError, OSError):
            return False

    # ── Spec loading ───────────────────────────────────────────────

    def _load_spec(self, root: Path) -> RldsSpec | None:
        """Cached feature-spec mapping; None when the directory is not RLDS."""
        if root not in self._spec_cache:
            spec = load_rlds_spec(root)
            if spec is None:
                return None
            self._spec_cache[root] = spec
        return self._spec_cache[root]

    def _require_spec(self, root: Path) -> RldsSpec:
        spec = self._load_spec(root)
        if spec is None:
            raise ValueError(
                f"{root}: no parsable features.json with a "
                "'steps' feature tree — not a readable RLDS dataset"
            )
        return spec

    # ── Index (episode table) ──────────────────────────────────────

    def _ensure_index(self, root: Path) -> _ScanResult:
        """Episode table for ``root``: sidecar, cache, or a fresh scan."""
        cached = self._index_cache.get(root)
        if cached is not None:
            return cached
        shards = shard_files(root)
        if not shards:
            raise FileNotFoundError(f"{root}: no *.tfrecord shard files found")
        result = self._try_load_sidecar(root, shards)
        if result is None:
            spec = self._require_spec(root)
            result = self._scan(root, spec, shards)
            self._write_sidecar(root, result)
        self._index_cache[root] = result
        return result

    def _scan(
        self, root: Path, spec: RldsSpec, shards: list[Path]
    ) -> _ScanResult:
        """One pass over every shard: episode spans, step counts, stats.

        Stats are computed only when the dataset ships no
        ``dataset_statistics.json`` — OXE-style distributions provide
        authoritative precomputed numbers that must not be shadowed.
        Records are parsed with a key filter so the scan never builds the
        per-step image lists (their payload bytes are skipped in place).
        """
        compute_stats = not (root / "dataset_statistics.json").exists()
        needed = {spec.action_path, spec.is_first_path}
        if spec.state_path:
            needed.add(spec.state_path)

        action_dim = spec.action_dim
        state_dim = spec.state_dim
        action_stats = StreamingStats() if compute_stats else None
        state_stats = StreamingStats() if compute_stats and spec.state_path else None

        started = time.perf_counter()
        episodes: list[EpisodeEntry] = []
        for shard in shards:
            for offset, length, data in tfrecord.scan_records(shard):
                feats = parse_example(data, keys=lambda k: k in needed)
                num_steps, action_dim = episode_steps(
                    feats, spec, action_dim, shard, offset
                )
                episodes.append(EpisodeEntry(shard, offset, length, num_steps))
                if action_stats is not None:
                    action_rows = vector_rows(
                        feats, spec.action_path, num_steps, action_dim
                    )
                    if action_rows is not None:
                        action_stats.add_rows(action_rows)
                if state_stats is not None and spec.state_path is not None:
                    state_rows = vector_rows(
                        feats, spec.state_path, num_steps, state_dim
                    )
                    if state_rows is not None:
                        state_stats.add_rows(state_rows)
                        if state_dim is None:
                            state_dim = state_rows.shape[1]

        stats = None
        if compute_stats and action_stats is not None:
            stats = NormStats(
                state=state_stats.finish() if state_stats is not None else None,
                action=action_stats.finish(),
                images=None,
                method="zscore",
            )

        logger.info(
            "RLDS index scan: %d episodes / %d frames across %d shards in %.1fs%s",
            len(episodes),
            sum(e.num_steps for e in episodes),
            len(shards),
            time.perf_counter() - started,
            " (stats computed)" if compute_stats else "",
        )
        return _ScanResult(root, tuple(shards), tuple(episodes), stats)

    def _try_load_sidecar(
        self, root: Path, shards: list[Path]
    ) -> _ScanResult | None:
        """Load a previously persisted scan when its fingerprint still matches."""
        path = root / _INDEX_FILENAME
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
            if raw.get("version") != _INDEX_VERSION:
                return None
            if raw.get("features") != _fingerprint(root / "features.json"):
                return None
            if raw.get("statistics") != _fingerprint(
                root / "dataset_statistics.json"
            ):
                return None
            infos = raw["shards"]
            if len(infos) != len(shards):
                return None
            for info, shard in zip(infos, shards):
                if info["name"] != shard.name:
                    return None
                stat = shard.stat()
                if info["size"] != stat.st_size or info["mtime_ns"] != stat.st_mtime_ns:
                    return None
            episodes = tuple(
                EpisodeEntry(
                    shards[int(entry[0])], int(entry[1]), int(entry[2]), int(entry[3])
                )
                for entry in raw["episodes"]
            )
            stats = NormStats.from_dict(raw["stats"]) if raw.get("stats") else None
        except (
            IndexError,
            json.JSONDecodeError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
        ):
            logger.debug("Unusable RLDS index sidecar %s; rescanning", path)
            return None
        logger.info("RLDS index loaded from sidecar %s", path)
        return _ScanResult(root, tuple(shards), episodes, stats)

    def _write_sidecar(self, root: Path, result: _ScanResult) -> None:
        """Persist the scan next to the shards (best effort, read-only safe)."""
        shard_positions = {shard: i for i, shard in enumerate(result.shards)}
        payload = {
            "version": _INDEX_VERSION,
            "features": _fingerprint(root / "features.json"),
            "statistics": _fingerprint(root / "dataset_statistics.json"),
            "shards": [
                {
                    "name": shard.name,
                    "size": shard.stat().st_size,
                    "mtime_ns": shard.stat().st_mtime_ns,
                }
                for shard in result.shards
            ],
            "episodes": [
                [
                    shard_positions[entry.shard],
                    entry.offset,
                    entry.length,
                    entry.num_steps,
                ]
                for entry in result.episodes
            ],
            "stats": result.stats.to_dict() if result.stats is not None else None,
        }
        target = root / _INDEX_FILENAME
        temporary: str | None = None
        try:
            fd, temporary = tempfile.mkstemp(prefix=f"{_INDEX_FILENAME}.", dir=root)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temporary, target)
            temporary = None
            logger.info("RLDS index persisted to %s", target)
        except OSError as exc:
            logger.info(
                "Cannot persist RLDS index (%s); the dataset will be "
                "rescanned on next start", exc,
            )
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    # ── Schema & stats ─────────────────────────────────────────────

    def get_schema(self, path: Path) -> DataSchema:
        root = locate_root(path)
        spec = self._require_spec(root)
        index = self._ensure_index(root)
        spec = self._resolve_dims(root, spec, index)

        cameras: list[CameraEntry] = []
        for cam in spec.cameras:
            height, width = cam.height, cam.width
            if height is None or width is None:
                height, width = self._probe_camera_hw(root, spec, index, cam)
            semantic = infer_camera_semantic(cam.key)
            cameras.append(CameraEntry(
                key=cam.key,
                resolution=(height, width) if height and width else None,
                encoding=cam.encoding,
                image_storage=ImageStorage.ENCODED_BYTES,
                semantic=semantic,
                semantic_source=DATA_INFERRED if semantic else DATA_UNDECLARED,
            ))

        state_leaf = spec.state_path.rsplit("/", 1)[-1] if spec.state_path else "state"
        state_dims = [
            StateDim(name=f"{state_leaf}_{i}", source_field=spec.state_path or "")
            for i in range(spec.state_dim or 0)
        ]
        # RLDS action semantics (joint-space vs end-effector, absolute vs
        # delta) vary per dataset and are not recorded in the spec — the
        # mode stays undeclared instead of guessed.
        action_dims = [
            ActionDim(name=f"action_{i}", source_field=spec.action_path or "")
            for i in range(spec.action_dim or 0)
        ]

        has_language = bool(spec.language_paths or spec.context_language_paths)
        return DataSchema(
            identity_name=dataset_name(root),
            source_format="rlds_tfrecord",
            episodes=len(index.episodes),
            total_frames=sum(e.num_steps for e in index.episodes),
            robot_ref=None,
            cameras_entries=tuple(cameras),
            state_dims=tuple(state_dims),
            action_dims=tuple(action_dims),
            action_frequency_hz=None,
            # RLDS metadata carries no fps; 30 is the neutral default the
            # same way RoboTwin pins its common 25.
            temporal_fps=30,
            instruction_task_field="task" if has_language else None,
            instruction_granularity=None,
        )

    def get_norm_stats(self, path: Path) -> NormStats:
        """Precomputed stats from ``dataset_statistics.json``, else the scan's.

        The OXE/dlimp convention ships one file per dataset with ``action``
        and ``state`` (or ``proprio``) blocks of mean/std/min/max/q01/q99.
        """
        root = locate_root(path)
        stats_file = root / "dataset_statistics.json"
        if stats_file.exists():
            try:
                raw = json.loads(stats_file.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                raise ValueError(f"Unparsable {stats_file}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{stats_file}: expected a JSON object")
            return norm_stats_from_dlimp(raw)
        return self._ensure_index(root).stats or NormStats()

    # ── Episode-level queries ──────────────────────────────────────

    def get_episode_lengths(self, path: Path) -> dict[int, int]:
        """Return ``{episode_index: num_frames}``."""
        index = self._ensure_index(locate_root(path))
        return {i: entry.num_steps for i, entry in enumerate(index.episodes)}

    def get_episode_ranges(self, path: Path) -> dict[int, tuple[int, int]]:
        """Return ``{ep_idx: (global_start, global_end)}`` (inclusive)."""
        ranges: dict[int, tuple[int, int]] = {}
        cursor = 0
        index = self._ensure_index(locate_root(path))
        for i, entry in enumerate(index.episodes):
            ranges[i] = (cursor, cursor + entry.num_steps - 1)
            cursor += entry.num_steps
        return ranges

    # ── Episode reading ────────────────────────────────────────────

    def read_episode(self, path: Path, episode_index: int) -> Episode:
        """Read one episode into canonical ``Frame`` objects.

        Images stay lazy as :class:`ImageRef` entries carrying the encoded
        bytes (decoded by an :class:`ImageDecoder`); state/action materialise from the
        float lists; language gathers every instruction variant recorded
        for the step.
        """
        root = locate_root(path)
        spec = self._require_spec(root)
        index = self._ensure_index(root)
        spec = self._resolve_dims(root, spec, index)
        if not 0 <= episode_index < len(index.episodes):
            raise KeyError(
                f"Episode {episode_index} not found in {root} "
                f"({len(index.episodes)} episodes)"
            )
        entry = index.episodes[episode_index]

        feats = parse_example(
            tfrecord.read_record(entry.shard, entry.offset, entry.length)
        )
        num_steps, _ = episode_steps(
            feats, spec, spec.action_dim, entry.shard, entry.offset
        )
        state_dim = spec.state_dim
        action_dim = spec.action_dim

        state_rows = vector_rows(feats, spec.state_path, num_steps, state_dim)
        action_rows = vector_rows(feats, spec.action_path, num_steps, action_dim)
        if action_rows is None:
            raise ValueError(
                f"{entry.shard}@{entry.offset}: feature '{spec.action_path}' "
                "missing from episode record"
            )

        default_first = [t == 0 for t in range(num_steps)]
        default_last = [t == num_steps - 1 for t in range(num_steps)]
        is_first_flags = flag_values(feats, spec.is_first_path, default_first)
        is_last_flags = flag_values(feats, spec.is_last_path, default_last)
        episode_language = context_language(feats, spec.context_language_paths)

        def frame_loader() -> Iterator[Frame]:
            for t in range(num_steps):
                images: dict[str, ImageRef] = {}
                for cam in spec.cameras:
                    payload = step_bytes(feats, cam.path, t, entry)
                    if payload is not None:
                        height, width = (
                            (cam.height, cam.width)
                            if cam.height and cam.width
                            else self._probe_camera_hw(root, spec, index, cam)
                        )
                        images[cam.key] = ImageRef(
                            path=entry.shard,
                            frame_index=t,
                            height=height,
                            width=width,
                            storage=ImageStorage.ENCODED_BYTES,
                            stream=cam.key,
                            payload=payload,
                        )
                language = step_language(feats, spec.language_paths, t)
                yield Frame(
                    index=t,
                    images=images,
                    state=state_rows[t] if state_rows is not None else None,
                    action=action_rows[t],
                    timestamp=None,
                    is_first=is_first_flags[t],
                    is_last=is_last_flags[t],
                    language=language or episode_language or None,
                )

        return Episode(
            episode_id=f"episode_{episode_index:06d}",
            episode_index=episode_index,
            num_frames=num_steps,
            _frame_loader=frame_loader,
        )

    # ── Internal helpers ───────────────────────────────────────────

    def _resolve_dims(
        self, root: Path, spec: RldsSpec, index: _ScanResult
    ) -> RldsSpec:
        """Fill dynamically-shaped vector widths from the first episode.

        features.json shapes are per-step and usually static; a dynamic
        width (``[-1, -1]`` or a missing shape) is recovered by dividing
        the first episode's action list by its step count.
        """
        dims_known = spec.action_dim is not None and (
            spec.state_dim is not None or spec.state_path is None
        )
        if dims_known or not index.episodes:
            return spec
        first = index.episodes[0]
        wanted = {spec.action_path, spec.state_path, spec.is_first_path}
        feats = parse_example(
            tfrecord.read_record(first.shard, first.offset, first.length),
            keys=lambda k: k in wanted,
        )
        _, action_dim = episode_steps(
            feats, spec, spec.action_dim, first.shard, first.offset
        )
        state_dim = spec.state_dim
        if spec.state_path is not None and state_dim is None:
            state_rows = vector_rows(feats, spec.state_path, first.num_steps, None)
            if state_rows is not None:
                state_dim = state_rows.shape[1]
        spec = replace(spec, action_dim=action_dim, state_dim=state_dim)
        self._spec_cache[root] = spec
        return spec

    def _probe_camera_hw(
        self, root: Path, spec: RldsSpec, index: _ScanResult, cam: CameraSpec
    ) -> tuple[int, int]:
        """Decode the camera's first frame to measure its resolution."""
        key = (root, cam.path)
        if key not in self._camera_hw_cache:
            if not index.episodes:
                raise ValueError(f"{root}: empty dataset cannot resolve camera sizes")
            entry = index.episodes[0]
            feats = parse_example(
                tfrecord.read_record(entry.shard, entry.offset, entry.length),
                keys=lambda k: k == cam.path,
            )
            values = feats.get(cam.path)
            if values is None or not values.bytes_values:
                raise ValueError(
                    f"Camera feature '{cam.path}' missing from the first "
                    f"episode of {root}"
                )
            image = cv2.imdecode(
                np.frombuffer(values.bytes_values[0], dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                raise ValueError(
                    f"Cannot decode the first frame of '{cam.path}' in "
                    f"{root} to measure its resolution"
                )
            self._camera_hw_cache[key] = (int(image.shape[0]), int(image.shape[1]))
        return self._camera_hw_cache[key]


# ── Module-level helpers ───────────────────────────────────────────


def _fingerprint(path: Path) -> dict[str, int] | None:
    """Return the fields that make a persisted index stale."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
