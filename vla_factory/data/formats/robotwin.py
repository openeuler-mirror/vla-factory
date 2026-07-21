"""RoboTwin 2.0 format reader — reads native episode hdf5 files.

RoboTwin stores each episode as ``data/{task}/{task_config}/data/episode{i}.hdf5``.
A single hdf5 holds every camera stream (JPEG byte streams) and the bimanual
joint trajectory:

  - ``/observation/{head,left,right}_camera/rgb``  — per-frame encoded JPEG
  - ``/joint_action/left_arm``   ``[T, A_left]``
  - ``/joint_action/left_gripper``  ``[T]`` or ``[T, 1]``
  - ``/joint_action/right_arm``  ``[T, A_right]``
  - ``/joint_action/right_gripper`` ``[T]`` or ``[T, 1]``

The joint vector is the concatenation, in this fixed order::

    [left_arm, left_gripper, right_arm, right_gripper]

which is the qpos control space RoboTwin's ``take_action(action_type='qpos')``
expects. Following RoboTwin/ACT convention, ``state[t]`` is the qpos at ``t``
and ``action[t]`` is the qpos at ``t+1`` (the joint target to reach next).

Images are left as :class:`VideoRef` pointing at the hdf5 file with the camera
name in ``stream``; the :class:`Hdf5JpegCodec` decodes them. RoboTwin ships no
precomputed normalisation stats, so :meth:`get_norm_stats` computes them by
scanning the joint trajectories.

``h5py`` is an optional (``[robotwin]``) dependency, imported lazily inside the
methods so the reader registry stays importable without it.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from numpy.typing import NDArray

from ..manifest import DataSchema, FeatureStats, NormStats
from ..codec.base import VideoCodec
from .base import Episode, Frame, VideoRef

logger = logging.getLogger(__name__)

# Fixed concatenation order of the bimanual joint vector.
_JOINT_ORDER = ("left_arm", "left_gripper", "right_arm", "right_gripper")
_EPISODE_RE = re.compile(r"episode(\d+)\.hdf5$")


def _load_h5py() -> Any:
    """Import ``h5py`` lazily with an actionable error if it is missing."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Reading RoboTwin hdf5 datasets requires 'h5py'. Install the "
            "RoboTwin extra: pip install -e \".[robotwin]\" (or pip install h5py)."
        ) from exc
    return h5py


def _episode_dir(path: Path) -> Path:
    """Locate the directory holding ``episode*.hdf5`` files.

    Accepts either a RoboTwin task-config root (``.../{task}/{config}/``, whose
    episodes live in ``data/``) or a directory that already contains the hdf5
    files directly.
    """
    data_sub = path / "data"
    if data_sub.is_dir() and any(data_sub.glob("episode*.hdf5")):
        return data_sub
    return path


def _episode_files(path: Path) -> dict[int, Path]:
    """Return ``{episode_index: hdf5_path}`` sorted by episode number."""
    ep_dir = _episode_dir(path)
    files: dict[int, Path] = {}
    for f in ep_dir.glob("episode*.hdf5"):
        m = _EPISODE_RE.search(f.name)
        if m:
            files[int(m.group(1))] = f
    return dict(sorted(files.items()))


def _as_2d(arr: NDArray) -> NDArray:
    """Coerce a per-frame joint array to ``[T, k]`` (grippers may be ``[T]``)."""
    a = np.asarray(arr, dtype=np.float32)
    return a.reshape(a.shape[0], -1) if a.ndim == 1 else a


def _read_qpos(f: Any) -> NDArray:
    """Read and concatenate the joint trajectory -> ``[T, state_dim]`` float32."""
    parts = [_as_2d(f[f"/joint_action/{name}"][()]) for name in _JOINT_ORDER]
    return np.concatenate(parts, axis=1)


def _joint_dims(f: Any) -> dict[str, int]:
    """Return the per-segment width of each joint group."""
    return {name: _as_2d(f[f"/joint_action/{name}"][()]).shape[1] for name in _JOINT_ORDER}


def _state_keys(dims: dict[str, int]) -> tuple[str, ...]:
    """Build per-dimension names, e.g. ``left_arm_0 .. left_gripper_0 ..``."""
    keys: list[str] = []
    for name in _JOINT_ORDER:
        width = dims[name]
        if width == 1:
            keys.append(name)
        else:
            keys.extend(f"{name}_{i}" for i in range(width))
    return tuple(keys)


def _camera_names(f: Any) -> list[str]:
    """Camera stream names under ``/observation`` that expose an ``rgb`` dataset."""
    obs = f.get("observation")
    if obs is None:
        return []
    return [name for name in obs.keys() if "rgb" in obs[name]]


def _decode_hw(f: Any, cam: str) -> tuple[int, int]:
    """Decode the first frame of a camera to get its ``(H, W)``."""
    raw = f[f"/observation/{cam}/rgb"][0]
    buf = np.frombuffer(bytes(raw), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to JPEG-decode first frame of camera '{cam}'.")
    return int(img.shape[0]), int(img.shape[1])


def _load_instruction(path: Path, ep_idx: int) -> str | None:
    """Best-effort load of an episode's language instruction.

    RoboTwin stores instructions separately under ``instructions/`` (used by
    language-conditioned policies such as pi0; ACT ignores it). The exact JSON
    schema varies across RoboTwin releases, so this is intentionally lenient:
    it returns the first available instruction string, or ``None``.
    """
    instr = path / "instructions" / f"episode{ep_idx}.json"
    if not instr.exists():
        return None
    try:
        data = json.loads(instr.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in ("seen", "unseen", "instruction", "instructions"):
            val = data.get(key)
            if isinstance(val, str):
                return val
            if isinstance(val, list) and val:
                return str(val[0])
    return None


class RoboTwinReader:
    """Read RoboTwin 2.0 native episode hdf5 datasets."""

    def can_read(self, path: Path) -> bool:
        """True if ``path`` holds RoboTwin ``episode*.hdf5`` files."""
        files = _episode_files(path)
        if not files:
            return False
        h5py = _load_h5py()
        first = next(iter(files.values()))
        try:
            with h5py.File(str(first), "r") as f:
                return "observation" in f and "joint_action" in f
        except (OSError, KeyError):
            return False

    # ── Schema & stats ────────────────────────────────────────────

    def get_schema(self, path: Path) -> DataSchema:
        """Infer ``DataSchema`` from the first episode's hdf5."""
        h5py = _load_h5py()
        files = _episode_files(path)
        if not files:
            raise FileNotFoundError(f"No RoboTwin episode*.hdf5 found under {path}")

        lengths = self.get_episode_lengths(path)
        total_frames = sum(lengths.values())

        first = next(iter(files.values()))
        with h5py.File(str(first), "r") as f:
            dims = _joint_dims(f)
            state_keys = _state_keys(dims)
            state_dim = len(state_keys)
            cameras = _camera_names(f)
            image_sizes = {cam: _decode_hw(f, cam) for cam in cameras}

        has_language = (path / "instructions").is_dir()

        return DataSchema(
            state_dim=state_dim,
            action_dim=state_dim,  # qpos action space == joint state space
            cameras=tuple(cameras),
            image_sizes=image_sizes,
            fps=25,  # RoboTwin does not persist fps; 25 is its common default
            has_language=has_language,
            total_episodes=len(files),
            total_frames=total_frames,
            robot_type="robotwin",
            state_keys=state_keys,
            action_keys=state_keys,
        )

    def get_norm_stats(self, path: Path) -> NormStats:
        """Compute state/action normalisation stats by scanning joint trajectories.

        RoboTwin ships no precomputed stats. State stats use all qpos; action
        stats use the next-step targets (qpos shifted by one), matching how
        :meth:`read_episode` assigns ``action[t] = qpos[t+1]``.
        """
        h5py = _load_h5py()
        files = _episode_files(path)
        if not files:
            return NormStats()

        state_rows: list[NDArray] = []
        action_rows: list[NDArray] = []
        for ep_path in files.values():
            with h5py.File(str(ep_path), "r") as f:
                qpos = _read_qpos(f)
            if qpos.shape[0] == 0:
                continue
            state_rows.append(qpos)
            # action[t] = qpos[t+1]; last step repeats the final qpos.
            nxt = np.concatenate([qpos[1:], qpos[-1:]], axis=0)
            action_rows.append(nxt)

        if not state_rows:
            return NormStats()

        state_all = np.concatenate(state_rows, axis=0)
        action_all = np.concatenate(action_rows, axis=0)
        return NormStats(
            state=_feature_stats(state_all),
            action=_feature_stats(action_all),
            images=None,  # images use ImageNet stats in the Normalize transform
            method="zscore",
        )

    # ── Episode-level queries ─────────────────────────────────────

    def get_episode_lengths(self, path: Path) -> dict[int, int]:
        """Return ``{episode_index: num_frames}``."""
        h5py = _load_h5py()
        lengths: dict[int, int] = {}
        for ep_idx, ep_path in _episode_files(path).items():
            with h5py.File(str(ep_path), "r") as f:
                lengths[ep_idx] = int(f["/joint_action/left_arm"].shape[0])
        return lengths

    def get_episode_ranges(self, path: Path) -> dict[int, tuple[int, int]]:
        """Return ``{ep_idx: (global_start, global_end)}`` (inclusive)."""
        ranges: dict[int, tuple[int, int]] = {}
        cursor = 0
        for ep_idx, length in self.get_episode_lengths(path).items():
            ranges[ep_idx] = (cursor, cursor + length - 1)
            cursor += length
        return ranges

    # ── Episode reading ───────────────────────────────────────────

    def read_episode(
        self, path: Path, episode_index: int, codec: VideoCodec
    ) -> Episode:
        """Read one episode into canonical ``Frame`` objects.

        Images stay lazy as :class:`VideoRef` (decoded later by ``codec``);
        state/action are materialised from the joint trajectory.
        """
        h5py = _load_h5py()
        files = _episode_files(path)
        if episode_index not in files:
            raise KeyError(f"Episode {episode_index} not found under {path}")
        ep_path = files[episode_index]

        with h5py.File(str(ep_path), "r") as f:
            qpos = _read_qpos(f)
            cameras = _camera_names(f)
            cam_hw = {cam: _decode_hw(f, cam) for cam in cameras}
        num_frames = qpos.shape[0]
        language = _load_instruction(path, episode_index)

        def frame_loader() -> Iterator[Frame]:
            for t in range(num_frames):
                images = {
                    cam: VideoRef(
                        video_path=ep_path,
                        frame_index=t,
                        height=cam_hw[cam][0],
                        width=cam_hw[cam][1],
                        stream=cam,
                    )
                    for cam in cameras
                }
                # action[t] = qpos[t+1]; final step repeats the last qpos.
                action = qpos[t + 1] if t + 1 < num_frames else qpos[t]
                yield Frame(
                    index=t,
                    images=images,
                    state=qpos[t],
                    action=action,
                    is_first=(t == 0),
                    is_last=(t == num_frames - 1),
                    language=language,
                )

        return Episode(
            episode_id=f"episode{episode_index}",
            episode_index=episode_index,
            num_frames=num_frames,
            _frame_loader=frame_loader,
        )


def _feature_stats(arr: NDArray) -> FeatureStats:
    """Per-dimension mean/std/min/max/q01/q99 over a ``[N, D]`` array."""
    return FeatureStats(
        mean=arr.mean(axis=0).tolist(),
        std=arr.std(axis=0).tolist(),
        min=arr.min(axis=0).tolist(),
        max=arr.max(axis=0).tolist(),
        q01=np.quantile(arr, 0.01, axis=0).tolist(),
        q99=np.quantile(arr, 0.99, axis=0).tolist(),
    )
