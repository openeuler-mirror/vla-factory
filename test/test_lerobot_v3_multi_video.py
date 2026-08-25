"""LeRobot v3 reader — multi-video-file frame mapping regression tests.

Builds synthetic LeRobot v3 datasets with several ``file-NNN.mp4`` videos per
camera, each covering a contiguous slice of the dataset-global index space, to
verify the reader maps a global parquet ``index`` to the correct video file and
within-file offset.

The old implementation unconditionally picked the first mp4 and used the global
index as the within-file index, which for any multi-file dataset broke torchcodec
(``IndexError``) and silently returned black/wrong frames through PyAV. These
tests fail against that code.

Run:
    pytest test/test_lerobot_v3_multi_video.py -v
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vla_factory.data.codec.pyav import PyAVCodec
from vla_factory.data.reader import LeRobotV3Reader
from vla_factory.data.data_schema import VideoRef

# torchcodec parity is optional: the torch.py codec lives on a separate PR
# (feat/torchcodec-codec). Only run cross-codec parity when both the
# torchcodec package AND the project module are importable.
TORCHCODEC_AVAILABLE = False
if importlib.util.find_spec("torchcodec") and importlib.util.find_spec(
    "vla_factory.data.codec.torch"
):
    try:
        from vla_factory.data.codec.torch import TorchCodec  # noqa: F401
        TORCHCODEC_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        TORCHCODEC_AVAILABLE = False

H, W = 16, 16
N_EPISODES = 3
EP_LEN = 10
N_FILES = 3
TOTAL = N_EPISODES * EP_LEN  # 30
FRAMES_PER_FILE = TOTAL // N_FILES  # 10


# ── Synthetic dataset builders ────────────────────────────────────


def _pick_encoder() -> str:
    """Return an mp4 encoder available in this PyAV build, or skip the test."""
    for name in ("mpeg4", "libx264"):
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:
            continue
    pytest.skip("No video encoder (mpeg4/libx264) available in this PyAV build")
    raise AssertionError("unreachable")  # pragma: no cover - pytest.skip raises


def _write_mp4(
    path: Path, frame_values: list[int], dims: tuple[int, int] = (H, W)
) -> None:
    """Write a solid-color mp4; frame k is RGB = (g, g, g) for g = frame_values[k].

    Encoding the global index into the pixel value makes every frame's video
    source verifiable from its decoded content.
    """
    h, w = dims
    container = av.open(str(path), "w")
    try:
        stream = container.add_stream(_pick_encoder(), rate=1)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        for g in frame_values:
            frame = av.VideoFrame.from_ndarray(
                np.full((h, w, 3), g, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _write_meta(root: Path, n_cams: int, total: int, n_episodes: int, per_episode: bool) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    cams = ("front", "wrist") if n_cams == 2 else ("front",)
    features = {
        f"observation.images.{cam}": {
            "dtype": "video",
            "shape": [total, H, W, 3],
            "names": ["index", "height", "width", "channels"],
            "video_info": {"video.height": H, "video.width": W, "video.channels": 3},
        }
        for cam in cams
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "test_robot",
        "total_frames": total,
        "total_episodes": n_episodes,
        "fps": 1,
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/episode_{episode_index:06d}.mp4"
            if per_episode
            else "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        ),
        "features": features,
    }
    (meta / "info.json").write_text(json.dumps(info))


def _write_parquet(root: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), data_dir / "file-000.parquet")


def _state(t: int) -> list[float]:
    return [float(t % 3), float(t), 0.0, 0.0, 0.0, 0.0]


def _write_dataset(root: Path, n_files: int = N_FILES) -> Path:
    """3 episodes × 10 frames split across ``n_files`` multi-episode videos/cam."""
    fpf = TOTAL // n_files
    for cam in ("front", "wrist"):
        cam_dir = root / "videos" / f"observation.images.{cam}" / "chunk-000"
        cam_dir.mkdir(parents=True, exist_ok=True)
        for fi in range(n_files):
            g0 = fi * fpf
            _write_mp4(cam_dir / f"file-{fi:03d}.mp4", list(range(g0, g0 + fpf)))

    rows = []
    for ep in range(N_EPISODES):
        for t in range(EP_LEN):
            rows.append(
                {
                    "index": ep * EP_LEN + t,
                    "episode_index": ep,
                    "frame_index": t,
                    "observation.state": _state(t),
                }
            )
    _write_parquet(root, rows)
    _write_meta(root, n_cams=2, total=TOTAL, n_episodes=N_EPISODES, per_episode=False)
    return root


def _write_per_episode_dataset(root: Path, n_episodes: int = 2, ep_len: int = 5) -> Path:
    """One ``episode_XXXXXX.mp4`` per episode (the per-episode video layout)."""
    total = n_episodes * ep_len
    for cam in ("front", "wrist"):
        cam_dir = root / "videos" / f"observation.images.{cam}" / "chunk-000"
        cam_dir.mkdir(parents=True, exist_ok=True)
        for ep in range(n_episodes):
            g0 = ep * ep_len
            _write_mp4(cam_dir / f"episode_{ep:06d}.mp4", list(range(g0, g0 + ep_len)))

    rows = []
    for ep in range(n_episodes):
        for t in range(ep_len):
            rows.append(
                {
                    "index": ep * ep_len + t,
                    "episode_index": ep,
                    "frame_index": t,
                    "observation.state": _state(t),
                }
            )
    _write_parquet(root, rows)
    _write_meta(root, n_cams=2, total=total, n_episodes=n_episodes, per_episode=True)
    return root


def _write_uncovered_dataset(root: Path) -> Path:
    """One 10-frame video (globals 0-9) but parquet ep1 claims globals 10-14."""
    cam_dir = root / "videos" / "observation.images.front" / "chunk-000"
    cam_dir.mkdir(parents=True, exist_ok=True)
    _write_mp4(cam_dir / "file-000.mp4", list(range(10)))

    rows = []
    for g in range(15):
        rows.append(
            {
                "index": g,
                "episode_index": 0 if g < 10 else 1,
                "frame_index": g % 10,
                "observation.state": _state(g),
            }
        )
    _write_parquet(root, rows)
    _write_meta(root, n_cams=1, total=15, n_episodes=2, per_episode=False)
    return root


def _expected_mapping(n_files: int = N_FILES) -> dict[int, tuple[int, int]]:
    """global index -> (file_index, within_file_index)."""
    fpf = TOTAL // n_files
    return {g: (g // fpf, g % fpf) for g in range(TOTAL)}


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return _write_dataset(tmp_path)


# ── Tests ─────────────────────────────────────────────────────────


def test_multi_video_maps_to_correct_file(dataset):
    """Each episode resolves to the video file containing its global index."""
    reader = LeRobotV3Reader()
    mapping = _expected_mapping()
    for ep in range(N_EPISODES):
        frames = reader.read_episode(dataset, ep, PyAVCodec()).load_frames()
        assert len(frames) == EP_LEN
        for frame in frames:
            file_idx, within = mapping[frame.index]
            assert len(frame.images) == 2  # front + wrist
            for ref in frame.images.values():
                assert ref.video_path.name == f"file-{file_idx:03d}.mp4"
                assert ref.frame_index == within


def test_decode_pixels_correct_both_codecs(dataset):
    """Reader refs decode to the expected file's content; codecs agree.

    Solid-color values encode the global frame index, so after decoding we
    compare the pixel value to the expected global index within a bounded
    yuv420p quantization tolerance. This catches wrong file/index mapping.
    """
    refs: dict[int, dict[str, VideoRef]] = {}
    reader = LeRobotV3Reader()
    for ep in range(N_EPISODES):
        frames = reader.read_episode(dataset, ep, PyAVCodec()).load_frames()
        for t in (0, EP_LEN // 2, EP_LEN - 1):
            refs[frames[t].index] = frames[t].images

    pyav = PyAVCodec(disk_cache=False)
    torch = None
    if TORCHCODEC_AVAILABLE:
        from vla_factory.data.codec.torch import TorchCodec

        torch = TorchCodec(disk_cache=False)

    for g, images in refs.items():
        file_idx, within = _expected_mapping()[g]
        for cam, ref in images.items():
            assert ref.video_path.name == f"file-{file_idx:03d}.mp4"
            assert ref.frame_index == within
            img = pyav.decode_frame(ref)
            assert img.shape == (H, W, 3)
            # The synthetic video encodes the global frame index into the pixel
            # value, so decode content must match the global index (within a
            # bounded yuv420p quantization tolerance).
            assert abs(int(img[0, 0, 0]) - g) <= 8, (
                f"g={g}/{cam}: decoded pixel {int(img[0, 0, 0])} does not "
                f"match expected global index (file={file_idx}, within={within})"
            )
            if torch is not None:
                assert np.array_equal(
                    img, torch.decode_frame(ref)
                ), f"g={g}/{cam}: pyav/torchcodec mismatch"


def test_old_mapping_would_fail(dataset):
    """The pre-fix mapping (file-000 + global index) must fail loudly."""
    file0 = dataset / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4"
    # Global index 15 is inside file-001; asking file-000 for it is the old bug.
    ref = VideoRef(video_path=file0, frame_index=15, height=H, width=W, channels=3)
    pyav = PyAVCodec(disk_cache=False)
    img = pyav.decode_frame(ref)
    assert not np.all(img == 15)  # wrong/black frame, never the true frame 15
    if TORCHCODEC_AVAILABLE:
        from vla_factory.data.codec.torch import TorchCodec

        torch = TorchCodec(disk_cache=False)
        with pytest.raises(IndexError):
            torch.decode_frame(ref)


def test_per_episode_video_uses_frame_index(tmp_path):
    """episode_XXXXXX.mp4 layout still maps via frame_index, unchanged."""
    ds = _write_per_episode_dataset(tmp_path)
    reader = LeRobotV3Reader()
    for ep in range(2):
        frames = reader.read_episode(ds, ep, PyAVCodec()).load_frames()
        assert len(frames) == 5
        for t, frame in enumerate(frames):
            for ref in frame.images.values():
                assert ref.video_path.name == f"episode_{ep:06d}.mp4"
                assert ref.frame_index == t


def test_uncovered_index_raises(tmp_path):
    """A global index beyond every video span fails fast instead of mis-decoding."""
    ds = _write_uncovered_dataset(tmp_path)
    reader = LeRobotV3Reader()
    # ep1's global start (10) is past the single file's range 0-9.
    with pytest.raises(ValueError, match="not covered"):
        reader.read_episode(ds, 1, PyAVCodec())


def test_episode_crossing_file_boundary_raises(tmp_path):
    """Episodes spanning multiple file-NNN.mp4 must fail fast, not black-frame."""
    ds = _write_dataset(tmp_path, n_files=2)
    reader = LeRobotV3Reader()
    with pytest.raises(ValueError, match="not covered"):
        reader.read_episode(ds, 1, PyAVCodec())
