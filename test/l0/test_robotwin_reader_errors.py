"""L0 error-path tests for the RoboTwin reader and the hdf5-JPEG codec.

``formats/robotwin.py`` and ``codec/hdf5_jpeg.py`` are native implementations —
there is no upstream API to compare against, so their correctness rests on
their own boundary behaviour. ``test_robotwin_reader.py`` covers the happy path
on a synthetic dataset; this file covers what happens when the dataset is
wrong: no episodes, a missing episode index, a camera that is not there, a
corrupt JPEG stream, a VideoRef with no camera name.

Every case is what a user hits after pointing the recipe at the wrong
directory or copying a truncated dataset, so the error has to say which of
those it was.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None,
    reason="h5py not installed (pip install -e \".[robotwin]\")",
)

import cv2  # noqa: E402 — after skip guard; cv2 is a core dep

from vla_factory.data.codec.hdf5_jpeg import Hdf5JpegCodec  # noqa: E402
from vla_factory.data.data_schema import VideoRef  # noqa: E402
from vla_factory.data.reader import RoboTwinReader  # noqa: E402

CAMERAS = ("head_camera",)
ARM_DIM = 6
STATE_DIM = 2 * (ARM_DIM + 1)
H, W = 16, 24


def _write_episode(path: Path, T: int = 3, *, corrupt_jpeg: bool = False) -> None:
    """Write a minimal but valid RoboTwin episode hdf5."""
    import h5py

    rng = np.random.default_rng(0)
    qpos = rng.standard_normal((T, STATE_DIM)).astype(np.float32)
    vlen = h5py.vlen_dtype(np.dtype("uint8"))

    with h5py.File(str(path), "w") as f:
        obs = f.create_group("observation")
        for cam in CAMERAS:
            ds = obs.create_group(cam).create_dataset("rgb", shape=(T,), dtype=vlen)
            for t in range(T):
                if corrupt_jpeg:
                    ds[t] = np.frombuffer(b"not-a-jpeg-at-all", dtype=np.uint8)
                    continue
                frame = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
                ok, enc = cv2.imencode(".jpg", frame)
                assert ok
                ds[t] = enc.flatten()
        ja = f.create_group("joint_action")
        ja.create_dataset("left_arm", data=qpos[:, 0:ARM_DIM])
        ja.create_dataset("left_gripper", data=qpos[:, ARM_DIM])
        ja.create_dataset("right_arm", data=qpos[:, ARM_DIM + 1:2 * ARM_DIM + 1])
        ja.create_dataset("right_gripper", data=qpos[:, 2 * ARM_DIM + 1])


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """A one-episode RoboTwin task-config root."""
    data_dir = tmp_path / "task" / "config" / "data"
    data_dir.mkdir(parents=True)
    _write_episode(data_dir / "episode0.hdf5")
    return tmp_path / "task" / "config"


# ── Reader: dataset-shaped failures ──────────────────────────────────


def test_schema_on_a_directory_with_no_episodes_raises(tmp_path):
    """Pointing the recipe at the wrong directory must name the directory."""
    empty = tmp_path / "not_robotwin"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="No RoboTwin episode"):
        RoboTwinReader().get_schema(empty)


def test_can_read_is_false_for_a_non_robotwin_directory(tmp_path):
    """can_read must answer False, not raise — 'auto' format probes every reader."""
    other = tmp_path / "lerobot_ish"
    (other / "meta").mkdir(parents=True)
    (other / "meta" / "info.json").write_text("{}")

    assert RoboTwinReader().can_read(other) is False


def test_can_read_is_false_for_a_truncated_hdf5(tmp_path):
    """A file named episode*.hdf5 that is not valid hdf5 must not crash the probe."""
    data_dir = tmp_path / "task" / "config" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "episode0.hdf5").write_bytes(b"truncated garbage")

    assert RoboTwinReader().can_read(tmp_path / "task" / "config") is False


def test_can_read_is_false_when_joint_action_is_missing(tmp_path):
    """A valid hdf5 without the joint_action group is not a RoboTwin episode."""
    import h5py

    data_dir = tmp_path / "task" / "config" / "data"
    data_dir.mkdir(parents=True)
    with h5py.File(str(data_dir / "episode0.hdf5"), "w") as f:
        f.create_group("observation")

    assert RoboTwinReader().can_read(tmp_path / "task" / "config") is False


def test_read_episode_with_unknown_index_raises(dataset_root):
    """An --episodes index outside the dataset must fail loudly, not return empty."""
    from vla_factory.data.codec import resolve_codec

    with pytest.raises(KeyError, match="Episode 42 not found"):
        RoboTwinReader().read_episode(dataset_root, 42, resolve_codec("hdf5_jpeg"))


# ── Codec: frame-shaped failures ─────────────────────────────────────


def test_decode_frame_without_a_stream_name_raises(dataset_root):
    """VideoRef.stream carries the camera name; None means the reader is broken."""
    ref = VideoRef(
        video_path=dataset_root / "data" / "episode0.hdf5",
        frame_index=0, stream=None, height=H, width=W,
    )

    with pytest.raises(ValueError, match="requires VideoRef.stream"):
        Hdf5JpegCodec().decode_frame(ref)


def test_decode_frame_with_unknown_camera_lists_available(dataset_root):
    """A camera_mapping typo must enumerate the cameras the file actually has."""
    ref = VideoRef(
        video_path=dataset_root / "data" / "episode0.hdf5",
        frame_index=0, stream="front_camera", height=H, width=W,
    )

    codec = Hdf5JpegCodec()
    try:
        with pytest.raises(KeyError) as exc:
            codec.decode_frame(ref)
        message = str(exc.value)
        assert "front_camera" in message
        assert "head_camera" in message, "the error must list the real cameras"
    finally:
        codec.close()


def test_corrupt_jpeg_stream_raises_runtime_error(tmp_path):
    """A frame that cv2 cannot decode must raise, not return None downstream."""
    data_dir = tmp_path / "task" / "config" / "data"
    data_dir.mkdir(parents=True)
    _write_episode(data_dir / "episode0.hdf5", corrupt_jpeg=True)

    ref = VideoRef(
        video_path=data_dir / "episode0.hdf5",
        frame_index=0, stream="head_camera", height=H, width=W,
    )
    codec = Hdf5JpegCodec()
    try:
        with pytest.raises(RuntimeError, match="Failed to JPEG-decode"):
            codec.decode_frame(ref)
    finally:
        codec.close()


def test_decode_frame_resizes_to_the_requested_shape(dataset_root):
    """The codec contract is (height, width) from the ref, not the stored size."""
    ref = VideoRef(
        video_path=dataset_root / "data" / "episode0.hdf5",
        frame_index=0, stream="head_camera", height=H * 2, width=W * 2,
    )
    codec = Hdf5JpegCodec()
    try:
        img = codec.decode_frame(ref)
        assert img.shape == (H * 2, W * 2, 3)
        assert img.dtype == np.uint8
    finally:
        codec.close()


def test_close_is_idempotent(dataset_root):
    """__del__ calls close() too; a double close must not raise."""
    ref = VideoRef(
        video_path=dataset_root / "data" / "episode0.hdf5",
        frame_index=0, stream="head_camera", height=H, width=W,
    )
    codec = Hdf5JpegCodec()
    codec.decode_frame(ref)
    codec.close()
    codec.close()
