"""RoboTwin reader + hdf5_jpeg codec smoke tests.

Builds a tiny synthetic RoboTwin-layout dataset (episode*.hdf5 with embedded
JPEG camera streams + /joint_action groups) so the default suite exercises the
reader without a real RoboTwin install. Skipped when h5py (the [robotwin]
extra) is not present.
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

from vla_factory.data.codec import resolve_codec  # noqa: E402
from vla_factory.data.reader import get_reader, RoboTwinReader  # noqa: E402
from vla_factory.data.data_schema import VideoRef  # noqa: E402

CAMERAS = ("head_camera", "left_camera", "right_camera")
ARM_DIM = 6
STATE_DIM = 2 * (ARM_DIM + 1)  # two arms + two 1-DOF grippers = 14
H, W = 48, 64


def _make_episode(path: Path, T: int, seed: int) -> np.ndarray:
    """Write one episode*.hdf5; return the [T, STATE_DIM] qpos it stored."""
    import h5py

    rng = np.random.default_rng(seed)
    qpos = rng.standard_normal((T, STATE_DIM)).astype(np.float32)
    vlen = h5py.vlen_dtype(np.dtype("uint8"))

    with h5py.File(str(path), "w") as f:
        obs = f.create_group("observation")
        for cam in CAMERAS:
            grp = obs.create_group(cam)
            ds = grp.create_dataset("rgb", shape=(T,), dtype=vlen)
            for t in range(T):
                frame = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
                ok, enc = cv2.imencode(".jpg", frame)
                assert ok
                ds[t] = enc.flatten()
        ja = f.create_group("joint_action")
        ja.create_dataset("left_arm", data=qpos[:, 0:ARM_DIM])
        ja.create_dataset("left_gripper", data=qpos[:, ARM_DIM])          # [T]
        ja.create_dataset("right_arm", data=qpos[:, ARM_DIM + 1:2 * ARM_DIM + 1])
        ja.create_dataset("right_gripper", data=qpos[:, 2 * ARM_DIM + 1]) # [T]
    return qpos


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, dict[int, np.ndarray]]:
    """RoboTwin task-config root with two episodes of lengths 5 and 7."""
    data_dir = tmp_path / "task" / "config" / "data"
    data_dir.mkdir(parents=True)
    qpos = {
        0: _make_episode(data_dir / "episode0.hdf5", T=5, seed=0),
        1: _make_episode(data_dir / "episode1.hdf5", T=7, seed=1),
    }
    return tmp_path / "task" / "config", qpos


def test_get_reader_and_can_read(dataset):
    root, _ = dataset
    reader = get_reader("robotwin")
    assert isinstance(reader, RoboTwinReader)
    assert reader.can_read(root)
    # auto-probe also resolves to the RoboTwin reader
    assert isinstance(get_reader("auto", path=root), RoboTwinReader)


def test_get_reader_cannot_read_empty(tmp_path):
    assert not RoboTwinReader().can_read(tmp_path)


def test_schema(dataset):
    root, _ = dataset
    schema = RoboTwinReader().get_schema(root)
    assert schema.state_dim == STATE_DIM
    assert schema.action_dim == STATE_DIM
    assert set(schema.cameras) == set(CAMERAS)
    assert schema.total_episodes == 2
    assert schema.total_frames == 5 + 7
    for cam in CAMERAS:
        assert schema.image_sizes[cam] == (H, W)
    # per-dimension keys include the named grippers
    assert "left_gripper" in schema.state_keys
    assert "right_gripper" in schema.state_keys
    assert len(schema.state_keys) == STATE_DIM


def test_schema_entry_table_facts(dataset):
    """RoboTwin reader surfaces the joint concatenation as explicit facts."""
    root, _ = dataset
    schema = RoboTwinReader().get_schema(root)

    assert schema.source_format == "robotwin_hdf5"
    assert schema.robot_ref == "robotwin"

    # action mode is measured joint_pos: the /joint_action/* spec binds qpos.
    assert len(schema.action_dims) == STATE_DIM
    assert all(d.mode == "joint_pos" and d.mode_source == "measured"
               for d in schema.action_dims)
    # the implicit _JOINT_ORDER concatenation is now a per-dim source_field fact
    assert all("/joint_action/" in d.source_field for d in schema.action_dims)
    assert len(schema.state_dims) == STATE_DIM

    # camera semantics: head_camera uniquely inferred; left/right unanchored → undeclared
    cams = {c.key: c for c in schema.cameras_entries}
    assert cams["head_camera"].semantic == "third_person_front"
    assert cams["left_camera"].semantic is None     # needs assembly.camera_mapping
    assert cams["right_camera"].semantic is None



def test_episode_lengths_and_ranges(dataset):
    root, _ = dataset
    reader = RoboTwinReader()
    assert reader.get_episode_lengths(root) == {0: 5, 1: 7}
    assert reader.get_episode_ranges(root) == {0: (0, 4), 1: (5, 11)}


def test_read_episode_state_action(dataset):
    root, qpos = dataset
    reader = RoboTwinReader()
    codec = resolve_codec("hdf5_jpeg")
    episode = reader.read_episode(root, 1, codec)
    frames = episode.load_frames()

    assert len(frames) == 7
    for t, frame in enumerate(frames):
        np.testing.assert_allclose(frame.state, qpos[1][t], rtol=1e-5)
        expected_action = qpos[1][t + 1] if t + 1 < 7 else qpos[1][t]
        np.testing.assert_allclose(frame.action, expected_action, rtol=1e-5)
    assert frames[0].is_first and frames[-1].is_last


def test_codec_decodes_frames(dataset):
    root, _ = dataset
    reader = RoboTwinReader()
    codec = resolve_codec("hdf5_jpeg")
    frame = reader.read_episode(root, 0, codec).load_frames()[0]

    assert set(frame.images.keys()) == set(CAMERAS)
    for cam, ref in frame.images.items():
        assert ref.stream == cam
        img = codec.decode_frame(ref)
        assert img.shape == (H, W, 3)
        assert img.dtype == np.uint8


def test_norm_stats(dataset):
    root, qpos = dataset
    stats = RoboTwinReader().get_norm_stats(root)
    assert stats.state is not None and stats.action is not None
    assert len(stats.state.mean) == STATE_DIM
    assert len(stats.action.q01) == STATE_DIM

    all_qpos = np.concatenate([qpos[0], qpos[1]], axis=0)
    np.testing.assert_allclose(stats.state.mean, all_qpos.mean(axis=0), rtol=1e-4)


# ── Hdf5JpegCodec LRU ──────────────────────────────────────────────


def _first_refs(root, episode: int) -> dict[str, VideoRef]:
    """Per-camera VideoRefs of the first frame of an episode."""
    reader = RoboTwinReader()
    codec = resolve_codec("hdf5_jpeg")
    frame = reader.read_episode(root, episode, codec).load_frames()[0]
    return frame.images


def test_lru_reuses_decoded_frame(dataset):
    """Re-decoding the same (stream, index) must return identical pixels."""
    from vla_factory.data.codec import resolve_codec

    root, _ = dataset
    codec = resolve_codec("hdf5_jpeg")
    ref = _first_refs(root, 0)["head_camera"]

    first = codec.decode_frame(ref)
    # Served from the LRU, not a fresh JPEG decode
    cache = codec._caches[ref.video_path]._cache
    assert (ref.stream, ref.frame_index) in cache

    second = codec.decode_frame(ref)
    np.testing.assert_array_equal(first, second)


def test_lru_eviction(dataset):
    """Overflowing the per-file LRU evicts the least-recently-used frame."""
    from vla_factory.data.codec.hdf5_jpeg import Hdf5JpegCodec

    root, _ = dataset
    codec = Hdf5JpegCodec(max_cached_per_video=3)
    refs = [
        RoboTwinReader().read_episode(root, 0, codec).load_frames()[t].images["head_camera"]
        for t in range(4)
    ]

    for ref in refs:
        codec.decode_frame(ref)

    cache = codec._caches[refs[0].video_path]._cache
    assert len(cache) == 3
    # Oldest frame (index 0) was evicted; the newest three remain
    assert ("head_camera", 0) not in cache
    assert all(("head_camera", t) in cache for t in (1, 2, 3))


def test_lru_key_includes_stream(dataset):
    """Same frame_index under different cameras must not collide in the LRU."""
    from vla_factory.data.codec.hdf5_jpeg import Hdf5JpegCodec

    root, _ = dataset
    codec = Hdf5JpegCodec()
    refs = _first_refs(root, 0)  # one ref per camera, all frame_index 0

    for ref in refs.values():
        codec.decode_frame(ref)

    cache = codec._caches[next(iter(refs.values())).video_path]._cache
    # Three distinct (stream, 0) keys coexist — same index, different cameras
    assert len(cache) == len(CAMERAS)
    for cam in CAMERAS:
        assert (cam, 0) in cache


def test_codec_default_max_cached(dataset, tmp_path):
    """resolve_codec must default to a *working* 32-frame LRU like PyAV."""
    from vla_factory.data.codec import resolve_codec

    root, _ = dataset
    codec = resolve_codec("hdf5_jpeg")
    assert codec._max_cached == 32

    # The cache is keyed per hdf5 file, so overflow must happen within a
    # single file (the shared fixture tops out at 7x3=21 keys < 32 per file).
    # Build one episode with T=12 timesteps x 3 cameras = 36 distinct
    # (stream, frame_index) keys > the default cap, so the eviction path
    # genuinely runs under default settings.
    big_root = tmp_path / "big" / "task" / "config"
    data_dir = big_root / "data"
    data_dir.mkdir(parents=True)
    _make_episode(data_dir / "episode0.hdf5", T=12, seed=2)

    reader = RoboTwinReader()
    episode = reader.read_episode(big_root, 0, codec).load_frames()
    assert len(episode) == 12
    for frame in episode:
        for ref in frame.images.values():
            assert codec.decode_frame(ref).shape == (H, W, 3)
    # 36 inserts into one file-level cache -> evicted down to exactly 32.
    caches = [c for c in codec._caches.values() if c.path.name == "episode0.hdf5"]
    assert len(caches) == 1
    assert len(caches[0]._cache) == 32

    # Decoding other episodes through the same codec still works and every
    # file-level cache stays within the default bound.
    for ep in range(2):  # fixture lengths 5 + 7 -> 15 / 21 keys per file
        ep_frames = reader.read_episode(root, ep, codec).load_frames()
        for frame in ep_frames:
            for ref in frame.images.values():
                codec.decode_frame(ref)
    assert all(len(c._cache) <= c.max_cached for c in codec._caches.values())
