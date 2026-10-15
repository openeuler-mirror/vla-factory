"""RLDS reader + rlds_jpeg codec tests over a synthetic TFRecord dataset.

Builds a tiny RLDS/TFDS-layout dataset (features.json + sharded
``train-*-of-*.tfrecord`` files written through ``tf.io.TFRecordWriter`` —
the reference framing with real CRCs — whose records are tf.train.Example
protos built with TF's own proto builders) so the suite exercises the full
path — detection, spec mapping, index scan, episode reads, stats, and
VLADataset samples — exactly as real TFDS data would flow. Camera frames
are PNG-encoded for lossless assertions. Skipped when TensorFlow (the
[rlds] extra) is not present.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import cv2  # core dep
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason='tensorflow not installed (pip install -e ".[rlds]")',
)

# Imported after the skip guard so collection itself stays TF-free; when
# TensorFlow is missing every test is skipped before touching this module
# level name. Silenced the same way the reader's lazy import does.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - skipif guard handles the user story
    tf = None  # type: ignore[assignment]

from vla_factory.data.codec import resolve_codec  # noqa: E402
from vla_factory.data.codec.rlds_jpeg import RldsJpegCodec  # noqa: E402
from vla_factory.data.reader import get_reader  # noqa: E402
from vla_factory.data.reader.rlds.reader import RldsReader  # noqa: E402

CAMERAS = ("image", "wrist_image")  # leaf names under steps/observation
STATE_DIM = 7
ACTION_DIM = 7
H, W = 48, 64

def _tensor(dtype: str, *shape: int) -> dict:
    return {
        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
        "tensor": {
            "dtype": dtype,
            "encoding": "none",
            "shape": {"dimensions": [str(dim) for dim in shape]},
        },
    }


def _image() -> dict:
    return {
        "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
        "image": {
            "dtype": "uint8",
            "encodingFormat": "png",
            "shape": {"dimensions": [str(H), str(W), "3"]},
        },
    }


def _features(values: dict) -> dict:
    return {
        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
        "featuresDict": {"features": values},
    }


def _sequence(values: dict) -> dict:
    return {
        "pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
        "sequence": {"feature": _features(values)},
    }


FEATURES_JSON = _features({
    "steps": _sequence({
        "observation": _features({
            "image": _image(),
            "wrist_image": _image(),
            "state": _tensor("float32", STATE_DIM),
        }),
        "action": _tensor("float32", ACTION_DIM),
        "reward": _tensor("float32"),
        "is_first": _tensor("bool"),
        "is_last": _tensor("bool"),
        "language_instruction": _tensor("string"),
        "language_instruction_2": _tensor("string"),
    }),
})


def _write_dataset_info(root: Path, name: str, splits: list[str]) -> None:
    (root / "dataset_info.json").write_text(json.dumps({
        "fileFormat": "tfrecord",
        "name": name,
        "splits": [
            {
                "name": split,
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
            }
            for split in splits
        ],
        "version": "1.0.0",
    }))


def _episode_arrays(T: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "state": rng.standard_normal((T, STATE_DIM)).astype(np.float32),
        "action": rng.standard_normal((T, ACTION_DIM)).astype(np.float32),
        "images": {
            cam: rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8)
            for cam in CAMERAS
        },
        "lang": f"pick up the cube {seed}",
        "lang2": f"grab the cube {seed}",
    }


def _encode_episode(data: dict, T: int) -> bytes:
    """Serialise one episode as a tf.train.Example via TF's proto builders."""
    def _floats(values: list[float]) -> tf.train.Feature:
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def _ints(values: list[int]) -> tf.train.Feature:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def _bytes(values: list[bytes]) -> tf.train.Feature:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    features = {
        "steps/observation/state": _floats(data["state"].reshape(-1).tolist()),
        "steps/action": _floats(data["action"].reshape(-1).tolist()),
        "steps/is_first": _ints([1 if t == 0 else 0 for t in range(T)]),
        "steps/is_last": _ints([1 if t == T - 1 else 0 for t in range(T)]),
        "steps/reward": _floats([0.0] * T),
        "steps/language_instruction": _bytes([data["lang"].encode()] * T),
        "steps/language_instruction_2": _bytes([data["lang2"].encode()] * T),
    }
    for cam in CAMERAS:
        encoded = []
        for t in range(T):
            # The arrays are RGB; cv2 encodes BGR, so flip channels first —
            # the codec contract returns decoded frames as RGB.
            bgr = cv2.cvtColor(data["images"][cam][t], cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".png", bgr)
            assert ok
            encoded.append(buf.tobytes())
        features[f"steps/observation/{cam}"] = _bytes(encoded)
    return tf.train.Example(
        features=tf.train.Features(feature=features)
    ).SerializeToString()


def _write_dataset(
    root: Path, episodes: list[dict], *, stats_file: dict | None = None
) -> None:
    """Lay out shards (2 per 3 episodes), the spec, optional statistics."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "features.json").write_text(json.dumps(FEATURES_JSON))
    _write_dataset_info(root, "bridge_mini", ["train"])
    shards = [
        ("bridge_mini-train.tfrecord-00000-of-00002", episodes[0:2]),
        ("bridge_mini-train.tfrecord-00001-of-00002", episodes[2:3]),
    ]
    for name, eps in shards:
        with tf.io.TFRecordWriter(str(root / name)) as writer:
            for ep in eps:
                writer.write(ep["proto"])
    if stats_file is not None:
        (root / "dataset_statistics.json").write_text(json.dumps(stats_file))


@pytest.fixture
def dataset(tmp_path: Path):
    """RLDS root with episodes of length 5 / 7 / 4 across two shards."""
    raw = [(5, 100), (7, 200), (4, 300)]
    episodes = []
    for T, seed in raw:
        data = _episode_arrays(T, seed)
        episodes.append({"data": data, "T": T, "proto": _encode_episode(data, T)})
    root = tmp_path / "bridge_mini" / "1.0.0"
    _write_dataset(root, episodes)
    return root, episodes


# ── Detection ──────────────────────────────────────────────────────


def test_get_reader_and_can_read(dataset):
    root, _ = dataset
    reader = get_reader("rlds")
    assert isinstance(reader, RldsReader)
    assert reader.can_read(root)
    assert isinstance(get_reader("auto", path=root), RldsReader)
    assert get_reader("rlds_tfrecord").can_read(root)


def test_can_read_negative(tmp_path):
    assert not RldsReader().can_read(tmp_path)
    (tmp_path / "features.json").write_text(json.dumps(FEATURES_JSON))
    assert not RldsReader().can_read(tmp_path)  # spec but no shards


def test_tfds_nested_layout_detected(dataset, tmp_path):
    root, _ = dataset
    # can_read already resolves through the <name>/<version> nesting.
    assert RldsReader().can_read(root.parent.parent)


def test_multiple_dataset_dirs_rejected(tmp_path):
    for name in ("dataset_a", "dataset_b"):
        root = tmp_path / name
        episodes = []
        data = _episode_arrays(3, abs(hash(name)) % 1000)
        episodes.append({"data": data, "T": 3, "proto": _encode_episode(data, 3)})
        _write_dataset(root, episodes)
    with pytest.raises(ValueError, match="multiple RLDS dataset"):
        RldsReader().get_schema(tmp_path)


def test_train_split_preferred(tmp_path):
    root = tmp_path / "mixed_splits"
    train_ep = _episode_arrays(5, 1)
    val_ep = _episode_arrays(3, 2)
    root.mkdir(parents=True)
    (root / "features.json").write_text(json.dumps(FEATURES_JSON))
    _write_dataset_info(root, "mixed_splits", ["train", "validation"])
    with tf.io.TFRecordWriter(str(root / "mixed_splits-train.tfrecord-00000-of-00001")) as w:
        w.write(_encode_episode(train_ep, 5))
    with tf.io.TFRecordWriter(str(root / "mixed_splits-validation.tfrecord-00000-of-00001")) as w:
        w.write(_encode_episode(val_ep, 3))
    assert RldsReader().get_episode_lengths(root) == {0: 5}


def test_missing_action_feature_rejected(tmp_path):
    root = tmp_path / "no_action"
    root.mkdir()
    spec = json.loads(json.dumps(FEATURES_JSON))
    step_features = spec["featuresDict"]["features"]["steps"]["sequence"]
    del step_features["feature"]["featuresDict"]["features"]["action"]
    (root / "features.json").write_text(json.dumps(spec))
    data = _episode_arrays(3, 1)
    _write_dataset_info(root, "no_action", ["train"])
    with tf.io.TFRecordWriter(str(root / "no_action-train.tfrecord-00000-of-00001")) as w:
        w.write(_encode_episode(data, 3))
    assert not RldsReader().can_read(root)
    with pytest.raises(ValueError, match="action"):
        RldsReader().get_schema(root)


# ── Schema ─────────────────────────────────────────────────────────


def test_schema(dataset):
    root, _ = dataset
    schema = RldsReader().get_schema(root)
    assert schema.source_format == "rlds_tfrecord"
    assert schema.identity_name == "bridge_mini"
    assert schema.episodes == 3
    assert schema.total_frames == 5 + 7 + 4
    assert set(schema.cameras) == set(CAMERAS)
    assert all(schema.image_sizes[cam] == (H, W) for cam in CAMERAS)
    assert schema.state_dim == STATE_DIM
    assert schema.action_dim == ACTION_DIM
    assert schema.state_keys == tuple(f"state_{i}" for i in range(STATE_DIM))
    assert schema.action_keys == tuple(f"action_{i}" for i in range(ACTION_DIM))
    assert schema.has_language
    # camera semantics: "wrist_image" matches the wrist rule; plain "image"
    # matches nothing and must stay undeclared for the resolver to ask.
    cams = {c.key: c for c in schema.cameras_entries}
    assert cams["wrist_image"].semantic == "wrist"
    assert cams["image"].semantic is None
    # RLDS action semantics are not recorded in the spec — undeclared, not guessed.
    assert all(d.mode is None and d.mode_source == "undeclared" for d in schema.action_dims)


def test_episode_lengths_and_ranges(dataset):
    root, _ = dataset
    reader = RldsReader()
    assert reader.get_episode_lengths(root) == {0: 5, 1: 7, 2: 4}
    assert reader.get_episode_ranges(root) == {0: (0, 4), 1: (5, 11), 2: (12, 15)}


# ── Episode reading ────────────────────────────────────────────────


def test_read_episode_values(dataset):
    root, episodes = dataset
    reader = RldsReader()
    codec = resolve_codec("auto", "rlds")
    assert isinstance(codec, RldsJpegCodec)

    frames = reader.read_episode(root, 1, codec).load_frames()
    data = episodes[1]["data"]
    assert len(frames) == 7
    for t, frame in enumerate(frames):
        np.testing.assert_allclose(frame.state, data["state"][t], rtol=1e-5)
        # RLDS action[t] is the command issued at step t — no shifting.
        np.testing.assert_allclose(frame.action, data["action"][t], rtol=1e-5)
        assert frame.language == (data["lang"], data["lang2"])
        assert frame.index == t
    assert frames[0].is_first and not frames[1].is_first
    assert frames[-1].is_last and not frames[0].is_last


def test_read_episode_decodes_images(dataset):
    root, episodes = dataset
    reader = RldsReader()
    codec = resolve_codec("auto", "rlds")
    frames = reader.read_episode(root, 0, codec).load_frames()

    for cam in CAMERAS:
        ref = frames[2].images[cam]
        assert ref.stream == cam
        assert ref.payload is not None
        img = codec.decode_frame(ref)
        assert img.shape == (H, W, 3)
        assert img.dtype == np.uint8
        # PNG encoding round-trips pixels exactly.
        np.testing.assert_array_equal(img, episodes[0]["data"]["images"][cam][2])


def test_read_episode_out_of_range(dataset):
    root, _ = dataset
    with pytest.raises(KeyError, match="Episode 99"):
        RldsReader().read_episode(root, 99, resolve_codec("auto", "rlds"))


# ── Codec caching ──────────────────────────────────────────────────


def test_codec_content_addressed_cache(dataset):
    """Same step index, different episodes: no cache aliasing."""
    root, episodes = dataset
    reader = RldsReader()
    codec = RldsJpegCodec()
    ref0 = reader.read_episode(root, 0, codec).load_frames()[0].images["image"]
    ref1 = reader.read_episode(root, 2, codec).load_frames()[0].images["image"]

    first = codec.decode_frame(ref0)
    second = codec.decode_frame(ref1)
    np.testing.assert_array_equal(first, episodes[0]["data"]["images"]["image"][0])
    np.testing.assert_array_equal(second, episodes[2]["data"]["images"]["image"][0])
    assert not np.array_equal(first, second)
    # Re-decode returns the cached pixels unchanged.
    np.testing.assert_array_equal(codec.decode_frame(ref0), first)


# ── Normalisation statistics ───────────────────────────────────────


def test_norm_stats_computed_by_scan(dataset):
    root, episodes = dataset
    stats = RldsReader().get_norm_stats(root)
    assert stats.state is not None and stats.action is not None
    all_state = np.concatenate([e["data"]["state"] for e in episodes])
    all_action = np.concatenate([e["data"]["action"] for e in episodes])
    np.testing.assert_allclose(stats.state.mean, all_state.mean(axis=0), rtol=1e-4)
    np.testing.assert_allclose(stats.action.std, all_action.std(axis=0), rtol=1e-3)
    np.testing.assert_allclose(stats.state.min, all_state.min(axis=0), rtol=1e-5)
    assert len(stats.state.q01) == STATE_DIM and len(stats.action.q99) == ACTION_DIM
    np.testing.assert_allclose(
        stats.action.q01, np.quantile(all_action, 0.01, axis=0), rtol=1e-4
    )


def test_norm_stats_from_dataset_statistics_file(tmp_path):
    root = tmp_path / "with_stats"
    data = _episode_arrays(4, 7)
    _write_dataset(
        root,
        [{"data": data, "T": 4, "proto": _encode_episode(data, 4)}],
        stats_file={
            "num_trajectories": 1,
            "action": {
                "mean": [0.1] * ACTION_DIM, "std": [0.2] * ACTION_DIM,
                "min": [-1.0] * ACTION_DIM, "max": [1.0] * ACTION_DIM,
                "q01": [-0.9] * ACTION_DIM, "q99": [0.9] * ACTION_DIM,
            },
            "proprio": {"mean": [0.5] * STATE_DIM, "std": [0.25] * STATE_DIM},
        },
    )
    stats = RldsReader().get_norm_stats(root)
    assert stats.action.mean == [0.1] * ACTION_DIM
    assert stats.action.q99 == [0.9] * ACTION_DIM
    # "proprio" spelling maps onto the state block.
    assert stats.state.mean == [0.5] * STATE_DIM
    assert stats.state.q01 == []


# ── Index sidecar ──────────────────────────────────────────────────


def test_sidecar_roundtrip(dataset):
    root, _ = dataset
    assert RldsReader().get_episode_lengths(root) == {0: 5, 1: 7, 2: 4}
    sidecar = root / ".vla_factory_rlds_index.json"
    assert sidecar.exists()

    # A fresh reader (and a scan wired to fail) must be served by the sidecar.
    reader = RldsReader()
    reader._scan = pytest.fail  # type: ignore[method-assign]
    assert reader.get_episode_lengths(root) == {0: 5, 1: 7, 2: 4}
    assert reader.get_norm_stats(root).action is not None  # stats persisted too


def test_sidecar_invalidated_on_shard_change(dataset):
    root, episodes = dataset
    assert RldsReader().get_episode_lengths(root) == {0: 5, 1: 7, 2: 4}

    extra = _episode_arrays(6, 999)
    proto = _encode_episode(extra, 6)
    with tf.io.TFRecordWriter(str(root / "bridge_mini-train.tfrecord-00001-of-00002")) as writer:
        writer.write(episodes[2]["proto"])
        writer.write(proto)

    assert RldsReader().get_episode_lengths(root) == {0: 5, 1: 7, 2: 4, 3: 6}


# ── End to end: VLADataset samples ─────────────────────────────────


def _build_vla_dataset(root: Path):
    from vla_factory.training.dataset import VLADataset, build_sample_windows

    reader = RldsReader()
    codec = resolve_codec("auto", "rlds")
    lengths = reader.get_episode_lengths(root)
    windows = build_sample_windows(
        episode_lengths=lengths, n_obs_steps=1, action_horizon=4
    )
    return VLADataset(windows, reader, codec, root, None)


def test_vla_dataset_samples(dataset):
    root, _ = dataset
    ds = _build_vla_dataset(root)

    # Episode 0 has 5 valid observation positions.
    assert len(ds) == 5 + 7 + 4
    sample = ds[6]  # episode 1, position 1
    assert sample["images.image"].shape == (H, W, 3)
    assert sample["images.wrist_image"].shape == (H, W, 3)
    assert sample["image_masks.image"]
    assert sample["state"].shape == (STATE_DIM,)
    assert sample["state"].dtype == np.float32
    assert sample["actions"].shape == (4, ACTION_DIM)
    assert sample["action_is_pad"].shape == (4,)
    assert sample["task"] in ("pick up the cube 200", "grab the cube 200")

    from vla_factory.training.dataset import collate_fn

    batch = collate_fn([ds[0], ds[6]])
    observation = batch["observation"]
    assert set(observation.images) == set(CAMERAS)
    assert observation.images["image"].shape == (2, H, W, 3)
    assert batch["actions"].shape == (2, 4, ACTION_DIM)
    # Each sample's task is one randomly-chosen variant of its episode.
    assert observation.task[0] in ("pick up the cube 100", "grab the cube 100")
    assert observation.task[1] in ("pick up the cube 200", "grab the cube 200")


def test_dataloader_forked_workers(dataset):
    """TF must stay usable inside forked DataLoader workers.

    The reader only touches TensorFlow's protobuf surface (no tf.data /
    eager / GPU), which is what keeps the lazy import fork-inheritable; this
    smoke test pins that property end to end.
    """
    import torch
    from torch.utils.data import DataLoader

    from vla_factory.training.dataset import collate_fn

    ds = _build_vla_dataset(dataset[0])
    loader = DataLoader(
        ds, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn,
    )
    batches = []
    for batch in loader:
        assert batch["observation"].images["image"].shape[1:] == (H, W, 3)
        assert batch["actions"].shape[1:] == (4, ACTION_DIM)
        batches.append(batch)
    assert sum(b["actions"].shape[0] for b in batches) == len(ds)
