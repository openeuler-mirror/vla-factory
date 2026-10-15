"""RLDS core tests that do not require the optional TensorFlow dependency."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vla_factory.data.codec import resolve_codec
from vla_factory.data.reader import get_reader
from vla_factory.data.reader.rlds import RldsReader
from vla_factory.data.reader.rlds.episode import episode_steps, vector_rows
from vla_factory.data.reader.rlds.layout import dataset_name, locate_root, shard_files
from vla_factory.data.reader.rlds.spec import RldsSpec, load_rlds_spec
from vla_factory.data.reader.rlds.stats import StreamingStats
from vla_factory.data.reader.rlds.tf_example import FeatureList


def _tensor(dtype: str, *shape: int) -> dict:
    return {
        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
        "tensor": {
            "dtype": dtype,
            "encoding": "none",
            "shape": {"dimensions": [str(dim) for dim in shape]},
        },
    }


def _image(height: int, width: int) -> dict:
    return {
        "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
        "image": {
            "dtype": "uint8",
            "encodingFormat": "jpeg",
            "shape": {"dimensions": [str(height), str(width), "3"]},
        },
    }


def _features(values: dict) -> dict:
    return {
        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
        "featuresDict": {"features": values},
    }


def _rlds_features() -> dict:
    steps = _features({
        "action": _tensor("float32", 7),
        "is_first": _tensor("bool"),
        "is_last": _tensor("bool"),
        "language_instruction": _tensor("string"),
        "observation": _features({
            "image": _image(256, 320),
            "state": _tensor("float32", 8),
            "wrist_image": _image(128, 160),
        }),
    })
    return _features({
        "episode_metadata": _features({"task": _tensor("string")}),
        "steps": {
            "pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
            "sequence": {"feature": steps},
        },
    })


def _write_metadata(root: Path, *, splits: list[str]) -> None:
    root.mkdir(parents=True)
    (root / "features.json").write_text(json.dumps(_rlds_features()))
    (root / "dataset_info.json").write_text(json.dumps({
        "fileFormat": "tfrecord",
        "name": "demo_robot",
        "splits": [
            {
                "name": split,
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
            }
            for split in splits
        ],
    }))


def test_parses_tfds_protobuf_json_features(tmp_path: Path):
    root = tmp_path / "demo_robot" / "default" / "1.0.0"
    _write_metadata(root, splits=["train"])
    (root / "demo_robot-train.tfrecord-00000-of-00001").touch()

    spec = load_rlds_spec(root)

    assert spec is not None
    assert spec.action_path == "steps/action"
    assert spec.action_dim == 7
    assert spec.state_path == "steps/observation/state"
    assert spec.state_dim == 8
    assert spec.language_paths == ("steps/language_instruction",)
    assert spec.context_language_paths == ("episode_metadata/task",)
    assert [(cam.key, cam.height, cam.width) for cam in spec.cameras] == [
        ("image", 256, 320),
        ("wrist_image", 128, 160),
    ]
    assert locate_root(root.parent.parent.parent) == root
    assert dataset_name(root) == "demo_robot"
    reader = get_reader("auto", path=root)
    assert isinstance(reader, RldsReader)
    schema = reader.get_schema(root)
    assert {camera.key: camera.image_storage.value for camera in schema.cameras_entries} == {
        "image": "encoded_bytes",
        "wrist_image": "encoded_bytes",
    }
    assert (
        resolve_codec(
            "auto",
            camera_storages={
                camera.key: camera.image_storage for camera in schema.cameras_entries
            },
            dataset_format=schema.source_format,
        ).name
        == "rlds_jpeg"
    )


def test_standard_tfds_names_do_not_mix_splits(tmp_path: Path):
    root = tmp_path / "1.0.0"
    _write_metadata(root, splits=["train", "validation"])
    train = root / "demo_robot-train.tfrecord-00000-of-00001"
    validation = root / "demo_robot-validation.tfrecord-00000-of-00001"
    train.touch()
    validation.touch()

    assert shard_files(root) == [train]


def test_multiple_non_train_splits_are_rejected(tmp_path: Path):
    root = tmp_path / "1.0.0"
    _write_metadata(root, splits=["validation", "test"])
    (root / "demo_robot-validation.tfrecord-00000-of-00001").touch()
    (root / "demo_robot-test.tfrecord-00000-of-00001").touch()

    with pytest.raises(ValueError, match="no 'train' split"):
        shard_files(root)


def test_ambiguous_shards_without_metadata_are_rejected(tmp_path: Path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "part-00000.tfrecord").touch()
    (root / "part-00001.tfrecord").touch()

    with pytest.raises(ValueError, match="cannot identify"):
        shard_files(root)


def test_episode_vectors_are_validated_without_tensorflow(tmp_path: Path):
    spec = RldsSpec(
        root=tmp_path,
        action_dim=2,
        state_path="steps/observation/state",
        state_dim=3,
        is_first_path="steps/is_first",
    )
    features = {
        "steps/action": FeatureList(
            "floats", float_values=[1.0, 2.0, 3.0, 4.0]
        ),
        "steps/observation/state": FeatureList(
            "floats", float_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ),
        "steps/is_first": FeatureList("ints", int_values=[1, 0]),
    }

    steps, width = episode_steps(features, spec, 2, tmp_path / "shard", 12)
    assert (steps, width) == (2, 2)
    np.testing.assert_array_equal(
        vector_rows(features, spec.state_path, steps, spec.state_dim),
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )

    features["steps/action"].float_values.append(5.0)
    with pytest.raises(ValueError, match="do not divide"):
        episode_steps(features, spec, 2, tmp_path / "shard", 12)


def test_streaming_stats_match_numpy():
    rows = np.asarray([[1.0, 5.0], [3.0, 1.0], [8.0, 2.0]])
    accumulator = StreamingStats()
    accumulator.add_rows(rows[:2])
    accumulator.add_rows(rows[2:])

    stats = accumulator.finish()
    assert stats is not None
    np.testing.assert_allclose(stats.mean, rows.mean(axis=0))
    np.testing.assert_allclose(stats.std, rows.std(axis=0))
    np.testing.assert_allclose(stats.min, rows.min(axis=0))
    np.testing.assert_allclose(stats.max, rows.max(axis=0))
