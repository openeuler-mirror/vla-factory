"""Storage-aware image-decoder selection and validation tests."""

from __future__ import annotations

import pytest

from vla_factory.data.codec import ImageDecoderRouter, resolve_image_decoder
from vla_factory.data.data_schema import DataSchema, ImageRef, ImageStorage


def test_image_ref_has_a_storage_neutral_path(tmp_path):
    ref = ImageRef(
        path=tmp_path / "episode.tfrecord",
        frame_index=4,
        height=240,
        width=320,
        storage=ImageStorage.ENCODED_BYTES,
        payload=b"encoded-image",
    )
    assert ref.path == tmp_path / "episode.tfrecord"
    assert ref.storage is ImageStorage.ENCODED_BYTES


def test_image_ref_requires_a_storage_path():
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'path'"):
        ImageRef(frame_index=0, height=240, width=320)


@pytest.mark.parametrize(
    ("source_format", "expected"),
    [
        ("lerobot_v3", ImageStorage.VIDEO_FILE),
        ("robotwin_hdf5", ImageStorage.HDF5_JPEG),
        ("rlds_tfrecord", ImageStorage.ENCODED_BYTES),
    ],
)
def test_old_schema_inferrs_image_storage_from_source_format(source_format, expected):
    schema = DataSchema.from_dict(
        {
            "identity": {"source_format": source_format},
            "observation": {"cameras": [{"key": "front"}]},
        }
    )
    assert schema.cameras_entries[0].image_storage is expected


def test_old_schema_with_unknown_format_requires_explicit_storage():
    with pytest.raises(ValueError, match="lacks required 'image_storage'"):
        DataSchema.from_dict(
            {
                "identity": {"source_format": "unknown_format"},
                "observation": {"cameras": [{"key": "front"}]},
            }
        )


def test_auto_selects_decoder_from_declared_storage():
    decoder = resolve_image_decoder(
        "auto",
        camera_storages={"front": ImageStorage.ENCODED_BYTES},
        dataset_format="rlds_tfrecord",
    )
    assert decoder.name == "rlds_jpeg"


def test_explicit_decoder_rejects_incompatible_rlds_storage():
    with pytest.raises(
        ValueError,
        match="pyav.*rlds_tfrecord.*front=encoded_bytes.*video_file",
    ):
        resolve_image_decoder(
            "pyav",
            camera_storages={"front": ImageStorage.ENCODED_BYTES},
            dataset_format="rlds_tfrecord",
        )


def test_auto_routes_heterogeneous_storages():
    decoder = resolve_image_decoder(
        "auto",
        camera_storages={
            "overhead": ImageStorage.HDF5_JPEG,
            "wrist": ImageStorage.ENCODED_BYTES,
        },
        dataset_format="mixed_fixture",
    )
    assert isinstance(decoder, ImageDecoderRouter)
    assert decoder.supported_storages == {
        ImageStorage.HDF5_JPEG,
        ImageStorage.ENCODED_BYTES,
    }
