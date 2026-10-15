"""Framing tests for the RLDS TFRecord offset reader.

The record *framing* (length-prefix + CRC layout) stays local because
streaming TF APIs expose no offsets for random access, so these tests pin
its behaviour directly: span math, truncated/gzip rejection. When
TensorFlow is installed (the ``[rlds]`` extra), records written by
``tf.io.TFRecordWriter`` — the reference implementation, real CRCs — must
round-trip through the same framing walk, which is the cross-validation
that matters for real TFDS data.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from vla_factory.data.reader.rlds.tfrecord import (
    _masked_crc32c,
    iter_record_spans,
    read_record,
    scan_records,
)

HAS_TF = importlib.util.find_spec("tensorflow") is not None


def _write_tfrecord(path: Path, records: list[bytes]) -> None:
    """Frame serialized protos by hand with TFRecord CRC32C checksums."""
    with open(path, "wb") as f:
        for record in records:
            length = len(record).to_bytes(8, "little")
            f.write(length)
            f.write(_masked_crc32c(length).to_bytes(4, "little"))
            f.write(record)
            f.write(_masked_crc32c(record).to_bytes(4, "little"))


# ── Framing (no TensorFlow involved) ───────────────────────────────


def test_framing_spans_and_reads(tmp_path: Path):
    records = [b"alpha", b"beta", b""]
    path = tmp_path / "train.tfrecord"
    _write_tfrecord(path, records)

    spans = list(iter_record_spans(path))
    assert [length for _, length in spans] == [5, 4, 0]
    # Each payload starts right after its 12-byte header; every record adds
    # header + payload + 4-byte CRC to the cursor.
    assert [offset for offset, _ in spans] == [
        12, 12 + 12 + 5 + 4, 12 + 12 + 5 + 4 + 12 + 4 + 4,
    ]
    for (offset, length), record in zip(spans, records):
        assert read_record(path, offset, length) == record
    assert list(scan_records(path))[0][2] == b"alpha"


def test_framing_truncated_header(tmp_path: Path):
    path = tmp_path / "bad.tfrecord"
    _write_tfrecord(path, [b"ok"])
    data = path.read_bytes()
    path.write_bytes(data[:-2])  # cut into the trailing CRC
    with pytest.raises(ValueError, match="truncated|declares"):
        list(iter_record_spans(path))


def test_framing_gzip_rejected(tmp_path: Path):
    path = tmp_path / "compressed.tfrecord"
    path.write_bytes(b"\x1f\x8b" + b"\0" * 32)
    with pytest.raises(ValueError, match="gzip"):
        list(iter_record_spans(path))


def test_payload_crc_rejected(tmp_path: Path):
    path = tmp_path / "corrupt.tfrecord"
    _write_tfrecord(path, [b"payload"])
    data = bytearray(path.read_bytes())
    data[12] ^= 1
    path.write_bytes(data)

    with pytest.raises(ValueError, match="invalid payload CRC"):
        list(scan_records(path))


# ── Cross-validation against TensorFlow's writer ───────────────────


@pytest.mark.skipif(not HAS_TF, reason='tensorflow not installed (pip install -e ".[rlds]")')
def test_tf_written_records_roundtrip(tmp_path: Path):
    """Real tf.io.TFRecordWriter output (true CRCs) must frame-read back."""
    import tensorflow as tf

    protos = [
        tf.train.Example(
            features=tf.train.Features(feature={
                "steps/action": tf.train.Feature(
                    float_list=tf.train.FloatList(value=[0.5, -1.25])
                ),
            })
        ).SerializeToString(),
        b"",
        tf.train.Example(
            features=tf.train.Features(feature={
                "steps/is_first": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[1, 0, 0])
                ),
            })
        ).SerializeToString(),
    ]
    path = tmp_path / "train-00000-of-00001.tfrecord"
    with tf.io.TFRecordWriter(str(path)) as writer:
        for proto in protos:
            writer.write(proto)

    read_back = [data for _, _, data in scan_records(path)]
    assert read_back == protos

    from vla_factory.data.reader.rlds.tf_example import parse_example
    parsed = [parse_example(data) for data in read_back]
    assert parsed[0]["steps/action"].float_values == [0.5, -1.25]
    assert parsed[2]["steps/is_first"].int_values == [1, 0, 0]
    assert parsed[1] == {}


@pytest.mark.skipif(not HAS_TF, reason='tensorflow not installed (pip install -e ".[rlds]")')
def test_parse_rejects_garbage():
    from vla_factory.data.reader.rlds.tf_example import parse_example

    with pytest.raises(ValueError, match="Cannot parse"):
        parse_example(b"\xde\xad\xbe\xef" * 8)
