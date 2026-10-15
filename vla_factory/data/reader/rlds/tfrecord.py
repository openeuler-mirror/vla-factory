"""TFRecord container framing — record offsets for random episode access.

A TFRecord file is a sequence of records, each framed as::

    uint64 length            (little endian)
    uint32 masked_crc32(length)   ┐ CRC32C of each part, masked with
    byte   data[length]           │ ((crc >> 15) | (crc << 17)) + 0xa282ead8
    uint32 masked_crc32(data)     ┘ (see TF's tf_record spec)

``tf.data.TFRecordDataset`` reads this framing but streams without exposing
record offsets, and the training data layer needs both: the index scan
records where every episode lives, and ``read_episode`` seeks straight to
one record. That framing walk is what stays local here — the proto *inside*
each record is parsed by TensorFlow (see :mod:`.tf_example`).

Gzip/zlib-compressed TFRecords are not supported — random access into a
compressed stream needs a full offset remap, and RLDS/OXE distributions
ship uncompressed shards. Compressed files are detected and rejected with
an actionable message instead of misparsing the magic bytes as a record
length.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Iterator

# 8-byte length + 4-byte length CRC precede each payload; 4-byte payload CRC
# trails it.
_HEADER_LEN = 12
_FOOTER_LEN = 4
_GZIP_MAGIC = b"\x1f\x8b"
_CRC_MASK_DELTA = 0xA282EAD8
_CRC_TABLE: tuple[int, ...] | None = None


def _iter_spans(f: IO[bytes], file_size: int, path: Path) -> Iterator[tuple[int, int]]:
    """Yield ``(payload_offset, payload_length)`` over an open shard file.

    Every iteration seeks to its header absolutely, so consumers may use the
    same handle between yields (``scan_records`` reads the payload) without
    desynchronising the walk. Header-only I/O otherwise — payloads are
    skipped, never read, on the span path.
    """
    offset = 0
    while offset < file_size:
        f.seek(offset)
        header = f.read(_HEADER_LEN)
        if len(header) < _HEADER_LEN:
            raise ValueError(
                f"{path}: truncated TFRecord header at offset {offset} "
                f"(got {len(header)} of {_HEADER_LEN} bytes)."
            )
        if offset == 0 and header[:2] == _GZIP_MAGIC:
            raise ValueError(
                f"{path}: gzip-compressed TFRecords are not supported "
                "(RLDS/OXE shards are uncompressed). Re-export the "
                "dataset without compression."
            )
        _verify_crc(header[:8], header[8:12], path, offset, "length")
        length = int.from_bytes(header[:8], "little")
        payload_offset = offset + _HEADER_LEN
        if payload_offset + length + _FOOTER_LEN > file_size:
            raise ValueError(
                f"{path}: TFRecord at offset {offset} declares "
                f"{length} payload bytes but only "
                f"{file_size - payload_offset} remain in the file "
                "(corrupt or truncated shard)."
            )
        yield payload_offset, length
        offset = payload_offset + length + _FOOTER_LEN


def iter_record_spans(path: Path) -> Iterator[tuple[int, int]]:
    """Yield ``(payload_offset, payload_length)`` for every record in ``path``."""
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        yield from _iter_spans(f, file_size, path)


def scan_records(path: Path) -> Iterator[tuple[int, int, bytes]]:
    """Yield ``(payload_offset, payload_length, payload)`` per record.

    Single file handle, payloads read in file order — the shape the index
    scan wants (it needs both the span for later random access and the
    proto bytes for per-episode step counts).
    """
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        for offset, length in _iter_spans(f, file_size, path):
            data = f.read(length)
            if len(data) != length:
                raise ValueError(
                    f"{path}: short read at offset {offset} "
                    f"({len(data)} of {length} bytes)."
                )
            checksum = f.read(_FOOTER_LEN)
            _verify_crc(data, checksum, path, offset, "payload")
            yield offset, length, data


def read_record(path: Path, offset: int, length: int) -> bytes:
    """Read one record's payload given its span from a previous scan."""
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(length)
        checksum = f.read(_FOOTER_LEN)
    if len(data) != length:
        raise ValueError(
            f"{path}: short read at offset {offset} "
            f"({len(data)} of {length} bytes)."
        )
    _verify_crc(data, checksum, path, offset, "payload")
    return data


def _verify_crc(
    data: bytes, stored: bytes, path: Path, offset: int, part: str
) -> None:
    if len(stored) != 4:
        raise ValueError(f"{path}: missing {part} CRC at offset {offset}")
    expected = int.from_bytes(stored, "little")
    actual = _masked_crc32c(data)
    if actual != expected:
        raise ValueError(
            f"{path}: invalid {part} CRC at offset {offset} "
            f"(expected 0x{expected:08x}, computed 0x{actual:08x})"
        )


def _masked_crc32c(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + _CRC_MASK_DELTA) & 0xFFFFFFFF


def _crc32c(data: bytes) -> int:
    """Castagnoli CRC32C, using the optional native module when available."""
    try:
        import google_crc32c  # type: ignore[import-not-found]
    except ImportError:
        return _crc32c_python(data)
    return int(google_crc32c.value(data))


def _crc32c_python(data: bytes) -> int:
    global _CRC_TABLE
    if _CRC_TABLE is None:
        table: list[int] = []
        for value in range(256):
            crc = value
            for _ in range(8):
                crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
            table.append(crc)
        _CRC_TABLE = tuple(table)
    crc = 0xFFFFFFFF
    for value in data:
        crc = _CRC_TABLE[(crc ^ value) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF
