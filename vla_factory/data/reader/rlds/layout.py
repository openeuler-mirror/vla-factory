"""Locate TFDS directories and select one dataset split's TFRecord shards."""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any


_MAX_DESCENT_DEPTH = 3
_PLACEHOLDERS = {
    "SHARD_INDEX": r"\d+",
    "NUM_SHARDS": r"\d+",
    "SHARD_X_OF_Y": r"\d+-of-\d+",
}


def is_tfrecord(path: Path) -> bool:
    name = path.name
    return (
        path.is_file()
        and not name.startswith("._")
        and ".tfrecord" in name
        and not name.endswith((".index", "_index.json"))
    )


def locate_root(path: Path) -> Path:
    """Resolve a config/version directory, allowing TFDS parent directories."""
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a directory")
    if _is_dataset_root(path):
        return path

    roots = _find_roots(path, _MAX_DESCENT_DEPTH)
    if len(roots) == 1:
        return roots[0]
    if roots:
        listed = ", ".join(str(root) for root in roots[:3])
        raise ValueError(
            f"{path} contains multiple RLDS datasets ({listed}); point "
            "data.path at one config/version directory"
        )
    raise ValueError(f"{path} contains no TFDS RLDS dataset")


def shard_files(root: Path, split: str = "train") -> list[Path]:
    """Return shards for ``split`` without mixing validation/test records."""
    files = sorted(path for path in root.iterdir() if is_tfrecord(path))
    if not files:
        return []

    info = _read_json(root / "dataset_info.json")
    if info is not None:
        selected = _select_from_metadata(files, info, split)
        if selected:
            return selected

    # Older third-party exports sometimes omit dataset_info.json. Accept the
    # two common naming forms, but never use an ambiguous partial match.
    marker = re.compile(rf"(?:^|[-_.]){re.escape(split)}(?:[-_.]|$)")
    selected = [path for path in files if marker.search(path.name)]
    if selected:
        return selected
    if len(files) == 1:
        return files
    raise ValueError(
        f"{root}: cannot identify the {split!r} split among {len(files)} "
        "TFRecord files; add dataset_info.json or use split-qualified names"
    )


def dataset_name(root: Path) -> str:
    info = _read_json(root / "dataset_info.json")
    name = info.get("name") if info else None
    return str(name) if name else root.name


def _is_dataset_root(path: Path) -> bool:
    return (path / "features.json").is_file() and any(
        is_tfrecord(child) for child in path.iterdir()
    )


def _find_roots(path: Path, depth: int) -> list[Path]:
    if _is_dataset_root(path):
        return [path]
    if depth == 0:
        return []
    roots: list[Path] = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            roots.extend(_find_roots(child, depth - 1))
    return roots


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _select_from_metadata(
    files: list[Path], info: dict[str, Any], split: str
) -> list[Path]:
    splits = info.get("splits")
    if not isinstance(splits, list):
        return []
    entry = next(
        (item for item in splits if isinstance(item, dict) and item.get("name") == split),
        None,
    )
    if entry is None:
        if len(splits) == 1 and isinstance(splits[0], dict):
            entry = splits[0]
        else:
            available = [item.get("name") for item in splits if isinstance(item, dict)]
            raise ValueError(
                f"dataset has no {split!r} split; available splits: {available}"
            )

    template = entry.get("filepathTemplate") or entry.get("filenameTemplate")
    if isinstance(template, str):
        pattern = _template_pattern(template, info, str(entry.get("name", split)))
        return [path for path in files if pattern.fullmatch(path.name)]
    return []


def _template_pattern(
    template: str, info: dict[str, Any], split: str
) -> re.Pattern[str]:
    values = {
        "DATASET": re.escape(str(info.get("name", ""))),
        "SPLIT": re.escape(split),
        "FILEFORMAT": re.escape(str(info.get("fileFormat", "tfrecord"))),
        **_PLACEHOLDERS,
    }
    pieces: list[str] = []
    for literal, field, _format_spec, _conversion in string.Formatter().parse(template):
        pieces.append(re.escape(literal))
        if field:
            pieces.append(values.get(field, r"[^/]+"))
    return re.compile("".join(pieces))
