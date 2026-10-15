"""Normalization statistics used by the RLDS index scan."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...data_schema import FeatureStats, NormStats


class StreamingStats:
    """Accumulate per-dimension moments and bounded-memory quantiles."""

    _SAMPLE_CAP = 1_000_000

    def __init__(self) -> None:
        self._count = 0
        self._sum: np.ndarray | None = None
        self._sum_squares: np.ndarray | None = None
        self._minimum: np.ndarray | None = None
        self._maximum: np.ndarray | None = None
        self._sample_stride = 1
        self._samples: list[np.ndarray] = []

    def add_rows(self, rows: np.ndarray) -> None:
        for row in rows.astype(np.float64, copy=False):
            if self._sum is None:
                self._sum = np.zeros_like(row)
                self._sum_squares = np.zeros_like(row)
                self._minimum = row.copy()
                self._maximum = row.copy()
            self._count += 1
            self._sum += row
            self._sum_squares += row * row
            np.minimum(self._minimum, row, out=self._minimum)
            np.maximum(self._maximum, row, out=self._maximum)
            if (self._count - 1) % self._sample_stride == 0:
                self._samples.append(row.copy())
                if len(self._samples) >= self._SAMPLE_CAP:
                    self._samples = self._samples[::2]
                    self._sample_stride *= 2

    def finish(self) -> FeatureStats | None:
        if self._count == 0 or self._sum is None:
            return None
        mean = self._sum / self._count
        variance = self._sum_squares / self._count - mean * mean
        sampled = np.stack(self._samples)
        return FeatureStats(
            mean=mean.tolist(),
            std=np.sqrt(np.maximum(variance, 0.0)).tolist(),
            min=self._minimum.tolist(),
            max=self._maximum.tolist(),
            q01=np.quantile(sampled, 0.01, axis=0).tolist(),
            q99=np.quantile(sampled, 0.99, axis=0).tolist(),
        )


def norm_stats_from_dlimp(raw: dict[str, Any]) -> NormStats:
    """Convert the common OXE/dlimp ``dataset_statistics.json`` shape."""

    def find_block(*names: str) -> dict[str, Any] | None:
        return next(
            (raw[name] for name in names if isinstance(raw.get(name), dict)),
            None,
        )

    def convert(block: dict[str, Any] | None) -> FeatureStats | None:
        if block is None:
            return None
        return FeatureStats(
            mean=[float(value) for value in block.get("mean", [])],
            std=[float(value) for value in block.get("std", [])],
            min=[float(value) for value in block.get("min", [])],
            max=[float(value) for value in block.get("max", [])],
            q01=[float(value) for value in block.get("q01", [])],
            q99=[float(value) for value in block.get("q99", [])],
        )

    return NormStats(
        state=convert(find_block("state", "proprio")),
        action=convert(find_block("action")),
        images=None,
        method="zscore",
    )
