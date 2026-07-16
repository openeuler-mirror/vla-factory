"""Z-score normalisation transform (numpy-based).

Uses the same formula as lerobot: ``(x - mean) / (std + eps)`` with ``eps=1e-8``.
This handles near-zero std dimensions (e.g. constant gripper) correctly — the
epsilon is additive rather than a max clamp, matching lerobot's behaviour.

Normalises **state** and **actions** with per-feature z-score from the dataset's
own statistics.  **Images** are normalised with ImageNet channel statistics
(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) matching lerobot's
``use_imagenet_stats=True`` default (see ``lerobot/datasets/factory.py:126``).
"""

from __future__ import annotations

import numpy as np

from ..manifest import NormStats
from .base import TransformStep
from .registry import TransformRegistry

_EPS = 1e-8  # matches lerobot's NormalizationProcessor default

# ImageNet normalization — lerobot overrides image stats with these when
# DatasetConfig.use_imagenet_stats=True (the default).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@TransformRegistry.register("normalize")
class Normalize(TransformStep):
    """Apply Z-score normalisation: ``(x - mean) / (std + eps)``.

    Operates on the ``"state"``, ``"actions"``, and ``"images.*"`` keys of the
    sample dict.

    - State / actions: dataset's own per-dimension z-score.
    - Images: per-camera stats from ``norm_stats.images`` if present,
      otherwise ImageNet channel mean/std as fallback.

    The image normalisation is **driven by saved metadata**: if
    ``norm_stats.images`` contains per-camera stats (as saved by the training
    pipeline), those are used.  If absent, ImageNet stats are the default.
    This ensures train/infer use identical normalisation without code
    duplication — change the training pipeline, and the saved metadata
    automatically propagates to inference.

    Expects numpy arrays as input.
    """

    def __init__(self, stats: NormStats, use_imagenet_stats: bool = True) -> None:
        self._stats = stats
        # When True (default, matching lerobot's ``use_imagenet_stats=True``),
        # images are normalised with fixed ImageNet channel constants and
        # ``stats.images`` is ignored entirely. The ImageNet decision is a
        # property of THIS step — train/infer never mutate ``norm_stats`` to
        # smuggle ImageNet constants into the per-camera stats.
        self._use_imagenet_stats = use_imagenet_stats

    def __call__(self, sample: dict) -> dict:
        # Normalise state (z-score with dataset stats)
        if self._stats.state is not None and sample.get("state") is not None:
            mean = np.array(self._stats.state.mean, dtype=np.float32)
            std = np.array(self._stats.state.std, dtype=np.float32) + _EPS
            sample["state"] = (sample["state"] - mean) / std

        # Normalise actions (z-score with dataset stats)
        if self._stats.action is not None and sample.get("actions") is not None:
            mean = np.array(self._stats.action.mean, dtype=np.float32)
            std = np.array(self._stats.action.std, dtype=np.float32) + _EPS
            # actions shape: [horizon, dim] — broadcasting works against [dim] mean/std
            sample["actions"] = (sample["actions"] - mean) / std

        # Normalise images: use per-camera stats if present, else ImageNet
        for key in list(sample.keys()):
            if not key.startswith("images."):
                continue
            img = sample[key]
            if not (isinstance(img, np.ndarray) and img.ndim == 3):
                continue

            if self._use_imagenet_stats:
                mean = IMAGENET_MEAN
                # +_EPS mirrors the legacy override path (ImageNet constants were
                # stored in stats.images and went through the `std + _EPS` branch).
                # Kept for bit-exact reproduction of prior training output.
                std = IMAGENET_STD + _EPS
            else:
                cam_name = key[len("images."):]
                cam_stats = (
                    self._stats.images.get(cam_name)
                    if self._stats.images is not None
                    else None
                )
                if cam_stats is not None and cam_stats.mean and cam_stats.std:
                    mean = np.array(cam_stats.mean, dtype=np.float32)
                    std = np.array(cam_stats.std, dtype=np.float32) + _EPS
                else:
                    # No per-camera dataset stats available — fall back to ImageNet
                    # so a missing stats.images never yields unnormalised images.
                    mean = IMAGENET_MEAN
                    std = IMAGENET_STD
            # img is CHW float32 [0, 1]
            sample[key] = (img - mean[:, None, None]) / std[:, None, None]

        return sample


@TransformRegistry.register("unnormalize_action")
class UnnormalizeActionStep(TransformStep):
    """Reverse of :class:`Normalize` for the ``actions`` field.

    ``actions * (std + _EPS) + mean`` — the exact inverse of Normalize's action
    branch, sharing the same ``_EPS`` and the same ``norm_stats.action``.

    Action-only: state / image normalisation has *no* reverse at inference (we
    never un-normalise an observation). This step is what the postprocessor
    pipeline runs on the model's action output.
    """

    def __init__(self, stats: NormStats) -> None:
        self._stats = stats

    def __call__(self, sample: dict) -> dict:
        actions = sample.get("actions")
        if actions is None or self._stats.action is None:
            return sample
        mean = np.array(self._stats.action.mean, dtype=np.float32)
        std = np.array(self._stats.action.std, dtype=np.float32) + _EPS
        # actions: [..., D]; mean/std: [D] — broadcasts over the leading dims.
        sample["actions"] = actions * std + mean
        return sample


@TransformRegistry.register("normalize_vector")
class NormalizeVector(TransformStep):
    """Normalize selected vector fields with dataset statistics.

    - ``method="zscore"`` (default): ``(x - mean) / (std + eps)`` — pi0/ACT.
    - ``method="quantile"``: ``(x - q01) / (q99 - q01 + eps) * 2 - 1`` — maps
      the 1st..99th percentile range to [-1, 1], matching openpi's
      ``use_quantile_norm`` for pi05 (openpi transforms.py).
    """

    def __init__(
        self,
        stats: NormStats,
        fields: list[str] | tuple[str, ...] = ("state", "actions"),
        method: str = "zscore",
    ) -> None:
        if method not in ("zscore", "quantile"):
            raise ValueError(f"Unsupported normalize_vector method: {method!r}")
        self._stats = stats
        self.fields = tuple(fields)
        self.method = method

    @classmethod
    def from_config(cls, cfg: dict, ctx=None) -> "NormalizeVector | None":
        stats = getattr(ctx, "norm_stats", None)
        if stats is None:
            return None
        return cls(
            stats=stats,
            fields=tuple(cfg.get("fields", ("state", "actions"))),
            method=cfg.get("method", "zscore"),
        )

    def _normalize(self, x, stats, field_name: str):
        if self.method == "quantile":
            q01, q99 = _require_quantiles(stats, field_name)
            return (x - q01) / (q99 - q01 + _QUANTILE_EPS) * 2.0 - 1.0
        mean = np.array(stats.mean, dtype=np.float32)
        std = np.array(stats.std, dtype=np.float32) + _EPS
        return (x - mean) / std

    def __call__(self, sample: dict) -> dict:
        if "state" in self.fields and self._stats.state is not None and sample.get("state") is not None:
            sample["state"] = self._normalize(sample["state"], self._stats.state, "state")

        if "actions" in self.fields and self._stats.action is not None and sample.get("actions") is not None:
            # actions shape: [horizon, dim] — stats broadcast against [dim]
            sample["actions"] = self._normalize(sample["actions"], self._stats.action, "actions")
        return sample

    def inverse_for_output(self, ctx=None) -> TransformStep | None:
        if "actions" not in self.fields or self._stats.action is None:
            return None
        if self.method == "quantile":
            return UnnormalizeActionQuantileStep(self._stats)
        return UnnormalizeActionStep(self._stats)


# openpi's quantile-normalisation epsilon (transforms.py _normalize_quantile).
_QUANTILE_EPS = 1e-6


def _require_quantiles(stats, field_name: str):
    """Return (q01, q99) arrays or fail early with an actionable message."""
    if not stats.q01 or not stats.q99:
        raise ValueError(
            f"normalize_vector method='quantile' needs q01/q99 statistics for "
            f"{field_name!r}, but the dataset stats do not provide them. "
            "lerobot v3 writes quantiles to meta/stats.json; regenerate the "
            "dataset stats or use method='zscore'."
        )
    return (
        np.array(stats.q01, dtype=np.float32),
        np.array(stats.q99, dtype=np.float32),
    )


@TransformRegistry.register("unnormalize_action_quantile")
class UnnormalizeActionQuantileStep(TransformStep):
    """Reverse of quantile normalisation for the ``actions`` field.

    ``(x + 1) / 2 * (q99 - q01 + eps) + q01`` — the exact inverse of
    NormalizeVector's quantile branch (matches openpi ``Unnormalize`` with
    ``use_quantiles=True``).
    """

    def __init__(self, stats: NormStats) -> None:
        self._stats = stats

    def __call__(self, sample: dict) -> dict:
        actions = sample.get("actions")
        if actions is None or self._stats.action is None:
            return sample
        q01, q99 = _require_quantiles(self._stats.action, "actions")
        sample["actions"] = (actions + 1.0) / 2.0 * (q99 - q01 + _QUANTILE_EPS) + q01
        return sample
