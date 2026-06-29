"""Orchestration: YAML recipe -> train/val DataLoaders.

``create_dataloaders()`` is the main entry point.  It wires together:

1. ``FormatReader``   — read raw dataset (e.g. LeRobot v3)
2. ``VideoCodec``     — pluggable video decoding (default PyAV)
3. Transforms         — assembled from YAML config + dataset stats
4. ``SlidingWindowSampler`` — episode -> sample indexing
5. ``DatasetManifest`` — train/val split + locator list
6. ``VLADataset``     — per-sample loading (numpy output)
7. ``DataLoader``     — batching & shuffling
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from vla_factory.config.recipe import TrainRecipe

from .formats import get_reader
from .codec import resolve_codec
from .manifest import build_manifest
from .dataset import VLADataset, collate_fn
from .transforms import build_preprocessor, BuildContext, TransformPipeline

logger = logging.getLogger(__name__)


def _build_transforms(
    norm_stats: Any | None = None,
    model_metadata: dict[str, Any] | None = None,
    action_dim: int = 0,
) -> TransformPipeline:
    """Assemble the preprocessor pipeline from model + dataset info.

    Thin wrapper over :func:`build_preprocessor`.  The ordered transform list
    comes from ``model_metadata["default_transforms"]`` (declared on the model
    registry entry, e.g. ACT's ``("normalize", "resize_images",
    "pad_dimensions")``), falling back to the legacy default list when absent.
    Each step self-skips when its precondition isn't met (no stats, zero
    image_size, no padding needed) — identical to the previous behaviour.
    """
    md = model_metadata or {}
    img_size = md.get("image_size")
    ctx = BuildContext(
        norm_stats=norm_stats,
        image_size=tuple(img_size) if img_size else None,
        model_action_dim=md.get("action_dim", 0),
        dataset_action_dim=action_dim,
    )
    return build_preprocessor(md.get("default_transforms") or None, ctx)


def create_dataloaders(
    recipe: TrainRecipe,
    model_metadata: dict[str, Any] | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train and val DataLoaders from a ``TrainRecipe``.

    Parameters
    ----------
    recipe : TrainRecipe
        Parsed YAML configuration.
    model_metadata : dict, optional
        Model capabilities (``action_dim``, ``image_size``, etc.) used to
        select transforms.

    Returns
    -------
    (train_loader, val_loader)
    """
    data_cfg = recipe.data
    sampler_cfg = data_cfg.sampler
    split_cfg = data_cfg.split
    path = Path(data_cfg.source.path)

    # 1. Reader + codec
    reader = get_reader(data_cfg.source.format, path=path)
    codec = resolve_codec(data_cfg.source.video_codec)

    # 2. Schema, stats, episode info
    schema = reader.get_schema(path)
    norm_stats = reader.get_norm_stats(path)
    episode_ranges = reader.get_episode_ranges(path)
    episode_lengths = reader.get_episode_lengths(path)

    # Note: ImageNet image normalisation is handled inside the Normalize
    # transform (use_imagenet_stats=True default). norm_stats is NOT mutated
    # here — it carries the dataset's own statistics straight into the pipeline.

    logger.info(
        "Dataset loaded: %d episodes, %d frames, action_dim=%d, state_dim=%d",
        schema.total_episodes, schema.total_frames,
        schema.action_dim, schema.state_dim,
    )

    # 3. Build transforms from config + stats
    transforms = _build_transforms(
        norm_stats=norm_stats,
        model_metadata=model_metadata,
        action_dim=schema.action_dim,
    )

    # 4. Manifest
    manifest = build_manifest(
        schema=schema,
        norm_stats=norm_stats,
        episode_ranges=episode_ranges,
        episode_lengths=episode_lengths,
        n_obs_steps=sampler_cfg.n_obs_steps,
        action_horizon=sampler_cfg.action_horizon,
        split_strategy=split_cfg.strategy,
        train_ratio=split_cfg.train_ratio,
        seed=split_cfg.seed,
    )

    logger.info(
        "Manifest: %d train samples, %d val samples",
        len(manifest._train_indices),
        len(manifest._val_indices),
    )

    # 5. Datasets
    train_ds = VLADataset(manifest, reader, codec, path, transforms, split="train")
    val_ds = VLADataset(manifest, reader, codec, path, transforms, split="val")

    # 6. DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=recipe.batch_size,
        shuffle=True,
        num_workers=recipe.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=recipe.batch_size,
        shuffle=False,
        num_workers=recipe.num_workers,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
