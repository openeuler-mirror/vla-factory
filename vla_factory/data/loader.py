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
import importlib
from pathlib import Path
from typing import Any

import torch

from vla_factory.config.recipe import TrainRecipe
from .formats import get_reader
from .codec import resolve_codec
from .manifest import build_manifest
from .dataset import VLADataset, collate_fn
from .transforms import build_preprocessor, TransformContext, TransformPipeline

logger = logging.getLogger(__name__)


def _build_transforms(
    norm_stats: Any | None = None,
    model_metadata: dict[str, Any] | None = None,
    action_dim: int = 0,
    recipe: TrainRecipe | None = None,
    schema: Any | None = None,
    split: str = "train",
) -> TransformPipeline:
    """Assemble the preprocessor pipeline from YAML transform config."""
    md = model_metadata or {}
    ctx = TransformContext(
        norm_stats=norm_stats,
        model_action_dim=md.get("action_dim", 0),
        dataset_action_dim=action_dim,
        recipe=recipe,
        schema=schema,
        model_config=md.get("profile") or {},
        split=split,
    )
    transform_items = md.get("transform_inputs")
    if transform_items is None:
        model_name = recipe.model_name if recipe is not None else "<unknown>"
        raise ValueError(
            "No transform pipeline configured. Add `transforms.inputs` to "
            f"vla_factory/config/model/{model_name}.yaml or "
            "`model.config.transforms.inputs` in the recipe."
        )
    return build_preprocessor(transform_items, ctx)


def _load_transform_imports(recipe: TrainRecipe) -> None:
    """Import user modules that register custom transform classes."""
    for module_name in recipe.transform_imports:
        importlib.import_module(module_name)


def _resolve_transform_inputs(recipe: TrainRecipe) -> tuple[list[dict] | None, dict]:
    """Read transform config from an already prepared recipe."""
    transform_inputs = (recipe.model_config.get("transforms") or {}).get("inputs")
    if transform_inputs is None:
        raise ValueError(
            "No transform pipeline configured. Add `transforms.inputs` to "
            f"vla_factory/config/model/{recipe.model_name}.yaml or "
            "`model.config.transforms.inputs` in the recipe."
        )
    return list(transform_inputs), recipe.model_config


def create_dataloaders(
    recipe: TrainRecipe,
    model_metadata: dict[str, Any] | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train and val DataLoaders from a ``TrainRecipe``.

    Parameters
    ----------
    recipe : TrainRecipe
        Prepared recipe. Training entrypoints should call ``resolve_recipe()``
        before passing authoring recipes here.
    model_metadata : dict, optional
        Model capabilities such as ``action_dim`` used to select transforms.

    Returns
    -------
    (train_loader, val_loader)
    """
    data_cfg = recipe.data
    sampler_cfg = data_cfg.sampler
    split_cfg = data_cfg.split
    path = Path(data_cfg.source.path)
    _load_transform_imports(recipe)

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
    transform_inputs, profile = _resolve_transform_inputs(recipe)
    transforms = _build_transforms(
        norm_stats=norm_stats,
        model_metadata={
            **(model_metadata or {}),
            "profile": profile,
            "transform_inputs": transform_inputs,
        },
        action_dim=schema.action_dim,
        recipe=recipe,
        schema=schema,
        split="train",
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
