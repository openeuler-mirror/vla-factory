"""Orchestration: recipe + resolved assembly -> train/val DataLoaders.

``create_dataloaders()`` is the main entry point.  It wires together:

1. ``FormatReader``   — read raw dataset (e.g. LeRobot v3)
2. ``VideoCodec``     — pluggable video decoding (default PyAV)
3. Transforms         — instantiated from the assembly's ``data_to_model`` plan
4. ``SlidingWindowSampler`` — episode -> sample indexing
5. ``DatasetManifest`` — train/val split + locator list
6. ``VLADataset``     — per-sample loading (numpy output)
7. ``DataLoader``     — batching & shuffling

The dataset description and the transform pipeline both come from the
``ResolvedAssembly``: this layer executes a resolved composition, it does not
re-derive one (architecture §4.2.6).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from vla_factory.assembly.resolver import ResolvedAssembly
from vla_factory.assembly.transforms import TransformContext, build_pipeline
from vla_factory.data.codec import resolve_codec
from vla_factory.data.formats import get_reader
from vla_factory.recipe.recipe import TrainRecipe
from vla_factory.training.dataset import VLADataset, collate_fn
from vla_factory.training.manifest import build_manifest

logger = logging.getLogger(__name__)


def create_dataloaders(
    recipe: TrainRecipe,
    assembly: ResolvedAssembly,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train and val DataLoaders from a recipe and its resolved assembly.

    Parameters
    ----------
    recipe : TrainRecipe
        Prepared recipe (``resolve_recipe()`` applied). Only execution config is
        read here — dataset location, batch size, workers.
    assembly : ResolvedAssembly
        The resolved data × model × robot composition: dataset description,
        statistics, the action horizon and the ``data_to_model`` plan.

    Returns
    -------
    (train_loader, val_loader)
    """
    data_cfg = recipe.data
    path = Path(data_cfg.path)

    # 1. Reader + codec (frame access; the descriptions come from the assembly)
    reader = get_reader(data_cfg.format, path=path)
    codec = resolve_codec(data_cfg.video_codec)

    schema = assembly.schema
    norm_stats = assembly.norm_stats
    episode_ranges = reader.get_episode_ranges(path)
    episode_lengths = reader.get_episode_lengths(path)

    logger.info(
        "Dataset loaded: %d episodes, %d frames, action_dim=%d, state_dim=%d",
        schema.total_episodes, schema.total_frames,
        schema.action_dim, schema.state_dim,
    )

    # 2. Transforms: the resolved plan, instantiated. The context carries only
    #    what a serialized call cannot — the dataset statistics. An unresolved
    #    plan is refused by build_pipeline; train() checks it earlier still, so
    #    the refusal lands before any output directory is touched.
    transforms = build_pipeline(
        assembly.data_to_model, TransformContext(norm_stats=norm_stats),
    )

    # 3. Manifest. Both window ends come from the model's temporal contract —
    #    how many frames it observes and how many actions it predicts — so a
    #    sample cannot be shaped differently from what the model consumes.
    manifest = build_manifest(
        schema=schema,
        norm_stats=norm_stats,
        episode_ranges=episode_ranges,
        episode_lengths=episode_lengths,
        n_obs_steps=assembly.model_io_spec.n_obs_steps,
        action_horizon=assembly.model_io_spec.action_horizon,
    )

    logger.info(
        "Manifest: %d train samples, %d val samples",
        len(manifest.train_indices),
        len(manifest.val_indices),
    )

    # 4. Datasets
    train_ds = VLADataset(manifest, reader, codec, path, transforms, split="train")
    val_ds = VLADataset(manifest, reader, codec, path, transforms, split="val")

    # 5. DataLoaders
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
