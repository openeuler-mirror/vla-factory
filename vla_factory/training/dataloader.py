"""Execute a resolved assembly as the training DataLoader.

``create_dataloader()`` wires together:

1. ``FormatReader``   — read raw dataset (e.g. LeRobot v3)
2. ``VideoCodec``     — pluggable video decoding (default PyAV)
3. Transforms         — instantiated from the assembly's ``data_to_model`` plan
4. Sample windows     — temporal indexing over every episode
5. ``VLADataset``     — per-sample loading (numpy output)
6. ``DataLoader``     — batching & shuffling

The dataset description and the transform pipeline both come from the
``ResolvedAssembly``: this layer executes a resolved composition, it does not
re-derive one (architecture §4.2.6).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from vla_factory.assembly import ResolvedAssembly
from vla_factory.assembly.transform import TransformContext, build_pipeline
from vla_factory.data.codec import resolve_codec
from vla_factory.data.reader import get_reader
from vla_factory.user_interface import TrainRecipe
from vla_factory.training.dataset import (
    VLADataset,
    build_sample_windows,
    collate_fn,
)

logger = logging.getLogger(__name__)


def create_dataloader(
    recipe: TrainRecipe,
    assembly: ResolvedAssembly,
) -> torch.utils.data.DataLoader:
    """Build the training DataLoader from a resolved recipe and assembly.

    Parameters
    ----------
    recipe : TrainRecipe
        Prepared recipe (``merge_model_config()`` applied). Only execution config is
        read here — dataset location, batch size, workers.
    assembly : ResolvedAssembly
        The resolved data × model × robot composition: dataset description,
        statistics, the action horizon and the ``data_to_model`` plan.

    Training currently has no evaluation pass, so every episode contributes
    windows to this loader. A validation split should return together with an
    implemented metric/evaluation loop, rather than silently withholding data.
    """
    data_cfg = recipe.data
    path = Path(data_cfg.path)

    # 1. Reader + codec (frame access; the descriptions come from the assembly)
    reader = get_reader(data_cfg.format, path=path)
    codec = resolve_codec(data_cfg.video_codec, data_cfg.format)

    schema = assembly.schema
    norm_stats = assembly.norm_stats
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

    # 3. Sample windows. Both ends come from the model's temporal contract —
    #    how many frames it observes and how many actions it predicts — so a
    #    sample cannot be shaped differently from what the model consumes.
    windows = build_sample_windows(
        episode_lengths=episode_lengths,
        n_obs_steps=assembly.model_io_spec.n_obs_steps,
        action_horizon=assembly.model_io_spec.action_horizon,
    )

    logger.info(
        "Sample windows: %d train",
        len(windows),
    )

    # 4. Dataset
    dataset = VLADataset(windows, reader, codec, path, transforms)

    # 5. DataLoader
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=recipe.training.batch_size,
        shuffle=True,
        num_workers=recipe.training.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )
