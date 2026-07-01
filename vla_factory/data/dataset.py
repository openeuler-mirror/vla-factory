"""VLA training dataset — converts canonical IR to numpy training samples.

Each sample is a **flat dict of numpy arrays** so that PyTorch's
``default_collate`` can automatically stack them into tensors::

    {
        "images.front":       ndarray[C, H, W],   # float32 [0,1]
        "images.wrist":       ndarray[C, H, W],
        "image_masks.front":  ndarray[C],          # bool, all True
        "image_masks.wrist":  ndarray[C],
        "state":              ndarray[state_dim],  # float32
        "actions":            ndarray[horizon, dim],
        "action_is_pad":      ndarray[horizon],    # bool, True for padded steps
    }

Use :func:`collate_fn` to assemble a batch back into
``{"observation": Observation, "actions": Tensor, "action_is_pad": Tensor}``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla_factory.model.protocols.observation import Observation

from .formats.base import FormatReader, Frame
from .codec.base import VideoCodec
from .manifest import DatasetManifest, SampleLocator
from .transforms.pipeline import TransformPipeline

logger = logging.getLogger(__name__)


class VLADataset(torch.utils.data.Dataset):
    """PyTorch Dataset that yields VLA training samples as numpy arrays.

    Internally uses a ``FormatReader`` + ``VideoCodec`` to load raw
    episodes as canonical ``Episode`` / ``Frame`` objects, then converts
    each ``SampleLocator`` into a flat dict of numpy arrays.

    Parameters
    ----------
    manifest : DatasetManifest
        Complete sample index with locators, schema, and norm stats.
    reader : FormatReader
        Dataset format reader (e.g. LeRobotV3Reader).
    codec : VideoCodec
        Video decoding strategy (e.g. PyAVCodec).
    path : Path
        Root directory of the dataset.
    transforms : TransformPipeline | list[TransformStep] | None
        Sample-level transforms (normalise, resize, pad, etc.).  A plain list is
        wrapped in a :class:`TransformPipeline`.
    split : str
        One of ``"train"`` or ``"val"``.
    """

    def __init__(
        self,
        manifest: DatasetManifest,
        reader: FormatReader,
        codec: VideoCodec,
        path: Path,
        transforms: TransformPipeline | list | None = None,
        split: str = "train",
    ) -> None:
        self.manifest = manifest
        self.reader = reader
        self.codec = codec
        self.path = path
        # Normalise to a TransformPipeline so __getitem__ has a single call site.
        if transforms is None:
            self.transforms = TransformPipeline()
        elif isinstance(transforms, TransformPipeline):
            self.transforms = transforms
        else:
            self.transforms = TransformPipeline(list(transforms))
        self.split = split

        # Select the appropriate locator subset
        if split == "train":
            self._indices = manifest._train_indices
        elif split == "val":
            self._indices = manifest._val_indices
        else:
            raise ValueError(f"Unknown split: {split!r}")

        # Episode frame cache: ep_index -> list[Frame]
        self._episode_cache: dict[int, list[Frame]] = {}

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx = self._indices[idx]
        locator = self.manifest.locators[real_idx]
        return self._load_sample(locator)

    # ── Internal ──────────────────────────────────────────────────

    def _get_episode_frames(self, ep_idx: int) -> list[Frame]:
        """Load and cache frames for an episode."""
        if ep_idx not in self._episode_cache:
            episode = self.reader.read_episode(self.path, ep_idx, self.codec)
            frames = episode.load_frames()
            self._episode_cache[ep_idx] = frames
            # Cache eviction: keep at most 64 episodes
            if len(self._episode_cache) > 64:
                oldest = next(iter(self._episode_cache))
                del self._episode_cache[oldest]
        return self._episode_cache[ep_idx]

    def _load_sample(self, loc: SampleLocator) -> dict[str, Any]:
        """Convert a ``SampleLocator`` into a flat dict of numpy arrays."""
        frames = self._get_episode_frames(loc.episode_index)

        # ── Observation frames ────────────────────────────────────
        obs_start = loc.start_frame_index
        obs_end = min(obs_start + loc.n_obs_steps, len(frames))

        # Use the last observation frame for images/state
        obs_frame = frames[min(obs_end - 1, len(frames) - 1)]

        sample: dict[str, Any] = {}

        # Decode images via VideoCodec: VideoRef → numpy HWC uint8 → CHW float32 [0,1]
        for cam_name, ref in obs_frame.images.items():
            img = self.codec.decode_frame(ref)  # numpy HWC uint8
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)  # HWC → CHW
            sample[f"images.{cam_name}"] = img
            sample[f"image_masks.{cam_name}"] = np.ones(img.shape[0], dtype=bool)

        # State vector
        if obs_frame.state is not None:
            sample["state"] = obs_frame.state.astype(np.float32)
        else:
            sample["state"] = None

        # ── Action frames ─────────────────────────────────────────
        # Action chunk starts from the last observation frame (delta=0),
        # matching lerobot's convention: the first predicted action is the
        # action to take at the current timestep.
        action_start = obs_end - 1
        action_end = action_start + loc.action_horizon

        actions_list: list[np.ndarray] = []
        is_pad: list[bool] = []
        for i in range(action_start, action_end):
            if i < len(frames) and frames[i].action is not None:
                actions_list.append(frames[i].action)
                is_pad.append(False)
            else:
                # Repeat-last padding — must have at least one valid action
                last_valid = None
                for f in reversed(frames):
                    if f.action is not None:
                        last_valid = f.action
                        break
                if last_valid is None:
                    raise ValueError(
                        f"All actions in episode {loc.episode_index} are None; "
                        "cannot pad action horizon without at least one valid action."
                    )
                actions_list.append(last_valid)
                is_pad.append(True)

        if actions_list:
            sample["actions"] = np.stack(actions_list, axis=0).astype(np.float32)
        else:
            sample["actions"] = np.zeros((loc.action_horizon, 0), dtype=np.float32)

        sample["action_is_pad"] = np.array(is_pad, dtype=bool)

        # Apply the transform pipeline (state, actions, images.* keys).
        sample = self.transforms(sample)

        return sample


# ── Collation helper ──────────────────────────────────────────────


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate that produces ``{"observation": Observation, "actions": Tensor, "action_is_pad": Tensor}``.

    Takes a list of flat numpy dict from ``VLADataset.__getitem__``
    and stacks them into a batched format.  PyTorch's ``torch.stack``
    automatically converts numpy arrays to tensors.
    """
    # Collect all keys
    keys = set()
    for item in batch:
        keys.update(item.keys())

    stacked: dict[str, torch.Tensor | None] = {}
    for key in keys:
        values = [item[key] for item in batch if item.get(key) is not None]
        if values:
            stacked[key] = torch.stack(
                [torch.as_tensor(v) for v in values]
            )
        else:
            stacked[key] = None

    # Separate images / masks / state
    images: dict[str, torch.Tensor] = {}
    image_masks: dict[str, torch.Tensor] = {}
    for key, val in stacked.items():
        if val is None:
            continue
        if key.startswith("images."):
            cam = key[len("images."):]
            images[cam] = val
        elif key.startswith("image_masks."):
            cam = key[len("image_masks."):]
            image_masks[cam] = val

    observation = Observation(
        images=images,
        image_masks=image_masks,
        state=stacked.get("state"),
    )

    return {
        "observation": observation,
        "actions": stacked["actions"],
        "action_is_pad": stacked["action_is_pad"],
    }
