"""Inference helpers for VLA Factory models."""

from __future__ import annotations

import json
import logging
import importlib
import re
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla_factory.recipe.parser import parse_recipe
from vla_factory.recipe.recipe import TrainRecipe
from vla_factory.data.manifest import DataSchema, FeatureStats, NormStats, resolve_vector_keys
from vla_factory.assembly.transforms import build_transforms, TransformContext
from vla_factory.data.formats import get_reader
from vla_factory.data.codec import resolve_codec
from vla_factory.model.interfaces.observation import Observation
from vla_factory.utils.constants import (
    INFERENCE_META_DIR, SCHEMA_FILE, NORM_STATS_FILE, RECIPE_FILE,
    FINAL_DIR, MODEL_WEIGHTS_FILE,
)
from vla_factory.model.registry import get_entry
logger = logging.getLogger(__name__)


def _validated_actions(value: Any, *, name: str) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim != 2 or 0 in actions.shape:
        raise ValueError(f"{name} must be a non-empty [steps, action_dim] array; got {actions.shape}.")
    if not np.isfinite(actions).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return np.ascontiguousarray(actions)


@dataclass(frozen=True)
class ActionChunk:
    values: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _validated_actions(self.values, name="ActionChunk"))

    @property
    def horizon(self) -> int:
        return self.values.shape[0]

    @property
    def action_dim(self) -> int:
        return self.values.shape[1]


@dataclass(frozen=True)
class ActionCommand:
    values: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _validated_actions(self.values, name="ActionCommand"))

    @property
    def num_steps(self) -> int:
        return self.values.shape[0]

    def single(self) -> np.ndarray:
        if self.num_steps != 1:
            raise ValueError(f"A single-step command was required, got {self.num_steps} steps.")
        return self.values[0]


class ExecutionStrategy(str, Enum):
    SYNCHRONOUS = "synchronous"
    TEMPORAL_ENSEMBLING = "temporal_ensembling"
    RECEDING_HORIZON = "receding_horizon"


class ExecutionPolicy:
    def __init__(self, strategy: ExecutionStrategy, action_horizon: int, action_dim: int, n_action_steps: int) -> None:
        if action_horizon < 1 or action_dim < 1:
            raise ValueError("action_horizon and action_dim must be positive")
        self.strategy, self.action_horizon = strategy, action_horizon
        self.action_dim, self.n_action_steps = action_dim, n_action_steps
        self._chunks: deque[np.ndarray] = deque()
        self._actions: deque[np.ndarray] = deque()

    @property
    def needs_chunk(self) -> bool:
        return self.strategy != ExecutionStrategy.RECEDING_HORIZON or not self._actions

    def _require(self, chunk: ActionChunk | None) -> np.ndarray:
        if chunk is None:
            raise ValueError("This execution step requires an ActionChunk.")
        expected = (self.action_horizon, self.action_dim)
        if chunk.values.shape != expected:
            raise ValueError(f"ActionChunk shape mismatch: expected {expected}, got {chunk.values.shape}.")
        return chunk.values

    def consume(self, chunk: ActionChunk | None) -> ActionCommand:
        if self.strategy == ExecutionStrategy.SYNCHRONOUS:
            return ActionCommand(self._require(chunk)[: self.n_action_steps])
        if self.strategy == ExecutionStrategy.TEMPORAL_ENSEMBLING:
            actions = self._require(chunk)
            self._chunks.append(actions)
            if len(self._chunks) > self.action_horizon:
                self._chunks.popleft()
            count = len(self._chunks)
            values = [self._chunks[i][count - 1 - i] for i in range(count)]
            weights = np.array([1.0 / (count - i) for i in range(count)])
            return ActionCommand(np.average(values, weights=weights, axis=0)[None, :])
        if self._actions:
            if chunk is not None:
                raise ValueError("A new chunk cannot be supplied during playback.")
        else:
            self._actions.extend(self._require(chunk)[: self.n_action_steps])
        return ActionCommand(self._actions.popleft()[None, :])

    def reset(self) -> None:
        self._chunks.clear()
        self._actions.clear()


def build_execution_policy(strategy: ExecutionStrategy | str, *, action_horizon: int, action_dim: int, n_action_steps: int | None = None) -> ExecutionPolicy:
    try:
        strategy = ExecutionStrategy(strategy)
    except ValueError as exc:
        raise ValueError(f"Unknown execution strategy {strategy!r}.") from exc
    steps = action_horizon if n_action_steps is None else n_action_steps
    if not 1 <= steps <= action_horizon:
        raise ValueError(f"n_action_steps must satisfy 1 <= n_action_steps <= action_horizon; got {steps}.")
    if strategy == ExecutionStrategy.TEMPORAL_ENSEMBLING and n_action_steps not in (None, 1):
        raise ValueError("temporal_ensembling always emits one step; n_action_steps must be omitted or 1.")
    return ExecutionPolicy(strategy, action_horizon, action_dim, steps)


class PolicyExecutor:
    def __init__(self, engine: "InferenceEngine", execution_policy: ExecutionPolicy) -> None:
        self.engine, self.execution_policy = engine, execution_policy

    def predict(self, obs: Any) -> ActionCommand:
        chunk = self.engine.predict(obs) if self.execution_policy.needs_chunk else None
        return self.execution_policy.consume(chunk)

    def reset(self) -> None:
        self.engine.reset()
        self.execution_policy.reset()


class ReplayPolicy:
    """Replay recorded actions without running model inference.

    An executable-policy stand-in occupying the same slot as
    :class:`PolicyExecutor` (``predict → ActionCommand``, ``reset``), used to
    validate deployment integrations end to end without a real model.
    """

    def __init__(self, episode_data: list[dict]) -> None:
        self.data = episode_data
        self._index = 0

    def predict(self, obs: "ObsDict") -> ActionCommand:
        if self._index >= len(self.data):
            raise StopIteration("Episode replay exhausted")
        action = np.asarray(self.data[self._index]["action"], dtype=np.float32)
        self._index += 1
        if action.ndim == 1:
            action = action[None, :]
        return ActionCommand(action)

    def reset(self) -> None:
        self._index = 0


def infer_from_dataset_sample(
    config: str | Path | TrainRecipe,
    *,
    checkpoint: str | Path,
    dataset_index: int = 0,
    split: str = "train",
    device: str | None = None,
    output: str | Path | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a trained model and run one dataset sample through inference.

    Intended as a post-training smoke test.  All inference logic (model
    loading, forward preprocessor, reverse postprocessor) is delegated to
    :class:`InferenceEngine`, so the transform pipeline is the single source
    of truth.  This function only adds the dataset-aware layer: reading a raw
    frame via the format reader and converting it to an :class:`ObsDict`.

    Works with any registered dataset format (currently ``lerobot-v3``;
    extend via ``FormatReader``).
    """
    recipe = parse_recipe(config) if isinstance(config, (str, Path)) else config
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. InferenceEngine handles model + transforms ────────────────
    engine = InferenceEngine(
        checkpoint_path=checkpoint,
        device=device,
    )

    # ── 2. Load raw frame from dataset via format reader ─────────────
    data_path = Path(recipe.data.source.path)
    reader = get_reader(recipe.data.source.format, path=data_path)
    codec = resolve_codec(recipe.data.source.video_codec)

    episode_lengths = reader.get_episode_lengths(data_path)
    sorted_eps = sorted(episode_lengths.items())
    total_frames = sum(length for _, length in sorted_eps)
    if total_frames == 0:
        raise ValueError(f"Dataset has no frames: {data_path}")
    if dataset_index < 0 or dataset_index >= total_frames:
        raise IndexError(
            f"dataset_index={dataset_index} out of range "
            f"(total {total_frames} frames)"
        )

    ep_idx, frame_idx = _map_index_to_episode_frame(sorted_eps, dataset_index)
    episode = reader.read_episode(data_path, ep_idx, codec)
    frames = episode.load_frames()
    obs_frame = frames[frame_idx]

    # ── 3. Raw frame → ObsDict ───────────────────────────────────────
    video: dict[str, np.ndarray] = {}
    for cam_name in engine.camera_keys:
        ref = obs_frame.images.get(cam_name)
        if ref is None:
            raise KeyError(
                f"Camera '{cam_name}' not found in dataset frame. "
                f"Available: {list(obs_frame.images.keys())}"
            )
        video[cam_name] = codec.decode_frame(ref)  # HWC uint8

    state = obs_frame.state.astype(np.float32) if obs_frame.state is not None else None
    obs = ObsDict(video=video, state=state, language=obs_frame.language)

    # ── 4. Ground-truth actions (raw / dataset scale) ────────────────
    action_horizon = recipe.action_spec.action_horizon
    gt_list: list[np.ndarray] = []
    for i in range(action_horizon):
        fi = frame_idx + i
        if fi < len(frames) and frames[fi].action is not None:
            gt_list.append(frames[fi].action.astype(np.float32))
        elif gt_list:
            gt_list.append(gt_list[-1])  # repeat-last padding
        else:
            gt_list.append(np.zeros(1, dtype=np.float32))
    gt_raw = np.stack(gt_list, axis=0)

    # ── 5. Predict via InferenceEngine (normalize → model → unnormalize)
    actions_raw = engine.predict(obs).values  # [H, D], raw scale

    # ── 6. Optionally save ───────────────────────────────────────────
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            actions_raw=actions_raw,
            target_actions_raw=gt_raw,
        )
    else:
        output_path = None

    return {
        "model_name": engine.recipe.model_name,
        "checkpoint": str(checkpoint),
        "device": device,
        "backend": getattr(engine._model, "_backend", type(engine._model).__name__),
        "split": split,
        "dataset_index": dataset_index,
        "episode_index": ep_idx,
        "frame_index": frame_idx,
        "action_shape": tuple(actions_raw.shape),
        "raw_action_shape": tuple(actions_raw.shape),
        "target_shape": tuple(gt_raw.shape),
        "first_action_raw": actions_raw[0].tolist() if actions_raw.ndim == 2 else actions_raw.tolist(),
        "first_target_action_raw": gt_raw[0].tolist(),
        "output": str(output_path) if output_path is not None else None,
    }


def _load_saved_metadata(checkpoint_path: Path) -> tuple[DataSchema | None, NormStats | None, TrainRecipe | None]:
    """Try loading recipe, schema, and norm_stats from checkpoint's inference_metadata/.

    Training saves these at start time, so they're available for any
    intermediate checkpoint.

    Returns (schema, norm_stats, recipe). Any may be None if not found.
    """
    # Walk up from checkpoint to find inference_metadata/ directory
    search_dir = checkpoint_path
    for _ in range(3):  # max 3 levels up
        meta_dir = search_dir / INFERENCE_META_DIR
        schema_file = meta_dir / SCHEMA_FILE
        norm_file = meta_dir / NORM_STATS_FILE
        if schema_file.exists() and norm_file.exists():
            break
        search_dir = search_dir.parent
    else:
        return None, None, None

    schema = None
    norm_stats = None
    recipe = None

    try:
        with open(schema_file) as f:
            schema_d = json.load(f)
        schema = DataSchema(
            state_dim=schema_d.get("state_dim", 0),
            action_dim=schema_d.get("action_dim", 0),
            cameras=tuple(schema_d.get("cameras", ())),
            image_sizes={k: tuple(v) for k, v in schema_d.get("image_sizes", {}).items()},
            fps=schema_d.get("fps", 30),
            has_language=schema_d.get("has_language", False),
            total_episodes=schema_d.get("total_episodes", 0),
            total_frames=schema_d.get("total_frames", 0),
            robot_type=schema_d.get("robot_type", "unknown"),
            state_keys=tuple(schema_d.get("state_keys", ())),
            action_keys=tuple(schema_d.get("action_keys", ())),
        )
    except Exception as e:
        logger.warning('Failed to load schema from %s: %s', schema_file, e)

    try:
        with open(norm_file) as f:
            ns_d = json.load(f)
        images_raw = ns_d.get("images")
        images_stats = None
        if isinstance(images_raw, dict):
            images_stats = {
                k: _parse_feature_stats(v)
                for k, v in images_raw.items()
            }
        norm_stats = NormStats(
            state=_parse_feature_stats(ns_d.get("state")),
            action=_parse_feature_stats(ns_d.get("action")),
            images=images_stats,
            method=ns_d.get("method", "zscore"),
        )
    except Exception as e:
        logger.warning('Failed to load norm_stats from %s: %s', norm_file, e)

    try:
        recipe_file = meta_dir / RECIPE_FILE
        if recipe_file.exists():
            recipe = parse_recipe(recipe_file)
    except Exception as e:
        logger.warning('Failed to load recipe from %s: %s', recipe_file, e)

    return schema, norm_stats, recipe


def _parse_feature_stats(d: dict | None) -> FeatureStats | None:
    if d is None:
        return None
    return FeatureStats(
        mean=d.get("mean", []),
        std=d.get("std", []),
        min=d.get("min", []),
        max=d.get("max", []),
        q01=d.get("q01", []),
        q99=d.get("q99", []),
    )


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve a checkpoint dir/root/file to an actual weight file."""
    path = Path(path)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    candidates = [
        path / FINAL_DIR / MODEL_WEIGHTS_FILE,
        path / MODEL_WEIGHTS_FILE,
        path / "model.safetensors",
    ]
    for ckpt_dir in sorted(path.glob("checkpoint-*"), key=_checkpoint_sort_key, reverse=True):
        candidates.extend([
            ckpt_dir / MODEL_WEIGHTS_FILE,
            ckpt_dir / "model.safetensors",
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No model weights found under {path}. Expected {FINAL_DIR}/{MODEL_WEIGHTS_FILE}, "
        f"{MODEL_WEIGHTS_FILE}, model.safetensors, or checkpoint-*/model.safetensors."
    )


def load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a PyTorch or safetensors checkpoint into a state_dict."""
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "This checkpoint is model.safetensors, but safetensors is not installed. "
                "Install it in the vla_factory runtime with: pip install safetensors"
            ) from exc
        return load_file(str(path), device="cpu")

    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint did not contain a state_dict: {path}")
    return state


def _map_index_to_episode_frame(
    sorted_eps: list[tuple[int, int]],
    index: int,
) -> tuple[int, int]:
    """Map a flat frame index to ``(episode_index, frame_within_episode)``."""
    remaining = index
    for ep_idx, ep_len in sorted_eps:
        if remaining < ep_len:
            return ep_idx, remaining
        remaining -= ep_len
    raise IndexError(f"Frame index {index} out of range")


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    step = int(match.group(1)) if match else -1
    return step, path.name


# ═══════════════════════════════════════════════════════════════════
# Inference layer — ObsDict, InferenceEngine
# (Architecture doc §12.4)
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ObsDict:
    """Unified observation input format (§12.4).

    Uses nested dict structure (reference: GR00T) instead of flat keys.
    """

    video: dict[str, np.ndarray]   # {"front": [H,W,3] uint8, ...}
    state: np.ndarray | None = None  # [state_dim] float32
    language: str | None = None


class InferenceEngine:
    """Transport-agnostic inference core (§12.4).

    Loads model + metadata from a checkpoint directory, runs prediction
    with configurable action-chunking strategies.

    Parameters
    ----------
    checkpoint_path : str | Path
        Checkpoint root (must contain ``inference_metadata/``).
    device : str
        Torch device.  Default: auto-detect.
    camera_names : list[str] | None
        Override camera key ordering.  Default: read from saved schema.
    ``predict`` always returns a complete :class:`ActionChunk`. Deployment
    execution strategies are composed with this engine by :class:`PolicyExecutor`.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        camera_names: list[str] | None = None,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── 1. Load saved metadata ────────────────────────────────
        schema, norm_stats, recipe = _load_saved_metadata(checkpoint_path)
        if recipe is None:
            raise FileNotFoundError(
                f"No inference_metadata found under {checkpoint_path}. "
                "Train a model first to generate it."
            )
        if schema is None:
            raise ValueError(
                "Schema is required for model creation but not found in metadata."
            )

        self.recipe = recipe
        self.norm_stats = norm_stats
        self.schema = schema
        for module_name in recipe.transform_imports:
            importlib.import_module(module_name)

        # ── 1.1 Clear model_path for inference ─────────────────────
        # The factory would try to load recipe.model_path (e.g. a pretrained
        # checkpoint) before the target checkpoint is applied — this fails when
        # the original pretrained path is unavailable (e.g. after copying the
        # checkpoint to a different machine).  At inference time the full model
        # state is already contained in the checkpoint being loaded, so
        # model_path must be None to avoid a spurious file-not-found error.
        recipe.model_path = None

        # ── 1.5 Resolve canonical state/action key order ─────────
        # The dimension→key mapping is a data/model contract, never invented by
        # sorting. It was resolved from the dataset feature ``names`` at train
        # time and saved into the checkpoint's schema.json — the sole source
        # here; the training dataset is never re-read at inference time.
        # Missing or mismatched keys mean the checkpoint metadata is incomplete
        # and fail here before any platform adapter is constructed.
        self.state_keys, self.action_keys = resolve_vector_keys(self.schema)
        self.schema = replace(
            self.schema,
            state_keys=self.state_keys,
            action_keys=self.action_keys,
        )

        # Camera key ordering: explicit > schema > default "front"
        if camera_names:
            self.camera_keys = tuple(camera_names)
        elif schema.cameras:
            self.camera_keys = schema.cameras
        else:
            self.camera_keys = ("front",)

        # ── 3. Build & load model ─────────────────────────────────
        entry = get_entry(recipe.model_name)
        model = entry.factory(recipe=recipe, schema=schema)
        ckpt_file = resolve_checkpoint_path(checkpoint_path)
        state_dict = load_checkpoint_state_dict(ckpt_file)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        self._model = model
        self.action_horizon = recipe.action_spec.action_horizon
        self.action_dim = recipe.action_spec.action_dim
        # Flow-matching / diffusion heads (pi0) denoise over N steps at inference
        # (ModelMetadata.inference_num_steps, e.g. pi0=10). Plumbed into
        # predict_actions so the adapter doesn't hardcode the count and openpi
        # doesn't receive num_steps=None (which crashes its time-step loop).
        self.num_inference_steps = entry.metadata.inference_num_steps

        # ── 3.5 Forward + reverse transform pipelines ────────────
        # Built solely from the resolved recipe saved in inference_metadata.
        transform_inputs = (recipe.model_config.get("transforms") or {}).get("inputs")
        if transform_inputs is None:
            raise ValueError(
                f"Resolved recipe at {checkpoint_path} does not contain "
                "`model.config.transforms.inputs`; checkpoint metadata is incomplete."
            )
        tctx = TransformContext(
            norm_stats=norm_stats,
            model_action_dim=entry.metadata.action_dim or recipe.action_spec.action_dim,
            dataset_action_dim=schema.action_dim,
            recipe=recipe,
            schema=schema,
            model_config=recipe.model_config,
            split="infer",
        )
        self.preprocessor, self.postprocessor = build_transforms(
            transform_inputs, tctx
        )

        logger.info(
            "InferenceEngine ready: model=%s checkpoint=%s cameras=%s "
            "action_dim=%d action_horizon=%d device=%s",
            recipe.model_name, ckpt_file, self.camera_keys,
            self.action_dim, self.action_horizon,
            self.device,
        )
        logger.info(
            "Resolved vector keys — state=%s action=%s",
            list(self.state_keys), list(self.action_keys),
        )

    # ── Public API ────────────────────────────────────────────────

    def predict(self, obs: ObsDict) -> ActionChunk:
        """Run inference and always return a strict ``[H, D]`` action chunk."""
        return self._predict_chunk(obs)

    def reset(self) -> None:
        """Reset model-side inference state.

        Chunk playback state belongs to the separate execution policy.
        """
        return None

    # ── Observation conversion ─────────────────────────────────────

    def _obs_to_observation(self, obs: ObsDict) -> Observation:
        """Convert ObsDict → Observation via the forward preprocessor pipeline.

        Builds a flat numpy sample (raw HWC images, raw state) and runs
        ``self.preprocessor`` — the *same* transform pipeline training uses
        (Normalize / ResizeImages / ...). All normalisation lives in the
        transforms; there is no inline math here. The normalised numpy arrays
        are then assembled into a torch ``Observation`` for the model.

        Note: the sample intentionally has no ``"actions"`` key (actions are the
        model's *output*), so action-affecting pre-steps (PadDimensions, future
        DeltaActions) no-op on it.
        """
        sample: dict[str, Any] = {}
        for cam_name in self.camera_keys:
            img = obs.video[cam_name]
            sample[f"images.{cam_name}"] = np.ascontiguousarray(img)
        if obs.state is not None:
            sample["state"] = obs.state.astype(np.float32)
        # task_tokenize reads sample["task"] → tokenized_prompt(_mask).
        # ObsDict.language carries the frame's task text (lerobot reader fills
        # Frame.language); without it, task_tokenize falls back to default_task
        # or an empty prompt (with a one-time warning) — the prompt tensor is
        # never missing, but language conditioning degrades for models like pi0.
        if obs.language is not None:
            sample["task"] = obs.language

        normalized = self.preprocessor(sample)

        images: dict[str, torch.Tensor] = {}
        image_masks: dict[str, torch.Tensor] = {}
        for cam_name in self.camera_keys:
            arr = normalized[f"images.{cam_name}"]
            tensor = torch.as_tensor(np.ascontiguousarray(arr)).unsqueeze(0).to(self.device)
            images[cam_name] = tensor
            image_masks[cam_name] = torch.ones((), dtype=torch.bool, device=self.device).unsqueeze(0)

        state: torch.Tensor | None = None
        if normalized.get("state") is not None:
            state = (
                torch.as_tensor(np.ascontiguousarray(normalized["state"]))
                .unsqueeze(0)
                .to(self.device)
            )

        tokenized_prompt: torch.Tensor | None = None
        tokenized_prompt_mask: torch.Tensor | None = None
        if normalized.get("tokenized_prompt") is not None:
            tokenized_prompt = (
                torch.as_tensor(np.ascontiguousarray(normalized["tokenized_prompt"]))
                .unsqueeze(0)
                .to(self.device)
            )
        if normalized.get("tokenized_prompt_mask") is not None:
            tokenized_prompt_mask = (
                torch.as_tensor(np.ascontiguousarray(normalized["tokenized_prompt_mask"]))
                .unsqueeze(0)
                .to(self.device)
            )

        return Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )

    # ── Chunk prediction ───────────────────────────────────────────

    @torch.inference_mode()
    def _predict_chunk(self, obs: ObsDict) -> ActionChunk:
        """Core inference: forward(obs) → model → reverse(actions) → [H, D] array."""
        observation = self._obs_to_observation(obs)
        actions = self._model.predict_actions(
            observation, num_steps=self.num_inference_steps
        )  # [B, H, D] or [H, D]

        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions)
        if actions_np.ndim == 3:
            actions_np = actions_np[0]
        elif actions_np.ndim == 1:
            # Non-chunk policies are represented as a one-step chunk. A model
            # configured with an action horizon > 1 must still return a chunk.
            actions_np = actions_np[None, :]

        # Reverse: un-normalise (and, in future, delta→absolute) via the
        # postprocessor. Raw obs.state is threaded through for the future
        # AbsoluteActions reverse (absolute = delta + state_raw); ACT's
        # UnnormalizeAction only reads "actions".
        post_sample: dict[str, Any] = {"actions": actions_np}
        if obs.state is not None:
            post_sample["state"] = obs.state.astype(np.float32)
        post_sample = self.postprocessor(post_sample)

        chunk = ActionChunk(post_sample["actions"])
        expected_shape = (self.action_horizon, self.action_dim)
        if chunk.values.shape != expected_shape:
            raise ValueError(
                "Model action chunk shape does not match checkpoint metadata: "
                f"expected {expected_shape}, got {chunk.values.shape}."
            )
        return chunk
