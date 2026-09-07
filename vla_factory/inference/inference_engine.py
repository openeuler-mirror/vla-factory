"""Transport-agnostic checkpoint inference."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from vla_factory.assembly.transform import TransformContext, build_pipeline
from vla_factory.data.data_schema import resolve_vector_keys
from vla_factory.inference.checkpoint import (
    load_checkpoint_state_dict,
    checkpoint_format,
    load_inference_metadata,
    resolve_checkpoint_path,
    validate_delta_load,
)
from vla_factory.inference.execution import ActionChunk
from vla_factory.model.model_interface import Observation
from vla_factory.model.registry import get_entry
from vla_factory.training.dataset import stack_frame_samples
from vla_factory.training.strategies import get_strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObsDict:
    """Raw observation shape shared by platform adapters and inference."""

    video: dict[str, np.ndarray]
    state: np.ndarray | None = None
    # Dataset frames may carry several equivalent prompts for one episode
    # (e.g. RoboTwin `seen` instructions). Training samples one per step;
    # inference deterministically resolves the first at the boundary below.
    language: str | tuple[str, ...] | None = None


def resolve_inference_language(
    language: str | tuple[str, ...] | None,
) -> str | None:
    """Deterministically pick the prompt for the inference boundary.

    Training samples one prompt per step from multi-instruction episodes;
    inference and offline evaluation always use the first so results stay
    comparable and reproducible (and the container is never stringified).
    """
    if language is None:
        return None
    if isinstance(language, (tuple, list)):
        return language[0] if language else None
    return language


class InferenceEngine:
    """Load and execute the resolved interface saved with a checkpoint.

    The saved assembly supplies camera keys, vector widths, and both transform
    plans. Deployment never re-resolves relationships between data, model, and
    robot against the currently installed declarations.

    There is deliberately no camera-name override. A platform adapter must map
    native camera names to the checkpoint's DataSchema keys before calling
    :meth:`predict`.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        assembly, recipe = load_inference_metadata(checkpoint_path)
        io_spec = assembly.model_io_spec

        self.assembly = assembly
        self.recipe = recipe
        self.schema = assembly.schema
        self.norm_stats = assembly.norm_stats

        entry = get_entry(recipe.model.name)
        checkpoint_file = resolve_checkpoint_path(checkpoint_path)
        weight_format = checkpoint_format(checkpoint_file)
        is_delta = weight_format == "lora_delta"
        # A delta checkpoint must first construct its declared base model, so
        # its path is always required. Full checkpoints are self-contained —
        # except for adapters that reconstruct structure + processors from the
        # base checkpoint (OpenVLA: HF AutoModel/AutoProcessor), which declare
        # inference_needs_base_checkpoint and keep the path.
        if is_delta:
            if not recipe.model.path:
                raise ValueError(
                    "Delta checkpoint requires model.path in its saved recipe"
                )
        elif not entry.metadata.inference_needs_base_checkpoint:
            recipe = replace(recipe, model=replace(recipe.model, path=None))
        self.recipe = recipe

        self.state_keys, self.action_keys = resolve_vector_keys(self.schema)
        if not io_spec.cameras:
            raise ValueError(
                f"The assembly saved with {checkpoint_path} declares no cameras; "
                "there is no observation contract to serve."
            )
        self.camera_keys = tuple(io_spec.cameras)

        assembly.check_model_compatibility(entry.metadata)
        model = entry.factory(recipe=recipe, assembly=assembly)
        # Trainer checkpoints retain strategy-owned wrappers so training can
        # resume. Recreate those wrappers before loading intermediate weights;
        if (
            recipe.finetuning.strategy == "lora"
            and (weight_format in {"lora_delta", "lora_wrapped_full"}
                 or (weight_format is None and checkpoint_file.parent.name.startswith("checkpoint-")))
        ):
            strategy = get_strategy(recipe.finetuning.strategy)
            strategy_config = strategy.parse_config(recipe.finetuning.config)
            model = strategy.prepare_model(
                model, strategy_config, entry.metadata
            )
        state_dict = load_checkpoint_state_dict(checkpoint_file)
        result = model.load_state_dict(state_dict, strict=not is_delta)
        if is_delta:
            validate_delta_load(model, result)
        model.to(self.device)
        model.eval()
        self._model = model

        # Padded models have distinct network-output and execution widths.
        self.action_horizon = io_spec.action_horizon
        self.model_output_dim = io_spec.action_dim
        self.execution_action_dim = self.schema.action_dim
        # Models with history_frames > 1 condition on a trailing window of
        # frames (diffusion_policy's To). Deployment sends one frame per
        # step, so the engine keeps the history itself; the time axis is
        # materialised only on this >1 branch (decision D8) and the
        # single-frame path stays shape-identical to before.
        self.n_obs_steps = max(1, int(io_spec.n_obs_steps))
        self._history: deque[dict[str, Any]] | None = (
            deque(maxlen=self.n_obs_steps) if self.n_obs_steps > 1 else None
        )
        self.num_inference_steps = int(
            (recipe.model.config or {}).get("num_inference_steps", 1)
        )

        transform_context = TransformContext(norm_stats=self.norm_stats)
        self.preprocessor = build_pipeline(
            assembly.robot_to_model, transform_context
        )
        self.postprocessor = build_pipeline(
            assembly.model_to_robot, transform_context
        )

        logger.info(
            "InferenceEngine ready: model=%s checkpoint=%s cameras=%s "
            "execution_action_dim=%d model_output_dim=%d "
            "action_horizon=%d n_obs_steps=%d inference_steps=%d device=%s",
            recipe.model.name,
            checkpoint_file,
            self.camera_keys,
            self.execution_action_dim,
            self.model_output_dim,
            self.action_horizon,
            self.n_obs_steps,
            self.num_inference_steps,
            self.device,
        )
        logger.info(
            "Resolved vector keys — state=%s action=%s",
            list(self.state_keys),
            list(self.action_keys),
        )

    def predict(self, observation: ObsDict) -> ActionChunk:
        """Run inference and return a strict ``[horizon, action_dim]`` chunk.

        For ``n_obs_steps > 1`` checkpoints the engine maintains the trailing
        observation window internally: consecutive calls step the history,
        and :meth:`reset` clears it at episode boundaries. Callers that own
        the frame sequence (offline evaluation, stepping far between
        predictions) should use :meth:`predict_window` instead.
        """
        return self._predict_chunk(observation)

    def predict_window(self, observations: Sequence[ObsDict]) -> ActionChunk:
        """Run inference on an explicitly assembled observation window.

        The caller supplies exactly ``n_obs_steps`` observations, oldest
        first, already aligned to the prediction position (with the episode
        start front-filled by frame repetition). This bypasses — and does not
        update — the internal history: an evaluation stepping by
        ``action_horizon`` between predictions must not leave strides of
        unseen frames in the window.
        """
        if len(observations) != self.n_obs_steps:
            raise ValueError(
                f"predict_window needs exactly n_obs_steps={self.n_obs_steps} "
                f"observations, got {len(observations)}."
            )
        if self._history is None:
            model_observation = self._single_observation(observations[0])
            return self._run_model(model_observation, observations[0])
        window = [self._transform_frame(obs) for obs in observations]
        model_observation = self._stacked_observation(window, observations[-1])
        return self._run_model(model_observation, observations[-1])

    def reset(self) -> None:
        """Reset model-side inference state.

        Chunk playback state belongs to the separate execution policy. This
        clears the observation history so a new episode's trailing window is
        not seeded with the previous episode's frames.
        """
        if self._history is not None:
            self._history.clear()

    def _obs_to_observation(self, observation: ObsDict) -> Observation:
        """Apply the saved forward pipeline and construct a model observation."""
        if self._history is None:
            return self._single_observation(observation)

        # Multi-frame deployment path: transform the incoming frame once,
        # let the history assemble the trailing window.
        self._history.append(self._transform_frame(observation))
        window = list(self._history)
        if len(window) < self.n_obs_steps:
            # Episode start: repeat the oldest frame. Training never sees a
            # partial window (build_episode_windows starts at full ones), so
            # this fill convention is what keeps step 0 well-defined — the
            # same first-frame repeat diffusion_policy's eval loops use.
            window = [window[0]] * (self.n_obs_steps - len(window)) + window
        return self._stacked_observation(window, observation)

    def _validated_frame_sample(self, observation: ObsDict) -> dict[str, Any]:
        """Validate one raw observation against the schema → flat frame dict."""
        missing_cameras = [
            key for key in self.camera_keys if key not in observation.video
        ]
        if missing_cameras:
            raise ValueError(
                "Observation does not satisfy the checkpoint DataSchema: "
                f"missing cameras {missing_cameras}; available cameras are "
                f"{sorted(observation.video)}. PlatformAdapter must emit "
                "DataSchema keys."
            )

        expected_state_dim = self.schema.state_dim
        if expected_state_dim and observation.state is None:
            raise ValueError(
                "Observation does not satisfy the checkpoint DataSchema: "
                f"state is required with width {expected_state_dim}."
            )
        if observation.state is not None:
            state = np.asarray(observation.state)
            if state.shape != (expected_state_dim,):
                raise ValueError(
                    "Observation does not satisfy the checkpoint DataSchema: "
                    f"expected state shape ({expected_state_dim},), got "
                    f"{state.shape}."
                )

        sample: dict[str, Any] = {
            f"images.{camera}": np.ascontiguousarray(observation.video[camera])
            for camera in self.camera_keys
        }
        if observation.state is not None:
            sample["state"] = observation.state.astype(np.float32)
        return sample

    def _transform_frame(self, observation: ObsDict) -> dict[str, Any]:
        """Forward-pipeline one frame. Window-level transform branches no-op
        on this frame-only dict; the language half runs in its own pass."""
        return self.preprocessor(self._validated_frame_sample(observation))

    def _single_observation(self, observation: ObsDict) -> Observation:
        """Single-frame path (``n_obs_steps == 1``) — the original contract."""
        sample = self._validated_frame_sample(observation)
        task = resolve_inference_language(observation.language)
        if task is not None:
            sample["task"] = task

        transformed = self.preprocessor(sample)
        images: dict[str, torch.Tensor] = {}
        image_masks: dict[str, torch.Tensor] = {}
        for camera in self.camera_keys:
            array = transformed[f"images.{camera}"]
            images[camera] = (
                torch.as_tensor(np.ascontiguousarray(array))
                .unsqueeze(0)
                .to(self.device)
            )
            image_masks[camera] = torch.ones(
                (1,), dtype=torch.bool, device=self.device
            )

        state_tensor = self._optional_tensor(transformed.get("state"))
        prompt_tensor = self._optional_tensor(transformed.get("tokenized_prompt"))
        prompt_mask_tensor = self._optional_tensor(
            transformed.get("tokenized_prompt_mask")
        )

        task = transformed.get("task")
        return Observation(
            images=images,
            image_masks=image_masks,
            state=state_tensor,
            task=[task] if task is not None else None,
            tokenized_prompt=prompt_tensor,
            tokenized_prompt_mask=prompt_mask_tensor,
            pixel_values=self._optional_tensor(transformed.get("pixel_values")),
        )

    def _stacked_observation(
        self, window: list[dict[str, Any]], current: ObsDict,
    ) -> Observation:
        """Build the time-stacked observation from transformed frames.

        Mirrors the training side exactly: per-frame transforms, then the
        time axis. The prompt is window-level — resolved from the current
        frame's language and run through the pipeline in its own pass,
        together with the current (last) frame's state, which
        ``task_tokenize``'s discrete_state mode reads.
        """
        stacked = stack_frame_samples(window)
        images: dict[str, torch.Tensor] = {}
        for camera in self.camera_keys:
            array = stacked[f"images.{camera}"]
            images[camera] = (
                torch.as_tensor(np.ascontiguousarray(array))
                .unsqueeze(0)
                .to(self.device)
            )
        image_masks = {
            camera: torch.ones(
                (1, self.n_obs_steps), dtype=torch.bool, device=self.device
            )
            for camera in self.camera_keys
        }

        state_tensor = self._optional_tensor(stacked.get("state"))

        language_sample: dict[str, Any] = {}
        task = resolve_inference_language(current.language)
        if task is not None:
            language_sample["task"] = task
        if stacked.get("state") is not None:
            language_sample["state"] = stacked["state"][-1]
        language = self.preprocessor(language_sample)
        prompt_tensor = self._optional_tensor(language.get("tokenized_prompt"))
        prompt_mask_tensor = self._optional_tensor(
            language.get("tokenized_prompt_mask")
        )

        return Observation(
            images=images,
            image_masks=image_masks,
            state=state_tensor,
            tokenized_prompt=prompt_tensor,
            tokenized_prompt_mask=prompt_mask_tensor,
        )

    def _optional_tensor(self, value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        return (
            torch.as_tensor(np.ascontiguousarray(value))
            .unsqueeze(0)
            .to(self.device)
        )

    @torch.inference_mode()
    def _predict_chunk(self, observation: ObsDict) -> ActionChunk:
        model_observation = self._obs_to_observation(observation)
        return self._run_model(model_observation, observation)

    @torch.inference_mode()
    def _run_model(
        self, model_observation: Observation, observation: ObsDict,
    ) -> ActionChunk:
        """Model call + planned postprocess — shared by every entry point."""
        actions = self._model.predict_actions(
            model_observation,
            num_steps=self.num_inference_steps,
        )

        if isinstance(actions, torch.Tensor):
            actions_array = actions.detach().cpu().numpy()
        else:
            actions_array = np.asarray(actions)
        if actions_array.ndim == 3:
            actions_array = actions_array[0]
        elif actions_array.ndim == 1:
            actions_array = actions_array[None, :]

        model_shape = (self.action_horizon, self.model_output_dim)
        if actions_array.shape != model_shape:
            raise ValueError(
                "Model output does not match the resolved IO spec: expected "
                f"{model_shape}, got {actions_array.shape}."
            )

        postprocess_sample: dict[str, Any] = {"actions": actions_array}
        if observation.state is not None:
            postprocess_sample["state"] = observation.state.astype(np.float32)
        postprocess_sample = self.postprocessor(postprocess_sample)

        chunk = ActionChunk(postprocess_sample["actions"])
        execution_shape = (
            self.action_horizon,
            self.execution_action_dim,
        )
        if chunk.values.shape != execution_shape:
            raise ValueError(
                "Post-processed action chunk does not match the planned command "
                f"space: expected {execution_shape}, got {chunk.values.shape}."
            )
        return chunk


__all__ = ["InferenceEngine", "ObsDict"]
