"""Transport-agnostic checkpoint inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla_factory.assembly.transform import TransformContext, build_pipeline
from vla_factory.data.data_schema import resolve_vector_keys
from vla_factory.inference.checkpoint import (
    load_checkpoint_state_dict,
    load_inference_metadata,
    resolve_checkpoint_path,
)
from vla_factory.inference.execution import ActionChunk
from vla_factory.model.model_interface import Observation
from vla_factory.model.registry import get_entry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObsDict:
    """Raw observation shape shared by platform adapters and inference."""

    video: dict[str, np.ndarray]
    state: np.ndarray | None = None
    language: str | None = None


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

        # At inference time the target checkpoint contains the complete model
        # state. Loading recipe.model.path first would unnecessarily depend on
        # the original pretrained checkpoint still being available.
        recipe = replace(recipe, model=replace(recipe.model, path=None))
        self.recipe = recipe

        self.state_keys, self.action_keys = resolve_vector_keys(self.schema)
        if not io_spec.cameras:
            raise ValueError(
                f"The assembly saved with {checkpoint_path} declares no cameras; "
                "there is no observation contract to serve."
            )
        self.camera_keys = tuple(io_spec.cameras)

        entry = get_entry(recipe.model.name)
        assembly.check_model_compatibility(entry.metadata)
        model = entry.factory(recipe=recipe, assembly=assembly)
        checkpoint_file = resolve_checkpoint_path(checkpoint_path)
        state_dict = load_checkpoint_state_dict(checkpoint_file)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        self._model = model

        # Padded models have distinct network-output and execution widths.
        self.action_horizon = io_spec.action_horizon
        self.model_output_dim = io_spec.action_dim
        self.execution_action_dim = self.schema.action_dim
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
            "action_horizon=%d inference_steps=%d device=%s",
            recipe.model.name,
            checkpoint_file,
            self.camera_keys,
            self.execution_action_dim,
            self.model_output_dim,
            self.action_horizon,
            self.num_inference_steps,
            self.device,
        )
        logger.info(
            "Resolved vector keys — state=%s action=%s",
            list(self.state_keys),
            list(self.action_keys),
        )

    def predict(self, observation: ObsDict) -> ActionChunk:
        """Run inference and return a strict ``[horizon, action_dim]`` chunk."""
        return self._predict_chunk(observation)

    def reset(self) -> None:
        """Reset model-side inference state.

        Chunk playback state belongs to the separate execution policy.
        """
        return None

    def _obs_to_observation(self, observation: ObsDict) -> Observation:
        """Apply the saved forward pipeline and construct a model observation."""
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
        if observation.language is not None:
            sample["task"] = observation.language

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
