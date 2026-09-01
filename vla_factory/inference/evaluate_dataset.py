"""Offline dataset inference and evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla_factory.data.codec import resolve_codec
from vla_factory.data.reader import get_reader
from vla_factory.inference.inference_engine import InferenceEngine, ObsDict
from vla_factory.user_interface import TrainRecipe, parse_recipe


def infer_dataset_sample(
    config: str | Path | TrainRecipe,
    *,
    checkpoint: str | Path,
    dataset_index: int = 0,
    device: str | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Run one flattened dataset frame through a trained checkpoint."""
    recipe = parse_recipe(config) if isinstance(config, (str, Path)) else config
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = InferenceEngine(checkpoint_path=checkpoint, device=device)

    data_path = Path(recipe.data.path)
    reader = get_reader(recipe.data.format, path=data_path)
    codec = resolve_codec(recipe.data.video_codec, recipe.data.format)

    episode_lengths = reader.get_episode_lengths(data_path)
    sorted_episodes = sorted(episode_lengths.items())
    total_frames = sum(length for _, length in sorted_episodes)
    if total_frames == 0:
        raise ValueError(f"Dataset has no frames: {data_path}")
    if dataset_index < 0 or dataset_index >= total_frames:
        raise IndexError(
            f"dataset_index={dataset_index} out of range "
            f"(total {total_frames} frames)"
        )

    episode_index, frame_index = _map_index_to_episode_frame(
        sorted_episodes, dataset_index
    )
    episode = reader.read_episode(data_path, episode_index, codec)
    frames = episode.load_frames()
    observation_frame = frames[frame_index]
    observation = _frame_observation(observation_frame, engine.camera_keys, codec)

    target_actions = _target_chunk(
        frames,
        frame_index=frame_index,
        action_horizon=engine.action_horizon,
        action_dim=engine.execution_action_dim,
    )
    predicted_actions = engine.predict(observation).values

    output_path: Path | None = None
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            actions_raw=predicted_actions,
            target_actions_raw=target_actions,
        )

    return {
        "model_name": engine.recipe.model.name,
        "checkpoint": str(checkpoint),
        "device": device,
        "backend": getattr(
            engine._model, "_backend", type(engine._model).__name__
        ),
        "dataset_index": dataset_index,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "action_shape": tuple(predicted_actions.shape),
        "raw_action_shape": tuple(predicted_actions.shape),
        "target_shape": tuple(target_actions.shape),
        "first_action_raw": predicted_actions[0].tolist(),
        "first_target_action_raw": target_actions[0].tolist(),
        "output": str(output_path) if output_path is not None else None,
    }


def evaluate_dataset(
    dataset: str | Path,
    *,
    checkpoint: str | Path,
    episode_indices: list[int] | None = None,
    device: str | None = None,
    save_dir: str | Path | None = None,
    include_frame_metrics: bool = False,
) -> dict[str, Any]:
    """Evaluate checkpoint actions against recorded actions using mean L1."""
    data_path = Path(dataset)
    engine = InferenceEngine(checkpoint_path=checkpoint, device=device)
    reader = get_reader(engine.recipe.data.format, path=data_path)
    codec = resolve_codec(engine.recipe.data.video_codec, engine.recipe.data.format)
    episode_lengths = reader.get_episode_lengths(data_path)
    selected = episode_indices or sorted(episode_lengths)

    reports: list[dict[str, Any]] = []
    total_l1 = 0.0
    total_frames = 0
    output_dir = Path(save_dir) if save_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for episode_index in selected:
        if episode_index not in episode_lengths:
            continue
        episode_length = episode_lengths[episode_index]
        frames = reader.read_episode(data_path, episode_index, codec).load_frames()
        episode_l1 = 0.0
        evaluated_frames = 0
        frame_reports: list[dict[str, Any]] = []

        for frame_index in range(0, episode_length, engine.action_horizon):
            observation = _frame_observation(
                frames[frame_index], engine.camera_keys, codec
            )
            valid_length = min(
                frame_index + engine.action_horizon, episode_length
            ) - frame_index
            targets = [
                frames[frame_index + offset].action.astype(np.float32)
                for offset in range(valid_length)
                if frames[frame_index + offset].action is not None
            ]
            if not targets:
                continue

            target_actions = np.stack(targets, axis=0)
            predicted_actions = engine.predict(observation).values[
                : len(target_actions)
            ]
            frame_losses = np.abs(predicted_actions - target_actions).mean(axis=1)
            episode_l1 += float(frame_losses.sum())
            evaluated_frames += len(target_actions)

            if include_frame_metrics:
                frame_reports.extend(
                    {
                        "frame_index": frame_index + offset,
                        "target": target_actions[offset].tolist(),
                        "prediction": predicted_actions[offset].tolist(),
                        "l1": float(frame_losses[offset]),
                    }
                    for offset in range(len(target_actions))
                )

        average_l1 = (
            episode_l1 / evaluated_frames if evaluated_frames else 0.0
        )
        report = {
            "episode_index": episode_index,
            "episode_length": episode_length,
            "total_l1": episode_l1,
            "num_frames": evaluated_frames,
            "average_l1": average_l1,
        }
        if include_frame_metrics:
            report["frames"] = frame_reports
        reports.append(report)
        total_l1 += episode_l1
        total_frames += evaluated_frames

        if output_dir is not None:
            np.savez(
                output_dir / f"episode_{episode_index}.npz",
                episode_index=episode_index,
                episode_length=episode_length,
                total_l1=episode_l1,
                num_frames=evaluated_frames,
                avg_l1=average_l1,
            )

    return {
        "action_horizon": engine.action_horizon,
        "episodes": reports,
        "total_l1": total_l1,
        "num_frames": total_frames,
        "average_l1": total_l1 / total_frames if total_frames else 0.0,
    }


def _frame_observation(frame, camera_keys: tuple[str, ...], codec) -> ObsDict:
    video: dict[str, np.ndarray] = {}
    for camera in camera_keys:
        reference = frame.images.get(camera)
        if reference is None:
            raise KeyError(
                f"Camera '{camera}' not found in dataset frame. "
                f"Available: {list(frame.images)}"
            )
        video[camera] = codec.decode_frame(reference)
    state = frame.state.astype(np.float32) if frame.state is not None else None
    return ObsDict(video=video, state=state, language=frame.language)


def _target_chunk(
    frames,
    *,
    frame_index: int,
    action_horizon: int,
    action_dim: int,
) -> np.ndarray:
    actions: list[np.ndarray] = []
    for offset in range(action_horizon):
        index = frame_index + offset
        if index < len(frames) and frames[index].action is not None:
            actions.append(frames[index].action.astype(np.float32))
        elif actions:
            actions.append(actions[-1])
        else:
            actions.append(np.zeros(action_dim, dtype=np.float32))
    return np.stack(actions, axis=0)


def _map_index_to_episode_frame(
    sorted_episodes: list[tuple[int, int]],
    index: int,
) -> tuple[int, int]:
    remaining = index
    for episode_index, episode_length in sorted_episodes:
        if remaining < episode_length:
            return episode_index, remaining
        remaining -= episode_length
    raise IndexError(f"Frame index {index} out of range")


__all__ = ["evaluate_dataset", "infer_dataset_sample"]
