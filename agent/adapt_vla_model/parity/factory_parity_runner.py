#!/usr/bin/env python3
"""Judge-owned Factory sample, capture, and comparison runner."""

from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path


def _seed(value: int) -> None:
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _codec_format(reader, path: Path, configured: str) -> str:
    """Choose an auto codec from the detected dataset, not the recipe placeholder."""
    if configured != "auto":
        return configured
    source_format = reader.get_schema(path).source_format
    return "robotwin" if source_format == "robotwin_hdf5" else source_format


def _sample(args: argparse.Namespace) -> None:
    from vla_factory.inference.checkpoint import load_inference_metadata
    from vla_factory.data.codec import resolve_codec
    from vla_factory.data.reader import get_reader
    from vla_factory.training.dataset import VLADataset, build_sample_windows
    from parity_io import write_sample

    _seed(args.seed)
    assembly, recipe = load_inference_metadata(args.bundle)
    recipe = replace(recipe, data=replace(recipe.data, path=args.dataset_path))
    reader = get_reader(recipe.data.format, path=Path(recipe.data.path))
    codec = resolve_codec(recipe.data.video_codec, _codec_format(reader, Path(recipe.data.path), recipe.data.format))
    lengths = reader.get_episode_lengths(Path(recipe.data.path))
    windows = build_sample_windows(
        episode_lengths=lengths, n_obs_steps=assembly.model_io_spec.n_obs_steps,
        action_horizon=assembly.model_io_spec.action_horizon,
    )
    if args.sample_index < 0 or args.sample_index >= len(windows):
        raise ValueError(f"sample index {args.sample_index} outside {len(windows)} windows")
    sample = VLADataset(windows, reader, codec, Path(recipe.data.path))[args.sample_index]
    state = sample["state"]
    if args.platform == "robotwin":
        cameras = {
            key.removeprefix("images."): {"rgb": value}
            for key, value in sample.items() if key.startswith("images.") and value is not None
        }
        observation = {
            "robotwin_observation": {
                "observation": cameras,
                "joint_action": {
                    "left_arm": state[:6], "left_gripper": state[6],
                    "right_arm": state[7:13], "right_gripper": state[13], "vector": state,
                },
                "pointcloud": [], "endpose": {},
            },
            "instruction": sample["task"],
            "step": 0,
        }
    elif args.platform == "lerobot":
        if len(args.state_key) != len(state):
            raise ValueError(
                f"LeRobot benchmark declares {len(args.state_key)} state keys for a {len(state)}-D state"
            )
        host = {
            key.removeprefix("images."): value
            for key, value in sample.items() if key.startswith("images.") and value is not None
        }
        host.update({key: float(value) for key, value in zip(args.state_key, state)})
        observation = {"lerobot_observation": host, "instruction": sample["task"], "step": 0}
    else:
        raise ValueError(f"unsupported benchmark platform: {args.platform}")
    observation["training"] = {"actions": sample["actions"], "action_is_pad": sample["action_is_pad"]}
    meta = {"dataset_path": args.dataset_path, "dataset_revision": args.dataset_revision,
            "sample_ids": [f"window:{args.sample_index}"],
            "seed": args.seed, "precision": args.precision, "mode": args.mode}
    write_sample(observation, meta, args.output)


def _capture(args: argparse.Namespace) -> None:
    from vla_factory.inference.inference_engine import InferenceEngine
    from vla_factory.inference.platforms.robotwin import RoboTwinAdapter
    from vla_factory.training.dataset import collate_fn
    from parity_io import read_sample, write_loss

    observation, meta = read_sample(args.sample)
    _seed(meta["seed"])
    engine = InferenceEngine(args.bundle)
    engine._model.train(meta["mode"] == "train")
    # Use Factory's production platform boundary to parse the native RoboTwin
    # request.  Do not duplicate its camera selection or qpos extraction in
    # this Judge runner.
    if args.platform == "robotwin":
        raw = RoboTwinAdapter(engine.camera_keys, engine.execution_action_dim)(observation)
    elif args.platform == "lerobot":
        from vla_factory.inference.platforms.lerobot import LerobotHostObsAdapter

        host = observation.get("lerobot_observation")
        if not isinstance(host, dict):
            raise ValueError("LeRobot parity sample has no lerobot_observation mapping")
        raw = LerobotHostObsAdapter(
            engine.camera_keys, engine.state_keys, engine.schema.state_dim,
        )(host, task=observation.get("instruction", ""))
    else:
        raise ValueError(f"unsupported benchmark platform: {args.platform}")
    sample = engine._validated_frame_sample(raw)
    if raw.language is not None:
        sample["task"] = raw.language
    sample["actions"] = observation["training"]["actions"]
    sample["action_is_pad"] = observation["training"]["action_is_pad"]
    transformed = engine.preprocessor(sample)
    inputs = collate_fn([transformed])
    observation = inputs["observation"]
    # Flow matching samples noise/time in compute_loss.  Construction and
    # preprocessing must not advance the stream shared with Reference.
    _seed(meta["seed"])
    loss, _ = engine._model.compute_loss(observation, inputs["actions"], inputs["action_is_pad"])
    write_loss(float(loss.detach().cpu()), args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--bundle", type=Path, required=True)
    sample.add_argument("--dataset-path", required=True)
    sample.add_argument("--dataset-revision", required=True)
    sample.add_argument("--sample-index", type=int, default=0)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--precision", default="fp32")
    sample.add_argument("--mode", choices=("eval", "train"), default="eval")
    sample.add_argument("--platform", choices=("robotwin", "lerobot"), required=True)
    sample.add_argument("--state-key", action="append", default=[])
    sample.add_argument("--output", type=Path, required=True)
    sample.set_defaults(func=_sample)
    capture = commands.add_parser("capture")
    capture.add_argument("--bundle", type=Path, required=True)
    capture.add_argument("--sample", type=Path, required=True)
    capture.add_argument("--platform", choices=("robotwin", "lerobot"), required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.set_defaults(func=_capture)
    compare = commands.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--factory", type=Path, required=True)
    compare.add_argument("--report", type=Path, required=True)
    compare.set_defaults(func=lambda args: __import__("parity_io").compare_loss(args.reference, args.factory, args.report))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
