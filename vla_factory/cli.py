"""VLA Factory CLI.

Usage::

    vlafactory-cli train --config recipe.yaml          # installed console script
    python -m vla_factory train --config recipe.yaml   # without install / from source
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="vlafactory-cli",
        description="VLA Factory: fine-tune robot models.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── train ──
    train_parser = subparsers.add_parser("train", help="Train a model from a YAML recipe.")
    train_parser.add_argument("--config", required=True, help="Path to YAML recipe file.")
    train_parser.add_argument("--steps", type=int, default=None, help="Override total_steps.")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size.")
    train_parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir.")

    # ── preprocess ──
    preproc_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess dataset videos to .npy disk cache.",
    )
    preproc_parser.add_argument("--config", required=True, help="Path to YAML recipe file.")

    # ── list ──
    list_parser = subparsers.add_parser(
        "list",
        help="List registered models, or describe one recipe with --config.",
    )
    list_parser.add_argument(
        "--config", default=None,
        help="Path to a YAML recipe. If set, print that recipe's model contract "
             "(base cameras, action_dim) + camera_mapping check, instead of listing all models.",
    )

    # ── evaluate ──
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate checkpoint on a dataset (L1 loss per episode).",
    )
    eval_parser.add_argument(
        "--checkpoint", required=True,
        help="Checkpoint root (must have inference_metadata/).",
    )
    eval_parser.add_argument(
        "--dataset", required=True,
        help="Path to the dataset.",
    )
    eval_parser.add_argument(
        "--episodes", type=int, nargs="*", default=None,
        help="Episode indices to evaluate. Default: all.",
    )
    eval_parser.add_argument("--device", default=None, help="Torch device.")
    eval_parser.add_argument(
        "--save-dir", default=None,
        help="Save per-episode results as .npz.",
    )
    eval_parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-frame predicted vs ground-truth actions.",
    )

    # ── infer ──
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference on a dataset sample (smoke test).",
    )
    infer_parser.add_argument(
        "--config",
        help="Path to YAML recipe file. If omitted, reads from checkpoint's inference_metadata/.",
    )
    infer_parser.add_argument(
        "--checkpoint", required=True,
        help="Checkpoint root (must have inference_metadata/).",
    )
    infer_parser.add_argument(
        "--dataset-index", type=int, default=0,
        help="Sample index in the dataset split.",
    )
    infer_parser.add_argument(
        "--split", default="train",
        help="Dataset split: train or val.",
    )
    infer_parser.add_argument("--device", default=None, help="Torch device.")
    infer_parser.add_argument("--output", default=None, help="Save results as .npz.")

    # ── serve ──
    serve_parser = subparsers.add_parser(
        "serve",
        help="Serve a checkpoint as a ZMQ inference client.",
    )
    serve_parser.add_argument(
        "--checkpoint", required=True,
        help="Checkpoint root (must have inference_metadata/).",
    )
    serve_parser.add_argument(
        "--remote-ip", default="127.0.0.1",
        help="Simulator host IP.",
    )
    serve_parser.add_argument(
        "--port-zmq-cmd", type=int, default=5555,
        help="Port to send actions.",
    )
    serve_parser.add_argument(
        "--port-zmq-observations", type=int, default=5556,
        help="Port to receive observations.",
    )
    serve_parser.add_argument("--device", default=None, help="Torch device.")
    serve_parser.add_argument(
        "--strategy", default="receding_horizon",
        choices=["synchronous", "temporal_ensembling", "receding_horizon"],
        help="Action chunking execution strategy.",
    )
    serve_parser.add_argument(
        "--camera-names", nargs="*", default=None,
        help="Camera names (default: from saved schema).",
    )
    serve_parser.add_argument(
        "--platform", default="simulator",
        choices=["simulator", "lerobot"],
        help="Observation/action wire format. 'simulator' uses observation.images.X / observation.state keys; "
             "'lerobot' uses the lerobot host format (per-motor state scalars + base64 JPEG cameras).",
    )
    serve_parser.add_argument(
        "--task", default="",
        help="Task instruction (for language-conditioned policies).",
    )
    serve_parser.add_argument(
        "--max-loop-freq-hz", type=float, default=60.0,
        help="Client loop frequency cap (must be positive).",
    )
    serve_parser.add_argument(
        "--polling-timeout-ms", type=int, default=1000,
        help="ZMQ observation polling timeout in milliseconds.",
    )
    serve_parser.add_argument(
        "--connect-timeout-s", type=float, default=0.0,
        help="Initial connection timeout. 0 = wait forever.",
    )
    serve_parser.add_argument(
        "--n-action-steps", type=int, default=None,
        help="Actions executed per predicted chunk before re-predicting (lerobot n_action_steps). "
             "Default: action_horizon (play the full chunk). Lower = more reactive feedback.",
    )


    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.command == "train":
        from vla_factory.training.train import train
        metrics = train(
            args.config,
            override_steps=args.steps,
            override_batch_size=args.batch_size,
            override_output_dir=args.output_dir,
        )
        print(f"Training complete. Final metrics: {metrics}")

    elif args.command == "preprocess":
        from vla_factory.config.parser import parse_recipe
        from vla_factory.data.codec.pyav import preprocess_dataset
        recipe = parse_recipe(args.config)
        data_path = Path(recipe.data.source.path)
        preprocess_dataset(data_path)
        print("Preprocessing complete.")

    elif args.command == "list":
        if args.config:
            # Describe one recipe: model contract (base cameras, action_dim) +
            # camera_mapping validation against the base + dataset cameras.
            from pathlib import Path as _Path
            from vla_factory.config.parser import parse_recipe
            from vla_factory.config.defaults import resolve_recipe
            from vla_factory.model.base_contract import describe_model_config

            recipe = resolve_recipe(parse_recipe(args.config))

            # Best-effort: read the dataset schema (lightweight, meta/info.json
            # only) for the camera diff. Skip silently if the dataset isn't set
            # or unreadable — the base-contract half still prints.
            schema = None
            data_path = recipe.data.source.path
            if data_path:
                try:
                    from vla_factory.data.formats import get_reader
                    reader = get_reader(recipe.data.source.format, path=_Path(data_path))
                    schema = reader.get_schema(_Path(data_path))
                except Exception as e:
                    print(f"(skipped dataset schema read: {e})")

            print(describe_model_config(recipe, schema))
        else:
            from vla_factory.model.registry import list_entries
            entries = list_entries()
            if not entries:
                print("No models registered.")
            for name, meta in sorted(entries.items()):
                install = meta.install_hint or "-"
                print(f"  {name:20s} backend={meta.backend}  head={meta.action_head_type}  install={install}")

    elif args.command == "evaluate":
        import numpy as np
        from pathlib import Path as _Path
        from vla_factory.deploy.infer import InferenceEngine, ObsDict
        from vla_factory.data.formats import get_reader
        from vla_factory.data.codec import resolve_codec

        data_path = _Path(args.dataset)
        engine = InferenceEngine(
            checkpoint_path=args.checkpoint,
            device=args.device,
            execution_strategy="synchronous",
        )
        reader = get_reader(engine.recipe.data.source.format, path=data_path)
        codec = resolve_codec(engine.recipe.data.source.video_codec)
        action_horizon = engine.action_horizon

        episode_lengths = reader.get_episode_lengths(data_path)
        ep_indices = args.episodes or sorted(episode_lengths.keys())

        print(f"Episodes: {len(ep_indices)}, action_horizon: {action_horizon}")
        print()

        total_count = 0
        total_loss = 0.0

        for ep_idx in ep_indices:
            if ep_idx not in episode_lengths:
                continue
            ep_len = episode_lengths[ep_idx]
            episode = reader.read_episode(data_path, ep_idx, codec)
            frames = episode.load_frames()

            ep_count = 0
            ep_loss = 0.0

            for t in range(0, ep_len, action_horizon):
                obs_frame = frames[t]
                video = {}
                for cam_name in engine.camera_keys:
                    ref = obs_frame.images.get(cam_name)
                    if ref is None:
                        raise KeyError(f"Camera '{cam_name}' not in frame. Available: {list(obs_frame.images.keys())}")
                    video[cam_name] = codec.decode_frame(ref)
                state = obs_frame.state.astype(np.float32) if obs_frame.state is not None else None
                # Pass the frame's task text so language-conditioned models (pi0)
                # tokenize the *actual* episode task, not the recipe's default_task
                # fallback. Same fix as infer_from_dataset_sample — without it every
                # episode is evaluated under default_task, which silently biases L1
                # when the dataset has more than one task.
                obs = ObsDict(video=video, state=state, language=obs_frame.language)

                valid_len = min(t + action_horizon, ep_len) - t
                gt_list = [frames[t + i].action.astype(np.float32) for i in range(valid_len) if frames[t + i].action is not None]
                if not gt_list:
                    continue
                gt_raw = np.stack(gt_list, axis=0)

                pred_raw = engine.predict(obs)[:len(gt_raw)]
                frame_losses = np.abs(pred_raw - gt_raw).mean(axis=1)
                ep_loss += frame_losses.sum()
                ep_count += len(gt_raw)

                if args.verbose:
                    for i in range(len(gt_raw)):
                        print(f"  Ep {ep_idx} frame {t+i}: gt={gt_raw[i].tolist()} pred={pred_raw[i].tolist()} L1={frame_losses[i]:.6f}")

            if ep_count > 0:
                avg = ep_loss / ep_count
                total_loss += ep_loss
                total_count += ep_count
                print(f"Episode {ep_idx}: {ep_len} frames, L1 = {ep_loss:.4f} / {ep_count} = {avg:.6f}")

            if args.save_dir:
                sp = _Path(args.save_dir)
                sp.mkdir(parents=True, exist_ok=True)
                np.savez(sp / f"episode_{ep_idx}.npz",
                    episode_index=ep_idx, episode_length=ep_len,
                    total_l1=float(ep_loss), num_frames=ep_count,
                    avg_l1=float(ep_loss / ep_count) if ep_count > 0 else 0.0)

        if total_count > 0:
            print(f"\n{'='*60}")
            print(f"Total: {total_count} frames across {len(ep_indices)} episodes")
            print(f"Average L1 = {total_loss:.4f} / {total_count} = {total_loss / total_count:.6f}")

    elif args.command == "infer":
        import torch
        from vla_factory.deploy.infer import infer_from_dataset_sample
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # If no --config given, use the saved recipe from inference_metadata
        config = args.config
        if config is None:
            from vla_factory.utils.constants import INFERENCE_META_DIR, RECIPE_FILE
            ckpt = Path(args.checkpoint)
            # Look in checkpoint dir first, then parent (for checkpoint-NNN/ subdirs)
            meta_recipe = ckpt / INFERENCE_META_DIR / RECIPE_FILE
            if not meta_recipe.exists():
                meta_recipe = ckpt.parent / INFERENCE_META_DIR / RECIPE_FILE
            if meta_recipe.exists():
                config = str(meta_recipe)
            else:
                print(f"Error: no --config provided and no saved recipe found")
                sys.exit(1)

        result = infer_from_dataset_sample(
            config=config,
            checkpoint=args.checkpoint,
            dataset_index=args.dataset_index,
            split=args.split,
            device=device,
            output=args.output,
        )
        print("Inference result:")
        for k, v in result.items():
            print(f"  {k}: {v}")

    elif args.command == "serve":
        import torch
        from vla_factory.deploy.infer import InferenceEngine
        from vla_factory.deploy.transport import ZMQTransport
        if args.max_loop_freq_hz <= 0:
            parser.error("--max-loop-freq-hz must be a positive number")
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        engine = InferenceEngine(
            checkpoint_path=args.checkpoint,
            device=device,
            camera_names=args.camera_names,
            execution_strategy=args.strategy,
            n_action_steps=args.n_action_steps,
        )

        if args.platform == "lerobot":
            import json as _json
            import time
            import zmq
            from vla_factory.deploy.lerobot_host_adapter import (
                LerobotHostObsAdapter,
                LerobotHostActionAdapter,
            )

            # Motor-key mapping is a resolved data/model contract (dataset
            # `names` → recipe embodiment), never invented by sorting. This is
            # what keeps each action dimension driving the motor it was
            # trained on instead of scrambling them.
            obs_adapter = LerobotHostObsAdapter(
                camera_keys=engine.camera_keys,
                state_keys=engine.state_keys,
                state_dim=engine.schema.state_dim,
            )
            act_adapter = LerobotHostActionAdapter(
                action_dim=engine.action_dim,
                action_keys=engine.action_keys,
            )

            context = zmq.Context()
            cmd_socket = context.socket(zmq.PUSH)
            cmd_socket.connect(f"tcp://{args.remote_ip}:{args.port_zmq_cmd}")
            cmd_socket.setsockopt(zmq.CONFLATE, 1)

            obs_socket = context.socket(zmq.PULL)
            obs_socket.connect(f"tcp://{args.remote_ip}:{args.port_zmq_observations}")
            obs_socket.setsockopt(zmq.CONFLATE, 1)

            poller = zmq.Poller()
            poller.register(obs_socket, zmq.POLLIN)

            print(f"[serve] Model: {engine.recipe.model_name}", flush=True)
            print(f"[serve] Strategy: {args.strategy}", flush=True)
            print(f"[serve] Device: {device}", flush=True)
            print(f"[serve] Platform: lerobot (state_keys={list(engine.state_keys)}, "
                  f"action_keys={list(engine.action_keys)})", flush=True)
            print(f"[serve] Connecting to {args.remote_ip}:{args.port_zmq_observations}/{args.port_zmq_cmd}", flush=True)

            # Wait for first observation (connection confirmation)
            if args.connect_timeout_s > 0:
                socks = dict(poller.poll(int(args.connect_timeout_s * 1000)))
                if obs_socket not in socks or socks[obs_socket] != zmq.POLLIN:
                    print("[serve] Timeout waiting for host observations.", flush=True)
                    sys.exit(1)
            else:
                last_log = 0.0
                while True:
                    socks = dict(poller.poll(1000))
                    if obs_socket in socks and socks[obs_socket] == zmq.POLLIN:
                        break
                    now = time.time()
                    if now - last_log >= 5.0:
                        print("[serve] Waiting for host observations...", flush=True)
                        last_log = now

            print("[serve] Connected. Running inference loop.", flush=True)

            try:
                while True:
                    loop_start = time.time()

                    # Recv latest observation: poll then drain. A bare NOBLOCK-drain loop
                    # starves a CONFLATE PULL (only the first message arrives); polling first
                    # matches lerobot's LeKiwiClient and keeps the pipe delivering.
                    poller = zmq.Poller()
                    poller.register(obs_socket, zmq.POLLIN)
                    socks = dict(poller.poll(args.polling_timeout_ms))
                    latest_raw = None
                    if obs_socket in socks and socks[obs_socket] == zmq.POLLIN:
                        while True:
                            try:
                                latest_raw = obs_socket.recv_string(zmq.NOBLOCK)
                            except zmq.Again:
                                break
                    if latest_raw is None:
                        time.sleep(max(1.0 / args.max_loop_freq_hz - (time.time() - loop_start), 0.0))
                        continue

                    # lerobot host format → ObsDict → predict → motor-key action dict
                    observation = _json.loads(latest_raw)
                    obs = obs_adapter(observation, task=args.task)
                    action = engine.predict(obs)
                    action_dict = act_adapter(action)
                    cmd_socket.send_string(_json.dumps(action_dict), flags=zmq.NOBLOCK)

                    elapsed = time.time() - loop_start
                    time.sleep(max(1.0 / args.max_loop_freq_hz - elapsed, 0.0))

            except KeyboardInterrupt:
                print("[serve] Keyboard interrupt. Exiting.", flush=True)
            finally:
                obs_socket.close(linger=0)
                cmd_socket.close(linger=0)
                context.term()

        else:
            # Default: simulator platform (observation.images.X keys)
            transport = ZMQTransport(
                remote_ip=args.remote_ip,
                port_zmq_cmd=args.port_zmq_cmd,
                port_zmq_observations=args.port_zmq_observations,
                polling_timeout_ms=args.polling_timeout_ms,
                connect_timeout_s=args.connect_timeout_s,
                max_loop_freq_hz=args.max_loop_freq_hz,
            )
            print(f"[serve] Model: {engine.recipe.model_name}", flush=True)
            print(f"[serve] Strategy: {args.strategy}", flush=True)
            print(f"[serve] Device: {device}", flush=True)
            transport.serve(engine)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
