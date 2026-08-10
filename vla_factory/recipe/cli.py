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
        description="VLA Factory: train and deploy robot models.",
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

    # ── resolve ──
    resolve_parser = subparsers.add_parser(
        "resolve",
        help="Dry-run: resolve the data × model × robot composition and print "
             "a summary or a structured ResolutionError. No GPU / no optional "
             "model extras required.",
    )
    resolve_parser.add_argument("--config", required=True, help="Path to YAML recipe file.")

    # ── inspect ──
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect one dimension's declared facts and their sources: "
             "`inspect data --path`, `inspect model --name [--path]`, "
             "`inspect robot --name`, or `inspect --config` for all three. "
             "No GPU / no optional extras / no robot connection required.",
    )
    inspect_parser.add_argument("dimension", nargs="?", choices=("data", "model", "robot"),
                                help="Which dimension to inspect.")
    inspect_parser.add_argument("--path", default=None,
                                help="Dataset path (data) or checkpoint path (model).")
    inspect_parser.add_argument("--name", default=None,
                                help="Registered model name (model) or robot profile (robot).")
    inspect_parser.add_argument("--format", default=None,
                                help="Dataset format hint for `inspect data` (default: auto).")
    inspect_parser.add_argument("--config", default=None,
                                help="Recipe YAML — inspects all three dimensions at once.")
    inspect_parser.add_argument("--json", action="store_true",
                                help="Emit machine-readable JSON (default: YAML).")
    inspect_parser.add_argument("--stats", action="store_true",
                                help="Include full NormStats (data only; default: summary).")

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

    # ── deploy ──
    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Deploy a checkpoint to a simulator or robot platform.",
    )
    deploy_parser.add_argument(
        "--checkpoint", required=True,
        help="Checkpoint root (must have inference_metadata/).",
    )
    deploy_parser.add_argument(
        "--remote-ip", default="127.0.0.1",
        help="Simulator host IP.",
    )
    deploy_parser.add_argument(
        "--port-zmq-cmd", type=int, default=5555,
        help="Port to send actions.",
    )
    deploy_parser.add_argument(
        "--port-zmq-observations", type=int, default=5556,
        help="Port to receive observations.",
    )
    deploy_parser.add_argument("--device", default=None, help="Torch device.")
    deploy_parser.add_argument(
        "--strategy", default=None,
        choices=["synchronous", "temporal_ensembling", "receding_horizon"],
        help="Action chunk execution strategy. Defaults to synchronous for "
             "RoboTwin and receding_horizon for other platforms.",
    )
    deploy_parser.add_argument(
        "--camera-names", nargs="*", default=None,
        help="Camera names (default: from saved schema).",
    )
    deploy_parser.add_argument(
        "--platform", default="simulator",
        choices=["simulator", "lerobot", "robotwin"],
        help="Target platform / wire format. 'simulator' uses observation.images.X / observation.state keys; "
             "'lerobot' uses the lerobot host format (per-motor state scalars + base64 JPEG cameras); "
             "'robotwin' runs a RoboTwin-compatible TCP model server (the RoboTwin simulator connects as client).",
    )
    deploy_parser.add_argument(
        "--host", default="0.0.0.0",
        help="Bind address for the 'robotwin' TCP model server.",
    )
    deploy_parser.add_argument(
        "--port", type=int, default=9999,
        help="TCP port for the 'robotwin' model server (must match the RoboTwin client's --port).",
    )
    deploy_parser.add_argument(
        "--task", default="",
        help="Task instruction (for language-conditioned policies).",
    )
    deploy_parser.add_argument(
        "--max-loop-freq-hz", type=float, default=60.0,
        help="Client loop frequency cap (must be positive).",
    )
    deploy_parser.add_argument(
        "--polling-timeout-ms", type=int, default=1000,
        help="ZMQ observation polling timeout in milliseconds.",
    )
    deploy_parser.add_argument(
        "--connect-timeout-s", type=float, default=0.0,
        help="Initial connection timeout. 0 = wait forever.",
    )
    deploy_parser.add_argument(
        "--n-action-steps", type=int, default=None,
        help="Steps selected from each predicted chunk. Synchronous returns this "
             "prefix; receding-horizon plays it one step per observation. "
             "Must be in [1, action_horizon]. Temporal ensembling requires 1 or omission.",
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
        from vla_factory.recipe.parser import parse_recipe
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
            from vla_factory.recipe.parser import parse_recipe
            from vla_factory.recipe.defaults import resolve_recipe
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

    elif args.command == "resolve":
        _run_resolve(args.config)

    elif args.command == "inspect":
        _run_inspect(args)

    elif args.command == "evaluate":
        import numpy as np
        from pathlib import Path as _Path
        from vla_factory.inference.infer import InferenceEngine, ObsDict
        from vla_factory.data.formats import get_reader
        from vla_factory.data.codec import resolve_codec

        data_path = _Path(args.dataset)
        engine = InferenceEngine(
            checkpoint_path=args.checkpoint,
            device=args.device,
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

                pred_raw = engine.predict(obs).values[:len(gt_raw)]
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
        from vla_factory.inference.infer import infer_from_dataset_sample
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

    elif args.command == "deploy":
        import torch
        from vla_factory.inference.infer import (
            PolicyExecutor,
            build_execution_policy,
        )
        from vla_factory.inference.infer import InferenceEngine
        if args.max_loop_freq_hz <= 0:
            parser.error("--max-loop-freq-hz must be a positive number")
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        engine = InferenceEngine(
            checkpoint_path=args.checkpoint,
            device=device,
            camera_names=args.camera_names,
        )
        strategy = args.strategy or (
            "synchronous" if args.platform == "robotwin" else "receding_horizon"
        )
        execution_policy = build_execution_policy(
            strategy,
            action_horizon=engine.action_horizon,
            action_dim=engine.action_dim,
            n_action_steps=args.n_action_steps,
        )
        policy = PolicyExecutor(engine, execution_policy)

        if args.platform == "robotwin":
            from vla_factory.inference.platforms.robotwin import RoboTwinAdapter
            from vla_factory.inference.policy_runtime import RemotePolicyModel
            from vla_factory.inference.transports.length_prefixed_json import (
                LengthPrefixedJsonRpcServer,
            )

            adapter = RoboTwinAdapter(
                camera_keys=engine.camera_keys,
                state_dim=engine.schema.state_dim,
            )
            model = RemotePolicyModel(
                policy, adapter, task=args.task,
            )
            server = LengthPrefixedJsonRpcServer(
                model, host=args.host, port=args.port,
            )

            print(f"[deploy] Model: {engine.recipe.model_name}", flush=True)
            print(f"[deploy] Device: {device}", flush=True)
            print(f"[deploy] Platform: robotwin (cameras={list(engine.camera_keys)}, "
                  f"state_dim={engine.schema.state_dim}, action_dim={engine.action_dim})", flush=True)
            print(f"[deploy] Listening on {args.host}:{args.port} — start the RoboTwin "
                  f"client with matching --port.", flush=True)
            server.serve_forever()

        else:
            # ZMQ-host platforms (simulator / lerobot) share one client-shaped
            # deployment loop; the platform difference is only the adapters.
            from vla_factory.inference.policy_runtime import PolicyRunner
            from vla_factory.inference.transports.zmq import (
                ZmqPolicyClient,
                ZmqPolicyClientConfig,
            )

            if args.platform == "lerobot":
                from vla_factory.inference.platforms.lerobot import (
                    LerobotHostObsAdapter,
                    LerobotHostActionAdapter,
                )

                # Motor-key mapping is a resolved data/model contract (dataset
                # `names` → recipe embodiment), never invented by sorting. This
                # is what keeps each action dimension driving the motor it was
                # trained on instead of scrambling them.
                obs_adapter = LerobotHostObsAdapter(
                    camera_keys=engine.camera_keys,
                    state_keys=engine.state_keys,
                    state_dim=engine.schema.state_dim,
                )
                action_adapter = LerobotHostActionAdapter(
                    action_dim=engine.action_dim,
                    action_keys=engine.action_keys,
                )
                platform_desc = (
                    f"lerobot (state_keys={list(engine.state_keys)}, "
                    f"action_keys={list(engine.action_keys)})"
                )
            else:
                # Default: simulator platform (observation.images.X keys)
                from vla_factory.inference.platforms.simulator import SimulatorAdapter

                obs_adapter = SimulatorAdapter(engine.camera_keys)
                action_adapter = None
                platform_desc = f"simulator (cameras={list(engine.camera_keys)})"

            runner = PolicyRunner(
                policy,
                obs_adapter,
                action_adapter,
                task=args.task,
                max_loop_freq_hz=args.max_loop_freq_hz,
            )
            client = ZmqPolicyClient(ZmqPolicyClientConfig(
                remote_ip=args.remote_ip,
                port_zmq_cmd=args.port_zmq_cmd,
                port_zmq_observations=args.port_zmq_observations,
                polling_timeout_ms=args.polling_timeout_ms,
                connect_timeout_s=args.connect_timeout_s,
            ))

            print(f"[deploy] Model: {engine.recipe.model_name}", flush=True)
            print(f"[deploy] Strategy: {strategy}", flush=True)
            print(f"[deploy] Device: {device}", flush=True)
            print(f"[deploy] Platform: {platform_desc}", flush=True)
            print(f"[deploy] Connecting to {args.remote_ip}:"
                  f"{args.port_zmq_observations}/{args.port_zmq_cmd}", flush=True)
            try:
                runner.run(client)
            except TimeoutError:
                print("[deploy] Timeout waiting for host observations.", flush=True)
                sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


def _run_resolve(config_path: str) -> None:
    """``vlafactory-cli resolve --config <recipe>`` dry-run handler.

    Parses the recipe, gathers the three descriptions (data schema/norm_stats,
    model metadata, optional base contract + robot profile), runs
    ``resolve_assembly`` and prints either a summary or a structured
    ``ResolutionError``. Runs without GPU and without optional model extras —
    it never triggers the model factory.
    """
    from pathlib import Path as _Path

    from vla_factory.recipe.parser import parse_recipe
    from vla_factory.recipe.defaults import resolve_recipe
    from vla_factory.model.registry import list_entries
    from vla_factory.assembly.resolver import (
        make_error,
        resolve_assembly,
        ResolutionError,
        UNKNOWN_MODEL,
        UNKNOWN_ROBOT,
    )

    recipe = resolve_recipe(parse_recipe(config_path))

    # ── Model metadata (registry only — no factory, no heavy deps) ──
    entries = list_entries()
    metadata = entries.get(recipe.model_name)
    if metadata is None:
        err = make_error(
            UNKNOWN_MODEL, "model.name",
            model_name=recipe.model_name, known=sorted(entries),
        )
        _print_resolution_error(err)
        sys.exit(1)

    # ── Optional base contract (from the checkpoint's config.json) ──
    base_contract = None
    if recipe.model_path:
        try:
            from vla_factory.model.base_contract import load_base_contract
            base_contract = load_base_contract(recipe.model_path)
        except Exception as e:  # unreadable / offline — keep going without it
            print(f"(skipped base contract read: {e})")

    # ── Optional robot profile ──
    robot_profile = None
    if recipe.robot.name:
        try:
            from vla_factory.robot import get_robot_profile, list_robot_profiles
            robot_profile = get_robot_profile(recipe.robot.name)
        except Exception:
            err = make_error(
                UNKNOWN_ROBOT, "robot.name",
                robot_name=recipe.robot.name, known=list_robot_profiles(),
            )
            _print_resolution_error(err)
            sys.exit(1)

    # ── Data schema + norm_stats (best-effort; meta files only) ──
    schema = None
    norm_stats = None
    data_path = recipe.data.source.path
    if data_path:
        try:
            from vla_factory.data.formats import get_reader
            reader = get_reader(recipe.data.source.format, path=_Path(data_path))
            schema = reader.get_schema(_Path(data_path))
            norm_stats = reader.get_norm_stats(_Path(data_path))
        except Exception as e:
            print(f"(skipped dataset read: {e})")

    # ── Controlled overrides from the recipe's ``assembly`` block ──
    overrides = {
        k: v for k, v in (
            ("camera_mapping", recipe.assembly.camera_mapping),
            ("accept_fps_mismatch", recipe.assembly.accept_fps_mismatch),
            ("gripper_flip", recipe.assembly.gripper_flip),
            ("default_task", recipe.assembly.default_task),
        ) if v is not None
    }

    try:
        assembly = resolve_assembly(
            schema=schema,
            norm_stats=norm_stats,
            metadata=metadata,
            base_contract=base_contract,
            robot_profile=robot_profile,
            overrides=overrides or None,
        )
    except ResolutionError as e:
        _print_resolution_error(e)
        sys.exit(1)

    _print_assembly_summary(assembly)


def _print_assembly_summary(assembly) -> None:
    ci = assembly.canonical_interface
    print("Resolved assembly:")
    print(f"  model:      {assembly.metadata_ref.get('name')}")
    contract = assembly.contract_ref
    print(f"  base:       {contract.get('repo_or_path') if contract else '(none)'}")
    print(f"  robot:      {assembly.robot_ref.get('name') if assembly.robot_ref else '(none)'}")
    print(f"  cameras:    {list(ci.cameras) or '(none)'}")
    print(f"  action:     dim={ci.action_dim} horizon={ci.action_horizon}")
    print(f"  state:      dim={ci.state_dim}")
    print(f"  language:   {'required' if ci.requires_language else 'not required'}")
    mapped = sum(
        1 for m in (
            assembly.camera_mapping, assembly.state_mapping,
            assembly.action_mapping, assembly.language_mapping,
            assembly.joint_mapping,
        ) if m.resolved
    )
    print(f"  mappings:   {mapped}/5 resolved (phase-0 skeleton)")


def _print_resolution_error(err: ResolutionError) -> None:
    print("Resolution failed:")
    print(f"  code: {err.code}")
    print(f"  path: {err.path}")
    print(f"  params: {err.params}")


# ── inspect (architecture §3.5) ───────────────────────────────────


def _emit(dimension: str, source: str, facts: object, as_json: bool) -> None:
    """Print one dimension's facts in the ``{dimension, source, facts}`` envelope.

    YAML by default (deterministic insertion order → diffable); ``--json``
    otherwise. Source is per-dimension: data facts carry their own per-fact
    ``*_source`` labels inside ``facts``; this envelope-level ``source`` records
    where the whole dimension came from.
    """
    import json as _json

    envelope = {"dimension": dimension, "source": source, "facts": facts}
    if as_json:
        print(_json.dumps(envelope, indent=2, ensure_ascii=False))
    else:
        import yaml as _yaml
        print(_yaml.safe_dump(envelope, sort_keys=False, allow_unicode=True), end="")


def _inspect_data(path: str, fmt: str | None, with_stats: bool, as_json: bool) -> None:
    from pathlib import Path as _Path

    from vla_factory.data.formats import get_reader

    p = _Path(path)
    reader = get_reader(fmt or "auto", path=p)
    schema = reader.get_schema(p)
    facts: dict = {"schema": schema.to_dict()}
    ns = reader.get_norm_stats(p)
    if with_stats:
        from dataclasses import asdict as _asdict
        facts["norm_stats"] = _asdict(ns)
    else:
        # Stats summary only: which vectors/images carry statistics.
        facts["norm_stats_summary"] = {
            "state": ns.state is not None,
            "action": ns.action is not None,
            "images": sorted((ns.images or {}).keys()),
        }
    _emit("data", "measured/inferred/undeclared (per fact)", facts, as_json)


def _tunables_view(params: dict, overrides: dict | None) -> dict:
    """Render the declared tunables as ``key -> {value, source}``.

    The point of the view is answering "what may I change, and did my change
    take effect" without reading code: every declared key is listed with the
    value actually in force and where it came from. ``transforms`` is summarised
    by step type — the full step list would drown the rest.
    """
    overrides = overrides or {}
    view: dict = {}
    for key in sorted(params):
        overridden = key in overrides
        value = overrides[key] if overridden else params[key]
        if key == "transforms":
            steps = (value or {}).get("inputs") or []
            value = [s.get("type") for s in steps if isinstance(s, dict)]
        view[key] = {
            "value": value,
            "source": "recipe" if overridden else "model default",
        }
    return view


def _inspect_model(
    name: str, path: str | None, as_json: bool, overrides: dict | None = None
) -> None:
    from dataclasses import asdict as _asdict

    from vla_factory.model.registry import list_entries

    entries = list_entries()
    meta = entries.get(name)
    if meta is None:
        print(f"Unknown model {name!r}. Known: {sorted(entries)}")
        sys.exit(1)
    meta_dict = _asdict(meta)
    # Facts and tunables are the two halves of a model declaration: named fields
    # the resolver reads and a recipe can never override, versus params a recipe
    # may override through model.config.
    params = meta_dict.pop("params", {}) or {}
    facts = {"metadata": meta_dict}
    source = "metadata"
    if path:
        try:
            from vla_factory.model.base_contract import load_base_contract
            contract = load_base_contract(path)
            if contract is not None:
                facts["base_contract"] = _asdict(contract)
                source = "metadata + base_contract"
        except Exception as e:
            facts["base_contract_error"] = str(e)
    facts["tunables"] = _tunables_view(params, overrides)
    _emit("model", source, facts, as_json)


def _inspect_robot(name: str, as_json: bool) -> None:
    from vla_factory.robot import get_robot_profile

    profile = get_robot_profile(name)  # raises FileNotFoundError if unknown
    _emit("robot", "declared", profile.to_dict(), as_json)


def _run_inspect(args) -> None:
    as_json = bool(args.json)
    if args.config:
        # Inspect all three dimensions from a recipe.
        from vla_factory.recipe.parser import parse_recipe
        from vla_factory.recipe.defaults import resolve_recipe

        parsed = parse_recipe(args.config)
        # What the user actually wrote, before the model's declared params are
        # merged underneath — that difference is exactly the "source" column.
        raw_overrides = dict(parsed.model_config or {})
        recipe = resolve_recipe(parsed)
        if recipe.data.source.path:
            # One unreadable dimension must not hide the other two: a recipe is
            # routinely inspected on a machine that has the model but not the
            # dataset. Report and carry on.
            try:
                _inspect_data(recipe.data.source.path, recipe.data.source.format,
                              bool(args.stats), as_json)
            except Exception as e:
                print(f"(skipped data dimension: {e})")
        if recipe.model_name:
            _inspect_model(recipe.model_name, recipe.model_path, as_json,
                           overrides=raw_overrides)
        if recipe.robot.name:
            _inspect_robot(recipe.robot.name, as_json)
        return

    if not args.dimension:
        sys.exit("inspect: provide a dimension (data/model/robot) or --config. See `inspect --help`.")
    if args.dimension == "data":
        if not args.path:
            sys.exit("inspect data: --path <dataset> is required.")
        _inspect_data(args.path, args.format, bool(args.stats), as_json)
    elif args.dimension == "model":
        if not args.name:
            sys.exit("inspect model: --name <model> is required.")
        _inspect_model(args.name, args.path, as_json)
    elif args.dimension == "robot":
        if not args.name:
            sys.exit("inspect robot: --name <robot> is required.")
        _inspect_robot(args.name, as_json)


if __name__ == "__main__":
    main()
