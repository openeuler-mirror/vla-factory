"""VLA Factory command-line interface.

Usage::

    vlafactory-cli train --config recipe.yaml          # installed console script
    python -m vla_factory train --config recipe.yaml   # without install / from source
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from vla_factory.assembly import MappingSource, ResolutionError, resolve_assembly
from vla_factory.assembly.resolve.mappings import validate_camera_override
from vla_factory.model.checkpoint_validation import (
    CheckpointCompatibilityError,
    validate_checkpoint_if_available,
)
from vla_factory.model.registry import list_entries
from vla_factory.recipe import merge_model_config, parse_recipe


def _describe_model_config(recipe, schema=None) -> str:
    """Describe model metadata, checkpoint status, and recipe camera mapping.

    Camera override validation stays in the assembly layer so this report and
    ``resolve_assembly`` cannot disagree about dynamic model slots.
    """
    entries = list_entries()
    metadata = entries.get(recipe.model.name)
    if metadata is None:
        return f"Model:   {recipe.model.name} (not registered; known: {sorted(entries)})"

    capabilities = [metadata.backend, metadata.action_head_type]
    if metadata.training_paradigm == "pretrained_finetune":
        capabilities.append("finetune-only")
    lines = [
        f"Model:   {recipe.model.name} (" + " · ".join(capabilities) + ")",
        f"Checkpoint: {recipe.model.path or '(not set)'}",
        f"Action:  dim={metadata.action_dim or 'from data'} "
        f"horizon={metadata.action_horizon or 'from recipe'} "
        f"policy={metadata.dim_policy}",
        "ModelMetadata vision slots:",
    ]
    for slot in metadata.vision_slots:
        resolution = "x".join(str(v) for v in slot.resolution) if slot.resolution else "?"
        lines.append(f"  {slot.name:20s} ({slot.channels}x{resolution})")
    if not metadata.vision_slots:
        lines.append("  (dynamic; follows dataset)")

    if recipe.model.path:
        try:
            check = validate_checkpoint_if_available(recipe.model.path, metadata)
            if check["status"] == "compatible":
                lines.append("Checkpoint check: compatible")
            else:
                lines.append(f"Checkpoint check: unavailable — {check['detail']}")
        except CheckpointCompatibilityError as exc:
            lines.append(f"Checkpoint check: incompatible — {'; '.join(exc.issues)}")

    dataset_cameras = list(schema.cameras) if schema is not None else []
    if schema is not None:
        lines.append(f"Dataset cameras: {dataset_cameras or '(none)'}")

    camera_mapping = recipe.overrides.camera_mapping
    if camera_mapping is not None:
        lines.append("camera_mapping:")
        lines.extend(
            f"  {role:20s} <- {camera}"
            for role, camera in camera_mapping.items()
        )
        if metadata.vision_slots:
            lines.extend(
                f"  {slot.name:20s} <- <EMPTY>"
                for slot in metadata.vision_slots
                if slot.name not in camera_mapping
            )
        if schema is None:
            lines.append("WARNING: dataset unavailable; camera names were not validated.")
        else:
            try:
                validate_camera_override(camera_mapping, schema, metadata)
            except ResolutionError as exc:
                lines.append(f"ERROR: {exc.path}: {exc.params}")
            unused = [camera for camera in dataset_cameras if camera not in camera_mapping.values()]
            if unused:
                lines.append(f"WARNING: dataset cameras unused: {unused}")
        if camera_mapping:
            pairs = ", ".join(
                f"{camera}->{role}" for role, camera in camera_mapping.items()
            )
            lines.append(
                f"WARNING: VLA models are sensitive to view semantics — verify {pairs}."
            )
    return "\n".join(lines)


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
        help="Flattened frame index in the dataset.",
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
    # No --camera-names: camera keys are part of the checkpoint's resolved
    # composition (see InferenceEngine). Renaming them at deploy time would
    # desynchronise them from the camera mapping the model was trained with.
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
        from vla_factory.data.codec.pyav import preprocess_dataset
        recipe = parse_recipe(args.config)
        data_path = Path(recipe.data.path)
        preprocess_dataset(data_path)
        print("Preprocessing complete.")

    elif args.command == "list":
        if args.config:
            # Describe one recipe: authoritative model metadata, optional
            # checkpoint consistency, and camera mapping.
            recipe = merge_model_config(parse_recipe(args.config))

            # Best-effort: read the dataset schema (lightweight, meta/info.json
            # only) for the camera diff. Skip silently if the dataset isn't set
            # or unreadable — the model-metadata half still prints.
            schema = None
            data_path = recipe.data.path
            if data_path:
                try:
                    from vla_factory.data.reader import get_reader
                    reader = get_reader(recipe.data.format, path=Path(data_path))
                    schema = reader.get_schema(Path(data_path))
                except Exception as e:
                    print(f"(skipped dataset schema read: {e})")

            print(_describe_model_config(recipe, schema))
        else:
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
        from vla_factory.inference.evaluate_dataset import evaluate_dataset

        report = evaluate_dataset(
            args.dataset,
            checkpoint=args.checkpoint,
            episode_indices=args.episodes,
            device=args.device,
            save_dir=args.save_dir,
            include_frame_metrics=args.verbose,
        )
        print(
            f"Episodes: {len(report['episodes'])}, "
            f"action_horizon: {report['action_horizon']}"
        )
        print()
        for episode in report["episodes"]:
            for frame in episode.get("frames", []):
                print(
                    f"  Ep {episode['episode_index']} frame "
                    f"{frame['frame_index']}: gt={frame['target']} "
                    f"pred={frame['prediction']} L1={frame['l1']:.6f}"
                )
            if episode["num_frames"]:
                print(
                    f"Episode {episode['episode_index']}: "
                    f"{episode['episode_length']} frames, "
                    f"L1 = {episode['total_l1']:.4f} / "
                    f"{episode['num_frames']} = {episode['average_l1']:.6f}"
                )

        if report["num_frames"]:
            print(f"\n{'='*60}")
            print(
                f"Total: {report['num_frames']} frames across "
                f"{len(report['episodes'])} episodes"
            )
            print(
                f"Average L1 = {report['total_l1']:.4f} / "
                f"{report['num_frames']} = {report['average_l1']:.6f}"
            )

    elif args.command == "infer":
        import torch
        from vla_factory.inference.evaluate_dataset import infer_dataset_sample
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

        result = infer_dataset_sample(
            config=config,
            checkpoint=args.checkpoint,
            dataset_index=args.dataset_index,
            device=device,
            output=args.output,
        )
        print("Inference result:")
        for k, v in result.items():
            print(f"  {k}: {v}")

    elif args.command == "deploy":
        if args.max_loop_freq_hz <= 0:
            parser.error("--max-loop-freq-hz must be a positive number")
        from vla_factory.inference.deploy import DeploymentConfig, deploy

        config = DeploymentConfig(
            checkpoint=args.checkpoint,
            platform=args.platform,
            device=args.device,
            strategy=args.strategy,
            task=args.task,
            n_action_steps=args.n_action_steps,
            max_loop_freq_hz=args.max_loop_freq_hz,
            remote_ip=args.remote_ip,
            port_zmq_cmd=args.port_zmq_cmd,
            port_zmq_observations=args.port_zmq_observations,
            polling_timeout_ms=args.polling_timeout_ms,
            connect_timeout_s=args.connect_timeout_s,
            host=args.host,
            port=args.port,
        )
        try:
            deploy(config)
        except TimeoutError:
            print("[deploy] Timeout waiting for host observations.", flush=True)
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


def _run_resolve(config_path: str) -> None:
    """``vlafactory-cli resolve --config <recipe>`` dry-run handler.

    Delegates the whole recipe → descriptions → assembly sequence to the
    public ``resolve_assembly()`` entry and only adds the
    CLI's own surface: a printed summary, a structured error dump, and the exit
    code. Runs without GPU and without optional model extras — it never triggers
    the model factory.
    """
    recipe = merge_model_config(parse_recipe(config_path))

    try:
        assembly = resolve_assembly(recipe)
    except ResolutionError as e:
        _print_resolution_error(e)
        sys.exit(1)
    except CheckpointCompatibilityError as e:
        print(f"Checkpoint compatibility failed: {e}")
        sys.exit(1)

    _print_assembly_summary(assembly, checkpoint_path=recipe.model.path)


def _camera_mapping_summary(mapping) -> str:
    total = len(mapping.entries)
    padded = [e["model_slot"] for e in mapping.entries if e.get("data_source") is None]
    overridden = sum(
        1 for entry in mapping.entries
        if entry.get("source") == MappingSource.OVERRIDE
    )
    parts = [f"{total - len(padded)}/{total} slots mapped"]
    if overridden:
        parts.append(f"{overridden} via override")
    if padded:
        parts.append(f"padding: {', '.join(padded)}")
    return "; ".join(parts)


def _vector_mapping_summary(mapping, model_width: int) -> str:
    mapped = len(mapping.entries)
    padded = max(0, int(model_width) - mapped)
    return f"{mapped}/{model_width} dims from data" + (
        f", {padded} padded" if padded else ""
    )


def _language_mapping_summary(mapping, plan) -> str:
    if not mapping.entries:
        return "not required"
    entry = mapping.entries[0]
    tokenize_call = next(
        (call for call in plan.calls if call.type == "task_tokenize"), None,
    )
    default_task = (
        tokenize_call.args.get("default_task") if tokenize_call is not None else None
    )
    if entry.get("data_field"):
        result = f"data field {entry['data_field']!r}"
        if default_task is not None:
            result += f"; fallback default_task {default_task!r}"
        return result
    if default_task is not None:
        return f"default_task {default_task!r} (override)"
    return "no task text and no default_task — empty prompt"


def _plan_summary(plan) -> str:
    if not plan.calls:
        return "no steps"
    count = len(plan.calls)
    return (f"{count} step{'s' if count > 1 else ''}: "
            + " → ".join(c.type for c in plan.calls))


def _print_assembly_summary(assembly, checkpoint_path: str | None = None) -> None:
    ci = assembly.model_io_spec
    print("Resolved assembly:")
    print(f"  model:      {assembly.metadata_ref.get('name')}")
    print(f"  checkpoint: {checkpoint_path or '(none)'}")
    print(f"  robot:      {assembly.robot_ref.get('name') if assembly.robot_ref else '(none)'}")
    print(f"  cameras:    {list(ci.cameras) or '(none)'}")
    print(f"  action:     dim={ci.action_dim} horizon={ci.action_horizon}")
    print(f"  state:      dim={ci.state_dim}")
    print(f"  language:   {'required' if ci.requires_language else 'not required'}")
    print("  mappings:")
    print(f"    camera:   {_camera_mapping_summary(assembly.camera_mapping)}")
    print(f"    state:    {_vector_mapping_summary(assembly.state_mapping, ci.state_dim)}")
    print(f"    action:   {_vector_mapping_summary(assembly.action_mapping, ci.action_dim)}")
    print(
        "    language: "
        f"{_language_mapping_summary(assembly.language_mapping, assembly.data_to_model)}"
    )
    print("  pipelines:")
    print(f"    data_to_model:  {_plan_summary(assembly.data_to_model)}")
    print(f"    robot_to_model: {_plan_summary(assembly.robot_to_model)} (shared)")
    print(f"    model_to_robot: {_plan_summary(assembly.model_to_robot)}")


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

    from vla_factory.data.reader import get_reader

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
    value actually in force and where it came from.
    """
    overrides = overrides or {}
    view: dict = {}
    for key in sorted(params):
        overridden = key in overrides
        value = overrides[key] if overridden else params[key]
        view[key] = {
            "value": value,
            "source": "recipe" if overridden else "model default",
        }
    return view


def _inspect_model(
    name: str, path: str | None, as_json: bool, overrides: dict | None = None
) -> None:
    from dataclasses import asdict as _asdict

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
    if path:
        try:
            facts["checkpoint_check"] = validate_checkpoint_if_available(path, meta)
        except CheckpointCompatibilityError as e:
            facts["checkpoint_check"] = {
                "status": "incompatible",
                "issues": list(e.issues),
            }
    facts["tunables"] = _tunables_view(params, overrides)
    _emit("model", "metadata", facts, as_json)


def _inspect_robot(name: str, as_json: bool) -> None:
    from vla_factory.robot import get_robot_profile

    profile = get_robot_profile(name)  # raises FileNotFoundError if unknown
    _emit("robot", "declared", profile.to_dict(), as_json)


def _run_inspect(args) -> None:
    as_json = bool(args.json)
    if args.config:
        # Inspect all three dimensions from a recipe.
        parsed = parse_recipe(args.config)
        # What the user actually wrote, before the model's declared params are
        # merged underneath — that difference is exactly the "source" column.
        raw_overrides = dict(parsed.model.config or {})
        recipe = merge_model_config(parsed)
        if recipe.data.path:
            # One unreadable dimension must not hide the other two: a recipe is
            # routinely inspected on a machine that has the model but not the
            # dataset. Report and carry on.
            try:
                _inspect_data(recipe.data.path, recipe.data.format,
                              bool(args.stats), as_json)
            except Exception as e:
                print(f"(skipped data dimension: {e})")
        if recipe.model.name:
            _inspect_model(recipe.model.name, recipe.model.path, as_json,
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
