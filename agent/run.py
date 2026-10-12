#!/usr/bin/env python3
"""Single-entry VLA adaptation workflow with resumable jobs.

Run this file with no arguments for interactive setup, or pass ``--resume``
with an existing job directory. The Judge Harness owns every execution,
repair decision, and final verdict.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from adapt_vla_model.benchmarks import DEFAULT_BENCHMARK
from adapt_vla_model.judge import run_workflow
from adapt_vla_model.runtime import DEFAULT_OPENCODE_MODEL, load_json, load_manifest, OPENCODE_MODELS


DEFAULTS = {
    "upstream_repo": "https://github.com/huggingface/lerobot",
    "vla": "smolvla",
    "checkpoint": "lerobot/smolvla_base",
    "factory_base": "master",
    "benchmark": DEFAULT_BENCHMARK,
    "max_repairs": "2",
    "opencode_model": DEFAULT_OPENCODE_MODEL,
}


def _ask(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _choose_opencode_model(default: str = DEFAULT_OPENCODE_MODEL) -> str:
    print("\nAvailable OpenCode models:")
    for index, model in enumerate(OPENCODE_MODELS, start=1):
        marker = " (default)" if model == default else ""
        print(f"  {index:2}. {model}{marker}")
    value = input(f"OpenCode model [1-{len(OPENCODE_MODELS)}, Enter={default}]: ").strip()
    if not value:
        return default
    if value.isdecimal():
        index = int(value)
        if 1 <= index <= len(OPENCODE_MODELS):
            return OPENCODE_MODELS[index - 1]
        raise ValueError(f"OpenCode model selection must be between 1 and {len(OPENCODE_MODELS)}")
    if "/" not in value:
        # Resolve a bare model name against the provider prefixes in the list.
        prefixes = sorted({model.split("/")[0] for model in OPENCODE_MODELS if "/" in model})
        value = next(
            (f"{prefix}/{value}" for prefix in prefixes if f"{prefix}/{value}" in OPENCODE_MODELS),
            value,
        )
    if value not in OPENCODE_MODELS:
        raise ValueError(f"unsupported OpenCode model: {value}")
    return value


def _job_id(vla: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", vla).strip("-") or "vla"
    return f"{value}-adapt"


def collect_inputs() -> argparse.Namespace:
    print("VLA Factory adaptation workflow")
    print("Press Enter to accept a default. The Judge selects all evaluation semantics.\n")
    upstream_repo = _ask("Upstream repository", DEFAULTS["upstream_repo"])
    vla = _ask("VLA name", DEFAULTS["vla"])
    checkpoint = _ask("Checkpoint source (HF ID/URL, ModelScope URL, or local directory)", DEFAULTS["checkpoint"])
    factory_base = _ask("Factory base revision", DEFAULTS["factory_base"])
    benchmark = _ask("Benchmark", DEFAULTS["benchmark"])
    opencode_model = _choose_opencode_model(DEFAULTS["opencode_model"])
    max_repairs_raw = _ask("Maximum automatic repair attempts", DEFAULTS["max_repairs"])
    try:
        max_repairs = int(max_repairs_raw)
        if max_repairs < 0:
            raise ValueError
    except ValueError as error:
        raise ValueError("maximum automatic repair attempts must be a non-negative integer") from error
    job_id = _ask("Job ID", _job_id(vla))
    default_dir = str(Path("runs") / job_id)
    job_dir = _ask("Job directory", default_dir)
    return argparse.Namespace(
        job_dir=job_dir,
        job_id=job_id,
        upstream_repo=upstream_repo,
        vla=vla,
        checkpoint=checkpoint,
        factory_base=factory_base,
        benchmark=benchmark,
        opencode_model=opencode_model,
        max_repairs=max_repairs,
    )


def _non_negative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", metavar="JOB_DIR",
        help="resume an interrupted job without repeating the interactive setup",
    )
    parser.add_argument(
        "--max-repairs", type=_non_negative_integer,
        help="override the saved repair budget when resuming",
    )
    return parser


def _resume_inputs(job_dir: str, max_repairs: int | None) -> argparse.Namespace:
    path = Path(job_dir).expanduser().resolve()
    if not (path / "manifest.json").is_file():
        raise ValueError(f"resume job has no manifest: {path}")
    # Validate the complete manifest now so malformed or mismatched paths fail
    # before the Harness mutates any resumable state.
    manifest = load_manifest(path)
    state_path = path / "workflow-state.json"
    saved_max: int | None = None
    if state_path.is_file():
        state = load_json(state_path)
        candidate = state.get("max_repairs")
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
            saved_max = candidate
    repair_budget = (
        max_repairs if max_repairs is not None
        else saved_max if saved_max is not None
        else int(DEFAULTS["max_repairs"])
    )
    return argparse.Namespace(
        job_dir=str(path), max_repairs=repair_budget,
        opencode_model=manifest.opencode_model, resume=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    options = parser.parse_args(argv)
    try:
        if options.resume:
            values = _resume_inputs(options.resume, options.max_repairs)
            print(f"Resuming VLA Factory job: {values.job_dir}")
            print(f"Repair budget: {values.max_repairs}")
        else:
            if options.max_repairs is not None:
                parser.error("--max-repairs requires --resume")
            values = collect_inputs()
            values.resume = False
        print(f"OpenCode model: {values.opencode_model}")
        print("\nStarting Judge-owned workflow. The final verdict will be written under:")
        print(Path(values.job_dir).resolve() / "artifacts" / "final-verdict.json")
        result = run_workflow(values, max_repairs=values.max_repairs)
    except (EOFError, KeyboardInterrupt):
        print("\nWorkflow cancelled before completion.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"FAIL: unable to start workflow: {error}", file=sys.stderr)
        return 1
    print(f"\nFINAL VERDICT: {result['verdict']} — {result['reason']} (repairs: {result['attempts']})")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
