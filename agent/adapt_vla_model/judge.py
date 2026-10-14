#!/usr/bin/env python3
"""Minimal Judge-owned VLA adaptation harness.

Agents are always launched with OpenCode's non-interactive JSON runner.
The Judge selects a named benchmark before either agent starts; agents cannot
change that recorded selection.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
from dataclasses import replace
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .benchmarks import (
    DEFAULT_BENCHMARK,
    action_schema as benchmark_action_schema,
    factory_contract_document,
    factory_root,
    parity_dataset,
    platform as benchmark_platform,
    resolve as resolve_benchmark,
)
from . import prompts
from .contract import (
    PARITY_GATES,
    VERSION as POLICY_CONTRACT_VERSION,
    document,
)
from .runtime import (
    agent_state_directory,
    DEFAULT_OPENCODE_MODEL,
    dependency_environment,
    JobManifest,
    load_json,
    load_manifest,
    normalized_network_environment,
    OPENCODE_MODELS,
    RetryableInfrastructureError,
    write_json,
    write_manifest,
)


POLICY_SPEC = "policy_spec.json"
FACTORY_BUNDLE = "factory_bundle.json"
JUDGE_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DOWNLOAD_STATE = "checkpoint-download.json"
CHECKPOINT_DOWNLOAD_ATTEMPTS = 4
CHECKPOINT_DOWNLOAD_BACKOFF_SECONDS = (5.0, 15.0, 30.0)
CHECKPOINT_DOWNLOAD_POLL_SECONDS = 2.0
CHECKPOINT_DOWNLOAD_STALL_SECONDS = 180.0


class ParityMismatchError(RuntimeError):
    """A repairable parity gate failure with non-numeric repair guidance."""

    def __init__(self) -> None:
        super().__init__("Judge loss_parity gate failed after both candidate smoke checks passed")

    @staticmethod
    def repair_context() -> dict[str, Any]:
        return {
            "gate": "loss_parity",
            "outcome": "mismatch",
            "assignment": "Factory-side investigation; this is not an attribution of fault",
            "sample": "Judge-owned random sample; do not create, select, modify, or replay a replacement sample",
            "investigation_order": [
                "preprocessing and action/state normalization",
                "action horizon, dimension mapping, and action_is_pad handling",
                "RNG state, eval/train mode, dtype, and device behavior",
                "checkpoint weight loading, model configuration, processor assets, and input mapping",
            ],
            "forbidden": [
                "changing Judge runners, sample selection, tolerances, benchmark contracts, or Reference artifacts",
                "hard-coding a scalar loss or altering a loss solely to match an unknown Judge result",
            ],
        }


def _judge_bootstrap_python() -> str:
    """Resolve the Judge-owned Python 3.12 used only to create candidate venvs.

    This interpreter is not a candidate runtime and is never used to load a
    Reference or Factory model.  Each candidate installer receives it through
    ``JUDGE_BOOTSTRAP_PYTHON`` and must create its own venv at the separate
    Judge-provided ``JUDGE_RUNTIME_DIR``.
    """
    configured = os.environ.get("VLA_JUDGE_BOOTSTRAP_PYTHON")
    candidates: list[str] = []
    if configured:
        candidates.append(str(Path(configured).expanduser()))
    else:
        candidates.extend(filter(None, (shutil.which("python3.12"), sys.executable)))
        # Resolve uv-managed interpreters from the Judge's real home *before*
        # child environments replace HOME with a hermetic per-role directory.
        # This lookup belongs solely to the Judge; installers never probe it.
        uv_install_dir = Path.home() / ".local" / "share" / "uv" / "python"
        candidates.extend(
            str(path) for path in sorted(uv_install_dir.glob("cpython-3.12*/bin/python3.12"))
        )
        uv = shutil.which("uv")
        if uv:
            found = subprocess.run(
                [uv, "python", "find", "3.12"], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
            )
            if found.returncode == 0 and found.stdout.strip():
                candidates.append(found.stdout.strip())
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_file() or not os.access(path, os.X_OK):
            continue
        valid = subprocess.run(
            [str(path), "-c", "import sys; raise SystemExit(sys.version_info[:2] != (3, 12))"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        if valid.returncode == 0:
            return str(path.resolve())
    setting = "VLA_JUDGE_BOOTSTRAP_PYTHON"
    if configured:
        raise RuntimeError(
            f"{setting} must name an executable Python 3.12, got {configured!r}"
        )
    raise RuntimeError(
        "Judge could not resolve a Python 3.12 bootstrap interpreter; "
        f"set {setting} to an executable Python 3.12 path"
    )


def _factory_compliance(job_dir: Path, manifest: JobManifest, artifact_dir: Path) -> dict[str, Any]:
    """Judge-owned proof that an Adapter stayed inside Factory extension points."""
    workspace = artifact_dir.parent
    status = subprocess.run(["git", "-C", str(workspace), "status", "--porcelain"], text=True,
                            capture_output=True, check=True).stdout.splitlines()
    name = manifest.vla.replace("-", "_")
    allow_list = {
        "files": (
            f"vla_factory/model/adapters/{name}.py",
            "pyproject.toml",
            "scripts/install.sh",
            "README.md",
            "README.cn.md",
            "vla_factory/inference/inference_engine.py",
            "vla_factory/assembly/resolve/pipelines.py",
        ),
        "prefixes": (
            f"examples/{name}",
            f"test/l0/test_{name}",
            f"test/l1/test_{name}",
            "vla_factory/assembly/transform/",
            "vla_factory/utils/",
        ),
    }
    changed = [line[3:] for line in status if line[3:] and not line[3:].startswith(".judge/")]
    violations = [path for path in changed if not (
        path in allow_list["files"]
        or any(path.startswith(prefix) for prefix in allow_list["prefixes"])
    )]
    banned: list[str] = []
    boundary_files = [workspace / f"vla_factory/model/adapters/{name}.py"] + [
        workspace / path for path in changed
        if path.startswith("vla_factory/assembly/transform/") or path == "vla_factory/assembly/resolve/pipelines.py"
    ]
    for source in boundary_files:
        if not source.is_file():
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            for value in ([module] if module else names):
                if value and (value in {"socket", "subprocess", "importlib"} or value.startswith(("vla_factory.inference", "vla_factory.deploy", "robotwin"))):
                    banned.append(f"{source.relative_to(workspace)}:{value}")
    assembly = load_json(artifact_dir / "factory_checkpoint/inference_metadata/assembly.json")
    metadata = assembly.get("metadata_ref", {})
    native = metadata.get("inference_needs_base_checkpoint") is False
    report = {"changed_paths": changed, "scope_violations": violations, "banned_imports": sorted(set(banned)),
              "native_bundle": native, "factory_bridge": "factory_policy_server.py"}
    if violations or banned or not native:
        raise RuntimeError(f"Factory architecture compliance failed: {json.dumps(report, sort_keys=True)}")
    return report


def _hermetic_environment(artifact_dir: Path) -> dict[str, str]:
    """Return the Factory Judge environment.

    Both candidate implementations must run on CPU for a reproducible parity
    result. In particular, do not let Factory opportunistically move a large
    model to whatever GPU happens to be visible on the host.
    """
    home = artifact_dir / "judge-hermetic-home"
    home.mkdir(parents=True, exist_ok=True)
    env = normalized_network_environment()
    env.update(dependency_environment(artifact_dir.parent))
    env.update({"HOME": str(home), "HF_HOME": str(home / "hf"),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": ""})
    return env


def _reference_environment(artifact_dir: Path | None = None) -> dict[str, str]:
    """Use the same CPU execution path as the Factory parity runtime."""
    env = normalized_network_environment()
    # Factory's isolated runtime installs CPU Torch.  Hiding CUDA here keeps
    # Reference's model device and mixed-precision kernels comparable.
    env["CUDA_VISIBLE_DEVICES"] = ""
    if artifact_dir is not None:
        env.update(dependency_environment(artifact_dir.parent))
        # Agent-created uv environments can point at /home/opencode, which is
        # a Bubblewrap-only path.  Judge owns a persistent home so a rerun of
        # the installer creates an executable runtime in this namespace.
        home = artifact_dir / "judge-reference-home"
        home.mkdir(parents=True, exist_ok=True)
        env.update({
            "HOME": str(home),
            "HF_HOME": str(home / "hf"),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_DATA_HOME": str(home / ".local" / "share"),
            "XDG_STATE_HOME": str(home / ".local" / "state"),
        })
        # Agent sandboxes see the checkpoint at /judge-inputs/checkpoint;
        # Judge processes are not in that mount namespace. Give reference
        # smoke tests and policy implementations the real locked path instead.
        checkpoint = artifact_dir.parents[2] / "judge-inputs" / "checkpoint"
        if checkpoint.is_dir():
            env["JUDGE_CHECKPOINT_DIR"] = str(checkpoint)
            # Compatibility for existing reference smoke tests. New policies
            # should prefer JUDGE_CHECKPOINT_DIR over this legacy name.
            env["CHECKPOINT"] = str(checkpoint)
    return env


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _revision(repository: Path, ref: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", f"{ref}^{{commit}}"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()


def resolve_factory_base(ref: str) -> str:
    try:
        return _revision(factory_root(), ref)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"factory base is not a commit: {ref}") from error


def _seed_bundle(workspace: Path, benchmark: str = DEFAULT_BENCHMARK, role: str = "reference") -> Path:
    bundle = workspace / ".judge"
    bundle.mkdir(parents=True, exist_ok=True)
    if role == "reference":
        (bundle / "REFERENCE_BUNDLE_CONTRACT.md").write_text(document(benchmark), encoding="utf-8")
        # The Reference implementation may import this Judge-facing protocol
        # for its public type surface. It exposes no Judge procedure.
        shutil.copy2(JUDGE_ROOT / "contract.py", bundle / "policy_contract.py")
        if benchmark_platform(benchmark) == "robotwin":
            # This is the fixed server named by the RoboTwin runtime.json. The
            # remaining Judge helpers are not Agent inputs.
            shutil.copy2(JUDGE_ROOT / "robotwin_policy_server.py", bundle / "robotwin_policy_server.py")
    elif role == "adapter":
        (bundle / "FACTORY_BUNDLE_CONTRACT.md").write_text(
            factory_contract_document(benchmark), encoding="utf-8"
        )
    else:
        raise ValueError(f"unknown agent role: {role}")
    return bundle


def _stage_upstream_source(job_dir: Path, bundle: Path) -> None:
    """Clone the complete, pinned upstream checkout into the Adapter bundle."""
    upstream = job_dir / "workspaces" / "upstream"
    if upstream.is_dir():
        destination = bundle / "upstream_source"
        manifest = load_manifest(job_dir)
        revision = _revision(upstream)
        if destination.exists():
            if not (destination / ".git").exists():
                raise ValueError(f"upstream source destination is incomplete: {destination}")
            if _revision(destination) != revision:
                raise ValueError(
                    f"upstream source destination has the wrong revision: {destination}"
                )
            return
        # --no-local avoids a shared-object clone whose source checkout may be
        # removed after preparation. The checkout is pinned below even if the
        # source repository's default branch later changes.
        _run(["git", "clone", "--no-local", str(upstream), str(destination)])
        _run(["git", "-C", str(destination), "checkout", "--detach", revision])
        _run(["git", "-C", str(destination), "remote", "set-url", "origin", manifest.upstream_repo])


def _remove_missing_worktree(repository: Path, path: Path) -> None:
    if path.exists():
        return
    listing = subprocess.run(
        ["git", "-C", str(repository), "worktree", "list", "--porcelain"],
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    if f"worktree {path}" in listing:
        _run(["git", "-C", str(repository), "worktree", "remove", "--force", str(path)])


def prepare_workspaces(job_dir: Path) -> None:
    manifest = load_manifest(job_dir)
    workspaces = job_dir / "workspaces"
    upstream, factory = workspaces / "upstream", workspaces / "factory"
    lock = job_dir / "workspace-lock.json"
    if lock.exists():
        raise ValueError(f"job workspaces already exist: {workspaces}")
    if upstream.exists() and not (upstream / ".git").exists():
        raise ValueError(f"upstream workspace is incomplete: {upstream}")
    if factory.exists() and not (factory / ".git").exists():
        raise ValueError(f"factory workspace is incomplete: {factory}")
    workspaces.mkdir(exist_ok=True)
    try:
        if not upstream.exists():
            _run(["git", "clone", manifest.upstream_repo, str(upstream)])
        source = factory_root()
        if factory.exists():
            if _revision(factory) != manifest.factory_base:
                raise ValueError(f"factory workspace has the wrong revision: {factory}")
        else:
            _remove_missing_worktree(source, factory)
            _run([
                "git", "-C", str(source), "worktree", "add", "--detach",
                str(factory), manifest.factory_base,
            ])
        _seed_bundle(upstream, manifest.benchmark_name, "reference")
        _seed_bundle(factory, manifest.benchmark_name, "adapter")
        _stage_upstream_source(job_dir, factory / ".judge")
        # The lock is the commit marker for workspace preparation. Write it
        # only after every resumable step has completed successfully.
        write_json(lock, {
            "upstream": {"path": str(upstream), "revision": _revision(upstream)},
            "factory": {"path": str(factory), "revision": _revision(factory)},
        })
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("failed to prepare Judge-owned workspaces") from error


def workspace_for_role(job_dir: Path, role: str) -> Path:
    if not (job_dir / "workspace-lock.json").is_file():
        raise ValueError("workspaces are not prepared; rerun `init` to resume this job")
    return job_dir / "workspaces" / {"reference": "upstream", "adapter": "factory"}[role]


def bundle_for_role(job_dir: Path, role: str) -> Path:
    manifest = load_manifest(job_dir)
    return _seed_bundle(workspace_for_role(job_dir, role), manifest.benchmark_name, role)


def _remove_residue(path: Path) -> None:
    """Remove one exact job-owned residue path without following symlinks."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _clear_agent_residue(job_dir: Path, role: str, workspace: Path, artifact_dir: Path) -> None:
    """Restore a role's job-owned workspace before a fresh Agent attempt.

    Agent executions can leave source edits, virtual environments, converted
    checkpoints, and partial scripts behind. A retry must not mistake any of
    those for new work. Judge-owned bundle inputs remain in place.
    """
    expected_workspace = (job_dir / "workspaces" / {"reference": "upstream", "adapter": "factory"}[role]).resolve()
    if workspace.resolve() != expected_workspace or artifact_dir.resolve() != (workspace / ".judge").resolve():
        raise ValueError("refusing to clear residue outside the job-owned workspace")

    agent_state = agent_state_directory(workspace)
    if agent_state.exists() or agent_state.is_symlink():
        _remove_residue(agent_state)

    # The worktree is created by this Harness, not supplied by the user. Keep
    # .judge while resetting code, then remove old Agent outputs below.
    _run(["git", "-C", str(workspace), "reset", "--hard", "HEAD"])
    _run(["git", "-C", str(workspace), "clean", "-fd", "-e", ".judge"])

    keep = {
        "reference": {
            "REFERENCE_BUNDLE_CONTRACT.md",
            "policy_contract.py",
            "robotwin_policy_server.py",
        },
        "adapter": {
            "FACTORY_BUNDLE_CONTRACT.md",
            "upstream_source",
        },
    }[role]
    for child in artifact_dir.iterdir():
        if child.name not in keep:
            _remove_residue(child)

    if role == "adapter":
        # `scripts/install.sh --model <name>` creates these ignored directories
        # in the Factory worktree; git clean deliberately leaves ignored files.
        name = load_manifest(job_dir).vla.replace("-", "_")
        for path in (workspace / f".{name}", workspace / ".local-deps", workspace / "outputs"):
            if path.exists() or path.is_symlink():
                _remove_residue(path)


def init_job(args: argparse.Namespace) -> None:
    job_dir = Path(args.job_dir).resolve()
    if job_dir.exists():
        if (job_dir / "manifest.json").is_file() and not (job_dir / "workspace-lock.json").exists():
            manifest = load_manifest(job_dir)
            # An interrupted initial download leaves the immutable manifest and
            # job-local partial files in place, but cannot write checkpoint_lock
            # until every file is complete. Resume that exact acquisition before
            # creating either candidate workspace.
            if manifest.checkpoint_lock is None:
                lock = acquire_checkpoint(job_dir, manifest.checkpoint)
                manifest = replace(manifest, checkpoint_lock=lock)
                write_manifest(job_dir / "manifest.json", manifest)
            factory_base = resolve_factory_base(manifest.factory_base)
            if factory_base != manifest.factory_base:
                manifest = replace(manifest, factory_base=factory_base)
                write_manifest(job_dir / "manifest.json", manifest)
            prepare_workspaces(job_dir)
            print(job_dir)
            return
        if (
            getattr(args, "resume", False)
            and (job_dir / "manifest.json").is_file()
            and (job_dir / "workspace-lock.json").is_file()
        ):
            # A later-stage resume reuses the immutable manifest/workspace
            # locks. run_workflow will safely start a fresh role execution.
            load_manifest(job_dir)
            print(job_dir)
            return
        raise ValueError(f"job directory already exists: {job_dir}")
    benchmark = resolve_benchmark(args.benchmark)
    manifest = JobManifest(
        job_id=args.job_id,
        upstream_repo=args.upstream_repo,
        vla=args.vla,
        checkpoint=args.checkpoint,
        factory_base=resolve_factory_base(args.factory_base),
        benchmark_name=benchmark.name,
        benchmark_revision=benchmark.revision,
        comparison=benchmark.comparison,
        opencode_model=getattr(args, "opencode_model", DEFAULT_OPENCODE_MODEL),
    )
    (job_dir / "artifacts").mkdir(parents=True)
    write_manifest(job_dir / "manifest.json", manifest)
    write_json(job_dir / "workflow-state.json", {
        "version": "workflow-state/v1", "stage": "init",
        "repairs_used": 0,
        "max_repairs": getattr(args, "max_repairs", 2),
    })
    lock = acquire_checkpoint(job_dir, manifest.checkpoint)
    write_manifest(job_dir / "manifest.json", replace(manifest, checkpoint_lock=lock))
    prepare_workspaces(job_dir)
    print(job_dir)


def parse_checkpoint_source(value: str) -> dict[str, str]:
    """Parse stable IDs plus Hub URLs without asking an Agent to infer a host."""
    source = value.strip()
    if not source:
        raise ValueError("checkpoint source is empty")
    local = Path(source).expanduser()
    if local.is_dir():
        return {"provider": "local", "model_id": str(local.resolve()), "revision": "local"}
    parsed = urlparse(source)
    host = parsed.netloc.lower()
    if host in {"modelscope.cn", "www.modelscope.cn", "modelscope.ai", "www.modelscope.ai"}:
        parts = [part for part in parsed.path.split("/") if part]
        try:
            index = parts.index("models")
            model_id = "/".join(parts[index + 1:index + 3])
        except ValueError:
            model_id = ""
        if not model_id or model_id.count("/") != 1:
            raise ValueError(f"invalid ModelScope checkpoint URL: {source}")
        return {"provider": "modelscope", "model_id": model_id,
                "revision": parse_qs(parsed.query).get("revision", ["master"])[0]}
    if host in {"huggingface.co", "www.huggingface.co"}:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"invalid Hugging Face checkpoint URL: {source}")
        revision = "main"
        if len(parts) >= 4 and parts[2] in {"tree", "resolve"}:
            revision = parts[3]
        return {"provider": "huggingface", "model_id": "/".join(parts[:2]), "revision": revision}
    if source.startswith("modelscope://"):
        model_id, _, revision = source.removeprefix("modelscope://").partition("@")
        return {"provider": "modelscope", "model_id": model_id, "revision": revision or "master"}
    if source.startswith("hf://"):
        model_id, _, revision = source.removeprefix("hf://").partition("@")
        return {"provider": "huggingface", "model_id": model_id, "revision": revision or "main"}
    # A plain `owner/model` remains Hugging Face-compatible for existing jobs.
    if source.count("/") == 1:
        return {"provider": "huggingface", "model_id": source, "revision": "main"}
    raise ValueError("checkpoint must be a Hugging Face ID/URL, ModelScope ID/URL, or local directory")


def acquire_checkpoint(job_dir: Path, checkpoint: str) -> dict[str, Any]:
    """Acquire or resume one immutable source snapshot before Agents start."""

    target = job_dir / "judge-inputs" / "checkpoint"
    source = parse_checkpoint_source(checkpoint)
    if target.is_symlink():
        if not (
            source["provider"] == "local"
            and target.resolve() == Path(source["model_id"]).resolve()
        ):
            raise ValueError(f"Judge checkpoint target is an unexpected symlink: {target}")
    elif target.exists() and not target.is_dir():
        raise ValueError(f"Judge checkpoint target is not a resumable directory: {target}")
    try:
        if source["provider"] == "local":
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(source["model_id"], target_is_directory=True)
        elif source["provider"] == "huggingface":
            source = _locked_huggingface_source(job_dir, checkpoint, source)
            _download_huggingface_snapshot(source, target)
        else:
            # Keep this provider dependency out of the Harness interpreter and
            # Agent sandboxes. It is a Judge-owned, job-local downloader only.
            runtime = job_dir / "judge-inputs" / ".modelscope-downloader"
            python = runtime / "bin" / "python"
            if not python.is_file():
                _run([sys.executable, "-m", "venv", str(runtime)])
                _run([str(python), "-m", "pip", "install", "modelscope"])
            program = (
                "from modelscope import snapshot_download; import sys; "
                "snapshot_download(model_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])"
            )
            _run([str(python), "-c", program, source["model_id"], source["revision"], str(target)])
    except RetryableInfrastructureError:
        raise
    except Exception as error:
        # Partial Hub files are deliberately retained. Both supported download
        # clients use the same local_dir on the next init attempt and can resume
        # instead of discarding multi-gigabyte shards.
        raise RuntimeError(f"checkpoint {checkpoint!r} could not be acquired") from error
    scan_root = target.resolve()
    files = sorted(
        {str(path.relative_to(scan_root)): path.stat().st_size for path in scan_root.rglob("*") if path.is_file()}.items()
    )
    if not files:
        raise RuntimeError(f"checkpoint {checkpoint!r} download is empty")
    state_path = job_dir / "judge-inputs" / CHECKPOINT_DOWNLOAD_STATE
    if state_path.is_file():
        state = load_json(state_path)
        state.update({"status": "complete", "completed_at": int(time.time())})
        write_json(state_path, state)
    return {"source": checkpoint, **source, "path": "judge-inputs/checkpoint", "files": dict(files)}


def _locked_huggingface_source(
    job_dir: Path, checkpoint: str, source: dict[str, str],
) -> dict[str, str]:
    """Resolve a floating Hub revision once and persist it across restarts."""
    state_path = job_dir / "judge-inputs" / CHECKPOINT_DOWNLOAD_STATE
    if state_path.is_file():
        state = load_json(state_path)
        expected = {
            "source": checkpoint,
            "provider": "huggingface",
            "model_id": source["model_id"],
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"checkpoint download state does not match the job manifest: {state_path}"
            )
        revision = state.get("revision")
        if not isinstance(revision, str) or not revision:
            raise ValueError(f"checkpoint download state has no locked revision: {state_path}")
        return {**source, "revision": revision}

    # Jobs created by older Harness versions have no download-state file, but
    # huggingface_hub records the resolved commit as the first line of every
    # local-dir metadata file. Recover it so resuming never re-resolves a moving
    # branch and mixes revisions in one checkpoint directory.
    target = job_dir / "judge-inputs" / "checkpoint"
    revision = _local_huggingface_revision(target)
    if revision is None:
        from huggingface_hub import HfApi

        info = HfApi().model_info(source["model_id"], revision=source["revision"])
        revision = info.sha
    locked = {**source, "revision": revision}
    write_json(state_path, {
        "version": "checkpoint-download/v1",
        "status": "downloading",
        "source": checkpoint,
        **locked,
    })
    return locked


def _local_huggingface_revision(target: Path) -> str | None:
    metadata_dir = target / ".cache" / "huggingface" / "download"
    if not metadata_dir.is_dir():
        return None
    revisions: set[str] = set()
    for metadata in metadata_dir.rglob("*.metadata"):
        try:
            revision = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
        except (OSError, IndexError, UnicodeDecodeError):
            continue
        if len(revision) == 40 and all(character in "0123456789abcdef" for character in revision.lower()):
            revisions.add(revision)
    if len(revisions) > 1:
        raise ValueError(
            f"partial Hugging Face checkpoint contains multiple revisions: {sorted(revisions)}"
        )
    return next(iter(revisions), None)


def _checkpoint_progress(path: Path) -> int:
    """Return downloaded bytes for progress/stall detection."""
    if not path.is_dir():
        return 0
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            stat = item.stat()
        except FileNotFoundError:
            continue
        total += stat.st_size
    return total


def _terminate_download(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _download_huggingface_snapshot(source: dict[str, str], target: Path) -> None:
    """Download via resumable HTTP with bounded retries and a stall watchdog."""
    target.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(os.environ.get("VLA_CHECKPOINT_DOWNLOAD_WORKERS", "2")))
    attempts = max(1, int(os.environ.get(
        "VLA_CHECKPOINT_DOWNLOAD_ATTEMPTS", str(CHECKPOINT_DOWNLOAD_ATTEMPTS),
    )))
    stall_seconds = max(10.0, float(os.environ.get(
        "VLA_CHECKPOINT_STALL_TIMEOUT", str(CHECKPOINT_DOWNLOAD_STALL_SECONDS),
    )))
    program = (
        "from huggingface_hub import snapshot_download; import sys; "
        "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], "
        "local_dir=sys.argv[3], max_workers=int(sys.argv[4]))"
    )
    command = [
        sys.executable, "-c", program, source["model_id"], source["revision"],
        str(target), str(workers),
    ]
    environment = {
        **os.environ,
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "HF_HUB_DOWNLOAD_TIMEOUT": os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "60"),
        "HF_HUB_ETAG_TIMEOUT": os.environ.get("HF_HUB_ETAG_TIMEOUT", "30"),
    }

    last_reason = "download failed"
    for attempt in range(1, attempts + 1):
        resumed_bytes = _checkpoint_progress(target)
        print(
            f"[judge:init] checkpoint download attempt {attempt}/{attempts}; "
            f"resuming from {resumed_bytes} local bytes"
        )
        process = subprocess.Popen(command, env=environment)
        progress = _checkpoint_progress(target)
        changed_at = time.monotonic()
        stalled = False
        try:
            while process.poll() is None:
                time.sleep(CHECKPOINT_DOWNLOAD_POLL_SECONDS)
                current = _checkpoint_progress(target)
                if current != progress:
                    progress = current
                    changed_at = time.monotonic()
                elif time.monotonic() - changed_at >= stall_seconds:
                    stalled = True
                    last_reason = (
                        f"checkpoint download made no filesystem progress for "
                        f"{stall_seconds:g}s"
                    )
                    print(f"[judge:init] {last_reason}; restarting downloader")
                    _terminate_download(process)
                    break
        except BaseException:
            _terminate_download(process)
            raise

        if not stalled and process.returncode == 0:
            return
        if not stalled:
            last_reason = f"checkpoint downloader exited with code {process.returncode}"
        if attempt < attempts:
            delay = CHECKPOINT_DOWNLOAD_BACKOFF_SECONDS[
                min(attempt - 1, len(CHECKPOINT_DOWNLOAD_BACKOFF_SECONDS) - 1)
            ]
            print(f"[judge:init] {last_reason}; retrying in {delay:g}s")
            time.sleep(delay)

    raise RetryableInfrastructureError(last_reason, attempts=attempts)


def run_agent(args: argparse.Namespace) -> None:
    job_dir = Path(args.job_dir).resolve()
    workspace = workspace_for_role(job_dir, args.role)
    artifact_dir = bundle_for_role(job_dir, args.role)
    _clear_agent_residue(job_dir, args.role, workspace, artifact_dir)
    {"reference": prompts.run_reference_agent, "adapter": prompts.run_adapter_agent}[args.role](job_dir, workspace, artifact_dir)


def _log_tail(path: Path, limit: int = 12_000) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def _failure_candidate(message: str) -> str | None:
    """Route candidate-owned failures; environment failures never go to an LLM."""
    lowered = message.lower()
    # Do not classify by generic path words such as ``workspace``: every
    # Judge log path contains it, including candidate-owned smoke failures.
    if any(token in lowered for token in (
        "dataset", "factory base", "judge runtime", "permission denied", "no space left",
    )):
        return None
    if "reference" in lowered or "policy_spec" in lowered or "policy_impl" in lowered:
        return "reference"
    if any(token in lowered for token in (
        "factory", "parity", "factory_bundle", "assembly", "weights.json", "model.pt",
    )):
        return "adapter"
    return None


def _failure_report(job_dir: Path, *, attempt: int, stage: str, error: BaseException,
                    candidate: str | None = None) -> dict[str, Any]:
    """Create the complete, bounded diagnostic object supplied to repair Agents."""
    message = str(error)
    # A parity mismatch is an observation about two candidates, not an
    # attribution. Its owner is selected only by the read-only diagnosis step.
    if isinstance(error, (ParityMismatchError, RetryableInfrastructureError)):
        candidate = None
    else:
        candidate = candidate if candidate is not None else _failure_candidate(message)
    roots = [job_dir / "artifacts" / "parity"]
    if candidate in {"reference", "adapter"} and (job_dir / "workspace-lock.json").is_file():
        roots.append(bundle_for_role(job_dir, candidate))
    logs = [
        {"path": str(path), "tail": tail}
        for root in roots if root.is_dir()
        for path in sorted(root.glob("*.log"))
        if (tail := _log_tail(path))
    ]
    report = {
        "version": "judge-failure/v1", "attempt": attempt, "stage": stage,
        "candidate": candidate, "repairable": candidate is not None,
        "error_type": type(error).__name__, "error_message": message,
        "logs": logs,
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }
    if isinstance(error, ParityMismatchError):
        report["repair_context"] = error.repair_context()
    if isinstance(error, RetryableInfrastructureError):
        report["infrastructure_context"] = {
            "retryable": True,
            "provider_attempts": error.attempts,
            "status_code": error.status_code,
        }
    return report


def _record_failure(job_dir: Path, report: dict[str, Any]) -> Path:
    path = job_dir / "artifacts" / "attempts" / f"{report['attempt']:02d}-failure.json"
    sequence = 2
    while path.exists():
        path = job_dir / "artifacts" / "attempts" / f"{report['attempt']:02d}-{sequence:02d}-failure.json"
        sequence += 1
    write_json(path, report)
    candidate = report.get("candidate")
    # This copy is inside the sealed candidate workspace; the Agent cannot see
    # job-level artifacts otherwise. Its content is Judge-owned evidence.
    if candidate in {"reference", "adapter"}:
        write_json(bundle_for_role(job_dir, candidate) / "judge-failure-report.json", report)
    return path


def _repair_inventory(workspace: Path, artifact_dir: Path) -> dict[str, Any]:
    """Return bounded, role-local context for a fresh repair Agent session.

    Repair Agents reuse files but not LLM conversation state.  Listing the
    meaningful handoff artifacts and current git changes lets the new session
    orient itself without scanning dependency trees, old runtimes, or the
    copied upstream checkout.
    """
    skipped_dirs = {
        ".git", "upstream_source", "judge-runtime", "judge-reference-home",
        "judge-hermetic-home", "_hf_cache_offline", "__pycache__",
    }
    files: list[dict[str, Any]] = []
    truncated = False
    for root, dirs, names in os.walk(artifact_dir):
        dirs[:] = sorted(name for name in dirs if name not in skipped_dirs)
        for name in sorted(names):
            path = Path(root) / name
            if not (path.is_file() or path.is_symlink()):
                continue
            try:
                size = path.lstat().st_size
            except OSError:
                continue
            files.append({"path": str(path.relative_to(workspace)), "bytes": size})
            if len(files) == 160:
                truncated = True
                break
        if truncated:
            break
    status = subprocess.run(
        ["git", "-C", str(workspace), "status", "--porcelain"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return {
        "workspace": str(workspace),
        "artifact_dir": str(artifact_dir),
        "git_status": status.stdout.splitlines() if status.returncode == 0 else [],
        "artifact_files": files,
        "artifact_files_truncated": truncated,
        "excluded": sorted(skipped_dirs),
    }


def _parity_repair_inputs(job_dir: Path) -> tuple[tuple[Path, Path], ...]:
    """Expose cross-role parity evidence only to a targeted repair session.

    The Agent's own workspace remains writable.  The other candidate and the
    diagnosis are mounted under explicit Judge-owned paths, so a repair can
    reproduce a Reference semantic setup without gaining authority to edit it.
    """
    return (
        (job_dir / "workspaces" / "upstream" / ".judge", Path("/judge-inputs/reference")),
        (job_dir / "workspaces" / "factory" / ".judge", Path("/judge-inputs/factory")),
        (job_dir / "artifacts" / "parity-diagnosis", Path("/judge-inputs/parity-diagnosis")),
        (job_dir / "artifacts" / "parity", Path("/judge-inputs/parity")),
    )


def _repair_agent(job_dir: Path, role: str, report: dict[str, Any]) -> None:
    """One scoped correction pass. Unlike the first pass, retain partial work."""
    workspace = workspace_for_role(job_dir, role)
    artifact_dir = bundle_for_role(job_dir, role)
    parity_diagnosis = report.get("parity_diagnosis")
    feedback_object = {
        "role": role,
        "previous_session": "ended; inspect the retained role files before editing",
        "failure_report": report,
        "current_role_inventory": _repair_inventory(workspace, artifact_dir),
    }
    readonly_inputs: tuple[tuple[Path, Path], ...] = ()
    if isinstance(parity_diagnosis, dict):
        feedback_object["parity_repair_handoff"] = parity_diagnosis["repair_handoffs"][role]
        feedback_object["parity_readonly_inputs"] = {
            "diagnosis": "/judge-inputs/parity-diagnosis/diagnosis.json",
            "reference_artifacts": "/judge-inputs/reference",
            "factory_artifacts": "/judge-inputs/factory",
            "judge_parity_evidence": "/judge-inputs/parity",
        }
        readonly_inputs = _parity_repair_inputs(job_dir)
    feedback = json.dumps(feedback_object, indent=2, sort_keys=True)
    {"reference": prompts.run_reference_agent, "adapter": prompts.run_adapter_agent}[role](
        job_dir, workspace, artifact_dir, feedback, readonly_inputs,
    )


def _repair_roles(job_dir: Path, roles: tuple[str, ...], report: dict[str, Any]) -> dict[str, BaseException]:
    """Repair independent selected roles concurrently and return role failures."""
    errors: dict[str, BaseException] = {}
    with ThreadPoolExecutor(max_workers=len(roles), thread_name_prefix="vla-repair") as executor:
        futures = {role: executor.submit(_repair_agent, job_dir, role, report) for role in roles}
        for role in roles:
            try:
                futures[role].result()
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                errors[role] = error
    return errors


def _parity_diagnosis(job_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Run the only cross-workspace, read-only attribution step."""
    diagnosis = prompts.run_diagnosis(job_dir, report)
    owner = diagnosis.get("owner")
    confidence = diagnosis.get("confidence")
    evidence = diagnosis.get("evidence")
    checks = diagnosis.get("next_checks")
    handoffs = diagnosis.get("repair_handoffs")
    if owner not in {"reference", "adapter", "both", "inconclusive"}:
        raise RuntimeError("parity diagnosis returned an invalid owner")
    if (diagnosis.get("version") != "parity-diagnosis/v1" or
            not isinstance(diagnosis.get("summary"), str) or
            confidence not in {"low", "medium", "high"} or not isinstance(evidence, list) or not isinstance(checks, list) or
            not isinstance(handoffs, dict)):
        raise RuntimeError("parity diagnosis returned an invalid schema")
    selected = set(("reference", "adapter") if owner == "both" else (() if owner == "inconclusive" else (owner,)))
    if set(handoffs) != selected:
        raise RuntimeError("parity diagnosis handoffs must match the selected repair role(s)")
    for role, handoff in handoffs.items():
        _validate_parity_handoff(role, handoff)
    write_json(job_dir / "artifacts" / "parity-diagnosis" / "diagnosis.json", diagnosis)
    return diagnosis


def _validate_parity_source(source: Any) -> None:
    if not isinstance(source, dict) or source.get("role") not in {"reference", "adapter", "judge"}:
        raise RuntimeError("parity diagnosis source must name reference, adapter, or judge")
    path = source.get("path")
    if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise RuntimeError("parity diagnosis source path must be a safe relative path")
    if "line_hint" in source and not isinstance(source["line_hint"], str):
        raise RuntimeError("parity diagnosis source line_hint must be a string")


def _validate_parity_handoff(role: str, handoff: Any) -> None:
    """Validate the stable envelope while leaving VLA facts open-ended."""
    if not isinstance(handoff, dict) or not isinstance(handoff.get("objective"), str):
        raise RuntimeError(f"parity diagnosis {role} handoff has no objective")
    facts, differences, order = (handoff.get("semantic_facts"), handoff.get("differences"),
                                 handoff.get("investigation_order"))
    if not all(isinstance(value, list) for value in (facts, differences, order)):
        raise RuntimeError(f"parity diagnosis {role} handoff lists are invalid")
    if not all(isinstance(item, str) for item in order):
        raise RuntimeError(f"parity diagnosis {role} investigation order is invalid")
    for fact in facts:
        if not isinstance(fact, dict) or not isinstance(fact.get("key"), str) or "value" not in fact:
            raise RuntimeError(f"parity diagnosis {role} semantic fact is invalid")
        _validate_parity_source(fact.get("source"))
    for difference in differences:
        required = ("dimension", "reference", "factory", "impact", "evidence")
        if not isinstance(difference, dict) or not all(key in difference for key in required):
            raise RuntimeError(f"parity diagnosis {role} difference is invalid")
        if not all(isinstance(difference[key], str) for key in required[:-1]) or not isinstance(difference["evidence"], list):
            raise RuntimeError(f"parity diagnosis {role} difference fields are invalid")
        for source in difference["evidence"]:
            _validate_parity_source(source)


def _final_verdict(job_dir: Path, verdict: str, *, attempts: int, reason: str,
                   last_failure: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "verdict": verdict, "attempts": attempts, "reason": reason,
        "completed_at": int(time.time()),
    }
    if last_failure is not None:
        result["last_failure"] = {
            key: last_failure[key] for key in ("stage", "candidate", "error_type", "error_message")
        }
    write_json(job_dir / "artifacts" / "final-verdict.json", result)
    return result


def run_workflow(init_args: argparse.Namespace, *, max_repairs: int = 2) -> dict[str, Any]:
    """Run the entire adaptation workflow and always return one final verdict."""
    if max_repairs < 0:
        raise ValueError("max_repairs must be non-negative")
    job_dir = Path(init_args.job_dir).resolve()
    repairs = 0
    if getattr(init_args, "resume", False):
        state_path = job_dir / "workflow-state.json"
        if state_path.is_file():
            saved_repairs = load_json(state_path).get("repairs_used", 0)
            if isinstance(saved_repairs, int) and not isinstance(saved_repairs, bool):
                repairs = max(0, saved_repairs)
    last_failure: dict[str, Any] | None = None

    try:
        init_job(init_args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        # A failed init has no safe candidate workspace for LLM repair.
        job_dir.mkdir(parents=True, exist_ok=True)
        last_failure = _failure_report(job_dir, attempt=repairs, stage="init", error=error)
        _record_failure(job_dir, last_failure)
        infrastructure = isinstance(error, RetryableInfrastructureError)
        write_json(job_dir / "workflow-state.json", {
            "version": "workflow-state/v1", "stage": "finished",
            "repairs_used": repairs, "max_repairs": max_repairs,
        })
        return _final_verdict(
            job_dir, "INCOMPLETE" if infrastructure else "FAIL", attempts=repairs,
            reason=(
                "checkpoint infrastructure retries exhausted"
                if infrastructure else "initialisation failed"
            ),
            last_failure=last_failure,
        )

    def save_state(stage: str) -> None:
        write_json(job_dir / "workflow-state.json", {
            "version": "workflow-state/v1", "stage": stage,
            "repairs_used": repairs, "max_repairs": max_repairs,
        })

    def infrastructure_verdict(stage: str, error: RetryableInfrastructureError) -> dict[str, Any]:
        nonlocal last_failure
        last_failure = _failure_report(job_dir, attempt=repairs, stage=stage, error=error)
        _record_failure(job_dir, last_failure)
        save_state("finished")
        return _final_verdict(
            job_dir, "INCOMPLETE", attempts=repairs,
            reason="OpenCode infrastructure retries exhausted", last_failure=last_failure,
        )

    # The two first passes own disjoint workspaces and OpenCode state, and share
    # only read-only Judge inputs. Run them concurrently; repairs remain ordered
    # because they preserve and mutate one failed role's retained work.
    roles = ("reference", "adapter")
    save_state("initial-agents:parallel")
    initial_errors: dict[str, BaseException] = {}
    with ThreadPoolExecutor(max_workers=len(roles), thread_name_prefix="vla-agent") as executor:
        futures = {
            role: executor.submit(run_agent, argparse.Namespace(job_dir=str(job_dir), role=role))
            for role in roles
        }
        for role in roles:
            try:
                futures[role].result()
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                initial_errors[role] = error

    # Infrastructure failure takes precedence over candidate repair: both roles
    # have already stopped, and retrying a provider outage must not spend budget.
    for role in roles:
        error = initial_errors.get(role)
        if isinstance(error, RetryableInfrastructureError):
            return infrastructure_verdict(f"infrastructure:agent:{role}", error)

    # Repair candidate-owned initial failures in a deterministic role order.
    for role in roles:
        error = initial_errors.get(role)
        if error is not None:
            last_failure = _failure_report(job_dir, attempt=repairs, stage=f"agent:{role}", error=error, candidate=role)
            _record_failure(job_dir, last_failure)
            while repairs < max_repairs:
                repairs += 1
                save_state(f"repair:{role}:{repairs}")
                try:
                    _repair_agent(job_dir, role, last_failure)
                    last_failure = None
                    break
                except RetryableInfrastructureError as infrastructure_error:
                    repairs -= 1
                    return infrastructure_verdict(
                        f"infrastructure:repair:{role}", infrastructure_error,
                    )
                except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as repair_error:
                    last_failure = _failure_report(job_dir, attempt=repairs, stage=f"repair:{role}",
                                                   error=repair_error, candidate=role)
                    _record_failure(job_dir, last_failure)
            if last_failure is not None:
                save_state("finished")
                return _final_verdict(job_dir, "FAIL", attempts=repairs,
                                      reason=f"{role} did not produce a valid candidate", last_failure=last_failure)

    while True:
        save_state("judge")
        try:
            judge_result = judge(argparse.Namespace(job_dir=str(job_dir)))
            if judge_result["verdict"] != "PASS":
                raise ParityMismatchError()
            save_state("finished")
            return _final_verdict(job_dir, "PASS", attempts=repairs, reason="Judge passed", last_failure=None)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
            last_failure = _failure_report(job_dir, attempt=repairs, stage="judge", error=error)
            _record_failure(job_dir, last_failure)
            if isinstance(error, ParityMismatchError):
                try:
                    diagnosis = _parity_diagnosis(job_dir, last_failure)
                except RetryableInfrastructureError as infrastructure_error:
                    return infrastructure_verdict(
                        "infrastructure:parity-diagnosis", infrastructure_error,
                    )
                owner = diagnosis["owner"]
                if owner == "inconclusive":
                    save_state("finished")
                    return _final_verdict(job_dir, "INCOMPLETE", attempts=repairs,
                                          reason="parity diagnosis inconclusive", last_failure=last_failure)
                # One parity diagnosis and its targeted role repairs form one
                # repair cycle, even when the diagnosis selects both roles.
                roles = ("reference", "adapter") if owner == "both" else (owner,)
                if repairs >= max_repairs:
                    save_state("finished")
                    return _final_verdict(job_dir, "FAIL", attempts=repairs,
                                          reason="maximum repair attempts reached", last_failure=last_failure)
                repairs += 1
                repair_stage = "repair:both" if len(roles) > 1 else f"repair:{roles[0]}"
                save_state(f"{repair_stage}:{repairs}")
                repair_errors = _repair_roles(
                    job_dir, roles, {**last_failure, "parity_diagnosis": diagnosis},
                )
                infrastructure_failure = next((
                    (role, error) for role, error in repair_errors.items()
                    if isinstance(error, RetryableInfrastructureError)
                ), None)
                if infrastructure_failure is not None:
                    role, infrastructure_error = infrastructure_failure
                    repairs -= 1
                    return infrastructure_verdict(
                        f"infrastructure:repair:{role}", infrastructure_error,
                    )
                for role in roles:
                    repair_error = repair_errors.get(role)
                    if repair_error is None:
                        continue
                    last_failure = _failure_report(
                        job_dir, attempt=repairs, stage=f"repair:{role}",
                        error=repair_error, candidate=role,
                    )
                    _record_failure(job_dir, last_failure)
                    if repairs >= max_repairs:
                        save_state("finished")
                        return _final_verdict(job_dir, "FAIL", attempts=repairs,
                                              reason="maximum repair attempts reached", last_failure=last_failure)
                continue
            candidate = last_failure["candidate"]
            if candidate not in {"reference", "adapter"}:
                save_state("finished")
                return _final_verdict(job_dir, "INCOMPLETE", attempts=repairs,
                                      reason="Judge-side or environment failure", last_failure=last_failure)
            if repairs >= max_repairs:
                save_state("finished")
                return _final_verdict(job_dir, "FAIL", attempts=repairs,
                                      reason="maximum repair attempts reached", last_failure=last_failure)
            repairs += 1
            save_state(f"repair:{candidate}:{repairs}")
            try:
                _repair_agent(job_dir, candidate, last_failure)
            except RetryableInfrastructureError as infrastructure_error:
                repairs -= 1
                return infrastructure_verdict(
                    f"infrastructure:repair:{candidate}", infrastructure_error,
                )
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as repair_error:
                last_failure = _failure_report(job_dir, attempt=repairs, stage=f"repair:{candidate}",
                                               error=repair_error, candidate=candidate)
                _record_failure(job_dir, last_failure)
                if repairs >= max_repairs:
                    save_state("finished")
                    return _final_verdict(job_dir, "FAIL", attempts=repairs,
                                          reason="maximum repair attempts reached", last_failure=last_failure)


def policy_spec(path: Path, manifest: JobManifest) -> dict[str, Any]:
    spec = load_json(path / POLICY_SPEC)
    required = ("contract_version", "implementation", "install_script", "smoke_test",
                "input_schema", "action_schema")
    missing = [field for field in required if field not in spec]
    if missing:
        raise ValueError(f"{path / POLICY_SPEC}: missing {', '.join(missing)}")
    if spec["contract_version"] != POLICY_CONTRACT_VERSION:
        raise ValueError(f"{path / POLICY_SPEC}: unsupported contract {spec['contract_version']!r}")
    if "checkpoint" in spec:
        raise ValueError(f"{path / POLICY_SPEC}: checkpoint is Judge-owned and must not appear in policy_spec.json")
    if "python" in spec:
        raise ValueError(f"{path / POLICY_SPEC}: runtime is Judge-owned and python must not appear in policy_spec.json")
    for field in ("implementation", "install_script", "smoke_test"):
        if not isinstance(spec[field], str):
            raise ValueError(f"{path / POLICY_SPEC}: invalid {field}")
        candidate = Path(spec[field])
        if candidate.is_absolute() or ".." in candidate.parts or not (path / candidate).is_file():
            raise ValueError(f"{path / POLICY_SPEC}: invalid {field}")
    if not isinstance(spec["input_schema"], dict) or not isinstance(spec["action_schema"], dict):
        raise ValueError(f"{path / POLICY_SPEC}: invalid policy schema")
    action = spec["action_schema"]
    expected = benchmark_action_schema(manifest.benchmark_name)
    expected_type = expected["type"]
    expected_dim = expected["dim"]
    if action.get("type") != expected_type or action.get("action_dim") != expected_dim:
        raise ValueError(
            f"{path / POLICY_SPEC}: action schema must be {expected_type}[{expected_dim}] for "
            f"{manifest.benchmark_name}"
        )
    if not isinstance(action.get("action_horizon"), int) or isinstance(action["action_horizon"], bool) or action["action_horizon"] <= 0:
        raise ValueError(f"{path / POLICY_SPEC}: action_horizon must be a positive integer")
    return spec


def factory_bundle_spec(path: Path, manifest: JobManifest) -> dict[str, Any]:
    spec = load_json(path / FACTORY_BUNDLE)
    required = ("format", "factory_checkpoint", "install_script", "smoke_test")
    missing = [field for field in required if field not in spec]
    if missing:
        raise ValueError(f"{path / FACTORY_BUNDLE}: missing {', '.join(missing)}")
    if spec["format"] != "vla-factory-inference/v1":
        raise ValueError(f"{path / FACTORY_BUNDLE}: invalid factory bundle identity")
    forbidden = {"checkpoint", "source_checkpoint", "base_checkpoint", "python"} & set(spec)
    if forbidden:
        raise ValueError(
            f"{path / FACTORY_BUNDLE}: source checkpoint is Judge-owned; forbidden fields: "
            f"{', '.join(sorted(forbidden))}"
        )
    for field in ("factory_checkpoint", "install_script", "smoke_test"):
        candidate = Path(spec[field])
        if not isinstance(spec[field], str) or candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"{path / FACTORY_BUNDLE}: invalid {field}")
    for field in ("install_script", "smoke_test"):
        if not (path / spec[field]).is_file():
            raise ValueError(f"{path / FACTORY_BUNDLE}: missing {field}")
    checkpoint = path / spec["factory_checkpoint"]
    required_files = ("model.pt", "weights.json", "inference_metadata/assembly.json", "inference_metadata/recipe.yaml")
    missing = [name for name in required_files if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"{path / FACTORY_BUNDLE}: incomplete factory checkpoint: {', '.join(missing)}")
    try:
        weight_format = load_json(checkpoint / "weights.json")["format"]
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path / FACTORY_BUNDLE}: invalid weights.json") from error
    if weight_format not in {"bare_full", "lora_wrapped_full", "lora_delta"}:
        raise ValueError(f"{path / FACTORY_BUNDLE}: invalid weight format")
    assembly = load_json(checkpoint / "inference_metadata/assembly.json")
    source_markers = (manifest.checkpoint, "/judge-inputs/checkpoint", "judge-inputs/checkpoint")
    if _contains_source_checkpoint(assembly, source_markers):
        raise ValueError(f"{path / FACTORY_BUNDLE}: assembly references Judge-owned source checkpoint")
    recipe = (checkpoint / "inference_metadata/recipe.yaml").read_text(encoding="utf-8", errors="replace")
    if any(marker and marker in recipe for marker in source_markers):
        raise ValueError(f"{path / FACTORY_BUNDLE}: recipe references Judge-owned source checkpoint")
    return spec


def _contains_source_checkpoint(value: Any, markers: tuple[str, ...]) -> bool:
    """Detect a source-checkpoint escape hatch in JSON metadata recursively."""
    if isinstance(value, str):
        return any(marker and marker in value for marker in markers)
    if isinstance(value, dict):
        return any(_contains_source_checkpoint(item, markers) for item in value.values())
    if isinstance(value, list):
        return any(_contains_source_checkpoint(item, markers) for item in value)
    return False


def _judge_python(path: Path, candidate: str) -> str:
    """Return the only runtime path Judge permits for a candidate."""
    executable = path / "judge-runtime" / "bin" / "python"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(
            f"{candidate} installer did not create Judge runtime {executable}; "
            "it must honor JUDGE_RUNTIME_DIR and JUDGE_PYTHON"
        )
    if str(executable.resolve()).startswith("/home/opencode/"):
        raise RuntimeError(
            f"{candidate} Judge runtime resolves into the Agent sandbox"
        )
    return str(executable)


def run_preflight(path: Path, manifest: JobManifest, candidate: str) -> dict[str, Any]:
    path = path.resolve()
    load_spec = factory_bundle_spec if candidate == "factory" else policy_spec
    spec = load_spec(path, manifest)
    environment = _hermetic_environment(path) if candidate == "factory" else _reference_environment(path)
    runtime = path / "judge-runtime"
    if runtime.exists() or runtime.is_symlink():
        _remove_residue(runtime)
    environment.update({
        "JUDGE_BOOTSTRAP_PYTHON": _judge_bootstrap_python(),
        "JUDGE_RUNTIME_DIR": str(runtime),
        "JUDGE_PYTHON": str(runtime / "bin" / "python"),
    })
    workspace = path.parent
    log = path / "judge-install.log"
    print(f"[judge:{candidate}] install started; log={log}")
    with log.open("w", encoding="utf-8") as file:
        result = subprocess.run(["bash", str(path / spec["install_script"])], cwd=workspace,
                                env=environment,
                                stdout=file, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{candidate} install failed; see {log}")
    print(f"[judge:{candidate}] install passed")
    spec = load_spec(path, manifest)
    python = _judge_python(path, candidate)
    log = path / "judge-smoke.log"
    print(f"[judge:{candidate}] smoke started; log={log}")
    with log.open("w", encoding="utf-8") as file:
        result = subprocess.run([python, str(path / spec["smoke_test"])], cwd=workspace,
                                env=environment,
                                stdout=file, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{candidate} smoke failed; see {log}")
    print(f"[judge:{candidate}] smoke passed")
    return {**spec, "judge_python": python}


def _run_preflights(reference_dir: Path, factory_dir: Path,
                    manifest: JobManifest) -> tuple[dict[str, Any], dict[str, Any]]:
    """Install and smoke-check the two isolated candidate runtimes concurrently."""
    candidates = {"reference": reference_dir, "factory": factory_dir}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="vla-preflight") as executor:
        futures = {
            candidate: executor.submit(run_preflight, path, manifest, candidate)
            for candidate, path in candidates.items()
        }
        return futures["reference"].result(), futures["factory"].result()


def _run_logged(command: list[str], cwd: Path, log: Path, env: dict[str, str] | None = None) -> None:
    with log.open("w", encoding="utf-8") as file:
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=file, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"parity command failed; see {log}")


def _run_parity(
    job_dir: Path, manifest: JobManifest, reference_dir: Path, factory_dir: Path,
    reference_spec: dict[str, Any], factory_spec: dict[str, Any], dataset: Path, revision: str,
) -> dict[str, Any]:
    artifacts = job_dir / "artifacts" / "parity"
    artifacts.mkdir(parents=True, exist_ok=True)
    bundle = factory_dir / factory_spec["factory_checkpoint"]
    benchmark = resolve_benchmark(manifest.benchmark_name)
    sample = {"dataset_path": str(dataset), "dataset_revision": revision,
               "sample_index": 0, "seed": 42, "precision": "fp32", "mode": "eval"}
    sample_path = artifacts / "sample.npz"
    sample_command = [
        factory_spec["judge_python"], str(JUDGE_ROOT / "parity" / "factory_parity_runner.py"), "sample", "--bundle", str(bundle),
        "--dataset-path", sample["dataset_path"], "--dataset-revision", sample["dataset_revision"],
        "--sample-index", str(sample["sample_index"]), "--seed", str(sample["seed"]),
        "--precision", sample["precision"], "--mode", sample["mode"], "--platform", benchmark.platform,
        "--output", str(sample_path),
    ]
    for key in benchmark.state_keys:
        sample_command.extend(("--state-key", key))
    _run_logged(
        sample_command,
        factory_dir.parent, artifacts / "sample.log", _hermetic_environment(factory_dir),
    )
    reference_capture, factory_capture = artifacts / "reference-loss.json", artifacts / "factory-loss.json"
    capture_jobs = {
        "reference": (
            [reference_spec["judge_python"], str(JUDGE_ROOT / "parity" / "native_parity_runner.py"),
             "--implementation", str(reference_dir / reference_spec["implementation"]),
             "--checkpoint", str(job_dir / "judge-inputs" / "checkpoint"),
             "--sample", str(sample_path), "--output", str(reference_capture)],
            reference_dir.parent, artifacts / "reference.log", _reference_environment(reference_dir),
        ),
        "factory": (
            [factory_spec["judge_python"], str(JUDGE_ROOT / "parity" / "factory_parity_runner.py"), "capture",
             "--bundle", str(bundle), "--sample", str(sample_path), "--platform", benchmark.platform,
             "--output", str(factory_capture)],
            factory_dir.parent, artifacts / "factory.log", _hermetic_environment(factory_dir),
        ),
    }
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="vla-loss") as executor:
        capture_futures = {
            role: executor.submit(_run_logged, *job) for role, job in capture_jobs.items()
        }
        for role in ("reference", "factory"):
            capture_futures[role].result()
    report = artifacts / "report.json"
    _run_logged(
        [factory_spec["judge_python"], str(JUDGE_ROOT / "parity" / "factory_parity_runner.py"), "compare",
         "--reference", str(reference_capture), "--factory", str(factory_capture), "--report", str(report)],
        factory_dir.parent, artifacts / "compare.log", _hermetic_environment(factory_dir),
    )
    result = load_json(report)
    return {"kind": "parity", **result, "gates": list(PARITY_GATES), "sample": sample,
            "reference_policy": str(reference_dir / reference_spec["implementation"]), "report": str(report)}


def judge(args: argparse.Namespace) -> dict[str, Any]:
    job_dir = Path(args.job_dir).resolve()
    manifest = load_manifest(job_dir)
    reference_dir = bundle_for_role(job_dir, "reference")
    factory_dir = bundle_for_role(job_dir, "adapter")
    artifacts = job_dir / "artifacts"
    dataset, revision = parity_dataset(manifest.benchmark_name)
    compliance = _factory_compliance(job_dir, manifest, factory_dir)
    write_json(artifacts / "compliance.json", compliance)
    reference_spec, factory_spec = _run_preflights(reference_dir, factory_dir, manifest)
    match = _run_parity(job_dir, manifest, reference_dir, factory_dir, reference_spec, factory_spec, dataset, revision)
    write_json(artifacts / "parity-match.json", match)
    compliance["hermetic_runtime"] = "PASS"
    compliance["match"] = "parity"
    write_json(artifacts / "compliance.json", compliance)
    status = match["status"]
    verdict = {"verdict": "PASS" if status == "PASS" else "FAIL" if status == "FAIL" else "INCOMPLETE", "match": match,
               "policy_contract": {"version": POLICY_CONTRACT_VERSION, "reference": reference_spec, "factory": factory_spec}}
    write_json(artifacts / "verdict.json", verdict)
    print(json.dumps(verdict, indent=2))
    return verdict


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="adapt_vla_model", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("job_dir")
    init.add_argument("--job-id", required=True)
    init.add_argument("--upstream-repo", required=True)
    init.add_argument("--vla", required=True)
    init.add_argument("--checkpoint", required=True)
    init.add_argument("--factory-base", required=True)
    init.add_argument("--benchmark", default=DEFAULT_BENCHMARK,
                      help=f"Judge-owned benchmark name (default: {DEFAULT_BENCHMARK})")
    init.add_argument("--opencode-model", choices=OPENCODE_MODELS,
                      default=DEFAULT_OPENCODE_MODEL)
    init.set_defaults(func=init_job)
    agent = commands.add_parser("agent")
    agent.add_argument("job_dir")
    agent.add_argument("role", choices=("reference", "adapter"))
    agent.set_defaults(func=run_agent)
    verdict = commands.add_parser("judge")
    verdict.add_argument("job_dir")
    verdict.set_defaults(func=judge)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        args.func(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
