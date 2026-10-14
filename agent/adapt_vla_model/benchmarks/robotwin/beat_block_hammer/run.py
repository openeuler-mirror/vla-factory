"""Judge runner for RoboTwin beat_block_hammer."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from adapt_vla_model.benchmarks import factory_root


SPEC = json.loads(Path(__file__).with_name("spec.json").read_text(encoding="utf-8"))
TIMEOUT_SECONDS = 6 * 60 * 60


def _robotwin_root() -> Path:
    value = os.environ.get("ROBOTWIN_ROOT")
    if not value:
        raise ValueError("ROBOTWIN_ROOT must point at the installed RoboTwin repository")
    root = Path(value).resolve()
    if not (root / "script" / "eval_policy_client.py").is_file():
        raise ValueError(f"ROBOTWIN_ROOT is not a RoboTwin checkout: {root}")
    return root


def _robotwin_python() -> str:
    value = os.environ.get("ROBOTWIN_PY")
    if not value:
        raise ValueError("ROBOTWIN_PY must point at the Python interpreter in the RoboTwin environment")
    path = Path(value).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"ROBOTWIN_PY is not executable: {path}")
    return str(path)


def revision() -> str:
    root = _robotwin_root()
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                capture_output=True, check=True).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"cannot resolve RoboTwin revision at {root}") from error
    return f"RoboTwin@{commit}"


def _result_rate(result_dir: Path) -> float:
    files = list(result_dir.glob("*/_result.txt"))
    if len(files) != 1:
        raise RuntimeError(f"expected one RoboTwin result under {result_dir}, found {len(files)}")
    try:
        rate = float(files[0].read_text(encoding="utf-8").strip())
    except ValueError as error:
        raise RuntimeError(f"invalid RoboTwin success rate in {files[0]}") from error
    if not 0.0 <= rate <= 1.0:
        raise RuntimeError(f"RoboTwin success rate is outside [0, 1]: {rate}")
    return rate


def _cuda_link_dir(robotwin_env: Path, artifacts: Path) -> Path:
    runtimes = [path for path in (robotwin_env / "lib").glob("libcudart.so.*") if path.is_file() and not path.is_symlink()]
    if not runtimes:
        raise RuntimeError(f"RoboTwin CUDA runtime is missing under {robotwin_env / 'lib'}")
    directory = artifacts / "cuda-link"
    directory.mkdir(parents=True, exist_ok=True)
    link = directory / "libcudart.so"
    if link.exists() and not link.is_symlink():
        raise RuntimeError(f"refusing non-link CUDA compatibility path: {link}")
    link.unlink(missing_ok=True)
    link.symlink_to(runtimes[0])
    return directory


def run(policy_port: int, candidate: str, job_id: str, result_path: Path) -> dict[str, object]:
    robotwin_root = _robotwin_root()
    robotwin_python = Path(_robotwin_python())
    robotwin_env = robotwin_python.parent.parent
    factory = factory_root()
    label = f"judge-{job_id}-{candidate}".replace("/", "-")
    result_dir = robotwin_root / "eval_result" / SPEC["task"] / "vla_factory" / SPEC["task_config"] / label
    if result_dir.exists():
        raise RuntimeError(f"refusing stale RoboTwin result directory: {result_dir}")
    torch_extensions = result_path.parent / "torch_extensions"
    torch_extensions.mkdir(parents=True, exist_ok=True)
    cuda_link = _cuda_link_dir(robotwin_env, result_path.parent)
    command = [
        str(robotwin_python), "script/eval_policy_client.py", "--port", str(policy_port),
        "--config", str(factory / "vla_factory" / "inference" / "connectors" / "robotwin.yml"),
        "--overrides", "task_name", SPEC["task"], "task_config", SPEC["task_config"], "ckpt_setting", label,
        "instruction_type", SPEC["instruction_type"], "seed", str(SPEC["seed"]),
    ]
    environment = {
        **os.environ,
        "PATH": str(robotwin_env / "bin") + os.pathsep + os.environ.get("PATH", ""),
        "LD_LIBRARY_PATH": str(cuda_link) + os.pathsep + str(robotwin_env / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""),
        "LIBRARY_PATH": str(cuda_link) + os.pathsep + os.environ.get("LIBRARY_PATH", ""),
        "CUDA_HOME": os.environ.get("CUDA_HOME", str(robotwin_env)),
        "PYTHONPATH": str(factory) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "TORCH_EXTENSIONS_DIR": str(torch_extensions),
    }
    subprocess.run(command, cwd=robotwin_root, env=environment, timeout=TIMEOUT_SECONDS, check=True)
    report = {"metrics": {"success_rate": _result_rate(result_dir)}, "task": SPEC["task"],
              "task_config": SPEC["task_config"], "seed": SPEC["seed"], "result_dir": str(result_dir)}
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
