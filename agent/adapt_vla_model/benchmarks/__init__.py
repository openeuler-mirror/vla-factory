"""Judge-owned benchmark discovery and execution."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


# Single source of truth for the default benchmark (README and run.py follow it).
# The LeRobot Isaac 6D benchmark is the default because it needs only a local
# LeRobot dataset; the RoboTwin benchmark additionally requires the RoboTwin
# simulation environment.
DEFAULT_BENCHMARK = "lerobot-leisaac-pickup-cude-30-v2"
# Benchmark data lives inside this package: <name>/<task>/spec.json + contracts.
BENCHMARK_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    revision: str
    comparison: dict[str, float]
    platform: str
    action_schema: dict[str, object]
    state_keys: tuple[str, ...]


def factory_root() -> Path:
    # benchmarks/ -> adapt_vla_model/ -> agent/ -> vla-factory repository root.
    root = Path(os.environ.get("VLA_FACTORY_PATH", Path(__file__).resolve().parents[3])).resolve()
    if not (root / "vla_factory" / "inference" / "connectors" / "robotwin.yml").is_file():
        raise ValueError(f"VLA_FACTORY_PATH has no RoboTwin connector: {root}")
    return root


def _directory(name: str) -> Path:
    for candidate in BENCHMARK_ROOT.glob("*/*/spec.json"):
        if json.loads(candidate.read_text(encoding="utf-8")).get("name") == name:
            return candidate.parent
    raise ValueError(f"unsupported Judge benchmark: {name}")


def _runner(name: str) -> ModuleType:
    path = _directory(name) / "run.py"
    spec = importlib.util.spec_from_file_location(f"judge_benchmark_{path.parent.name}", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load benchmark runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def platform(name: str) -> str:
    """Read a benchmark boundary without resolving its installed revision."""
    value = json.loads((_directory(name) / "spec.json").read_text(encoding="utf-8")).get("platform", "robotwin")
    if value not in {"robotwin", "lerobot"}:
        raise ValueError(f"unsupported benchmark platform: {value}")
    return value


def action_schema(name: str) -> dict[str, object]:
    """Read the static output contract without probing the installed dataset."""
    value = json.loads((_directory(name) / "spec.json").read_text(encoding="utf-8")).get("action_schema")
    if not isinstance(value, dict) or not isinstance(value.get("type"), str) or not isinstance(value.get("dim"), int):
        raise ValueError(f"benchmark {name} has an invalid action schema")
    return value


def resolve(name: str = DEFAULT_BENCHMARK) -> BenchmarkSpec:
    data = json.loads((_directory(name) / "spec.json").read_text(encoding="utf-8"))
    action = action_schema(name)
    state_keys = data.get("state_keys", [])
    if not isinstance(state_keys, list) or not all(isinstance(key, str) for key in state_keys):
        raise ValueError(f"benchmark {name} has invalid state keys")
    return BenchmarkSpec(
        data["name"], _runner(name).revision(), data["comparison"], platform(name), action,
        tuple(state_keys),
    )


def contract_document(name: str, version: str) -> str:
    return (_directory(name) / "REFERENCE_BUNDLE_CONTRACT.md").read_text(encoding="utf-8").replace("__VERSION__", version)


def factory_contract_document(name: str) -> str:
    """Return the complete Factory contract owned by one benchmark."""
    path = _directory(name) / "FACTORY_BUNDLE_CONTRACT.md"
    if not path.is_file():
        raise ValueError(f"benchmark {name} has no Factory Bundle Contract")
    return path.read_text(encoding="utf-8")


def parity_dataset(name: str) -> tuple[Path, str]:
    """Return the platform-owned benchmark data location; Agents never choose it."""
    data = json.loads((_directory(name) / "spec.json").read_text(encoding="utf-8"))
    relative = data.get("parity_dataset_relative_path")
    revision = data.get("parity_dataset_revision")
    if not isinstance(relative, str) or not isinstance(revision, str):
        raise ValueError(f"benchmark {name} has no parity dataset")
    root_name = data.get("parity_dataset_root_env", "ROBOTWIN_ROOT")
    if not isinstance(root_name, str):
        raise ValueError(f"benchmark {name} has an invalid parity dataset root")
    root = os.environ.get(root_name)
    if not root:
        default_root = data.get("parity_dataset_default_root")
        root = os.path.expanduser(default_root) if isinstance(default_root, str) else None
    if not root:
        raise ValueError(f"{root_name} must point at the installed benchmark dataset root")
    dataset = Path(root).resolve() / relative
    if not dataset.is_dir():
        raise ValueError(f"benchmark parity dataset is missing: {dataset}")
    return dataset, revision


def run(name: str, policy_port: int, candidate: str, job_id: str, result_path: Path) -> dict[str, object]:
    return _runner(name).run(policy_port, candidate, job_id, result_path)
