#!/usr/bin/env python3
"""Run the same environment-by-tier matrix as the PR CI daemon locally.

The daemon executes a clean detached PR checkout. This runner therefore
refuses a dirty worktree by default, so a local green result represents the
current commit rather than uncommitted edits. Use ``--allow-dirty`` only while
iterating before a commit.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_FILE = Path.home() / ".vlaf_ci.conf"
PYTHON_ENV_KEYS = ("VLAF_ENV_BASE", "VLAF_ENV_ACT", "VLAF_ENV_PI")
ENV_KEYS = (*PYTHON_ENV_KEYS, "HF_TOKEN")


def load_environment_config(path: Path = CONF_FILE) -> None:
    """Load only CI interpreter paths, without evaluating shell content."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key in ENV_KEYS:
            os.environ.setdefault(key, value)


def ensure_clean_worktree() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise SystemExit(
            "Working tree is dirty. Commit or stash changes before local CI, "
            "or pass --allow-dirty for an explicitly non-CI-equivalent run."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the same L0/L1/L2 matrix as the PR CI daemon.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="test uncommitted edits; this differs from the detached CI checkout",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="new directory for junit XML; default is a fresh /tmp directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_environment_config()
    if not args.allow_dirty:
        ensure_clean_worktree()

    missing = [key for key in ENV_KEYS if not os.environ.get(key)]
    invalid = [
        key for key in PYTHON_ENV_KEYS
        if os.environ.get(key) and not Path(os.environ[key]).is_file()
    ]
    if missing or invalid:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if invalid:
            details.append("invalid " + ", ".join(invalid))
        raise SystemExit("Configured CI interpreters and HF_TOKEN are required: " + "; ".join(details))

    report_dir = args.report_dir
    if report_dir is None:
        report_dir = Path(tempfile.mkdtemp(prefix="vlaf-ci-local-"))
    else:
        if report_dir.exists():
            raise SystemExit(f"Report directory already exists: {report_dir}")
        report_dir.mkdir(parents=True)

    # Import after loading configuration: daemon.ENVS is built at import time.
    sys.path.insert(0, str(Path(__file__).parent))
    import daemon

    all_ok = True
    for label, python_bin in daemon.ENVS:
        print(f"== {label}: {python_bin} ==")
        if not daemon.run_gate(REPO_ROOT, label, python_bin, report_dir / label):
            all_ok = False

    rows = daemon.collect_results(report_dir)
    for row in rows:
        print(
            f"{row['env']}: L0 {row['l0']} | L1 {row['l1']} | "
            f"L2 {row['l2']} | {row['time']}"
        )
    all_ok = all_ok and all(row["ok"] for row in rows)
    print(f"JUnit reports: {report_dir}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
