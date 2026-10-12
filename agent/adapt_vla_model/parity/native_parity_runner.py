#!/usr/bin/env python3
"""Judge-owned Reference runtime launcher for Policy.compute_loss()."""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path


def _seed(value: int) -> None:
    import numpy as np

    random.seed(value)
    np.random.seed(value)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.implementation.parent))
    spec = importlib.util.spec_from_file_location("judge_reference_policy", args.implementation)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load policy implementation: {args.implementation}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = getattr(module, "build_policy", None)
    if not callable(build):
        raise ValueError("policy_impl.py must export build_policy")
    from parity_io import read_sample, write_loss

    observation, meta = read_sample(args.sample)
    policy = build(args.checkpoint)
    _seed(meta["seed"])
    write_loss(policy.compute_loss(observation), args.output)


if __name__ == "__main__":
    main()
