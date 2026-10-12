#!/usr/bin/env python3
"""Judge-owned VLA-Factory RoboTwin server for a native checkpoint bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _install_trace(path: Path) -> None:
    """Instrument the fixed Factory chain; Agent code cannot emit these events."""
    from vla_factory.inference.inference_engine import InferenceEngine
    from vla_factory.inference.platforms.robotwin import RoboTwinAdapter

    def emit(event: str, **data) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"event": event, **data}, sort_keys=True) + "\n")

    adapter_call = RoboTwinAdapter.__call__
    def traced_adapter(self, observation, task=""):
        result = adapter_call(self, observation, task)
        emit("platform_adapter", cameras=sorted(result.video),
             state_dim=len(result.state) if result.state is not None else 0)
        return result
    RoboTwinAdapter.__call__ = traced_adapter

    run_model = InferenceEngine._run_model
    def traced_run(self, model_observation, observation):
        emit("robot_to_model", plan=[call.type for call in self.assembly.robot_to_model.calls],
             model_action_dim=self.model_output_dim)
        result = run_model(self, model_observation, observation)
        emit("model_to_robot", plan=[call.type for call in self.assembly.model_to_robot.calls],
             qpos_shape=list(result.values.shape))
        return result
    InferenceEngine._run_model = traced_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    _install_trace(args.trace)
    from vla_factory.inference.deploy import DeploymentConfig, deploy
    deploy(DeploymentConfig(checkpoint=args.checkpoint, platform="robotwin", host="127.0.0.1", port=args.port))


if __name__ == "__main__":
    main()
