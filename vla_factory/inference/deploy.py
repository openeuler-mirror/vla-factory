"""Compose inference, action execution, platform adapters, and transports."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch

from vla_factory.inference.execution import (
    ActionCommand,
    PolicyExecutor,
    build_execution_policy,
)
from vla_factory.inference.inference_engine import InferenceEngine, ObsDict
from vla_factory.inference.platforms.base import PlatformObservationAdapter
from vla_factory.inference.transports.base import PolicyClientTransport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeploymentConfig:
    """Runtime options shared by the supported deployment forms."""

    checkpoint: str | Path
    platform: str = "simulator"
    device: str | None = None
    strategy: str | None = None
    task: str = ""
    n_action_steps: int | None = None
    max_loop_freq_hz: float = 60.0
    remote_ip: str = "127.0.0.1"
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    polling_timeout_ms: int = 1000
    connect_timeout_s: float = 0.0
    host: str = "0.0.0.0"
    port: int = 9999

    def __post_init__(self) -> None:
        if self.platform not in {"simulator", "lerobot", "robotwin"}:
            raise ValueError(f"Unknown deployment platform {self.platform!r}.")
        if self.max_loop_freq_hz <= 0:
            raise ValueError("max_loop_freq_hz must be a positive number")


class ExecutablePolicy(Protocol):
    def predict(self, observation: ObsDict) -> ActionCommand:
        ...

    def reset(self) -> None:
        ...


class PolicyRunner:
    """Drive receive → adapt → predict → adapt → send over a client transport."""

    def __init__(
        self,
        policy: ExecutablePolicy,
        observation_adapter: PlatformObservationAdapter,
        action_adapter: Callable[[np.ndarray], Any] | None = None,
        task: str = "",
        max_loop_freq_hz: float = 60.0,
    ) -> None:
        if max_loop_freq_hz <= 0:
            raise ValueError(
                f"max_loop_freq_hz must be positive, got {max_loop_freq_hz}"
            )
        self.policy = policy
        self.observation_adapter = observation_adapter
        self.action_adapter = action_adapter
        self.task = task
        self.max_loop_freq_hz = max_loop_freq_hz

    def run(self, client: PolicyClientTransport) -> None:
        """Run until interrupted and always close the transport on exit."""
        try:
            client.wait_for_connection()
            logger.info("Connected. Waiting for observations.")
            while True:
                loop_start = time.time()
                observation = client.recv_observation()
                if observation is None:
                    self._pace(loop_start)
                    continue
                if observation.get("__control__") == "reset":
                    self.policy.reset()
                    logger.info(
                        "Reset episode=%d.", observation.get("episode_index", -1)
                    )
                    continue

                command = self.policy.predict(
                    self.observation_adapter(observation, self.task)
                )
                if not isinstance(command, ActionCommand):
                    raise TypeError(
                        "ExecutablePolicy.predict() must return ActionCommand, got "
                        f"{type(command).__name__}."
                    )
                action = (
                    self.action_adapter(command.single())
                    if self.action_adapter is not None
                    else command.values
                )
                client.send_action(action)
                self._pace(loop_start)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt. Exiting.")
        finally:
            client.close()

    def _pace(self, loop_start: float) -> None:
        elapsed = time.time() - loop_start
        time.sleep(max(1.0 / self.max_loop_freq_hz - elapsed, 0.0))


class RemotePolicyModel:
    """Expose the RPC method convention expected by RoboTwin's model client."""

    def __init__(
        self,
        policy: ExecutablePolicy,
        adapter: PlatformObservationAdapter,
        task: str = "",
    ) -> None:
        self.policy = policy
        self.adapter = adapter
        self.task = task
        self._last_observation: Any = None

    def reset_model(self, obs: Any = None) -> None:
        self.policy.reset()
        self._last_observation = None
        return None

    def update_obs(self, obs: Any) -> None:
        self._last_observation = obs
        return None

    def get_action(self, obs: Any = None) -> np.ndarray:
        observation = obs if obs is not None else self._last_observation
        if observation is None:
            raise RuntimeError(
                "get_action called before any observation (no obs arg and no "
                "prior update_obs)."
            )
        command = self.policy.predict(self.adapter(observation, self.task))
        if not isinstance(command, ActionCommand):
            raise TypeError(
                "ExecutablePolicy.predict() must return ActionCommand, got "
                f"{type(command).__name__}."
            )
        return command.values


def deploy(config: DeploymentConfig) -> None:
    """Start the configured client- or server-shaped deployment runtime."""
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = InferenceEngine(checkpoint_path=config.checkpoint, device=device)
    strategy = config.strategy or (
        "synchronous" if config.platform == "robotwin" else "receding_horizon"
    )
    execution_policy = build_execution_policy(
        strategy,
        action_horizon=engine.action_horizon,
        action_dim=engine.execution_action_dim,
        n_action_steps=config.n_action_steps,
    )
    policy = PolicyExecutor(engine, execution_policy)

    if config.platform == "robotwin":
        _deploy_robotwin(config, engine, policy, device)
    else:
        _deploy_zmq(config, engine, policy, strategy, device)


def _deploy_robotwin(
    config: DeploymentConfig,
    engine: InferenceEngine,
    policy: PolicyExecutor,
    device: str,
) -> None:
    from vla_factory.inference.platforms.robotwin import RoboTwinAdapter
    from vla_factory.inference.transports.length_prefixed_json import (
        LengthPrefixedJsonRpcServer,
    )

    adapter = RoboTwinAdapter(
        camera_keys=engine.camera_keys,
        state_dim=engine.schema.state_dim,
    )
    model = RemotePolicyModel(policy, adapter, task=config.task)
    server = LengthPrefixedJsonRpcServer(
        model,
        host=config.host,
        port=config.port,
    )

    print(f"[deploy] Model: {engine.recipe.model.name}", flush=True)
    print(f"[deploy] Device: {device}", flush=True)
    print(
        f"[deploy] Platform: robotwin (cameras={list(engine.camera_keys)}, "
        f"state_dim={engine.schema.state_dim}, "
        f"action_dim={engine.execution_action_dim})",
        flush=True,
    )
    print(
        f"[deploy] Listening on {config.host}:{config.port} — start the "
        "RoboTwin client with the matching port.",
        flush=True,
    )
    server.serve_forever()


def _deploy_zmq(
    config: DeploymentConfig,
    engine: InferenceEngine,
    policy: PolicyExecutor,
    strategy: str,
    device: str,
) -> None:
    from vla_factory.inference.transports.zmq import (
        ZmqPolicyClient,
        ZmqPolicyClientConfig,
    )

    if config.platform == "lerobot":
        from vla_factory.inference.platforms.lerobot import (
            LerobotHostActionAdapter,
            LerobotHostObsAdapter,
        )

        observation_adapter = LerobotHostObsAdapter(
            camera_keys=engine.camera_keys,
            state_keys=engine.state_keys,
            state_dim=engine.schema.state_dim,
        )
        action_adapter: Callable[[np.ndarray], Any] | None = (
            LerobotHostActionAdapter(
                action_dim=engine.execution_action_dim,
                action_keys=engine.action_keys,
            )
        )
        platform_description = (
            f"lerobot (state_keys={list(engine.state_keys)}, "
            f"action_keys={list(engine.action_keys)})"
        )
    else:
        from vla_factory.inference.platforms.simulator import SimulatorAdapter

        observation_adapter = SimulatorAdapter(engine.camera_keys)
        action_adapter = None
        platform_description = f"simulator (cameras={list(engine.camera_keys)})"

    runner = PolicyRunner(
        policy,
        observation_adapter,
        action_adapter,
        task=config.task,
        max_loop_freq_hz=config.max_loop_freq_hz,
    )
    client = ZmqPolicyClient(
        ZmqPolicyClientConfig(
            remote_ip=config.remote_ip,
            port_zmq_cmd=config.port_zmq_cmd,
            port_zmq_observations=config.port_zmq_observations,
            polling_timeout_ms=config.polling_timeout_ms,
            connect_timeout_s=config.connect_timeout_s,
        )
    )

    print(f"[deploy] Model: {engine.recipe.model.name}", flush=True)
    print(f"[deploy] Strategy: {strategy}", flush=True)
    print(f"[deploy] Device: {device}", flush=True)
    print(f"[deploy] Platform: {platform_description}", flush=True)
    print(
        f"[deploy] Connecting to {config.remote_ip}:"
        f"{config.port_zmq_observations}/{config.port_zmq_cmd}",
        flush=True,
    )
    runner.run(client)


__all__ = [
    "DeploymentConfig",
    "PolicyRunner",
    "RemotePolicyModel",
    "deploy",
]
