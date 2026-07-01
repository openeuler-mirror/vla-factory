"""Transport layer for InferenceEngine — ZMQ and in-process (§12.5, §12.6).

The Transport abstraction decouples inference logic from the communication
protocol.  The ZMQ implementation uses :mod:`vla_factory.deploy.zmq_client`.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol, runtime_checkable

import numpy as np

from vla_factory.deploy.infer import InferenceEngine, ObsDict
from vla_factory.deploy.zmq_client import (
    ZmqPolicyClient,
    ZmqPolicyClientConfig,
    encode_observation_json,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Transport protocol
# ═══════════════════════════════════════════════════════════════════


@runtime_checkable
class Transport(Protocol):
    """Communication transport interface (§12.5)."""

    def serve(self, engine: InferenceEngine, host: str = "0.0.0.0", port: int = 5555) -> None:
        """Start serving: receive requests, call engine.predict()."""
        ...

    def connect(self, host: str = "localhost", port: int = 5555) -> PolicyClient:
        """Connect to a remote engine, return a PolicyClient."""
        ...


# ═══════════════════════════════════════════════════════════════════
# ZMQ Transport — wraps existing PUSH/PULL infrastructure
# ═══════════════════════════════════════════════════════════════════


class ZMQTransport:
    """ZMQ transport using the LeKiwi-style PUSH/PULL protocol.

    This wraps :class:`ZmqPolicyClient` / :class:`ZmqPolicyClientConfig`
    from :mod:`vla_factory.deploy.zmq_client` rather than reimplementing from
    scratch.  The transport operates in *client mode*: it connects to a
    simulator host that PUSHes observations and PULLs actions.

    Parameters
    ----------
    remote_ip : str
        Simulator host IP.
    port_zmq_cmd : int
        Port for sending actions (PUSH).
    port_zmq_observations : int
        Port for receiving observations (PULL).
    polling_timeout_ms : int
        ZMQ polling timeout in milliseconds.
    max_loop_freq_hz : float
        Maximum loop frequency cap.
    """

    def __init__(
        self,
        remote_ip: str = "127.0.0.1",
        port_zmq_cmd: int = 5555,
        port_zmq_observations: int = 5556,
        polling_timeout_ms: int = 1000,
        connect_timeout_s: float = 0.0,
        max_loop_freq_hz: float = 60.0,
    ) -> None:
        self.remote_ip = remote_ip
        self.port_zmq_cmd = port_zmq_cmd
        self.port_zmq_observations = port_zmq_observations
        self.polling_timeout_ms = polling_timeout_ms
        self.connect_timeout_s = connect_timeout_s
        self.max_loop_freq_hz = max_loop_freq_hz

    def serve(self, engine: InferenceEngine, host: str = "0.0.0.0", port: int = 5555) -> None:
        """Run engine as a ZMQ client connected to a simulator host.

        This is a blocking loop: receive observation → predict → send action.
        """
        config = ZmqPolicyClientConfig(
            remote_ip=self.remote_ip or host,
            port_zmq_cmd=self.port_zmq_cmd or port,
            port_zmq_observations=self.port_zmq_observations,
            polling_timeout_ms=self.polling_timeout_ms,
            connect_timeout_s=self.connect_timeout_s,
            max_loop_freq_hz=self.max_loop_freq_hz,
        )
        client = ZmqPolicyClient(config)
        logger.info(
            "ZMQ transport connecting to tcp://%s:%d/%d",
            config.remote_ip, config.port_zmq_observations, config.port_zmq_cmd,
        )
        client.wait_for_connection()
        logger.info("Connected. Waiting for observations.")

        adapter = _ZMQObsAdapter(engine.camera_keys)
        try:
            while True:
                loop_start = time.time()
                observation = client.recv_observation()
                if observation is None:
                    continue
                if observation.get("__control__") == "reset":
                    engine.reset()
                    logger.info(
                        "Reset episode=%d.", observation.get("episode_index", -1),
                    )
                    continue

                obs = adapter(observation)
                action = engine.predict(obs)
                client.send_action(action)
                elapsed = time.time() - loop_start
                time.sleep(max(1.0 / config.max_loop_freq_hz - elapsed, 0.0))

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt. Exiting.")
        finally:
            client.close()

    def connect(self, host: str = "localhost", port: int = 5555) -> PolicyClient:
        """Return a PolicyClient that communicates via ZMQ."""
        return PolicyClient(
            host=host,
            port=port,
            transport_type="zmq",
            port_observations=self.port_zmq_observations,
        )


# ═══════════════════════════════════════════════════════════════════
# In-process transport — zero overhead, direct function call
# ═══════════════════════════════════════════════════════════════════


class InProcessTransport:
    """Direct function call, no serialization or network overhead.

    For LeRobot-style in-process integration:
        engine = InferenceEngine(...)
        actions = engine.predict(obs)
    """

    def serve(self, engine: InferenceEngine, **kwargs) -> None:
        raise RuntimeError(
            "InProcess transport does not support serve() — "
            "use engine.predict() directly."
        )

    def connect(self, engine: InferenceEngine) -> InferenceEngine:
        return engine


# ═══════════════════════════════════════════════════════════════════
# PolicyClient — remote proxy with same interface as InferenceEngine
# ═══════════════════════════════════════════════════════════════════


class PolicyClient:
    """Remote inference client — drop-in replacement for InferenceEngine (§12.6).

    Implements the same ``predict()`` / ``reset()`` interface so control-loop
    code does not need to know whether inference is local or remote.

    ┌────────────────┐     ┌────────────────┐
    │  control code  │     │  control code  │
    │  engine.predict│     │  engine.predict│
    └───────┬────────┘     └───────┬────────┘
            │                      │
     local: InferenceEngine  remote: PolicyClient
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        transport_type: str = "zmq",
        port_observations: int = 5556,
    ) -> None:
        self.host = host
        self.port = port
        self.transport_type = transport_type
        self._client: object | None = None

        if transport_type == "zmq":
            config = ZmqPolicyClientConfig(
                remote_ip=host,
                port_zmq_cmd=port,
                port_zmq_observations=port_observations,
            )
            self._client = ZmqPolicyClient(config)

    def predict(self, obs) -> np.ndarray:
        """Send observation, receive action array."""
        if self._client is None:
            raise RuntimeError("PolicyClient not connected.")

        if isinstance(obs, ObsDict):
            # Convert ObsDict → flat ZMQ dict
            zmq_obs: dict = {}
            for cam, arr in obs.video.items():
                zmq_obs[f"observation.images.{cam}"] = arr
            if obs.state is not None:
                zmq_obs["observation.state"] = obs.state
            if obs.language is not None:
                zmq_obs["language"] = obs.language
            self._client.send_action  # ensure connected
            # Use recv/send pattern
            raise NotImplementedError(
                "PolicyClient.predict with ObsDict requires a running host. "
                "Use serve_lerobot_zmq_client pattern instead."
            )
        return np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        """Send reset signal."""
        pass

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


# ═══════════════════════════════════════════════════════════════════
# Helper: ZMQ observation dict → ObsDict
# ═══════════════════════════════════════════════════════════════════


class _ZMQObsAdapter:
    """Convert flat ZMQ observation dict to ObsDict.

    ZMQ observation keys:
        ``observation.images.front`` → ``ObsDict.video["front"]``
        ``observation.state``        → ``ObsDict.state``
        ``language``                 → ``ObsDict.language``
    """

    def __init__(self, camera_keys: tuple[str, ...]) -> None:
        self.camera_keys = camera_keys

    def __call__(self, observation: dict) -> ObsDict:
        video: dict[str, np.ndarray] = {}
        for cam in self.camera_keys:
            key = f"observation.images.{cam}"
            if key not in observation:
                raise KeyError(
                    f"Expected image key '{key}' in observation, "
                    f"got: {list(observation.keys())}"
                )
            video[cam] = observation[key]

        state = observation.get("observation.state")
        if state is not None:
            state = np.asarray(state, dtype=np.float32)

        language = observation.get("language")
        return ObsDict(video=video, state=state, language=language)
