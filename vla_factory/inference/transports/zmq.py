"""LeKiwi-style ZMQ PUSH/PULL policy transport.

Pure transport: owns the sockets and moves observation/action JSON. It does
not interpret observations, choose adapters, or drive inference — that
orchestration lives in ``deploy/policy_runtime.py``.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import zmq

logger = logging.getLogger(__name__)


class ZmqPolicyClientConfig:
    """Connection and polling settings for :class:`ZmqPolicyClient`."""

    def __init__(
        self,
        remote_ip: str = "127.0.0.1",
        port_zmq_cmd: int = 5555,
        port_zmq_observations: int = 5556,
        polling_timeout_ms: int = 1000,
        connect_timeout_s: float = 0.0,
    ) -> None:
        self.remote_ip = remote_ip
        self.port_zmq_cmd = port_zmq_cmd
        self.port_zmq_observations = port_zmq_observations
        self.polling_timeout_ms = polling_timeout_ms
        self.connect_timeout_s = connect_timeout_s


class ZmqPolicyClient:
    """Low-level LeKiwi-style ZMQ PUSH/PULL client.

    Observations are pulled from the host and actions are pushed back. The
    client keeps only the newest observation when multiple frames are queued.
    """

    def __init__(self, config: ZmqPolicyClientConfig) -> None:
        self.config = config
        self._context = zmq.Context()

        self._cmd_socket = self._context.socket(zmq.PUSH)
        self._cmd_socket.connect(
            f"tcp://{config.remote_ip}:{config.port_zmq_cmd}"
        )
        self._cmd_socket.setsockopt(zmq.CONFLATE, 1)

        self._obs_socket = self._context.socket(zmq.PULL)
        self._obs_socket.connect(
            f"tcp://{config.remote_ip}:{config.port_zmq_observations}"
        )
        self._obs_socket.setsockopt(zmq.CONFLATE, 1)

    def wait_for_connection(self) -> None:
        """Wait until the first observation is available."""
        poller = zmq.Poller()
        poller.register(self._obs_socket, zmq.POLLIN)

        if self.config.connect_timeout_s > 0:
            sockets = dict(
                poller.poll(int(self.config.connect_timeout_s * 1000))
            )
            if sockets.get(self._obs_socket) != zmq.POLLIN:
                raise TimeoutError(
                    "Timeout waiting for observations from "
                    f"{self.config.remote_ip}:"
                    f"{self.config.port_zmq_observations}"
                )
        else:
            last_log = 0.0
            while True:
                sockets = dict(poller.poll(1000))
                if sockets.get(self._obs_socket) == zmq.POLLIN:
                    break
                now = time.time()
                if now - last_log >= 5.0:
                    logger.info("Waiting for host observations...")
                    last_log = now

        logger.info("Connected to host.")

    def recv_observation(self) -> dict | None:
        """Return the newest observation, or ``None`` on polling timeout."""
        poller = zmq.Poller()
        poller.register(self._obs_socket, zmq.POLLIN)
        sockets = dict(poller.poll(self.config.polling_timeout_ms))

        latest_raw = None
        if sockets.get(self._obs_socket) == zmq.POLLIN:
            while True:
                try:
                    latest_raw = self._obs_socket.recv_string(zmq.NOBLOCK)
                except zmq.Again:
                    break

        return json.loads(latest_raw) if latest_raw is not None else None

    def send_action(self, action: np.ndarray | dict) -> None:
        """Send a JSON-encoded action to the host."""
        payload = action.tolist() if isinstance(action, np.ndarray) else action
        self._cmd_socket.send_string(json.dumps(payload), flags=zmq.NOBLOCK)

    def close(self) -> None:
        """Release sockets and their context immediately."""
        self._obs_socket.close(linger=0)
        self._cmd_socket.close(linger=0)
        self._context.term()


__all__ = [
    "ZmqPolicyClientConfig",
    "ZmqPolicyClient",
]
