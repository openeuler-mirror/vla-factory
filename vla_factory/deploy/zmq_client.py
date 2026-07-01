"""ZMQ client for LeKiwi-style PUSH/PULL protocol (§12.5).

This module implements the ZMQ inference client that connects to a
simulator host: receive observations (PULL) → predict → send actions (PUSH).
"""

from __future__ import annotations

import json as _json
import logging
import time

import numpy as np
import zmq

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


class ZmqPolicyClientConfig:
    """Configuration for :class:`ZmqPolicyClient`."""

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


# ═══════════════════════════════════════════════════════════════════
# ZMQ Client
# ═══════════════════════════════════════════════════════════════════


class ZmqPolicyClient:
    """ZMQ inference client — connects to a simulator host via PUSH/PULL.

    The client operates in *pull observations, push actions* mode:
    it connects to a host that PUSHes observations on ``port_zmq_observations``
    and PULLs actions from ``port_zmq_cmd``.

    Typical usage::

        config = ZmqPolicyClientConfig(remote_ip="192.168.1.10")
        client = ZmqPolicyClient(config)
        client.wait_for_connection()
        while True:
            obs = client.recv_observation()
            if obs is not None:
                action = engine.predict(obs_from_dict(obs))
                client.send_action(action)
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

        self._connected = False

    def wait_for_connection(self) -> None:
        """Block until the first observation arrives (connection confirmation).

        Uses ``connect_timeout_s`` from config; 0 means wait forever.
        """
        poller = zmq.Poller()
        poller.register(self._obs_socket, zmq.POLLIN)

        if self.config.connect_timeout_s > 0:
            socks = dict(
                poller.poll(int(self.config.connect_timeout_s * 1000))
            )
            if self._obs_socket not in socks or socks[self._obs_socket] != zmq.POLLIN:
                raise TimeoutError(
                    f"Timeout waiting for observations from "
                    f"{self.config.remote_ip}:{self.config.port_zmq_observations}"
                )
        else:
            last_log = 0.0
            while True:
                socks = dict(poller.poll(1000))
                if self._obs_socket in socks and socks[self._obs_socket] == zmq.POLLIN:
                    break
                now = time.time()
                if now - last_log >= 5.0:
                    logger.info("Waiting for host observations...")
                    last_log = now

        self._connected = True
        logger.info("Connected to host.")

    def recv_observation(self) -> dict | None:
        """Receive the latest observation dict from the host.

        Returns ``None`` if no observation is available within the
        polling timeout.
        """
        poller = zmq.Poller()
        poller.register(self._obs_socket, zmq.POLLIN)
        socks = dict(poller.poll(self.config.polling_timeout_ms))

        latest_raw = None
        if self._obs_socket in socks and socks[self._obs_socket] == zmq.POLLIN:
            # Drain stale messages, keep only the latest
            while True:
                try:
                    latest_raw = self._obs_socket.recv_string(zmq.NOBLOCK)
                except zmq.Again:
                    break

        if latest_raw is None:
            return None

        return _json.loads(latest_raw)

    def send_action(self, action: np.ndarray | dict) -> None:
        """Send an action to the host.

        Accepts either a numpy array (serialized as JSON list) or a
        dict (serialized as JSON object).
        """
        if isinstance(action, np.ndarray):
            payload = _json.dumps(action.tolist())
        else:
            payload = _json.dumps(action)
        self._cmd_socket.send_string(payload, flags=zmq.NOBLOCK)

    def close(self) -> None:
        """Release ZMQ sockets and context."""
        self._obs_socket.close(linger=0)
        self._cmd_socket.close(linger=0)
        self._context.term()
        self._connected = False


# ═══════════════════════════════════════════════════════════════════
# Observation encoding
# ═══════════════════════════════════════════════════════════════════


def encode_observation_json(obs: dict) -> str:
    """Serialize a flat observation dict to JSON for ZMQ transport.

    Handles numpy arrays by converting them to lists.
    """
    serializable: dict = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value
    return _json.dumps(serializable)
