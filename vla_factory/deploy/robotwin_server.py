"""RoboTwin-compatible model server for the ``robotwin`` serve platform.

RoboTwin's ``policy_model_server.py`` runs the policy in its own process and lets
the simulator connect as a client (``ModelClient`` in ``eval_policy_client.py``).
This module implements the *same wire protocol* so a VLA Factory checkpoint can
stand in as that server — keeping the heavy model deps (lerobot/openpi) out of
the RoboTwin/SAPIEN environment (cross-process isolation).

Wire protocol (verbatim from RoboTwin, so its stock client interoperates):
  - framing: 4-byte big-endian length prefix, then a UTF-8 JSON payload;
  - numpy arrays encode as ``{"__numpy_array__": True, "data": <base64>,
    "dtype": ..., "shape": ...}``;
  - request ``{"cmd": <method>, "obs": <obs>}`` → ``getattr(model, cmd)(obs)``
    (called with no arg when ``obs`` is null) → response ``{"res": <result>}``;
    errors return ``{"error": ..., "traceback": ...}``.

The served "model" is :class:`RobotwinEngineModel`, a thin wrapper exposing the
methods a RoboTwin ``deploy_policy`` calls (``reset_model`` / ``update_obs`` /
``get_action``) over an :class:`InferenceEngine`.
"""

from __future__ import annotations

import base64
import json
import logging
import socket
import threading
import traceback
from typing import Any

import numpy as np

from vla_factory.deploy.infer import InferenceEngine
from vla_factory.deploy.robotwin_adapter import RobotwinObsAdapter

logger = logging.getLogger(__name__)


# ── Wire codec (RoboTwin NumpyEncoder-compatible) ─────────────────


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that serialises numpy arrays with reconstruction metadata."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(np.ascontiguousarray(obj).tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_to_json(data: Any) -> str:
    """Serialise Python data (with numpy arrays) to a JSON string."""
    return json.dumps(data, cls=_NumpyEncoder)


def json_to_numpy(json_str: str) -> Any:
    """Deserialise a JSON string, reconstructing numpy arrays."""

    def _hook(dct: dict) -> Any:
        if "__numpy_array__" in dct:
            raw = base64.b64decode(dct["data"])
            return np.frombuffer(raw, dtype=dct["dtype"]).reshape(dct["shape"])
        return dct

    return json.loads(json_str, object_hook=_hook)


def _recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly ``n`` bytes; return None on clean EOF before any byte."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 4096))
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# ── Engine wrapper exposing the RoboTwin deploy_policy method contract ──


class RobotwinEngineModel:
    """Adapt an :class:`InferenceEngine` to RoboTwin's RPC method contract.

    Methods mirror what a RoboTwin ``deploy_policy`` invokes on the model proxy:

    - ``reset_model()``   — clear engine chunk buffers (per-episode reset);
    - ``update_obs(obs)`` — cache the latest encoded observation;
    - ``get_action(obs)`` — predict and return an action **chunk**
      ``[n_steps, action_dim]`` (RoboTwin iterates ``for action in actions``).
      ``obs`` may be omitted to reuse the last ``update_obs``.

    The engine must use the ``synchronous`` execution strategy so ``predict``
    returns the full ``[action_horizon, action_dim]`` chunk.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        adapter: RobotwinObsAdapter,
        task: str = "",
        n_action_steps: int | None = None,
    ) -> None:
        self.engine = engine
        self.adapter = adapter
        self.task = task
        self.n_action_steps = n_action_steps
        self._last_obs: Any = None

    def reset_model(self, obs: Any = None) -> None:
        self.engine.reset()
        self._last_obs = None
        return None

    def update_obs(self, obs: Any) -> None:
        self._last_obs = obs
        return None

    def get_action(self, obs: Any = None) -> np.ndarray:
        if obs is None:
            obs = self._last_obs
        if obs is None:
            raise RuntimeError(
                "get_action called before any observation (no obs arg and no "
                "prior update_obs)."
            )
        chunk = np.asarray(self.engine.predict(self.adapter(obs, self.task)), dtype=np.float32)
        if chunk.ndim == 1:  # defensive: a single-step strategy → make it a 1-step chunk
            chunk = chunk[None, :]
        if self.n_action_steps is not None:
            chunk = chunk[: self.n_action_steps]
        return chunk


# ── TCP server ────────────────────────────────────────────────────


class RobotwinModelServer:
    """Serve a model object over RoboTwin's TCP protocol.

    Blocking: :meth:`serve_forever` listens and dispatches each client in a
    thread until interrupted. Any public method on ``model`` is reachable as a
    ``cmd``; only the deploy_policy contract (reset_model/update_obs/get_action)
    is expected in practice.
    """

    def __init__(self, model: Any, host: str = "0.0.0.0", port: int = 9999) -> None:
        self.model = model
        self.host = host
        self.port = port
        self._server_socket: socket.socket | None = None
        self._running = False

    def serve_forever(self) -> None:
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.settimeout(1.0)
        self._server_socket.listen(5)
        self._running = True
        logger.info("RoboTwin model server listening on %s:%d", self.host, self.port)

        threads: list[threading.Thread] = []
        try:
            while self._running:
                try:
                    client, addr = self._server_socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    # Socket closed by stop() during shutdown — exit cleanly.
                    if not self._running:
                        break
                    raise
                logger.info("RoboTwin client connected from %s", addr)
                t = threading.Thread(target=self._handle_client, args=(client,), daemon=True)
                t.start()
                threads.append(t)
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down RoboTwin server.")
        finally:
            self.stop()
            for t in threads:
                t.join(timeout=1)

    def stop(self) -> None:
        self._running = False
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None

    def _handle_client(self, client: socket.socket) -> None:
        with client:
            while self._running:
                header = _recv_exactly(client, 4)
                if header is None:
                    logger.info("RoboTwin client disconnected.")
                    return
                length = int.from_bytes(header, "big")
                payload = _recv_exactly(client, length)
                if payload is None:
                    logger.info("RoboTwin client disconnected mid-message.")
                    return
                try:
                    data = json_to_numpy(payload.decode("utf-8"))
                    cmd = data.get("cmd")
                    obs = data.get("obs")
                    method = getattr(self.model, cmd, None)
                    if not callable(method):
                        raise AttributeError(f"No model method named '{cmd}'")
                    result = method(obs) if obs is not None else method()
                    response = {"res": result}
                except Exception as exc:  # noqa: BLE001 — report to client, keep serving
                    logger.warning("Error handling '%s': %s", locals().get("cmd"), exc)
                    response = {"error": str(exc), "traceback": traceback.format_exc()}
                    self._send(client, response)
                    return
                self._send(client, response)

    @staticmethod
    def _send(client: socket.socket, response: dict) -> None:
        body = numpy_to_json(response).encode("utf-8")
        client.sendall(len(body).to_bytes(4, "big"))
        client.sendall(body)
