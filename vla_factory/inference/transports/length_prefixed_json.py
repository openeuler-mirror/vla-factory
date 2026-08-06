"""Length-prefixed JSON RPC transport with numpy array support.

The wire format is compatible with RoboTwin's ``ModelClient`` but the transport
does not know anything about RoboTwin observations, cameras, joints or actions.
It only decodes ``{cmd, obs}``, dispatches a method on the supplied handler and
encodes ``{res}`` or a structured error response.
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

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """Encode numpy values with enough metadata for exact reconstruction."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(
                    np.ascontiguousarray(obj).tobytes()
                ).decode("ascii"),
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
    """Serialize Python data containing numpy values to JSON."""
    return json.dumps(data, cls=_NumpyEncoder)


def json_to_numpy(json_str: str) -> Any:
    """Deserialize JSON and reconstruct encoded numpy arrays."""

    def _hook(dct: dict) -> Any:
        if "__numpy_array__" in dct:
            raw = base64.b64decode(dct["data"])
            return np.frombuffer(raw, dtype=dct["dtype"]).reshape(dct["shape"])
        return dct

    return json.loads(json_str, object_hook=_hook)


def _recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly ``n`` bytes; return ``None`` if the peer closes."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 4096))
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class LengthPrefixedJsonRpcServer:
    """Serve an RPC handler using 4-byte length-prefixed numpy-aware JSON."""

    def __init__(self, handler: Any, host: str = "0.0.0.0", port: int = 9999) -> None:
        self.handler = handler
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
        logger.info("RPC server listening on %s:%d", self.host, self.port)

        threads: list[threading.Thread] = []
        try:
            while self._running:
                try:
                    client, addr = self._server_socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if not self._running:
                        break
                    raise
                logger.info("RPC client connected from %s", addr)
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client,),
                    daemon=True,
                )
                thread.start()
                threads.append(thread)
        except KeyboardInterrupt:
            logger.info("Interrupted; shutting down RPC server.")
        finally:
            self.stop()
            for thread in threads:
                thread.join(timeout=1)

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
                    logger.info("RPC client disconnected.")
                    return
                length = int.from_bytes(header, "big")
                payload = _recv_exactly(client, length)
                if payload is None:
                    logger.info("RPC client disconnected mid-message.")
                    return
                try:
                    request = json_to_numpy(payload.decode("utf-8"))
                    response = {"res": self._dispatch(request)}
                except Exception as exc:  # report error to peer before closing
                    logger.warning("RPC request failed: %s", exc)
                    response = {
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    self._send(client, response)
                    return
                self._send(client, response)

    def _dispatch(self, request: dict[str, Any]) -> Any:
        cmd = request.get("cmd")
        obs = request.get("obs")
        if not isinstance(cmd, str):
            raise AttributeError(f"No model method named {cmd!r}")
        method = getattr(self.handler, cmd, None)
        if not callable(method):
            raise AttributeError(f"No model method named '{cmd}'")
        return method(obs) if obs is not None else method()

    @staticmethod
    def _send(client: socket.socket, response: dict[str, Any]) -> None:
        body = numpy_to_json(response).encode("utf-8")
        client.sendall(len(body).to_bytes(4, "big"))
        client.sendall(body)
