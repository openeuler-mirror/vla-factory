#!/usr/bin/env python3
"""Serve a Policy Contract implementation over RoboTwin's native TCP protocol."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import socket
from pathlib import Path
from typing import Any


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as error:
        raise RuntimeError("RoboTwin Policy server needs numpy in the candidate runtime") from error
    return numpy


def _encode(value: Any) -> Any:
    np = _numpy()
    if isinstance(value, np.ndarray):
        return {"__numpy_array__": True, "data": base64.b64encode(value.tobytes()).decode(),
                "dtype": str(value.dtype), "shape": value.shape}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _decode(value: dict[str, Any]) -> Any:
    if value.get("__numpy_array__"):
        np = _numpy()
        return np.frombuffer(base64.b64decode(value["data"]), dtype=value["dtype"]).reshape(value["shape"])
    return value


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    while size:
        chunk = connection.recv(size)
        if not chunk:
            raise ConnectionError("client closed connection")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _load(path: Path, checkpoint: str) -> Any:
    spec = importlib.util.spec_from_file_location("candidate_policy", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load policy implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module.build_policy(checkpoint)
    if not all(callable(getattr(policy, name, None)) for name in ("reset", "act", "compute_loss")):
        raise TypeError("build_policy() must return an object with reset(), act(), and compute_loss()")
    return policy


def serve(policy: Any, port: int) -> None:
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen()
        while True:
            connection, _ = server.accept()
            with connection:
                try:
                    while True:
                        size = int.from_bytes(_recv_exact(connection, 4), "big")
                        request = json.loads(_recv_exact(connection, size), object_hook=_decode)
                        if request["cmd"] == "reset_model":
                            policy.reset({})
                            response: Any = None
                        elif request["cmd"] == "get_action":
                            response = policy.act(request["obs"])
                        else:
                            raise ValueError(f"unsupported RoboTwin command: {request['cmd']}")
                        body = json.dumps({"res": response}, default=_encode).encode()
                        connection.sendall(len(body).to_bytes(4, "big") + body)
                except ConnectionError:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    serve(_load(args.implementation, args.checkpoint), args.port)


if __name__ == "__main__":
    main()
