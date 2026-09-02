"""L0 tests for the transport layer (framing / serialization only).

``deploy/transports/`` is pure connection + framing: no adapters, no
inference. Its failure modes are also the least visible ones — a short read or
a mis-sized length prefix hangs the deployment loop instead of raising, so it
looks like "the robot stopped responding" rather than a bug. These tests drive
both transports over real loopback sockets on ephemeral ports.

Before Issue #7 ``transports/zmq.py`` had no direct test at all, and
``length_prefixed_json.py`` was only exercised through well-formed RoboTwin
requests.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np
import pytest
import zmq

import vla_factory.inference.transports.zmq as zmq_transport
from vla_factory.inference.transports.length_prefixed_json import (
    LengthPrefixedJsonRpcServer,
    _recv_exactly,
    json_to_numpy,
    numpy_to_json,
)
from vla_factory.inference.transports.zmq import ZmqPolicyClient, ZmqPolicyClientConfig


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ══════════════════════════════════════════════════════════════════════
#  length-prefixed JSON: frame boundaries
# ══════════════════════════════════════════════════════════════════════


def test_recv_exactly_reassembles_a_split_payload():
    """A payload arriving in several TCP segments must be rejoined, not truncated.

    TCP gives no message boundaries: a 4096+ byte frame routinely arrives in
    pieces. Returning the first chunk would silently corrupt every large
    observation.
    """
    left, right = socket.socketpair()
    with left, right:
        def _dribble():
            for piece in (b"abc", b"de", b"fghij"):
                right.sendall(piece)
                time.sleep(0.01)

        sender = threading.Thread(target=_dribble, daemon=True)
        sender.start()
        assert _recv_exactly(left, 10) == b"abcdefghij"
        sender.join(timeout=2)


def test_recv_exactly_returns_none_when_peer_closes_early():
    """A peer that disappears mid-frame must surface as None, not block forever."""
    left, right = socket.socketpair()
    with left:
        right.sendall(b"only-four")
        right.close()
        assert _recv_exactly(left, 100) is None


def test_recv_exactly_zero_bytes_is_a_noop():
    left, right = socket.socketpair()
    with left, right:
        assert _recv_exactly(left, 0) == b""


class _EchoHandler:
    """Minimal RPC handler: one no-arg method, one that takes an observation."""

    def __init__(self) -> None:
        self.calls: list = []

    def ping(self):
        self.calls.append(("ping", None))
        return "pong"

    def echo(self, obs):
        self.calls.append(("echo", obs))
        return obs


@pytest.fixture
def rpc_server():
    handler = _EchoHandler()
    port = _free_port()
    server = LengthPrefixedJsonRpcServer(handler, host="127.0.0.1", port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, handler, port
    server.stop()
    thread.join(timeout=2)


def _connect(port: int, tries: int = 100) -> socket.socket:
    last: OSError | None = None
    for _ in range(tries):
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=5)
        except OSError as exc:  # server thread not listening yet
            last = exc
            time.sleep(0.05)
    raise last  # type: ignore[misc]


def _send_frame(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(len(payload).to_bytes(4, "big"))
    sock.sendall(payload)


def _read_frame(sock: socket.socket) -> dict:
    header = _recv_exactly(sock, 4)
    assert header is not None, "server closed without answering"
    body = _recv_exactly(sock, int.from_bytes(header, "big"))
    assert body is not None
    return json_to_numpy(body.decode("utf-8"))


def test_frame_split_across_writes_is_served(rpc_server):
    """The server must reassemble a request that arrives in pieces."""
    _, handler, port = rpc_server
    body = numpy_to_json({"cmd": "ping"}).encode("utf-8")

    with _connect(port) as sock:
        sock.sendall(len(body).to_bytes(4, "big"))
        for i in range(0, len(body), 3):  # dribble the payload out
            sock.sendall(body[i:i + 3])
            time.sleep(0.005)
        assert _read_frame(sock) == {"res": "pong"}

    assert handler.calls == [("ping", None)]


def test_two_requests_on_one_connection_keep_their_boundaries(rpc_server):
    """Back-to-back frames must not bleed into each other."""
    _, handler, port = rpc_server
    with _connect(port) as sock:
        _send_frame(sock, numpy_to_json({"cmd": "ping"}).encode("utf-8"))
        assert _read_frame(sock) == {"res": "pong"}
        _send_frame(sock, numpy_to_json({"cmd": "echo", "obs": {"a": 1}}).encode("utf-8"))
        assert _read_frame(sock) == {"res": {"a": 1}}


def test_malformed_json_returns_a_structured_error(rpc_server):
    """Garbage on the wire must come back as an error frame, not a silent hang."""
    _, _, port = rpc_server
    with _connect(port) as sock:
        _send_frame(sock, b"{not json at all")
        response = _read_frame(sock)

    assert "error" in response
    assert "traceback" in response, "the peer needs the server-side trace to debug"


def test_oversized_length_prefix_does_not_wedge_the_server(rpc_server):
    """A header promising more bytes than are sent must end that connection cleanly.

    The server treats it as a mid-message disconnect; crucially it must keep
    serving afterwards rather than leaving the accept loop stuck.
    """
    _, handler, port = rpc_server
    liar = _connect(port)
    liar.sendall((10_000).to_bytes(4, "big"))
    liar.sendall(b"far too short")
    liar.close()

    # The server must still answer a well-formed client.
    with _connect(port) as sock:
        _send_frame(sock, numpy_to_json({"cmd": "ping"}).encode("utf-8"))
        assert _read_frame(sock) == {"res": "pong"}


def test_client_disconnect_before_any_frame_is_clean(rpc_server):
    _, _, port = rpc_server
    _connect(port).close()

    with _connect(port) as sock:
        _send_frame(sock, numpy_to_json({"cmd": "ping"}).encode("utf-8"))
        assert _read_frame(sock) == {"res": "pong"}


def test_numpy_roundtrip_preserves_dtype_and_shape():
    """The wire codec must reconstruct arrays exactly — a float64/float32 slip
    silently changes what the model is fed."""
    original = {
        "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "state": np.linspace(-1, 1, 7, dtype=np.float32),
        "count": np.int64(5),
        "flag": np.bool_(True),
    }
    restored = json_to_numpy(numpy_to_json(original))

    for key in ("image", "state"):
        assert restored[key].dtype == original[key].dtype
        assert restored[key].shape == original[key].shape
        np.testing.assert_array_equal(restored[key], original[key])
    assert restored["count"] == 5 and isinstance(restored["count"], int)
    assert restored["flag"] is True


def test_non_contiguous_array_survives_the_roundtrip():
    """Slices/transposes are non-contiguous; tobytes() must not reorder them."""
    original = np.arange(12, dtype=np.float32).reshape(3, 4).T
    restored = json_to_numpy(numpy_to_json({"a": original}))["a"]
    np.testing.assert_array_equal(restored, original)


# ══════════════════════════════════════════════════════════════════════
#  ZMQ PUSH/PULL client
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def zmq_host():
    """A LeKiwi-shaped host: PULL for actions, PUSH for observations."""
    context = zmq.Context()
    action_sink = context.socket(zmq.PULL)      # receives what the client PUSHes
    obs_source = context.socket(zmq.PUSH)       # feeds what the client PULLs
    cmd_port = action_sink.bind_to_random_port("tcp://127.0.0.1")
    obs_port = obs_source.bind_to_random_port("tcp://127.0.0.1")

    yield action_sink, obs_source, cmd_port, obs_port

    action_sink.close(linger=0)
    obs_source.close(linger=0)
    context.term()


def _client(cmd_port: int, obs_port: int, **kw) -> ZmqPolicyClient:
    return ZmqPolicyClient(ZmqPolicyClientConfig(
        remote_ip="127.0.0.1",
        port_zmq_cmd=cmd_port,
        port_zmq_observations=obs_port,
        **kw,
    ))


def test_recv_observation_parses_host_json(zmq_host):
    _, obs_source, cmd_port, obs_port = zmq_host
    client = _client(cmd_port, obs_port, polling_timeout_ms=2000)
    try:
        obs_source.send_string(json.dumps({"observation.state": [1.0, 2.0]}))
        assert client.recv_observation() == {"observation.state": [1.0, 2.0]}
    finally:
        client.close()


def test_recv_observation_returns_none_on_timeout(zmq_host):
    """A silent host must yield None so the runner can retry, not raise."""
    _, _, cmd_port, obs_port = zmq_host
    client = _client(cmd_port, obs_port, polling_timeout_ms=50)
    try:
        assert client.recv_observation() is None
    finally:
        client.close()


def test_conflate_is_configured_before_connect(monkeypatch):
    """ZMQ applies CONFLATE only when it is set before connect/bind."""
    sockets = []

    class RecordingSocket:
        def __init__(self):
            self.events = []

        def setsockopt(self, option, value):
            self.events.append(("setsockopt", option, value))

        def connect(self, endpoint):
            self.events.append(("connect", endpoint))

    class RecordingContext:
        def socket(self, _kind):
            socket = RecordingSocket()
            sockets.append(socket)
            return socket

    monkeypatch.setattr(zmq_transport.zmq, "Context", RecordingContext)
    ZmqPolicyClient(ZmqPolicyClientConfig())

    assert len(sockets) == 2
    for socket in sockets:
        assert socket.events[0] == ("setsockopt", zmq.CONFLATE, 1)
        assert socket.events[1][0] == "connect"


def test_recv_observation_keeps_only_the_newest_frame(zmq_host):
    """A backlog must yield the freshest observation.

    Acting on a stale frame is worse than skipping it — the arm would move on
    where the scene used to be.
    """
    _, obs_source, cmd_port, obs_port = zmq_host
    client = _client(cmd_port, obs_port, polling_timeout_ms=2000)
    try:
        for i in range(5):
            obs_source.send_string(json.dumps({"seq": i}))
        time.sleep(0.2)  # let the whole burst land before reading
        assert client.recv_observation() == {"seq": 4}
    finally:
        client.close()


def test_send_action_serializes_ndarray_as_json_list(zmq_host):
    action_sink, _, cmd_port, obs_port = zmq_host
    client = _client(cmd_port, obs_port)
    try:
        client.send_action(np.array([0.5, -1.5], dtype=np.float32))
        assert json.loads(action_sink.recv_string()) == [0.5, -1.5]
    finally:
        client.close()


def test_send_action_passes_dicts_through(zmq_host):
    """The lerobot host format is a per-motor dict, not an array."""
    action_sink, _, cmd_port, obs_port = zmq_host
    client = _client(cmd_port, obs_port)
    try:
        client.send_action({"shoulder.pos": 1.0, "gripper.pos": 0.0})
        assert json.loads(action_sink.recv_string()) == {
            "shoulder.pos": 1.0, "gripper.pos": 0.0
        }
    finally:
        client.close()


def test_wait_for_connection_times_out_when_host_is_silent(zmq_host):
    """A bounded connect timeout must raise TimeoutError — the CLI maps it to exit 1."""
    _, _, cmd_port, obs_port = zmq_host
    client = _client(cmd_port, obs_port, connect_timeout_s=0.1)
    try:
        with pytest.raises(TimeoutError, match="Timeout waiting for observations"):
            client.wait_for_connection()
    finally:
        client.close()


def test_wait_for_connection_returns_once_an_observation_arrives(zmq_host):
    _, obs_source, cmd_port, obs_port = zmq_host
    client = _client(cmd_port, obs_port, connect_timeout_s=5.0)
    try:
        obs_source.send_string(json.dumps({"observation.state": [0.0]}))
        client.wait_for_connection()  # must return, not raise
    finally:
        client.close()
