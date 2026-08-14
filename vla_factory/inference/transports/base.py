"""Transport contract driven by the client-shaped deployment loop."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PolicyClientTransport(Protocol):
    """Move observation and action payloads without interpreting them."""

    def wait_for_connection(self) -> None:
        ...

    def recv_observation(self) -> dict | None:
        ...

    def send_action(self, action: Any) -> None:
        ...

    def close(self) -> None:
        ...


__all__ = ["PolicyClientTransport"]
