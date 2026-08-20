"""Serializable transform calls planned by assembly resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TransformStepCall:
    """One registered transform name and its resolved constructor arguments."""

    type: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "args": dict(self.args)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TransformStepCall":
        if not isinstance(value, dict) or not value.get("type"):
            raise ValueError("transform call must be an object with a non-empty 'type'")
        args = value.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("transform call 'args' must be an object")
        return cls(type=str(value["type"]), args=dict(args))


@dataclass(frozen=True)
class TransformPipelinePlan:
    """A complete ordered transform plan.

    Empty calls are a valid identity plan. Failure to compile a declaration is
    an exception, not a partially-resolved value.
    """

    calls: tuple[TransformStepCall, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"calls": [call.to_dict() for call in self.calls]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TransformPipelinePlan":
        if not isinstance(value, dict) or "calls" not in value:
            raise ValueError("transform plan must be an object containing 'calls'")
        calls = value["calls"]
        if not isinstance(calls, list):
            raise ValueError("transform plan 'calls' must be a list")
        return cls(calls=tuple(TransformStepCall.from_dict(call) for call in calls))
