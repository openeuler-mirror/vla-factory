"""Judge-owned boundaries for policy serving and model parity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


VERSION = "robotwin-policy/v1"
ActionChunk = Sequence[Sequence[float]]
PARITY_GATES = (
    "loss_parity",
)


class Policy(Protocol):
    """Judge-facing benchmark policy operations and native loss capture."""

    def reset(self, task_info: Mapping[str, Any]) -> None:
        """Clear per-episode state; task_info may be empty."""

    def act(self, observation: Mapping[str, Any]) -> ActionChunk:
        """Convert one native benchmark observation to an action chunk."""

    def compute_loss(self, observation: Mapping[str, Any]) -> float:
        """Apply official preprocessing and native_model.forward()."""


def document(benchmark: str) -> str:
    """Load the benchmark-owned contract that is copied into each workspace."""
    from .benchmarks import contract_document
    return contract_document(benchmark, VERSION)
