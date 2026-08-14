"""Turn model action chunks into commands for a deployment platform."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np


def _validated_actions(value: Any, *, name: str) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim != 2 or 0 in actions.shape:
        raise ValueError(
            f"{name} must be a non-empty [steps, action_dim] array; "
            f"got {actions.shape}."
        )
    if not np.isfinite(actions).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return np.ascontiguousarray(actions)


@dataclass(frozen=True)
class ActionChunk:
    """A complete ``[horizon, action_dim]`` model prediction."""

    values: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "values", _validated_actions(self.values, name="ActionChunk")
        )

    @property
    def horizon(self) -> int:
        return self.values.shape[0]

    @property
    def action_dim(self) -> int:
        return self.values.shape[1]


@dataclass(frozen=True)
class ActionCommand:
    """The ``[num_steps, action_dim]`` actions for one platform interaction."""

    values: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "values", _validated_actions(self.values, name="ActionCommand")
        )

    @property
    def num_steps(self) -> int:
        return self.values.shape[0]

    def single(self) -> np.ndarray:
        if self.num_steps != 1:
            raise ValueError(
                f"A single-step command was required, got {self.num_steps} steps."
            )
        return self.values[0]


class ExecutionStrategy(str, Enum):
    SYNCHRONOUS = "synchronous"
    TEMPORAL_ENSEMBLING = "temporal_ensembling"
    RECEDING_HORIZON = "receding_horizon"


class ExecutionPolicy(ABC):
    """Stateful policy selecting commands from predicted action chunks."""

    def __init__(
        self,
        *,
        action_horizon: int,
        action_dim: int,
        n_action_steps: int,
    ) -> None:
        if action_horizon < 1 or action_dim < 1:
            raise ValueError("action_horizon and action_dim must be positive")
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps

    @property
    @abstractmethod
    def needs_chunk(self) -> bool:
        """Whether the next command requires a new model prediction."""

    @abstractmethod
    def consume(self, chunk: ActionChunk | None) -> ActionCommand:
        """Select the command for the current platform interaction."""

    @abstractmethod
    def reset(self) -> None:
        """Discard temporal state at an episode boundary."""

    def _require_chunk(self, chunk: ActionChunk | None) -> np.ndarray:
        if chunk is None:
            raise ValueError("This execution step requires an ActionChunk.")
        expected = (self.action_horizon, self.action_dim)
        if chunk.values.shape != expected:
            raise ValueError(
                f"ActionChunk shape mismatch: expected {expected}, "
                f"got {chunk.values.shape}."
            )
        return chunk.values


class SynchronousExecution(ExecutionPolicy):
    @property
    def needs_chunk(self) -> bool:
        return True

    def consume(self, chunk: ActionChunk | None) -> ActionCommand:
        return ActionCommand(self._require_chunk(chunk)[: self.n_action_steps])

    def reset(self) -> None:
        return None


class TemporalEnsemblingExecution(ExecutionPolicy):
    def __init__(self, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self._chunks: deque[np.ndarray] = deque()

    @property
    def needs_chunk(self) -> bool:
        return True

    def consume(self, chunk: ActionChunk | None) -> ActionCommand:
        actions = self._require_chunk(chunk)
        self._chunks.append(actions)
        if len(self._chunks) > self.action_horizon:
            self._chunks.popleft()
        count = len(self._chunks)
        values = [self._chunks[i][count - 1 - i] for i in range(count)]
        weights = np.array([1.0 / (count - i) for i in range(count)])
        averaged = np.average(values, weights=weights, axis=0)
        return ActionCommand(averaged[None, :])

    def reset(self) -> None:
        self._chunks.clear()


class RecedingHorizonExecution(ExecutionPolicy):
    def __init__(self, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self._actions: deque[np.ndarray] = deque()

    @property
    def needs_chunk(self) -> bool:
        return not self._actions

    def consume(self, chunk: ActionChunk | None) -> ActionCommand:
        if self._actions:
            if chunk is not None:
                raise ValueError("A new chunk cannot be supplied during playback.")
        else:
            self._actions.extend(self._require_chunk(chunk)[: self.n_action_steps])
        return ActionCommand(self._actions.popleft()[None, :])

    def reset(self) -> None:
        self._actions.clear()


_EXECUTION_POLICIES: dict[ExecutionStrategy, type[ExecutionPolicy]] = {
    ExecutionStrategy.SYNCHRONOUS: SynchronousExecution,
    ExecutionStrategy.TEMPORAL_ENSEMBLING: TemporalEnsemblingExecution,
    ExecutionStrategy.RECEDING_HORIZON: RecedingHorizonExecution,
}


def build_execution_policy(
    strategy: ExecutionStrategy | str,
    *,
    action_horizon: int,
    action_dim: int,
    n_action_steps: int | None = None,
) -> ExecutionPolicy:
    """Build one action-execution policy from its public strategy name."""
    try:
        strategy = ExecutionStrategy(strategy)
    except ValueError as exc:
        raise ValueError(f"Unknown execution strategy {strategy!r}.") from exc

    steps = action_horizon if n_action_steps is None else n_action_steps
    if not 1 <= steps <= action_horizon:
        raise ValueError(
            "n_action_steps must satisfy 1 <= n_action_steps <= action_horizon; "
            f"got {steps}."
        )
    if (
        strategy == ExecutionStrategy.TEMPORAL_ENSEMBLING
        and n_action_steps not in (None, 1)
    ):
        raise ValueError(
            "temporal_ensembling always emits one step; "
            "n_action_steps must be omitted or 1."
        )

    return _EXECUTION_POLICIES[strategy](
        action_horizon=action_horizon,
        action_dim=action_dim,
        n_action_steps=steps,
    )


class ChunkPolicy(Protocol):
    def predict(self, observation: Any) -> ActionChunk:
        ...

    def reset(self) -> None:
        ...


class PolicyExecutor:
    """Compose a chunk-producing inference engine with an execution policy."""

    def __init__(
        self,
        engine: ChunkPolicy,
        execution_policy: ExecutionPolicy,
    ) -> None:
        self.engine = engine
        self.execution_policy = execution_policy

    def predict(self, observation: Any) -> ActionCommand:
        chunk = (
            self.engine.predict(observation)
            if self.execution_policy.needs_chunk
            else None
        )
        return self.execution_policy.consume(chunk)

    def reset(self) -> None:
        self.engine.reset()
        self.execution_policy.reset()


class ReplayPolicy:
    """Replay recorded actions without running model inference."""

    def __init__(self, episode_data: list[dict]) -> None:
        self.data = episode_data
        self._index = 0

    def predict(self, observation: Any) -> ActionCommand:
        if self._index >= len(self.data):
            raise StopIteration("Episode replay exhausted")
        action = np.asarray(self.data[self._index]["action"], dtype=np.float32)
        self._index += 1
        if action.ndim == 1:
            action = action[None, :]
        return ActionCommand(action)

    def reset(self) -> None:
        self._index = 0


__all__ = [
    "ActionChunk",
    "ActionCommand",
    "ExecutionPolicy",
    "ExecutionStrategy",
    "PolicyExecutor",
    "ReplayPolicy",
    "build_execution_policy",
]
