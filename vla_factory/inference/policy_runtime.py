"""Policy runtime: the two serving forms around a PolicyExecutor.

Orchestration composes adapters + an executable policy and owns episode
reset, pacing and request handling — but never a wire protocol. The two
forms mirror each other; which one applies depends on who initiates the
connection (see the deploy-module doc, section 1.2):

============  ==========================  =================================
Form          Orchestrator                Transport (owned elsewhere)
============  ==========================  =================================
client-shaped ``PolicyRunner``            an injected client satisfying
              (active loop: recv → adapt  ``PolicyClientTransport``
              → predict → adapt → send)   (e.g. ``transports.zmq``)
server-shaped ``RemotePolicyModel``       an RPC server dispatching
              (passive handler:           ``{cmd, obs}`` to handler methods
              reset/update/get_action)    (e.g. ``transports.
                                          length_prefixed_json``)
============  ==========================  =================================

Platform differences live entirely in the injected adapters; both forms
consume only ``ActionCommand`` produced by the executable policy.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from vla_factory.inference.infer import ActionCommand, PolicyExecutor
from vla_factory.inference.platforms.base import PlatformObservationAdapter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Client-shaped form — active loop driving an injected client transport
# ═══════════════════════════════════════════════════════════════════


@runtime_checkable
class PolicyClientTransport(Protocol):
    """The client-transport contract the runner drives.

    Implementations move observation/action payloads and own the connection;
    they must not interpret observations or model semantics.
    """

    def wait_for_connection(self) -> None:
        ...

    def recv_observation(self) -> dict | None:
        ...

    def send_action(self, action: Any) -> None:
        ...

    def close(self) -> None:
        ...


class PolicyRunner:
    """Drive receive → adapt → predict → adapt → send over a client transport.

    Parameters
    ----------
    policy : PolicyExecutor
        A composed inference + execution policy that always returns
        ``ActionCommand`` with shape ``[N, D]``.
    obs_adapter : PlatformObservationAdapter
        Platform observation → ``ObsDict``.
    action_adapter : Callable[[np.ndarray], Any] | None
        Optional action vector → platform command (e.g. per-motor dict for the
        lerobot host). ``None`` sends the raw action array.
    task : str
        Task instruction forwarded to the observation adapter (for
        language-conditioned policies).
    max_loop_freq_hz : float
        Loop frequency cap. Pacing is an orchestration concern, so it lives
        here rather than on the transport.
    """

    def __init__(
        self,
        policy: PolicyExecutor,
        obs_adapter: PlatformObservationAdapter,
        action_adapter: Callable[[np.ndarray], Any] | None = None,
        task: str = "",
        max_loop_freq_hz: float = 60.0,
    ) -> None:
        if max_loop_freq_hz <= 0:
            raise ValueError(
                f"max_loop_freq_hz must be positive, got {max_loop_freq_hz}"
            )
        self.policy = policy
        self.obs_adapter = obs_adapter
        self.action_adapter = action_adapter
        self.task = task
        self.max_loop_freq_hz = max_loop_freq_hz

    def run(self, client: PolicyClientTransport) -> None:
        """Run the deployment loop over ``client`` until interrupted.

        The client is closed on exit. A ``TimeoutError`` from
        ``wait_for_connection`` (connect timeout configured on the transport)
        propagates to the caller.
        """
        try:
            client.wait_for_connection()
            logger.info("Connected. Waiting for observations.")
            while True:
                loop_start = time.time()
                observation = client.recv_observation()
                if observation is None:
                    self._pace(loop_start)
                    continue
                if (
                    isinstance(observation, dict)
                    and observation.get("__control__") == "reset"
                ):
                    self.policy.reset()
                    logger.info(
                        "Reset episode=%d.",
                        observation.get("episode_index", -1),
                    )
                    continue

                command = self.policy.predict(
                    self.obs_adapter(observation, self.task)
                )
                if not isinstance(command, ActionCommand):
                    raise TypeError(
                        "PolicyExecutor.predict() must return ActionCommand, got "
                        f"{type(command).__name__}."
                    )
                if self.action_adapter is not None:
                    action = self.action_adapter(command.single())
                else:
                    action = command.values
                client.send_action(action)
                self._pace(loop_start)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt. Exiting.")
        finally:
            client.close()

    def _pace(self, loop_start: float) -> None:
        elapsed = time.time() - loop_start
        time.sleep(max(1.0 / self.max_loop_freq_hz - elapsed, 0.0))


# ═══════════════════════════════════════════════════════════════════
# Server-shaped form — passive handler behind an RPC transport
# ═══════════════════════════════════════════════════════════════════


class RemotePolicyModel:
    """Expose reset/update/predict-style RPC methods over an executable policy.

    The method set (``reset_model`` / ``update_obs`` / ``get_action``) is the
    calling convention of RoboTwin's remote-policy client, not an invention of
    this framework.
    """

    def __init__(
        self,
        policy: PolicyExecutor,
        adapter: PlatformObservationAdapter,
        task: str = "",
    ) -> None:
        self.policy = policy
        self.adapter = adapter
        self.task = task
        self._last_obs: Any = None

    def reset_model(self, obs: Any = None) -> None:
        self.policy.reset()
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
        adapted = self.adapter(obs, self.task)
        command = self.policy.predict(adapted)
        if not isinstance(command, ActionCommand):
            raise TypeError(
                "PolicyExecutor.predict() must return ActionCommand, got "
                f"{type(command).__name__}."
            )
        return command.values


__all__ = [
    "PolicyClientTransport",
    "PolicyRunner",
    "RemotePolicyModel",
]
