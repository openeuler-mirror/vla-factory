"""Dependency-free RoboCasa connector and closed-loop benchmark client.

This module runs in the RoboCasa gymnasium environment and drives the
RoboCasa365 benchmark: it makes one gym env per task, forwards each
observation to the VLA Factory model server (speaking the same
length-prefixed JSON-RPC protocol RoboTwin uses), executes the returned action
chunk via ``env.step``, and aggregates the binary success flag into a
per-task / per-split success-rate report.

It deliberately imports only the standard library and numpy at module level so
the RoboCasa environment can import it without installing VLA Factory's model
dependencies. ``gym`` / ``robocasa`` are imported lazily inside
:func:`run_benchmark` and probed via ``importlib.util.find_spec`` so a missing
simulator surfaces an actionable install hint instead of a traceback.

Run as a module from the RoboCasa environment::

    python -m vla_factory.inference.connectors.robocasa \\
        --port 9999 --task-set atomic_seen --split target \\
        --tasks PickPlaceCounterToCabinet --trials 50 \\
        --report robocasa_report.json
"""


def _encode_numpy(obj):
    """Serialize numpy scalars/arrays to a JSON-safe structure (std lib only).

    Mirrors ``vla_factory.inference.transports.length_prefixed_json`` so the
    server decodes arrays the connector emits exactly. Kept inline (rather than
    imported) to honour the dependency-free contract of this connector.
    """
    import base64
    import json

    import numpy as np

    class _Encoder(json.JSONEncoder):
        def default(self, value):
            if isinstance(value, np.ndarray):
                return {
                    "__numpy_array__": True,
                    "data": base64.b64encode(
                        np.ascontiguousarray(value).tobytes()
                    ).decode("ascii"),
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                }
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            if isinstance(value, np.bool_):
                return bool(value)
            return super().default(value)

    return json.dumps(obj, cls=_Encoder)


def _decode_numpy(text):
    """Reconstruct numpy arrays the server emits (mirrors the transport)."""
    import base64
    import json

    import numpy as np

    def _hook(dct):
        if "__numpy_array__" in dct:
            raw = base64.b64decode(dct["data"])
            return np.frombuffer(raw, dtype=dct["dtype"]).reshape(dct["shape"])
        return dct

    return json.loads(text, object_hook=_hook)


class ModelClient:
    """Length-prefixed JSON-RPC client for the VLA Factory model server.

    The wire format matches
    :class:`vla_factory.inference.transports.length_prefixed_json.LengthPrefixedJsonRpcServer`:
    a 4-byte big-endian length prefix framing a numpy-aware JSON payload. The
    server dispatches ``{"cmd": <method>, "obs": <arg>}`` and returns
    ``{"res": <result>}`` or ``{"error": ..., "traceback": ...}``.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9999) -> None:
        self.host = host
        self.port = port
        self._sock = None

    def connect(self) -> None:
        import socket

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))

    def call(self, func_name: str, obs=None):
        """Invoke a remote method and return its result (or raise on error)."""
        if self._sock is None:
            raise RuntimeError("ModelClient.call before connect().")
        payload = _encode_numpy({"cmd": func_name, "obs": obs}).encode("utf-8")
        self._sock.sendall(len(payload).to_bytes(4, "big"))
        self._sock.sendall(payload)
        response = self._recv_message()
        decoded = _decode_numpy(response.decode("utf-8"))
        if isinstance(decoded, dict) and "error" in decoded and "res" not in decoded:
            raise RuntimeError(
                f"Model server error calling {func_name!r}: {decoded['error']}"
            )
        return decoded.get("res") if isinstance(decoded, dict) else decoded

    def _recv_message(self) -> bytes:
        header = self._recv_exactly(4)
        if header is None:
            raise ConnectionError("Model server closed the connection.")
        length = int.from_bytes(header, "big")
        body = self._recv_exactly(length)
        if body is None:
            raise ConnectionError("Model server closed mid-message.")
        return body

    def _recv_exactly(self, n: int):
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = self._sock.recv(min(remaining, 4096))
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()


def _ensure_simulator():
    """Probe for ``robocasa`` and ``gymnasium``; raise an actionable error."""
    import importlib.util

    missing = [
        name for name in ("gymnasium", "robocasa")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise ModuleNotFoundError(
            "RoboCasa benchmark needs the gymnasium simulator: package(s) "
            f"{missing} are not installed in this environment. Install "
            "robocasa (and its gymnasium dependency) in the RoboCasa "
            "environment, then re-run from there. See "
            "docs/tutorial/robocasa.md for the isolated-environment recipe."
        )


def _resolve_tasks(tasks, task_set):
    """Explicit task list wins; otherwise fetch the task-set from robocasa."""
    if tasks:
        return list(tasks)
    _ensure_simulator()
    try:
        from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
        return list(TASK_SET_REGISTRY[task_set])
    except KeyError as exc:
        raise ValueError(
            f"Unknown RoboCasa task-set {task_set!r}; pass --tasks explicitly "
            "or use a key from robocasa.utils.dataset_registry.TASK_SET_REGISTRY."
        ) from exc


def _make_progress_bar(total, desc):
    """Build a tqdm progress bar, falling back to a stdlib print-based shim.

    The RoboCasa environment has no obligation to install ``tqdm``; this
    keeps the connector importable everywhere while still giving rich
    progress feedback when tqdm is present. The returned object exposes
    ``update()``, ``set_postfix()``, and ``close()`` on both paths.
    """
    try:
        from tqdm.auto import trange as _trange
        return _trange(total, desc=desc)
    except ImportError:
        class _PlainBar:
            def __init__(self, total, desc):
                self.total = total
                self.desc = desc
                self.n = 0

            def update(self, n=1):
                self.n += n
                print(f"[{self.desc}] {self.n}/{self.total}", flush=True)

            def set_postfix(self, mapping):
                postfix = " ".join(f"{k}={v}" for k, v in mapping.items())
                print(f"[{self.desc}] {self.n}/{self.total}  {postfix}", flush=True)

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()
                return False
        return _PlainBar(total, desc)


def _to_robocasa_action(action):
    """Convert raw RoboCasa365 action order to RoboCasa's gym dict."""
    import numpy as np

    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.shape != (12,):
        raise ValueError(f"RoboCasa expects a 12-D action, got {values.shape}.")
    return {
        "action.base_motion": values[0:4],
        "action.control_mode": values[4:5],
        "action.end_effector_position": values[5:8],
        "action.end_effector_rotation": values[8:11],
        "action.gripper_close": values[11:12],
    }


def run_benchmark(
    *,
    model=None,
    host: str = "127.0.0.1",
    port: int = 9999,
    tasks: list[str] | None = None,
    task_set: str = "atomic_seen",
    split: str = "target",
    trials: int = 50,
    max_steps: int | None = None,
    seed: int = 0,
    report: str | None = None,
):
    """Drive the RoboCasa365 closed-loop benchmark and return a report dict.

    For each task, makes ``trials`` gym scenes (``gym.make("robocasa/<task>",
    split=...)``), forwards each observation to the model, executes the
    returned action chunk via ``env.step``, and records the binary success
    flag from the episode's terminal ``info``. Results are aggregated by task
    and across the task-set, and written to ``report`` as JSON when given.

    Parameters
    ----------
    model : object | None
        A model backend exposing ``.call(func_name, obs)`` and a context
        manager (``__enter__``/``__exit__``). ``None`` (default) builds a
        :class:`ModelClient` connecting to the remote server at ``host:port``;
        the in-process benchmark path passes a wrapper around
        :class:`~vla_factory.inference.deploy.RemotePolicyModel` so the gym loop
        is shared between the two-process and in-process flows.
    """
    _ensure_simulator()
    import time
    import gymnasium as gym
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    task_list = _resolve_tasks(tasks, task_set)
    per_task: dict[str, dict] = {}
    successes = 0
    total = 0
    all_successes: list[bool] = []
    benchmark_start = time.time()

    backend = model if model is not None else ModelClient(host=host, port=port)
    with backend:
        for task in task_list:
            env_id = task if task.startswith("robocasa/") else f"robocasa/{task}"
            task_name = task.removeprefix("robocasa/")
            step_limit = max_steps if max_steps is not None else get_task_horizon(task_name)
            try:
                env = gym.make(env_id, split=split)
            except Exception as exc:
                per_task[task] = {
                    "error": f"gym.make({env_id!r}, split={split!r}) failed: {exc}",
                    "trials": 0, "successes": 0, "success_rate": 0.0,
                }
                continue

            task_successes = 0
            progbar = _make_progress_bar(
                total=trials, desc=f"{task_name}",
            )
            try:
                for trial in range(trials):
                    episode_start = time.time()
                    obs, info = env.reset(seed=seed + trial)
                    backend.call("reset_model")
                    done = False
                    steps = 0
                    while not done and steps < step_limit:
                        request = {
                            "robocasa_observation": obs,
                            "instruction": obs.get("annotation.human.task_description"),
                            "step": steps,
                        }
                        actions = backend.call("get_action", request)
                        for action in actions:
                            obs, reward, terminated, truncated, info = env.step(
                                _to_robocasa_action(action)
                            )
                            steps += 1
                            if terminated or truncated or _success_flag(info):
                                done = True
                                break
                            if steps >= step_limit:
                                done = True
                                break
                    success = bool(_success_flag(info))
                    if success:
                        task_successes += 1
                    all_successes.append(success)
                    episode_s = time.time() - episode_start
                    progbar.set_postfix({
                        "last": f"{'✓' if success else '✗'}({steps}st,{episode_s:.0f}s)",
                        "task_rate": f"{task_successes}/{trial + 1}",
                        "running": f"{100.0 * sum(all_successes) / len(all_successes):.1f}%",
                    })
                    progbar.update()
            finally:
                progbar.close()
                env.close()

            rate = task_successes / trials if trials else 0.0
            per_task[task] = {
                "trials": trials,
                "successes": task_successes,
                "success_rate": rate,
            }
            successes += task_successes
            total += trials

    benchmark_s = time.time() - benchmark_start
    overall = {
        "task_set": task_set,
        "split": split,
        "trials_per_task": trials,
        "tasks": per_task,
        "overall_success_rate": successes / total if total else 0.0,
        "total_successes": successes,
        "total_trials": total,
        "eval_s": benchmark_s,
        "eval_ep_s": (benchmark_s / total) if total else 0.0,
    }
    if report:
        import json

        with open(report, "w") as f:
            json.dump(overall, f, indent=2)
    return overall


def _success_flag(info):
    """Read the binary success flag from a robosuite/robocasa terminal info.

    robosuite exposes ``info["success"]`` (a 0/1 float); some envs nest it
    under ``info["eval"]["success_rate"]``. Any truthy numeric value counts.
    """
    if not isinstance(info, dict):
        return False
    for key in ("success", "is_success"):
        if key in info:
            return bool(info[key])
    eval_info = info.get("eval")
    if isinstance(eval_info, dict):
        for key in ("success_rate", "success"):
            if key in eval_info:
                return bool(eval_info[key])
    return False


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m vla_factory.inference.connectors.robocasa",
        description="Drive the RoboCasa365 closed-loop benchmark against a "
                    "VLA Factory model server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Model server host.")
    parser.add_argument("--port", type=int, default=9999, help="Model server port.")
    parser.add_argument(
        "--task-set", default="atomic_seen",
        help="Task-group label for the report "
             "(atomic_seen / atomic_unseen / composite_seen / composite_unseen).",
    )
    parser.add_argument(
        "--split", default="target",
        help="Scene split passed to gym.make (e.g. target).",
    )
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="Task names (e.g. PickPlaceCounterToCabinet). If omitted, a "
             "--task-set must resolve to a task list.",
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of gym scenes per task (default 50, matching v1.0.1).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Per-episode step cap; default uses RoboCasa's registered task horizon.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--report", default=None,
        help="Write the JSON success-rate report to this path.",
    )
    args = parser.parse_args()

    try:
        report = run_benchmark(
            host=args.host,
            port=args.port,
            tasks=args.tasks,
            task_set=args.task_set,
            split=args.split,
            trials=args.trials,
            max_steps=args.max_steps,
            seed=args.seed,
            report=args.report,
        )
    except ModuleNotFoundError as exc:
        import sys

        print(f"[robocasa] {exc}", file=sys.stderr)
        sys.exit(2)

    import json

    print(
        f"\n[robocasa] {report['total_successes']}/{report['total_trials']} succeeded "
        f"({report['overall_success_rate']*100:.1f}%) "
        f"in {report['eval_s']:.1f}s ({report['eval_ep_s']:.1f}s/ep)"
    )
    for task_name, task_report in report["tasks"].items():
        if "error" in task_report:
            print(f"  {task_name}: ERROR ({task_report['error']})")
            continue
        print(
            f"  {task_name}: {task_report['successes']}/{task_report['trials']} "
            f"({task_report['success_rate']*100:.1f}%)"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    _main()
