"""SimplerEnv connector and closed-loop benchmark client.

This module runs in the SimplerEnv (ManiSkill / SAPIEN) environment and
drives the SimplerEnv benchmark: it makes one env per task, reads the
third-person camera frame and the end-effector state from the observation
dict, forwards them to the VLA Factory model server (speaking the same
length-prefixed JSON-RPC protocol RoboTwin / RoboCasa use), executes the
returned action chunk via ``env.step``, and aggregates the binary success
flag into a per-task success-rate report.

The runtime contract mirrors the upstream SimplerEnv code and the NVIDIA
GR00T reference integration (``gr00t/eval/sim/SimplerEnv``):

- env entry: ``simpler_env.make(task)``;
- images: ``obs["image"][camera]["rgb"]`` — each embodiment exposes a
  single third-person camera, ``overhead_camera`` for Google-robot tasks
  and ``3rd_view_camera`` for WidowX tasks;
- robot state: ``obs["agent"]["eef_pos"]`` — 8-D: translation (3) +
  quaternion (4, wxyz) + gripper (1);
- instruction: ``env.unwrapped.get_language_instruction()``;
- actions: flat 7-D vectors (end-effector delta + gripper) consumed
  directly by ``env.step``; 4-D checkpoints (xyz + gripper) are expanded
  with zero rotations, and an optional gripper-convention post-processor
  covers checkpoints whose gripper semantics differ from the env's.

It deliberately imports only the standard library and numpy at module level
(the shared :class:`ModelClient` lives in the robocasa connector, which has
the same dependency-free contract) so the SimplerEnv environment can import
it without installing VLA Factory's model dependencies. ``simpler_env`` is
imported lazily inside :func:`run_benchmark` and probed via
``importlib.util.find_spec`` so a missing simulator surfaces an actionable
install hint instead of a traceback.

Run as a module from the SimplerEnv environment::

    python -m vla_factory.inference.connectors.simplerenv \\
        --port 9999 --tasks google_robot_pick_coke_can \\
        --trials 25 --report simplerenv_report.json
"""

from vla_factory.inference.connectors.robocasa import (
    ModelClient,
    _make_progress_bar,
)


# Embodiment task-name prefix → its single third-person camera. SimplerEnv
# has no wrist camera on either embodiment.
EMBODIMENT_CAMERAS: dict[str, str] = {
    "google_robot": "overhead_camera",
    "widowx": "3rd_view_camera",
}

ACTION_DIM = 7


def _ensure_simulator():
    """Probe for ``simpler_env``; raise an actionable error."""
    import importlib.util

    if importlib.util.find_spec("simpler_env") is None:
        raise ModuleNotFoundError(
            "The SimplerEnv benchmark needs the simpler_env package in this "
            "environment. Install SimplerEnv (github.com/simpler-env/"
            "SimplerEnv) in the simulator environment, then re-run from "
            "there. See docs/tutorial/simplerenv.md for the "
            "isolated-environment recipe."
        )


def _embodiment_camera(task: str, cameras: tuple[str, ...] | None) -> str:
    """Resolve the single camera to forward for a task.

    Explicit ``cameras`` wins (one entry); otherwise the task-name prefix
    selects the embodiment's registered camera.
    """
    if cameras:
        if len(cameras) != 1:
            raise ValueError(
                "SimplerEnv exposes a single third-person camera per "
                "embodiment; pass at most one --cameras entry, got "
                f"{list(cameras)}."
            )
        return cameras[0]
    for prefix, camera in EMBODIMENT_CAMERAS.items():
        if task.startswith(prefix):
            return camera
    known = " / ".join(sorted(EMBODIMENT_CAMERAS))
    raise ValueError(
        f"Cannot infer the SimplerEnv embodiment camera for task {task!r}; "
        f"pass --cameras explicitly. Known task prefixes: {known}."
    )


def _get_image(obs: dict, camera: str):
    """Read one camera's RGB frame from a ManiSkill obs dict."""
    import numpy as np

    image = obs.get("image", {}).get(camera)
    if image is None:
        raise KeyError(
            f"Camera {camera!r} not found in obs['image']. Available: "
            f"{sorted(obs.get('image', {}).keys())}. SimplerEnv exposes one "
            "third-person camera per embodiment (overhead_camera for Google "
            "robot, 3rd_view_camera for WidowX)."
        )
    frame = image.get("rgb") if isinstance(image, dict) else image
    if frame is None:
        raise KeyError(
            f"obs['image'][{camera!r}] has no 'rgb' entry; run SimplerEnv "
            "with an rgb-bearing obs_mode (e.g. the default 'rgbd')."
        )
    return np.ascontiguousarray(frame)


def _get_state(obs: dict):
    """Read the 8-D end-effector state from a ManiSkill obs dict."""
    import numpy as np

    agent = obs.get("agent", {})
    state = agent.get("eef_pos")
    if state is None:
        raise KeyError(
            "obs['agent']['eef_pos'] not found. Available agent keys: "
            f"{sorted(agent.keys())}."
        )
    return np.asarray(state, dtype=np.float32)


def _success_flag(info, terminated: bool, env) -> bool:
    """Read the binary success flag from a SimplerEnv step.

    ManiSkill2_real2sim sets ``done = info["success"]``; SimplerEnv also
    terminates successful episodes, and some wrappers latch
    ``success_once``. ``truncated`` alone (TimeLimit) is not success.
    """
    if isinstance(info, dict):
        for key in ("success", "is_success"):
            if key in info and bool(info[key]):
                return True
    if terminated:
        return True
    return bool(getattr(env, "success_once", False))


class _GripperPostProcessor:
    """Adapt checkpoint gripper semantics to the env's [-1, 1] convention.

    ``raw`` forwards the model action unchanged (default — correct for
    checkpoints whose actions already follow the env convention). ``google``
    maps a [0, 1] gripper to [-1, 1] and, once the model commands a close,
    keeps the gripper closed for ``sticky_steps`` steps (the sticky-gripper
    protocol SimplerEnv's Google-robot tasks need). ``widowx`` binarises the
    command around 0.5.
    """

    def __init__(self, mode: str = "raw", sticky_steps: int = 15) -> None:
        if mode not in ("raw", "google", "widowx"):
            raise ValueError(
                f"Unknown gripper mode {mode!r}; expected raw / google / widowx."
            )
        self.mode = mode
        self.sticky_steps = sticky_steps
        self._sticky_remaining = 0

    def reset(self) -> None:
        self._sticky_remaining = 0

    def process(self, action):
        import numpy as np

        values = np.array(action, dtype=np.float32, copy=True)
        if self.mode == "raw":
            return values
        gripper = values[-1]
        if self.mode == "widowx":
            values[-1] = 2.0 * (gripper > 0.5) - 1.0
            return values
        # google: [0, 1] open → [-1, 1], closing latched for sticky_steps.
        if self._sticky_remaining > 0:
            values[-1] = -1.0
            self._sticky_remaining -= 1
        elif gripper <= 0.5:
            values[-1] = -1.0
            self._sticky_remaining = self.sticky_steps - 1
        else:
            values[-1] = 1.0
        return values


def _to_simplerenv_action(action, gripper: _GripperPostProcessor):
    """Convert a raw model action to the env's flat 7-D action vector.

    SimplerEnv consumes ``[dx, dy, dz, droll, dpitch, dyaw, gripper]`` on
    both embodiments. 7-D actions pass through (after gripper
    post-processing); 4-D checkpoints (xyz + gripper) are expanded with zero
    rotations. The result is clipped to [-1, 1].
    """
    import numpy as np

    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.shape[0] == 4:
        values = np.concatenate([values[:3], np.zeros(3, np.float32), values[3:4]])
    if values.shape[0] != ACTION_DIM:
        raise ValueError(
            f"SimplerEnv expects a {ACTION_DIM}-D action "
            "(end-effector delta + gripper), got {values.shape[0]}. "
            "Checkpoints must emit either 7-D actions or 4-D (xyz + gripper)."
        )
    return np.clip(gripper.process(values), -1.0, 1.0)


def run_benchmark(
    *,
    model=None,
    host: str = "127.0.0.1",
    port: int = 9999,
    tasks: list[str] | None = None,
    trials: int = 25,
    max_steps: int | None = None,
    seed: int = 0,
    cameras: tuple[str, ...] | None = None,
    instruction: str | None = None,
    gripper_mode: str = "raw",
    report: str | None = None,
):
    """Drive the SimplerEnv closed-loop benchmark and return a report dict.

    For each task, makes one env (``simpler_env.make(task)``) and runs
    ``trials`` reset/eval episodes on it, forwarding the single
    third-person camera frame plus ``obs["agent"]["eef_pos"]`` to the model,
    executing the returned action chunk via ``env.step``, and recording the
    binary success flag from the episode's terminal step. Episodes end at
    the env's registered TimeLimit unless ``max_steps`` overrides it.
    Results are aggregated by task and overall, and written to ``report``
    as JSON when given.

    Note: this loop resets with ``env.reset(seed=seed + trial)``, i.e. the
    prepackaged-initial-state protocol. SimplerEnv's official benchmark
    additionally varies robot/object init states via ``options`` (variant
    aggregation) and the visual-matching setting; numbers from this runner
    are not directly comparable to the published benchmark.

    Parameters
    ----------
    model : object | None
        A model backend exposing ``.call(func_name, obs)`` and a context
        manager. ``None`` (default) builds a :class:`ModelClient` connecting
        to the remote server at ``host:port``.
    cameras : tuple[str, ...] | None
        Single SimplerEnv camera to forward; by default inferred from the
        task-name prefix (``overhead_camera`` / ``3rd_view_camera``).
    instruction : str | None
        Override the language instruction sent to the model; by default
        ``env.unwrapped.get_language_instruction()`` per episode.
    gripper_mode : str
        Checkpoint gripper convention: ``raw`` passthrough (default),
        ``google`` ([0, 1] → [-1, 1] + sticky close), ``widowx``
        (binarise around 0.5).
    """
    _ensure_simulator()
    import time

    import simpler_env

    if not tasks:
        raise ValueError(
            "SimplerEnv benchmark needs an explicit --tasks list (e.g. "
            "google_robot_pick_coke_can, widowx_put_eggplant_in_basket)."
        )

    per_task: dict[str, dict] = {}
    successes = 0
    total = 0
    all_successes: list[bool] = []
    benchmark_start = time.time()

    backend = model if model is not None else ModelClient(host=host, port=port)
    with backend:
        for task in tasks:
            camera = _embodiment_camera(task, cameras)
            gripper = _GripperPostProcessor(gripper_mode)
            try:
                env = simpler_env.make(task)
            except Exception as exc:
                per_task[task] = {
                    "error": f"simpler_env.make({task!r}) failed: {exc}",
                    "trials": 0, "successes": 0, "success_rate": 0.0,
                }
                continue

            task_successes = 0
            progbar = _make_progress_bar(total=trials, desc=task)
            try:
                for trial in range(trials):
                    episode_start = time.time()
                    obs, info = env.reset(seed=seed + trial)
                    backend.call("reset_model")
                    gripper.reset()
                    task_instruction = instruction or (
                        env.unwrapped.get_language_instruction()
                    )
                    terminated = truncated = False
                    info: dict = info or {}
                    steps = 0
                    while (
                        not (terminated or truncated)
                        and (max_steps is None or steps < max_steps)
                    ):
                        env_obs = {
                            f"image.{camera}": _get_image(obs, camera),
                            "state": _get_state(obs),
                        }
                        request = {
                            "simplerenv_observation": env_obs,
                            "instruction": task_instruction,
                            "step": steps,
                        }
                        actions = backend.call("get_action", request)
                        for action in actions:
                            obs, reward, terminated, truncated, info = env.step(
                                _to_simplerenv_action(action, gripper)
                            )
                            steps += 1
                            if terminated or truncated:
                                break
                            if max_steps is not None and steps >= max_steps:
                                break
                    success = _success_flag(info, terminated, env)
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
        "benchmark": "simplerenv",
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


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m vla_factory.inference.connectors.simplerenv",
        description="Drive the SimplerEnv closed-loop benchmark against a "
                    "VLA Factory model server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Model server host.")
    parser.add_argument("--port", type=int, default=9999, help="Model server port.")
    parser.add_argument(
        "--tasks", nargs="+", required=True,
        help="SimplerEnv task names (e.g. google_robot_pick_coke_can, "
             "widowx_put_eggplant_in_basket).",
    )
    parser.add_argument(
        "--trials", type=int, default=25,
        help="Number of episodes per task (default 25).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Per-episode step cap; by default the env's registered "
             "TimeLimit ends the episode.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--cameras", nargs="*", default=None,
        help="Camera to forward (at most one — SimplerEnv exposes a single "
             "third-person camera per embodiment). Default: inferred from "
             "the task name (overhead_camera / 3rd_view_camera).",
    )
    parser.add_argument(
        "--instruction", default=None,
        help="Override the language instruction sent to the model (default: "
             "env.unwrapped.get_language_instruction()).",
    )
    parser.add_argument(
        "--gripper-mode", default="raw", choices=["raw", "google", "widowx"],
        help="Checkpoint gripper convention: raw passthrough, google "
             "([0,1]→[-1,1] + sticky close), widowx (binarise around 0.5).",
    )
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
            trials=args.trials,
            max_steps=args.max_steps,
            seed=args.seed,
            cameras=tuple(args.cameras) if args.cameras else None,
            instruction=args.instruction,
            gripper_mode=args.gripper_mode,
            report=args.report,
        )
    except ModuleNotFoundError as exc:
        import sys

        print(f"[simplerenv] {exc}", file=sys.stderr)
        sys.exit(2)

    import json

    print(
        f"\n[simplerenv] {report['total_successes']}/{report['total_trials']} succeeded "
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
