"""SimplerEnv connector and closed-loop benchmark client.

This module runs in the SimplerEnv (ManiSkill / SAPIEN) environment and
drives the SimplerEnv benchmark: it makes one env per task, reads the agent
cameras and robot state, forwards them to the VLA Factory model server
(speaking the same length-prefixed JSON-RPC protocol RoboTwin / RoboCasa
use), executes the returned action chunk via ``env.step``, and aggregates
the binary success flag into a per-task success-rate report.

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


# Per-task step caps from the SimplerEnv evaluation scripts; tasks absent
# from this table fall back to --max-steps / DEFAULT_MAX_STEPS.
TASK_MAX_STEPS: dict[str, int] = {
    "google_robot_pick_coke_can": 280,
    "google_robot_move_near": 280,
    "google_robot_open_drawer": 280,
    "google_robot_close_drawer": 280,
    "google_robot_place_apple": 280,
    "widowx_spoon_on_towel": 150,
    "widowx_carrot_on_plate": 150,
    "widowx_stack_blocks": 150,
    "widowx_put_eggplant_in_basket": 150,
    "widowx_put_item_in_basket": 150,
}

DEFAULT_MAX_STEPS = 300

# SimplerEnv camera names exposed by env.get_image().
DEFAULT_CAMERAS = ("agentview", "robot0_eye_in_hand")


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


def _success_flag(info, terminated: bool, env) -> bool:
    """Read the binary success flag from a SimplerEnv terminal step.

    SimplerEnv terminates the episode on success; ``info["success"]`` and
    the wrapper's ``success_once`` latching cover the non-terminating
    variants.
    """
    if terminated:
        return True
    if isinstance(info, dict):
        for key in ("success", "is_success"):
            if key in info and bool(info[key]):
                return True
        eval_info = info.get("eval")
        if isinstance(eval_info, dict):
            for key in ("success_rate", "success"):
                if key in eval_info and bool(eval_info[key]):
                    return True
    return bool(getattr(env, "success_once", False))


def _resolve_max_steps(task: str, max_steps: int | None) -> int:
    if max_steps is not None:
        return max_steps
    return TASK_MAX_STEPS.get(task, DEFAULT_MAX_STEPS)


def run_benchmark(
    *,
    model=None,
    host: str = "127.0.0.1",
    port: int = 9999,
    tasks: list[str] | None = None,
    trials: int = 25,
    max_steps: int | None = None,
    seed: int = 0,
    image_size: tuple[int, int] = (224, 224),
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
    report: str | None = None,
):
    """Drive the SimplerEnv closed-loop benchmark and return a report dict.

    For each task, makes ``trials`` SimplerEnv envs
    (``make_sim_env(task)``), reads the ``cameras`` images plus
    ``env.get_robot_state()``, forwards them to the model, executes the
    returned action chunk via ``env.step``, and records the binary success
    flag from the episode's terminal step. Results are aggregated by task
    and overall, and written to ``report`` as JSON when given.

    Parameters
    ----------
    model : object | None
        A model backend exposing ``.call(func_name, obs)`` and a context
        manager. ``None`` (default) builds a :class:`ModelClient` connecting
        to the remote server at ``host:port`.
    image_size : tuple[int, int]
        (W, H) passed to ``env.get_image``; must match the resolution the
        checkpoint was trained on.
    cameras : tuple[str, ...]
        SimplerEnv camera names to forward. Must match (after aliasing) the
        checkpoint's trained camera set.
    """
    _ensure_simulator()
    import time

    from simpler_env.utils.env.env_builder import make_sim_env

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
            step_limit = _resolve_max_steps(task, max_steps)
            try:
                env = make_sim_env(task)
            except Exception as exc:
                per_task[task] = {
                    "error": f"make_sim_env({task!r}) failed: {exc}",
                    "trials": 0, "successes": 0, "success_rate": 0.0,
                }
                continue
            action_dim = env.action_space.shape[0]

            task_successes = 0
            progbar = _make_progress_bar(total=trials, desc=task)
            try:
                for trial in range(trials):
                    episode_start = time.time()
                    env.reset(seed=seed + trial)
                    backend.call("reset_model")
                    info: dict = {}
                    terminated = truncated = False
                    steps = 0
                    while not (terminated or truncated) and steps < step_limit:
                        env_obs = {
                            f"image.{cam}": env.get_image(
                                size=image_size, camera=cam
                            )
                            for cam in cameras
                        }
                        env_obs["state"] = env.get_robot_state()
                        request = {
                            "simplerenv_observation": env_obs,
                            "instruction": task,
                            "step": steps,
                        }
                        actions = backend.call("get_action", request)
                        for action in actions:
                            obs, reward, terminated, truncated, info = env.step(
                                _to_simplerenv_action(action, action_dim)
                            )
                            steps += 1
                            if (
                                terminated or truncated
                                or _success_flag(info, terminated, env)
                                or steps >= step_limit
                            ):
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


def _to_simplerenv_action(action, action_dim: int):
    """Validate a raw model action against the env's action space.

    SimplerEnv consumes flat vectors in the env's native action space
    (4-D for the Google robot: xyz-delta + gripper; 7-D for WidowX:
    end-effector delta + gripper), clipped to [-1, 1]. No re-ordering is
    applied — the checkpoint's action semantics must already match the
    embodiment.
    """
    import numpy as np

    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.shape[0] != action_dim:
        raise ValueError(
            f"SimplerEnv expects a {action_dim}-D action, got "
            f"{values.shape[0]}. The checkpoint's action space must match "
            "the embodiment (Google robot: 4, WidowX: 7)."
        )
    return np.clip(values, -1.0, 1.0)


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
        help="Per-episode step cap; default uses the SimplerEnv per-task "
             "horizon table.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--image-size", type=int, nargs=2, default=(224, 224),
        metavar=("WIDTH", "HEIGHT"),
        help="Camera resolution requested from env.get_image (default 224 224).",
    )
    parser.add_argument(
        "--cameras", nargs="+", default=list(DEFAULT_CAMERAS),
        help="SimplerEnv cameras to forward (agentview, "
             "robot0_eye_in_hand).",
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
            image_size=tuple(args.image_size),
            cameras=tuple(args.cameras),
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
