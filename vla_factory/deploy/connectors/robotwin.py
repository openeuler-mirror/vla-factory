"""Dependency-free RoboTwin connector for a VLA Factory model server.

RoboTwin imports this module as a policy callback while its
``eval_policy_client.py`` injects a raw-TCP ``ModelClient`` as ``model``. The
connector forwards the native observation plus instruction and drives the
returned action chunk. Camera selection, qpos extraction, validation and model
preprocessing all stay in the VLA Factory server process.

This module deliberately has no imports so it can run in a RoboTwin/SAPIEN
environment without installing VLA Factory's model dependencies.
"""


def encode_obs(observation, instruction=None, step=None):
    """Wrap, but do not transform, a native RoboTwin observation."""
    return {
        "robotwin_observation": observation,
        "instruction": instruction,
        "step": step,
    }


def get_model(usr_args):
    """Unused in client/server mode; RoboTwin injects its ``ModelClient``."""
    return None


def eval(TASK_ENV, model, observation):
    """Forward one observation and execute the remote action chunk as qpos."""
    obs = encode_obs(
        observation,
        instruction=TASK_ENV.get_instruction(),
        step=TASK_ENV.take_action_cnt,
    )
    actions = model.call(func_name="get_action", obs=obs)
    for action in actions:
        TASK_ENV.take_action(action, action_type="qpos")
        observation = TASK_ENV.get_obs()
    return observation


def reset_model(model):
    """No-op: RoboTwin's client already calls remote ``reset_model``."""
    pass
