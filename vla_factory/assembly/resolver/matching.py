"""Deterministic candidate derivation shared by Check Pairs and Resolve Mapping.

Both stages ask the same two questions — *which cameras can feed this slot* and
*which robot joint does this dataset dim name refer to* — and they must never
answer them differently: a check that passes where the mapping cannot be built
(or vice versa) is the failure mode this module exists to prevent. So the rules
live here once, and each stage only decides what to do with the answer.
"""

from __future__ import annotations

from vla_factory.data.manifest import DataSchema
from vla_factory.data.semantics import infer_camera_semantic, strip_known_suffix
from vla_factory.model.interfaces.model import VisionSlot
from vla_factory.robot.profile import RobotProfile


def camera_semantic_satisfies(semantic: str, accepts: tuple[str, ...]) -> bool:
    """A data/robot camera satisfies a slot if its semantic is directly
    accepted, or is a specific third-person view and the slot accepts the
    ``third_person`` generalization (architecture §4.1.2 / vocabulary.py)."""
    if semantic in accepts:
        return True
    return semantic.startswith("third_person") and "third_person" in accepts


def camera_candidates(slot: VisionSlot, candidates: list[tuple[str, str]]) -> list[str]:
    """Cameras that can feed *slot*, sorted (determinism, §1.7)."""
    return sorted(
        key for key, sem in candidates
        if camera_semantic_satisfies(sem, slot.semantic_accepts)
    )


def data_camera_candidates(schema: DataSchema) -> list[tuple[str, str]]:
    """``[(camera_key, inferred_semantic), ...]`` for data cameras with a
    resolved semantic. A camera whose semantic could not be inferred (e.g.
    RoboTwin's ``left_camera``) simply is not a candidate — it neither
    satisfies nor conflicts with anything."""
    return [(e.key, e.semantic) for e in schema.cameras_entries if e.semantic]


def robot_camera_candidates(robot_profile: RobotProfile) -> list[tuple[str, str]]:
    """Same shape as :func:`data_camera_candidates`, inferring the semantic
    from the robot's stable camera names."""
    return [
        (cam, sem) for cam in robot_profile.cameras
        if (sem := infer_camera_semantic(cam)) is not None
    ]


def embed_joints(
    names: list[str], robot_joint_names: tuple[str, ...],
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Embed dataset dim names into a robot's joint set (decision D4).

    Names are compared after stripping a known suffix (``.pos``/``.vel``/
    ``.delta`` — data-module §8.3 keeps the suffix, robot joint names don't
    carry one; ``data/semantics.py:strip_known_suffix`` is the single source
    for that table). This is a *subset* embedding, not set equality: the robot
    is allowed to have joints the dataset never recorded (e.g. LeKiwi's mobile
    base absent from arm-only training data).

    Returns ``(pairs, unmatched, duplicates)``: the matched
    ``(dataset_name, robot_joint_name)`` correspondences in dataset order, plus
    the two failure sets.
    """
    stripped = [strip_known_suffix(n) for n in names]
    robot_names = set(robot_joint_names)
    unmatched = sorted({orig for orig, s in zip(names, stripped) if s not in robot_names})
    duplicates = sorted({s for s in stripped if stripped.count(s) > 1})
    pairs = [(orig, s) for orig, s in zip(names, stripped) if s in robot_names]
    return pairs, unmatched, duplicates
