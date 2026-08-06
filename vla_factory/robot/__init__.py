"""Robot body-description module (``RobotProfile``).

Describes a robot body — the physical facts a composition resolver and the
inference layer need: identity / body variant, stable camera semantic names,
joint names/order/units/types/limits, control modes, gripper convention,
coordinate frame + URDF reference, static safety bounds and the recommended
control frequency.

``RobotProfile`` deliberately does **not** describe which process the robot is
attached to or which transport/ROS-topic/IP/port it uses — those runtime
concerns live in the inference module. The composition resolver only consumes
the static body facts declared here.
"""

from .profile import (
    GripperConvention,
    JointGroup,
    RobotProfile,
    load_robot_profile,
    profile_from_dict,
)
from .registry import get_robot_profile, list_robot_profiles

__all__ = [
    "GripperConvention",
    "JointGroup",
    "RobotProfile",
    "load_robot_profile",
    "profile_from_dict",
    "get_robot_profile",
    "list_robot_profiles",
]
