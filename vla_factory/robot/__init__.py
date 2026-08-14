"""Public API for static robot body descriptions.

``RobotProfile`` records body facts such as cameras, joints, control modes,
gripper conventions, limits, frames, and recommended control frequency. The
current composition layer validates the profile, checks declared control-mode
compatibility, and stores a snapshot in ``ResolvedAssembly``.

``RobotProfile`` deliberately does **not** describe which process the robot is
attached to or which transport/ROS-topic/IP/port it uses — those runtime
concerns live in ``vla_factory.inference``. Platform adapters, rather than this
module, translate live observations and actions. Static fields not named above
as active checks are declarations only; they do not currently drive runtime
transforms, safety enforcement, or loop timing.
"""

from .profile import (
    GripperConvention,
    JointGroup,
    RobotProfile,
)
from .registry import get_robot_profile, list_robot_profiles, load_robot_profile

__all__ = [
    "GripperConvention",
    "JointGroup",
    "RobotProfile",
    "load_robot_profile",
    "get_robot_profile",
    "list_robot_profiles",
]
