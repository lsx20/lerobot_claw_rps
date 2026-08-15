from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("piper_rh56f2_follower")
@dataclass
class PiperRH56F2FollowerConfig(RobotConfig):
    """Piper arm + Inspire RH56F2 hand follower robot."""

    can_port: str = "can0"
    speed_rate: int = 30
    max_arm_delta_deg: float | None = 5.0
    max_ee_delta_mm: float | None = 20.0
    max_ee_delta_deg: float | None = 10.0
    prompt_before_disable: bool = True
    clip_arm_to_sdk_limits: bool = False
    clip_joint6_to_sdk_limits: bool = False

    hand_port: str = "/dev/ttyUSB0"
    hand_baudrate: int = 115200
    hand_id: int = 1
    hand_speed: int = 800
    hand_force: int = 1500
    hand_mode: int = 0
    max_hand_delta: float | None = 120.0

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
