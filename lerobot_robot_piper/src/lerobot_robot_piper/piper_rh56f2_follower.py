import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_piper_rh56f2_follower import PiperRH56F2FollowerConfig
from .piper_follower import JOINT_LIMITS_DEG, JOINT_NAMES, load_piper_interface_v2
from .rh56f2_hand import HAND_NAMES, RH56F2Hand, RH56F2HandConfig

logger = logging.getLogger(__name__)

EE_POSE_NAMES = ["ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz"]


def _clip_step(goal: float, current: float, max_delta: float | None) -> float:
    if max_delta is None:
        return goal
    return current + float(np.clip(goal - current, -max_delta, max_delta))


class PiperRH56F2Follower(Robot):
    """LeRobot-compatible Piper arm + RH56F2 dexterous hand.

    API units:
      - arm joint positions: degrees
      - end-effector pose: mm for x/y/z, degrees for rx/ry/rz
      - hand positions: RH56F2 register angle units
    """

    config_class = PiperRH56F2FollowerConfig
    name = "piper_rh56f2_follower"

    def __init__(self, config: PiperRH56F2FollowerConfig):
        super().__init__(config)
        self.config = config
        self.piper: Any = None
        self.hand: RH56F2Hand | None = None
        self._is_connected = False
        self._active_move_mode: int | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

    def _arm_status_code(self) -> int:
        return int(self.piper.GetArmStatus().arm_status.arm_status)

    def _ctrl_mode_code(self) -> int:
        return int(self.piper.GetArmStatus().arm_status.ctrl_mode)

    def _mode_feed_code(self) -> int:
        return int(self.piper.GetArmStatus().arm_status.mode_feed)

    def _has_real_feedback(self) -> bool:
        status = self.piper.GetArmStatus()
        pose = self.piper.GetArmEndPoseMsgs()
        joints = self.piper.GetArmJointMsgs()
        ep = pose.end_pose
        js = joints.joint_state
        return (
            status.Hz > 0
            or pose.Hz > 0
            or joints.Hz > 0
            or any(
                value != 0
                for value in (
                    ep.X_axis,
                    ep.Y_axis,
                    ep.Z_axis,
                    ep.RX_axis,
                    ep.RY_axis,
                    ep.RZ_axis,
                    js.joint_1,
                    js.joint_2,
                    js.joint_3,
                    js.joint_4,
                    js.joint_5,
                    js.joint_6,
                )
            )
        )

    def _wait_for_real_feedback(self, timeout_s: float = 5.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._has_real_feedback():
                return
            time.sleep(0.05)
        raise RuntimeError("No real Piper feedback received; refusing to command motion.")

    def _enable_all(self, timeout_s: float = 5.0) -> bool:
        deadline = time.time() + timeout_s
        last_status: list[bool] = []
        while time.time() < deadline:
            self.piper.EnableArm(7, 0x02)
            time.sleep(0.02)
            last_status = list(self.piper.GetArmEnableStatus())
            if last_status and all(last_status):
                return True
        logger.warning("Piper enable failed; final enable status=%s", last_status)
        return False

    def _wait_for_mode_ready(self, move_mode: int, timeout_s: float = 5.0) -> None:
        deadline = time.time() + timeout_s
        last = ""
        while time.time() < deadline:
            self.piper.MotionCtrl_2(0x01, move_mode, self.config.speed_rate, 0x00)
            self.piper.EnableArm(7, 0x02)
            time.sleep(0.05)
            enable_status = list(self.piper.GetArmEnableStatus())
            ctrl_mode = self._ctrl_mode_code()
            mode_feed = self._mode_feed_code()
            arm_status = self._arm_status_code()
            last = (
                f"ctrl=0x{ctrl_mode:x} mode=0x{mode_feed:x} "
                f"arm=0x{arm_status:x} enable={enable_status}"
            )
            if (
                ctrl_mode == 0x01
                and mode_feed == move_mode
                and arm_status == 0x00
                and enable_status
                and all(enable_status)
            ):
                self._active_move_mode = move_mode
                return
        raise RuntimeError(f"Piper did not enter move mode 0x{move_mode:x}: {last}")

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {f"{name}.pos": float for name in JOINT_NAMES}
        features.update({name: float for name in EE_POSE_NAMES})
        features.update({f"hand.{name}.pos": float for name in HAND_NAMES})
        features.update({f"hand.{name}.force": float for name in HAND_NAMES})
        for cam_name in self.cameras:
            cam_cfg = self.config.cameras[cam_name]
            features[cam_name] = (cam_cfg.height, cam_cfg.width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {f"{name}.pos": float for name in JOINT_NAMES}
        features["arm.speed_rate"] = float
        features.update({name: float for name in EE_POSE_NAMES})
        features.update({f"hand.{name}.pos": float for name in HAND_NAMES})
        features.update({f"hand.{name}.speed": float for name in HAND_NAMES})
        features.update({f"hand.{name}.force_limit": float for name in HAND_NAMES})
        features["hand.mode"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self.hand is not None
            and self.hand.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        C_PiperInterface_V2 = load_piper_interface_v2()

        self.piper = C_PiperInterface_V2(
            self.config.can_port,
            judge_flag=False,
            can_auto_init=False,
            dh_is_offset=1,
            start_sdk_fk_cal=True,
        )
        self.piper.ConnectPort()
        time.sleep(0.2)
        self._wait_for_real_feedback()

        self.piper.MotionCtrl_1(0x02, 0x00, 0x02)
        time.sleep(0.05)
        start = self._arm_current_deg()
        if not self._enable_all():
            raise RuntimeError("Failed to enable Piper arm.")
        self._wait_for_mode_ready(0x01)
        self._send_arm_deg(start, clip_limits=False)

        self.hand = RH56F2Hand(
            RH56F2HandConfig(
                port=self.config.hand_port,
                baudrate=self.config.hand_baudrate,
                hand_id=self.config.hand_id,
                speed=self.config.hand_speed,
                force=self.config.hand_force,
                mode=self.config.hand_mode,
            )
        )
        self.hand.connect()

        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        logger.info(
            "PiperRH56F2Follower connected: can=%s hand=%s",
            self.config.can_port,
            self.config.hand_port,
        )

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        if self.hand is not None:
            self.hand.configure()

    def _arm_current_deg(self) -> dict[str, float]:
        joint_msgs = self.piper.GetArmJointMsgs()
        js = joint_msgs.joint_state
        values = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
        return {
            f"{name}.pos": value / 1000.0
            for name, value in zip(JOINT_NAMES, values, strict=True)
        }

    @check_if_not_connected
    def get_arm_joint_positions(self) -> RobotObservation:
        return self._arm_current_deg()

    def _ee_current_mm_deg(self) -> dict[str, float]:
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        values = [
            end_pose.X_axis,
            end_pose.Y_axis,
            end_pose.Z_axis,
            end_pose.RX_axis,
            end_pose.RY_axis,
            end_pose.RZ_axis,
        ]
        return {name: value / 1000.0 for name, value in zip(EE_POSE_NAMES, values, strict=True)}

    def _send_arm_deg(self, arm_action: dict[str, float], clip_limits: bool = True) -> None:
        if self._active_move_mode != 0x01:
            self._wait_for_mode_ready(0x01)
        current = self._arm_current_deg()
        values: list[int] = []
        for name in JOINT_NAMES:
            key = f"{name}.pos"
            goal = float(arm_action.get(key, current[key]))
            should_clip = (
                clip_limits
                and self.config.clip_arm_to_sdk_limits
                and (name != "joint_6" or self.config.clip_joint6_to_sdk_limits)
            )
            if should_clip:
                lo, hi = JOINT_LIMITS_DEG[name]
                goal = float(np.clip(goal, lo, hi))
            goal = _clip_step(goal, current[key], self.config.max_arm_delta_deg)
            values.append(int(round(goal * 1000)))
        self.piper.MotionCtrl_2(0x01, 0x01, self.config.speed_rate, 0x00)
        self.piper.JointCtrl(*values)

    def _send_ee_mm_deg(self, ee_action: dict[str, float]) -> None:
        if self._active_move_mode != 0x00:
            self._wait_for_mode_ready(0x00)
        current = self._ee_current_mm_deg()
        values: list[int] = []
        for name in EE_POSE_NAMES:
            goal = float(ee_action.get(name, current[name]))
            max_delta = (
                self.config.max_ee_delta_mm
                if name in {"ee.x", "ee.y", "ee.z"}
                else self.config.max_ee_delta_deg
            )
            goal = _clip_step(goal, current[name], max_delta)
            values.append(int(round(goal * 1000.0)))
        self.piper.MotionCtrl_2(0x01, 0x00, self.config.speed_rate, 0x00)
        self.piper.EndPoseCtrl(*values)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: RobotObservation = {}
        obs.update(self._arm_current_deg())
        obs.update(self._ee_current_mm_deg())

        hand_pos = self.hand.read_positions("angleAct")
        obs.update({f"hand.{name}.pos": value for name, value in hand_pos.items()})
        hand_force = self.hand.read_positions("forceAct")
        obs.update({f"hand.{name}.force": value for name, value in hand_force.items()})

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        sent: RobotAction = {}

        if "arm.speed_rate" in action:
            speed_rate = int(round(float(action["arm.speed_rate"])))
            self.config.speed_rate = int(np.clip(speed_rate, 0, 100))
            sent["arm.speed_rate"] = float(self.config.speed_rate)

        arm_action = {
            key: float(value)
            for key, value in action.items()
            if key in {f"{n}.pos" for n in JOINT_NAMES}
        }
        ee_action = {
            key: float(value)
            for key, value in action.items()
            if key in set(EE_POSE_NAMES)
        }
        if arm_action and ee_action:
            raise ValueError(
                "Use either joint position actions or end-effector pose actions, not both."
            )
        if arm_action:
            self._send_arm_deg(arm_action)
            sent.update(arm_action)
        if ee_action:
            self._send_ee_mm_deg(ee_action)
            sent.update(ee_action)

        hand_speed_action = {}
        for name in HAND_NAMES:
            key = f"hand.{name}.speed"
            if key in action:
                hand_speed_action[name] = float(action[key])
        if hand_speed_action:
            self.hand.write_positions("speedSet", hand_speed_action)
            sent.update({f"hand.{name}.speed": value for name, value in hand_speed_action.items()})

        hand_force_action = {}
        for name in HAND_NAMES:
            key = f"hand.{name}.force_limit"
            if key in action:
                hand_force_action[name] = float(action[key])
        if hand_force_action:
            self.hand.write_positions("forceSet", hand_force_action)
            sent.update(
                {f"hand.{name}.force_limit": value for name, value in hand_force_action.items()}
            )

        if "hand.mode" in action:
            mode = int(round(float(action["hand.mode"])))
            if mode not in (0, 1, 2):
                raise ValueError(f"Unsupported RH56F2 hand mode: {mode}")
            self.hand.write_positions("mode", {name: mode for name in HAND_NAMES})
            sent["hand.mode"] = float(mode)

        hand_action = {}
        for name in HAND_NAMES:
            key = f"hand.{name}.pos"
            if key in action:
                hand_action[name] = float(action[key])
        if hand_action:
            current = self.hand.read_positions("angleAct")
            clipped = {
                name: _clip_step(value, current[name], self.config.max_hand_delta)
                for name, value in hand_action.items()
            }
            self.hand.set_angles(clipped)
            sent.update({f"hand.{name}.pos": value for name, value in clipped.items()})

        return sent

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.piper is not None:
            print()
            print("WARNING: disabling Piper motors may make the arm drop.")
            print("Hold/support the arm and hand before disabling.")
            confirm = input(
                "Type D then Enter to disable arm motors, or press Enter to keep motors enabled: "
            ).strip()
            if confirm == "D":
                self.piper.DisableArm(7)
                logger.info("Piper arm motors disabled by user confirmation.")
            else:
                logger.info("Piper arm motors left enabled; no DisableArm command sent.")
        if self.hand is not None:
            self.hand.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()
        self._is_connected = False
        logger.info("PiperRH56F2Follower disconnected.")
