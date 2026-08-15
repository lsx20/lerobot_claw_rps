#!/usr/bin/env python3
"""Recover Piper CAN/control/enable state without commanding motion.

This script deliberately does not send JointCtrl or EndPoseCtrl. It only:
  1. restores normal master/slave output mode,
  2. exits drag-teach / clears pending trajectories,
  3. enables motors,
  4. prints status for human verification.

Disabling is never automatic. Type uppercase D at the final prompt to disable.
"""

from __future__ import annotations

import argparse
import sys
import time

from piper_sdk import C_PiperInterface_V2


def print_status(piper: C_PiperInterface_V2, label: str) -> None:
    print(f"\n=== {label} ===")
    for name in (
        "GetArmStatus",
        "GetArmEnableStatus",
        "GetArmJointMsgs",
        "GetArmEndPoseMsgs",
        "GetArmCtrlCode151",
        "GetArmModeCtrl",
    ):
        try:
            print(f"{name}: {getattr(piper, name)()}")
        except Exception as exc:
            print(f"{name}: failed: {exc}")


def wait_for_any_feedback(piper: C_PiperInterface_V2, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status = piper.GetArmStatus()
            joints = piper.GetArmJointMsgs()
            pose = piper.GetArmEndPoseMsgs()
            if status.Hz > 0 or joints.Hz > 0 or pose.Hz > 0:
                return True
            js = joints.joint_state
            ep = pose.end_pose
            if any(
                value != 0
                for value in (
                    js.joint_1,
                    js.joint_2,
                    js.joint_3,
                    js.joint_4,
                    js.joint_5,
                    js.joint_6,
                    ep.X_axis,
                    ep.Y_axis,
                    ep.Z_axis,
                    ep.RX_axis,
                    ep.RY_axis,
                    ep.RZ_axis,
                )
            ):
                return True
        except Exception:
            pass
        time.sleep(0.05)
    return False


def restore_normal_feedback(piper: C_PiperInterface_V2, repeat: int) -> None:
    print("Restoring normal output-arm mode: MasterSlaveConfig(0xFC, 0, 0, 0)")
    for _ in range(repeat):
        piper.MasterSlaveConfig(0xFC, 0, 0, 0)
        time.sleep(0.02)


def recover_control_mode(piper: C_PiperInterface_V2, speed: int, installation_pos: int) -> None:
    print("Exiting teach/trajectory state and selecting CAN MOVE_J mode.")
    for _ in range(20):
        piper.MotionCtrl_1(0x02, 0x06, 0x06)
        time.sleep(0.02)
    for _ in range(20):
        piper.MotionCtrl_1(0x02, 0x04, 0x02)
        time.sleep(0.02)
    for _ in range(20):
        piper.MotionCtrl_1(0x02, 0x00, 0x02)
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00, 0, installation_pos)
        time.sleep(0.02)


def wait_for_can_ctrl(piper: C_PiperInterface_V2, speed: int, installation_pos: int, timeout_s: float) -> bool:
    print("Waiting for Control Mode to become CAN_CTRL(0x1).")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        piper.MotionCtrl_1(0x02, 0x00, 0x02)
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00, 0, installation_pos)
        time.sleep(0.05)
        try:
            status = piper.GetArmStatus().arm_status
            print(f"ctrl_mode={status.ctrl_mode}, teach_status={status.teach_status}")
            if int(status.ctrl_mode) == 0x01:
                return True
        except Exception as exc:
            print(f"[warn] failed reading ctrl_mode: {exc}")
    return False


def enable_until_feedback_true(piper: C_PiperInterface_V2, timeout_s: float) -> list[bool]:
    print("Sending EnableArm(7, 0x02) until feedback reports all joints enabled.")
    deadline = time.time() + timeout_s
    last_status: list[bool] = []
    while time.time() < deadline:
        piper.EnableArm(7, 0x02)
        time.sleep(0.05)
        try:
            last_status = list(piper.GetArmEnableStatus())
        except Exception:
            last_status = []
        print(f"enable feedback: {last_status}")
        if last_status and all(last_status):
            return last_status
    return last_status


def prompt_before_disable(piper: C_PiperInterface_V2) -> None:
    print()
    print("WARNING: disabling Piper motors may make the arm drop.")
    print("Hold/support the arm before disabling.")
    answer = input(
        "Type D then Enter to disable arm motors, or press Enter to keep motors enabled: "
    ).strip()
    if answer == "D":
        piper.DisableArm(7)
        print("Piper arm motors disabled.")
    else:
        print("Piper arm motors left enabled.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0")
    parser.add_argument("--speed", type=int, default=5)
    parser.add_argument(
        "--installation-pos",
        type=lambda value: int(value, 0),
        default=0x01,
        choices=(0x01, 0x02, 0x03),
    )
    parser.add_argument("--skip-master-slave-restore", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip initial YES prompt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("This recovery script sends no JointCtrl and no EndPoseCtrl.")
    print("Keep the arm supported and emergency stop reachable.")
    if not args.yes:
        answer = input("Type YES to continue: ").strip()
        if answer != "YES":
            print("Aborted.")
            return 1

    piper = C_PiperInterface_V2(
        args.can,
        judge_flag=False,
        can_auto_init=False,
        dh_is_offset=1,
        start_sdk_fk_cal=True,
    )
    piper.ConnectPort()
    time.sleep(0.5)

    if not wait_for_any_feedback(piper, 3.0):
        print("No SDK-readable feedback yet. If candump shows 3A*, run with normal-mode restore and power-cycle if needed.")

    print_status(piper, "before recovery")
    if not args.skip_master_slave_restore:
        restore_normal_feedback(piper, repeat=50)
        time.sleep(0.5)

    recover_control_mode(piper, args.speed, args.installation_pos)
    if not wait_for_can_ctrl(piper, args.speed, args.installation_pos, timeout_s=8.0):
        print("Control Mode did not become CAN_CTRL. Exit teach mode from the arm/official tool, then retry.")
    enable_status = enable_until_feedback_true(piper, timeout_s=8.0)
    print(f"after enable feedback: {enable_status}")

    for _ in range(20):
        piper.MotionCtrl_1(0x02, 0x00, 0x00)
        piper.MotionCtrl_2(0x01, 0x01, args.speed, 0x00, 0, args.installation_pos)
        time.sleep(0.02)

    print_status(piper, "after recovery")
    print("Now gently test by hand whether each joint has holding torque. Do not force it.")

    try:
        prompt_before_disable(piper)
    except KeyboardInterrupt:
        print("\nInterrupted. Motors were not disabled by this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
