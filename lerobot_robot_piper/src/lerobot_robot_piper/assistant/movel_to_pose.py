#!/usr/bin/env python3
"""Move Piper to an end-effector pose with MOVE_L.

This intentionally follows the official SDK MOVE_L demo style:
  MotionCtrl_2(0x01, 0x02, speed, 0x00)
  EndPoseCtrl(X, Y, Z, RX, RY, RZ)

It does not use MOVE_J and does not disable motors automatically.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from piper_sdk import C_PiperInterface_V2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant.movej_to_joint_pose import (  # noqa: E402
    enable_all,
    fmt,
    joints_deg,
    print_status,
    prompt_before_disable,
    wait_for_real_feedback,
)


MOVE_L_MODE = 0x02
MOVE_L_BRANCH_LIMITS_DEG = {
    1: (-150.0, 150.0),
    2: (0.0, 180.0),
    3: (-170.0, 0.0),
    4: (-100.0, 100.0),
    5: (-70.0, 70.0),
    6: (-120.0, 120.0),
}


def parse_pose_mm_deg(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("expected X,Y,Z,RX,RY,RZ")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pose values must be numbers") from exc
    return [int(round(item * 1000.0)) for item in values]


def parse_pose_raw(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("expected X,Y,Z,RX,RY,RZ raw values")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("raw pose values must be integers") from exc


def end_pose_raw(piper: C_PiperInterface_V2) -> list[int]:
    ep = piper.GetArmEndPoseMsgs().end_pose
    return [ep.X_axis, ep.Y_axis, ep.Z_axis, ep.RX_axis, ep.RY_axis, ep.RZ_axis]


def arm_status_code(piper: C_PiperInterface_V2) -> int:
    return int(piper.GetArmStatus().arm_status.arm_status)


def ctrl_mode_code(piper: C_PiperInterface_V2) -> int:
    return int(piper.GetArmStatus().arm_status.ctrl_mode)


def mode_feed_code(piper: C_PiperInterface_V2) -> int:
    return int(piper.GetArmStatus().arm_status.mode_feed)


def pose_error_mm_deg(actual: list[int], target: list[int]) -> tuple[float, float]:
    xyz_error = max(abs(actual[idx] - target[idx]) for idx in range(3)) / 1000.0
    rpy_error = max(abs(actual[idx] - target[idx]) for idx in range(3, 6)) / 1000.0
    return xyz_error, rpy_error


def joint_limit_violations(joints: list[float]) -> list[str]:
    violations = []
    for idx, value in enumerate(joints, start=1):
        lo, hi = MOVE_L_BRANCH_LIMITS_DEG[idx]
        if value < lo or value > hi:
            violations.append(f"J{idx}={value:.3f} outside [{lo:.1f},{hi:.1f}]")
    return violations


def wait_for_movel_ready(
    piper: C_PiperInterface_V2,
    speed: int,
    timeout_s: float,
) -> bool:
    deadline = time.time() + timeout_s
    count = 0
    while time.time() < deadline:
        piper.MotionCtrl_2(0x01, MOVE_L_MODE, speed, 0x00)
        piper.EnableArm(7, 0x02)
        time.sleep(0.05)

        status = piper.GetArmStatus()
        enable_status = list(piper.GetArmEnableStatus())
        ctrl_mode = ctrl_mode_code(piper)
        mode_feed = mode_feed_code(piper)
        arm_status = arm_status_code(piper)
        if count % 10 == 0:
            print(
                "mode ready check: "
                f"ctrl=0x{ctrl_mode:x} mode=0x{mode_feed:x} "
                f"arm=0x{arm_status:x} hz={status.Hz:.1f} enable={enable_status}"
            )
        if (
            ctrl_mode == 0x01
            and mode_feed == MOVE_L_MODE
            and arm_status == 0x00
            and all(enable_status)
        ):
            return True
        count += 1
    return False


def send_target(
    piper: C_PiperInterface_V2,
    target: list[int],
    speed: int,
    duration_s: float,
    rate_hz: float,
) -> bool:
    interval_s = 1.0 / rate_hz
    deadline = time.time() + duration_s
    count = 0
    while time.time() < deadline:
        piper.MotionCtrl_2(0x01, MOVE_L_MODE, speed, 0x00)
        piper.EndPoseCtrl(*target)
        time.sleep(interval_s)

        if count % max(1, int(rate_hz / 2)) == 0:
            actual = end_pose_raw(piper)
            xyz_error, rpy_error = pose_error_mm_deg(actual, target)
            enable_status = list(piper.GetArmEnableStatus())
            arm_status = arm_status_code(piper)
            motion_status = int(piper.GetArmStatus().arm_status.motion_status)
            print(
                "target: "
                f"enable={enable_status} arm=0x{arm_status:x} "
                f"motion=0x{motion_status:x} xyz_err={xyz_error:.3f}mm "
                f"rpy_err={rpy_error:.3f}deg joints={fmt(joints_deg(piper))} "
                f"pose={actual}"
            )
            if not all(enable_status):
                print("[warn] Arm is no longer fully enabled; stopping.")
                return False
            if arm_status != 0x00:
                print("[warn] Arm Status is no longer NORMAL; stopping.")
                return False
        count += 1
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0")
    parser.add_argument("--speed", type=int, default=3)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--feedback-timeout", type=float, default=8.0)
    parser.add_argument("--target", type=parse_pose_mm_deg)
    parser.add_argument("--target-raw", type=parse_pose_raw)
    parser.add_argument("--dx-mm", type=float)
    parser.add_argument("--dy-mm", type=float)
    parser.add_argument("--dz-mm", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_modes = [
        args.target is not None,
        args.target_raw is not None,
        any(value is not None for value in (args.dx_mm, args.dy_mm, args.dz_mm)),
    ]
    if sum(target_modes) != 1:
        raise ValueError("provide exactly one of --target, --target-raw, or --dx/--dy/--dz")
    if not 0 <= args.speed <= 100:
        raise ValueError("--speed must be in [0, 100]")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")

    target = args.target_raw if args.target_raw is not None else args.target
    print("This script sends one MOVE_L target.")
    if target is not None:
        print(f"  target raw: {target}")
        print(
            "  target mm/deg: "
            f"{target[0] / 1000:.3f}, {target[1] / 1000:.3f}, "
            f"{target[2] / 1000:.3f}, {target[3] / 1000:.3f}, "
            f"{target[4] / 1000:.3f}, {target[5] / 1000:.3f}"
        )
    else:
        print(
            "  relative delta mm: "
            f"dx={args.dx_mm or 0.0:.3f}, dy={args.dy_mm or 0.0:.3f}, dz={args.dz_mm or 0.0:.3f}"
        )
    print("It follows the official MOVE_L demo control call and never disables motors automatically.")
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
    time.sleep(1)

    try:
        wait_for_real_feedback(piper, args.feedback_timeout)
        print_status(piper, "initial")

        print("Selecting CAN_CTRL + MOVE_L...")
        if not enable_all(piper, 120, 0.02):
            print("[warn] Arm did not enable; refusing to send EndPoseCtrl.")
            print_status(piper, "failed enable before MOVE_L")
            return 1
        if not wait_for_movel_ready(piper, args.speed, args.feedback_timeout):
            print("[warn] MOVE_L mode did not become ready.")
            print_status(piper, "failed MOVE_L ready")
            return 1

        current_joints = joints_deg(piper)
        violations = joint_limit_violations(current_joints)
        if violations:
            print("[warn] Current joints are outside the selected MOVE_L branch.")
            print(f"Current joints deg: {fmt(current_joints)}")
            for item in violations:
                print(f"  [joint limit] {item}")
            print("Refusing to send EndPoseCtrl. Reset/repose the arm, then try again.")
            return 1

        if target is None:
            target = end_pose_raw(piper)
            start = list(target)
            target[0] += int(round((args.dx_mm or 0.0) * 1000.0))
            target[1] += int(round((args.dy_mm or 0.0) * 1000.0))
            target[2] += int(round((args.dz_mm or 0.0) * 1000.0))
            print(f"Captured current start raw xyz/rpy: {start}")
            print(f"Generated MOVE_L target raw xyz/rpy: {target}")

        print_status(piper, "before target")
        if not send_target(piper, target, args.speed, args.duration, args.rate_hz):
            print_status(piper, "failed target")
            return 1
        print_status(piper, "final")
    except KeyboardInterrupt:
        print("\nInterrupted. Motors were not disabled by this script.")
    finally:
        try:
            prompt_before_disable(piper)
        except Exception as exc:
            print(f"[warn] final disable prompt failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
