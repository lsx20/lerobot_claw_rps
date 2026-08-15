#!/usr/bin/env python3
"""Run MOVE_P demo points without MOVE_J.

Official demo points:
  start  = [57, 0, 215, 0, 85, 0] mm/deg
  target = [57, 0, 260, 0, 85, 0] mm/deg

By default this sends the official EndPoseCtrl targets directly, matching the
SDK demo's intent. With --use-current-start it captures the current feedback
pose after MOVE_P is ready and sends only a small Z offset from there.
"""

from __future__ import annotations

import argparse
import sys
import time

from piper_sdk import C_PiperInterface_V2

from movej_home_then_movep import (
    arm_status_code,
    ctrl_mode_code,
    enable_all,
    end_pose_raw,
    fmt,
    joints_deg,
    mode_feed_code,
    print_driver_summary,
    print_status,
    prompt_before_disable,
    select_mode,
    status_feedback_is_fresh,
    wait_for_mode_ready,
    wait_for_real_feedback,
)


OFFICIAL_START_RAW = [57000, 0, 215000, 0, 85000, 0]
OFFICIAL_TARGET_RAW = [57000, 0, 260000, 0, 85000, 0]
MOVE_P_BRANCH_LIMITS_DEG = {
    1: (-150.0, 150.0),
    2: (0.0, 180.0),
    3: (-170.0, 0.0),
    4: (-100.0, 100.0),
    5: (-70.0, 70.0),
    6: (-120.0, 120.0),
}


def parse_pose_raw(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("expected 6 comma-separated raw pose values")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("all pose values must be integers") from exc


def pose_error_mm_deg(actual: list[int], target: list[int]) -> tuple[float, float]:
    xyz_error = max(abs(actual[idx] - target[idx]) for idx in range(3)) / 1000.0
    rpy_error = max(abs(actual[idx] - target[idx]) for idx in range(3, 6)) / 1000.0
    return xyz_error, rpy_error


def joint_limit_violations(joints: list[float]) -> list[str]:
    violations = []
    for idx, value in enumerate(joints, start=1):
        lo, hi = MOVE_P_BRANCH_LIMITS_DEG[idx]
        if value < lo or value > hi:
            violations.append(f"J{idx}={value:.3f} outside [{lo:.1f},{hi:.1f}]")
    return violations


def send_movep_target(
    piper: C_PiperInterface_V2,
    target: list[int],
    speed: int,
    installation_pos: int,
    official_motion_ctrl: bool,
    duration_s: float,
    rate_hz: float,
    label: str,
) -> bool:
    print(f"\nSending official MOVE_P {label}: {target}")
    interval_s = 1.0 / rate_hz
    end_t = time.time() + duration_s
    count = 0
    while time.time() < end_t:
        if official_motion_ctrl:
            piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
        else:
            piper.MotionCtrl_2(0x01, 0x00, speed, 0x00, 0, installation_pos)
        piper.EndPoseCtrl(*target)
        time.sleep(interval_s)

        if count % max(1, int(rate_hz / 2)) == 0:
            actual = end_pose_raw(piper)
            xyz_error, rpy_error = pose_error_mm_deg(actual, target)
            enable_status = list(piper.GetArmEnableStatus())
            arm_status = arm_status_code(piper)
            motion_status = int(piper.GetArmStatus().arm_status.motion_status)
            print(
                f"{label}: enable={enable_status} arm=0x{arm_status:x} "
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


def select_mode_official(
    piper: C_PiperInterface_V2,
    move_mode: int,
    speed: int,
    count: int,
    interval_s: float,
) -> None:
    for _ in range(count):
        piper.MotionCtrl_2(0x01, move_mode, speed, 0x00)
        time.sleep(interval_s)


def wait_for_mode_ready_official(
    piper: C_PiperInterface_V2,
    move_mode: int,
    speed: int,
    timeout_s: float,
) -> bool:
    deadline = time.time() + timeout_s
    first_status_time = float(piper.GetArmStatus().time_stamp)
    count = 0
    while time.time() < deadline:
        piper.MotionCtrl_2(0x01, move_mode, speed, 0x00)
        piper.EnableArm(7, 0x02)
        time.sleep(0.05)
        enable_status = list(piper.GetArmEnableStatus())
        ctrl_mode = ctrl_mode_code(piper)
        mode_feed = mode_feed_code(piper)
        arm_status = arm_status_code(piper)
        status = piper.GetArmStatus()
        fresh = status_feedback_is_fresh(status, first_status_time)
        if count % 10 == 0:
            print(
                "mode ready check: "
                f"ctrl=0x{ctrl_mode:x} mode=0x{mode_feed:x} "
                f"arm=0x{arm_status:x} hz={status.Hz:.1f} "
                f"fresh={fresh} enable={enable_status}"
            )
        if (
            fresh
            and ctrl_mode == 0x01
            and mode_feed == move_mode
            and arm_status == 0x00
            and all(enable_status)
        ):
            return True
        count += 1
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0")
    parser.add_argument("--speed", type=int, default=3)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--feedback-timeout", type=float, default=8.0)
    parser.add_argument("--start-duration", type=float, default=4.0)
    parser.add_argument("--target-duration", type=float, default=4.0)
    parser.add_argument(
        "--official-start-raw",
        type=parse_pose_raw,
        default=list(OFFICIAL_START_RAW),
    )
    parser.add_argument(
        "--official-target-raw",
        type=parse_pose_raw,
        default=list(OFFICIAL_TARGET_RAW),
    )
    parser.add_argument(
        "--use-current-start",
        action="store_true",
        help="capture current feedback pose after MOVE_P ready, then target current Z + --z-offset-mm",
    )
    parser.add_argument(
        "--z-offset-mm",
        type=float,
        default=10.0,
        help="Z offset used with --use-current-start",
    )
    parser.add_argument(
        "--send-start-first",
        action="store_true",
        help="also send the captured/official start pose before target",
    )
    parser.add_argument(
        "--official-motion-ctrl",
        action="store_true",
        help="match the official demo by calling MotionCtrl_2 with 4 args only",
    )
    parser.add_argument(
        "--installation-pos",
        type=lambda value: int(value, 0),
        default=0x01,
        choices=(0x01, 0x02, 0x03),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.speed <= 100:
        raise ValueError("--speed must be in [0, 100]")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")

    print("This script runs MOVE_P EndPoseCtrl targets only.")
    if args.use_current_start:
        print("  start raw:  will be captured from current feedback after MOVE_P ready")
        print(f"  target raw: captured start + Z {args.z_offset_mm:.3f} mm")
    else:
        print(f"  start raw:  {args.official_start_raw}")
        print(f"  target raw: {args.official_target_raw}")
    print("It does not run MOVE_J and does not disable motors automatically.")
    print("Start with the arm physically near the official start/zero branch.")
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

        print("Selecting CAN_CTRL + MOVE_P...")
        if args.official_motion_ctrl:
            select_mode_official(piper, 0x00, args.speed, 50, 0.02)
        else:
            select_mode(piper, 0x00, args.speed, args.installation_pos, 50, 0.02)
        print("Enabling arm motors...")
        if not enable_all(piper, 120, 0.02):
            print("[warn] Arm did not enable; refusing to send EndPoseCtrl.")
            print_status(piper, "failed enable before MOVE_P")
            print_driver_summary(piper, "failed enable before MOVE_P")
            return 1
        if args.official_motion_ctrl:
            mode_ready = wait_for_mode_ready_official(
                piper,
                0x00,
                args.speed,
                args.feedback_timeout,
            )
        else:
            mode_ready = wait_for_mode_ready(
                piper,
                0x00,
                args.speed,
                args.installation_pos,
                args.feedback_timeout,
            )
        if not mode_ready:
            print("[warn] MOVE_P mode did not become ready.")
            print_status(piper, "failed MOVE_P ready")
            return 1
        print_status(piper, "before MOVE_P target")

        start_raw = list(args.official_start_raw)
        target_raw = list(args.official_target_raw)
        if args.use_current_start:
            start_raw = end_pose_raw(piper)
            target_raw = list(start_raw)
            target_raw[2] += int(round(args.z_offset_mm * 1000.0))
            print(f"Captured current MOVE_P start raw xyz/rpy: {start_raw}")
            print(f"Generated MOVE_P target raw xyz/rpy:       {target_raw}")

        current_joints = joints_deg(piper)
        violations = joint_limit_violations(current_joints)
        if violations:
            print(
                "[warn] Current joints are outside the selected MOVE_P branch; "
                "refusing to send EndPoseCtrl."
            )
            print(f"Current joints deg: {fmt(current_joints)}")
            for item in violations:
                print(f"  [joint limit] {item}")
            return 1

        if args.send_start_first:
            if not send_movep_target(
                piper,
                start_raw,
                args.speed,
                args.installation_pos,
                args.official_motion_ctrl,
                args.start_duration,
                args.rate_hz,
                "start",
            ):
                print_status(piper, "failed start")
                return 1

        if not send_movep_target(
            piper,
            target_raw,
            args.speed,
            args.installation_pos,
            args.official_motion_ctrl,
            args.target_duration,
            args.rate_hz,
            "target",
        ):
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
