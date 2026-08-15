#!/usr/bin/env python3
"""Move Piper to an absolute end-effector pose with MOVE_P.

This intentionally follows the official SDK MOVE_P demo style:
  MotionCtrl_2(0x01, 0x00, speed, 0x00)
  EndPoseCtrl(X, Y, Z, RX, RY, RZ)

It does not use MOVE_J and does not disable motors automatically.
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
    status_feedback_is_fresh,
    wait_for_real_feedback,
)


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


def pose_error_mm_deg(actual: list[int], target: list[int]) -> tuple[float, float]:
    xyz_error = max(abs(actual[idx] - target[idx]) for idx in range(3)) / 1000.0
    rpy_error = max(abs(actual[idx] - target[idx]) for idx in range(3, 6)) / 1000.0
    return xyz_error, rpy_error


def wait_for_movep_ready(
    piper: C_PiperInterface_V2,
    speed: int,
    timeout_s: float,
) -> bool:
    deadline = time.time() + timeout_s
    first_status_time = float(piper.GetArmStatus().time_stamp)
    count = 0
    while time.time() < deadline:
        piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
        piper.EnableArm(7, 0x02)
        time.sleep(0.05)

        status = piper.GetArmStatus()
        enable_status = list(piper.GetArmEnableStatus())
        ctrl_mode = ctrl_mode_code(piper)
        mode_feed = mode_feed_code(piper)
        arm_status = arm_status_code(piper)
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
            and mode_feed == 0x00
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
        piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
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
    parser.add_argument(
        "--target",
        type=parse_pose_mm_deg,
        help="absolute target X,Y,Z,RX,RY,RZ in mm and degrees",
    )
    parser.add_argument(
        "--target-raw",
        type=parse_pose_raw,
        help="absolute target X,Y,Z,RX,RY,RZ in raw 0.001mm/0.001deg units",
    )
    parser.add_argument(
        "--target-z",
        type=float,
        help="absolute target Z in mm; keep current X,Y,RX,RY,RZ",
    )
    parser.add_argument(
        "--target-z-raw",
        type=int,
        help="absolute target Z in raw 0.001mm units; keep current X,Y,RX,RY,RZ",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_modes = [
        args.target is not None,
        args.target_raw is not None,
        args.target_z is not None,
        args.target_z_raw is not None,
    ]
    if sum(target_modes) != 1:
        raise ValueError(
            "provide exactly one of --target, --target-raw, --target-z, or --target-z-raw"
        )
    if not 0 <= args.speed <= 100:
        raise ValueError("--speed must be in [0, 100]")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")

    target = args.target_raw if args.target_raw is not None else args.target
    if args.target_z is not None:
        target = None
        target_z_raw = int(round(args.target_z * 1000.0))
    elif args.target_z_raw is not None:
        target = None
        target_z_raw = args.target_z_raw
    else:
        target_z_raw = None
    print("This script sends one absolute MOVE_P target.")
    if target_z_raw is None:
        print(f"  target raw: {target}")
        print(
            "  target mm/deg: "
            f"{target[0] / 1000:.3f}, {target[1] / 1000:.3f}, "
            f"{target[2] / 1000:.3f}, {target[3] / 1000:.3f}, "
            f"{target[4] / 1000:.3f}, {target[5] / 1000:.3f}"
        )
    else:
        print(
            "  target: keep current X,Y,RX,RY,RZ and move to "
            f"Z={target_z_raw / 1000.0:.3f} mm"
        )
    print("It follows the official MOVE_P demo control call and never disables motors automatically.")
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
        if not enable_all(piper, 120, 0.02):
            print("[warn] Arm did not enable; refusing to send EndPoseCtrl.")
            print_driver_summary(piper, "failed enable before MOVE_P")
            return 1
        if not wait_for_movep_ready(piper, args.speed, args.feedback_timeout):
            print("[warn] MOVE_P mode did not become ready.")
            print_status(piper, "failed MOVE_P ready")
            return 1

        if target_z_raw is not None:
            target = end_pose_raw(piper)
            start = list(target)
            target[2] = target_z_raw
            print(f"Captured current start raw xyz/rpy: {start}")
            print(f"Generated vertical target raw xyz/rpy: {target}")

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
