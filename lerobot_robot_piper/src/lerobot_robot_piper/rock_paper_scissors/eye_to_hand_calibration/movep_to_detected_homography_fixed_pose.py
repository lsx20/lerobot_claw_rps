#!/usr/bin/env python3
"""Detect a tabletop ball, map pixel to base XY, and move with a fixed RPY."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time

from scipy.spatial.transform import Rotation as R

from homography_tabletop_runtime import (
    CLAW_START_POSE_MM_DEG,
    PACKAGE_DIR,
    add_common_args,
    apply_homography,
    build_movep_command,
    build_movep_command_from_pose,
    detect_stable_pixel,
    load_homography,
    parse_rpy,
    print_and_maybe_execute_sequence,
)
from planar_grasp_geometry import radial_target, solve_planar_joint_target
from movej_home_then_movep import (
    arm_status_code,
    ctrl_mode_code,
    enable_all,
    end_pose_raw,
    fmt,
    joints_deg,
    mode_feed_code,
    print_status,
    prompt_before_disable,
    wait_for_real_feedback,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--rpy", type=parse_rpy, default=(172.0, 55.0, 180.0), help="RX,RY,RZ. In radial mode only RX,RY are used.")
    parser.add_argument(
        "--target-mode",
        choices=("radial", "xy_offset", "planar_joint"),
        default="radial",
        help="radial/xy_offset use MOVE_P target; planar_joint uses MOVE_P start then MOVE_J target",
    )
    parser.add_argument("--radial-offset-mm", type=float, default=45.0, help="flange distance behind ball along the base-origin ray")
    parser.add_argument("--rz-offset-deg", type=float, default=180.0, help="RZ = wrap(rz_offset_deg + atan2(Y, X)) in radial mode")
    parser.add_argument("--x-offset-mm", type=float, default=-90.0, help="added to mapped base X in xy_offset mode")
    parser.add_argument("--y-offset-mm", type=float, default=0.0, help="added to mapped base Y before MOVE_P")
    parser.add_argument("--start-duration", type=float, default=10.0, help="seconds for the initial start-pose MOVE_P")
    parser.add_argument("--joint-duration", type=float, default=8.0, help="seconds for planar_joint target MOVE_J")
    parser.add_argument("--joint-settle-timeout", type=float, default=20.0)
    parser.add_argument("--joint-tolerance-deg", type=float, default=1.5)
    parser.add_argument("--feedback-timeout", type=float, default=12.0)
    parser.add_argument("--planar-j4-deg", type=float, default=0.0)
    parser.add_argument("--planar-j6-deg", type=float, default=0.0)
    parser.add_argument("--planar-j5-seed-deg", type=float, default=13.0)
    parser.add_argument(
        "--auto-keep-enabled-after-joint",
        action="store_true",
        help="auto-press Enter at the final MOVE_J disable prompt. Default leaves the prompt interactive so D can be typed.",
    )
    parser.add_argument(
        "--start-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="move to claw-machine DEFAULT_START_POSE before moving to the detected target",
    )
    return parser.parse_args()


def build_movej_command(args: argparse.Namespace, joints_deg: list[float]) -> list[str]:
    joint_target = ",".join(f"{value:.3f}" for value in joints_deg)
    return [
        sys.executable,
        str(PACKAGE_DIR / "assistant" / "movej_to_joint_pose.py"),
        "--can",
        args.can,
        f"--target-joints-deg={joint_target}",
        "--speed",
        str(args.speed),
        "--duration",
        str(args.joint_duration),
        "--rate-hz",
        str(args.rate_hz),
        "--settle-timeout",
        str(args.joint_settle_timeout),
        "--joint-tolerance-deg",
        str(args.joint_tolerance_deg),
    ]


def print_and_maybe_execute_movej(args: argparse.Namespace, cmd: list[str]) -> int:
    print("target MOVE_J command:")
    print(" ".join(cmd))
    if not args.execute:
        print("dry run only. Add --execute to run MOVE_J.")
        return 0
    print("running target MOVE_J...")
    if args.auto_keep_enabled_after_joint:
        return subprocess.run(cmd, input="YES\n\n", text=True, check=False).returncode
    return subprocess.run(cmd, input="YES\n", text=True, check=False).returncode


def movep_target_raw(pose_mm_deg: tuple[float, float, float, float, float, float]) -> list[int]:
    return [int(round(value * 1000.0)) for value in pose_mm_deg]


def pose_error_mm_deg(actual: list[int], target: list[int]) -> tuple[float, float]:
    xyz_error = max(abs(actual[idx] - target[idx]) for idx in range(3)) / 1000.0
    rpy_error = max(abs(actual[idx] - target[idx]) for idx in range(3, 6)) / 1000.0
    return xyz_error, rpy_error


def format_status_value(value: object) -> str:
    try:
        return f"0x{int(value):x}"
    except (TypeError, ValueError):
        return str(value)


def wait_for_motion_mode(piper: object, move_mode: int, speed: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    count = 0
    while time.time() < deadline:
        piper.MotionCtrl_2(0x01, move_mode, speed, 0x00)
        piper.EnableArm(7, 0x02)
        time.sleep(0.05)
        enable_status = list(piper.GetArmEnableStatus())
        ctrl_mode = ctrl_mode_code(piper)
        mode_feed = mode_feed_code(piper)
        arm_status = arm_status_code(piper)
        if count % 10 == 0:
            print(
                "mode ready check: "
                f"ctrl=0x{ctrl_mode:x} mode=0x{mode_feed:x} "
                f"arm=0x{arm_status:x} enable={enable_status}"
            )
        if ctrl_mode == 0x01 and mode_feed == move_mode and arm_status == 0x00 and all(enable_status):
            return True
        count += 1
    return False


def send_movep_once(piper: object, target_raw: list[int], speed: int) -> None:
    piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
    piper.EndPoseCtrl(*target_raw)


def send_movej_once(piper: object, joints: list[float], speed: int) -> None:
    raw = [int(round(value * 1000.0)) for value in joints]
    piper.JointCtrl(*raw)


def run_movep_target(
    piper: object,
    target_raw: list[int],
    speed: int,
    duration_s: float,
    rate_hz: float,
    label: str,
) -> bool:
    interval_s = 1.0 / rate_hz
    deadline = time.time() + duration_s
    while time.time() < deadline:
        send_movep_once(piper, target_raw, speed)
        time.sleep(interval_s)
        actual = end_pose_raw(piper)
        xyz_error, rpy_error = pose_error_mm_deg(actual, target_raw)
        enable_status = list(piper.GetArmEnableStatus())
        status = piper.GetArmStatus().arm_status
        print(
            f"{label}: enable={enable_status} arm={format_status_value(getattr(status, 'arm_status', ''))} "
            f"motion={format_status_value(getattr(status, 'motion_status', ''))} xyz_err={xyz_error:.3f}mm "
            f"rpy_err={rpy_error:.3f}deg joints={fmt(joints_deg(piper))}"
        )
        if not all(enable_status) or arm_status_code(piper) != 0x00:
            return False
        if xyz_error <= 2.0 and rpy_error <= 2.0:
            return True
    return True


def max_joint_error(actual: list[float], target: list[float]) -> float:
    return max(abs(actual[idx] - target[idx]) for idx in range(6))


def run_movej_target(
    piper: object,
    target: list[float],
    speed: int,
    duration_s: float,
    settle_timeout_s: float,
    rate_hz: float,
    tolerance_deg: float,
) -> bool:
    start = joints_deg(piper)
    print(f"MOVE_J start joints:  {fmt(start)}")
    print(f"MOVE_J target joints: {fmt(target)}")
    interval_s = 1.0 / rate_hz
    steps = max(1, int(duration_s * rate_hz))
    for step in range(steps + 1):
        alpha = step / steps
        waypoint = [start[idx] + (target[idx] - start[idx]) * alpha for idx in range(6)]
        send_movej_once(piper, waypoint, speed)
        if step % max(1, int(rate_hz)) == 0:
            print(f"movej {alpha * 100:5.1f}% joints={fmt(joints_deg(piper))}")
        if not all(piper.GetArmEnableStatus()) or arm_status_code(piper) != 0x00:
            return False
        time.sleep(interval_s)
    deadline = time.time() + settle_timeout_s
    while time.time() < deadline:
        send_movej_once(piper, target, speed)
        actual = joints_deg(piper)
        error = max_joint_error(actual, target)
        print(f"movej settling error={error:.3f} deg joints={fmt(actual)}")
        if error <= tolerance_deg:
            return True
        if not all(piper.GetArmEnableStatus()) or arm_status_code(piper) != 0x00:
            return False
        time.sleep(interval_s)
    return False


def run_start_movep_then_planar_movej(args: argparse.Namespace, target_joints: list[float]) -> int:
    print("single-session planar_joint execution: start MOVE_P -> target MOVE_J")
    if not args.execute:
        print("dry run only. Add --execute to move.")
        return 0
    from piper_sdk import C_PiperInterface_V2

    command_pregrasp_args = args
    from homography_tabletop_runtime import command_pregrasp

    command_pregrasp(command_pregrasp_args)
    piper = C_PiperInterface_V2(args.can, judge_flag=False, can_auto_init=False, dh_is_offset=1, start_sdk_fk_cal=True)
    piper.ConnectPort()
    time.sleep(1.0)
    try:
        wait_for_real_feedback(piper, args.feedback_timeout)
        print_status(piper, "initial")
        print("Enabling Piper arm...")
        if not enable_all(piper, 120, 0.02):
            print("[warn] Arm did not enable.")
            print_status(piper, "failed enable")
            return 1
        if args.start_first:
            print("Selecting MOVE_P for start pose...")
            if not wait_for_motion_mode(piper, 0x00, args.speed, args.feedback_timeout):
                print("[warn] MOVE_P mode did not become ready for start.")
                return 1
            if not run_movep_target(
                piper,
                movep_target_raw(CLAW_START_POSE_MM_DEG),
                args.speed,
                args.start_duration,
                args.rate_hz,
                "start",
            ):
                print("[warn] start MOVE_P failed.")
                return 1
            print_status(piper, "after start MOVE_P")

        print("Switching to MOVE_J in the same SDK session...")
        current = joints_deg(piper)
        if not wait_for_motion_mode(piper, 0x01, args.speed, args.feedback_timeout):
            print("[warn] MOVE_J mode did not become ready.")
            return 1
        for _ in range(5):
            send_movej_once(piper, current, 1)
            time.sleep(0.02)
        if not run_movej_target(
            piper,
            target_joints,
            args.speed,
            args.joint_duration,
            args.joint_settle_timeout,
            args.rate_hz,
            args.joint_tolerance_deg,
        ):
            print("[warn] target MOVE_J failed.")
            return 1
        print_status(piper, "final")
    finally:
        try:
            if args.auto_keep_enabled_after_joint:
                print("Piper arm motors left enabled by --auto-keep-enabled-after-joint.")
            else:
                prompt_before_disable(piper)
        finally:
            try:
                piper.DisconnectPort()
            except Exception:
                pass
    return 0


def main() -> int:
    args = parse_args()
    homography = load_homography(args.calibration)
    detection = detect_stable_pixel(args, homography)
    ball_xy_m = apply_homography(homography, detection.pixel)
    if args.target_mode in {"radial", "planar_joint"}:
        target_xy_m, rpy, theta_deg = radial_target(
            ball_xy_m,
            args.radial_offset_mm,
            args.rz_offset_deg,
            args.rpy[0],
            args.rpy[1],
        )
    else:
        theta_deg = math.degrees(math.atan2(ball_xy_m[1], ball_xy_m[0]))
        target_xy_m = (
            ball_xy_m[0] + args.x_offset_mm / 1000.0,
            ball_xy_m[1] + args.y_offset_mm / 1000.0,
        )
        rpy = args.rpy
    print(f"pixel={detection.pixel} conf={detection.confidence:.3f}")
    print(f"ball_xy_m=({ball_xy_m[0]:.6f}, {ball_xy_m[1]:.6f}) theta_deg={theta_deg:.3f}")
    if args.target_mode in {"radial", "planar_joint"}:
        print(f"radial_offset_mm={args.radial_offset_mm:.1f} rz_offset_deg={args.rz_offset_deg:.1f}")
    else:
        print(f"xy_offset_mm=({args.x_offset_mm:.1f}, {args.y_offset_mm:.1f})")
    print(f"flange_target_xy_m=({target_xy_m[0]:.6f}, {target_xy_m[1]:.6f})")
    if args.target_mode == "planar_joint":
        joints, fk, error_mm = solve_planar_joint_target(
            target_xy_m,
            args.fixed_z_mm,
            args.planar_j4_deg,
            args.planar_j6_deg,
            args.planar_j5_seed_deg,
        )
        fk_xyz = fk[:3, 3]
        fk_rpy = R.from_matrix(fk[:3, :3]).as_euler("xyz", degrees=True)
        print("planar_joint_target_deg=" + ",".join(f"{value:.3f}" for value in joints))
        print(f"planar_joint_fk_xyz_m=({fk_xyz[0]:.6f}, {fk_xyz[1]:.6f}, {fk_xyz[2]:.6f}) error={error_mm:.2f}mm")
        print(f"planar_joint_fk_rpy_deg=({fk_rpy[0]:.3f}, {fk_rpy[1]:.3f}, {fk_rpy[2]:.3f})")
        print(f"planar constraints: J1=atan2(target_Y,target_X), J4={args.planar_j4_deg:.1f}, J6={args.planar_j6_deg:.1f}")
        if args.start_first:
            start_cmd, start_target = build_movep_command_from_pose(args, CLAW_START_POSE_MM_DEG)
            print("start MOVE_P command:")
            print(" ".join(start_cmd))
            print(f"start_movep_target_mm_deg={start_target}")
        print("target MOVE_J command: same SDK session, no child process")
        print("target_joints_deg=" + ",".join(f"{value:.3f}" for value in joints))
        return run_start_movep_then_planar_movej(args, joints)
    print(f"target_rpy_deg=({rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f})")
    target_cmd, target = build_movep_command(args, target_xy_m, rpy)
    commands = []
    if args.start_first:
        target_duration = args.duration
        args.duration = args.start_duration
        start_cmd, start_target = build_movep_command_from_pose(args, CLAW_START_POSE_MM_DEG)
        args.duration = target_duration
        commands.append(("start", start_cmd, start_target))
    commands.append(("target", target_cmd, target))
    return print_and_maybe_execute_sequence(args, commands)


if __name__ == "__main__":
    raise SystemExit(main())
