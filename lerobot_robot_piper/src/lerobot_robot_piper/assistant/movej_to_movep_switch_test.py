#!/usr/bin/env python3
"""Verify switching from MOVE_J to MOVE_P.

MOVE_J side follows the official joint demo style:
  MotionCtrl_2(0x01, 0x01, speed, 0x00)
  JointCtrl(j1, j2, j3, j4, j5, j6)

MOVE_P side follows the official end-pose demo style:
  MotionCtrl_2(0x01, 0x00, speed, 0x00)
  EndPoseCtrl(X, Y, Z, RX, RY, RZ)

It never disables motors automatically.
"""

from __future__ import annotations

import argparse
import sys
import time

from piper_sdk import C_PiperInterface_V2


JOINT_LIMITS_DEG = {
    1: (-150.0, 150.0),
    2: (0.0, 180.0),
    3: (-170.0, 0.0),
    4: (-100.0, 100.0),
    5: (-70.0, 70.0),
    6: (-120.0, 120.0),
}


def parse_joint_target(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("expected 6 comma-separated joint degrees")
    try:
        joints = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("all joint targets must be numbers") from exc
    for idx, joint in enumerate(joints, start=1):
        lo, hi = JOINT_LIMITS_DEG[idx]
        if joint < lo or joint > hi:
            raise argparse.ArgumentTypeError(f"J{idx}={joint} outside [{lo}, {hi}]")
    return joints


def fmt(values: list[float]) -> str:
    return " ".join(f"{value:9.3f}" for value in values)


def joints_deg(piper: C_PiperInterface_V2) -> list[float]:
    js = piper.GetArmJointMsgs().joint_state
    return [
        js.joint_1 / 1000.0,
        js.joint_2 / 1000.0,
        js.joint_3 / 1000.0,
        js.joint_4 / 1000.0,
        js.joint_5 / 1000.0,
        js.joint_6 / 1000.0,
    ]


def end_pose_raw(piper: C_PiperInterface_V2) -> list[int]:
    ep = piper.GetArmEndPoseMsgs().end_pose
    return [ep.X_axis, ep.Y_axis, ep.Z_axis, ep.RX_axis, ep.RY_axis, ep.RZ_axis]


def pose_mm_deg(pose: list[int]) -> str:
    return (
        f"X={pose[0] / 1000.0:8.3f} Y={pose[1] / 1000.0:8.3f} "
        f"Z={pose[2] / 1000.0:8.3f} RX={pose[3] / 1000.0:8.3f} "
        f"RY={pose[4] / 1000.0:8.3f} RZ={pose[5] / 1000.0:8.3f}"
    )


def pose_error_mm_deg(actual: list[int], target: list[int]) -> tuple[float, float]:
    xyz_error = max(abs(actual[idx] - target[idx]) for idx in range(3)) / 1000.0
    rpy_error = max(abs(actual[idx] - target[idx]) for idx in range(3, 6)) / 1000.0
    return xyz_error, rpy_error


def arm_status_code(piper: C_PiperInterface_V2) -> int:
    return int(piper.GetArmStatus().arm_status.arm_status)


def ctrl_mode_code(piper: C_PiperInterface_V2) -> int:
    return int(piper.GetArmStatus().arm_status.ctrl_mode)


def mode_feed_code(piper: C_PiperInterface_V2) -> int:
    return int(piper.GetArmStatus().arm_status.mode_feed)


def has_real_feedback(piper: C_PiperInterface_V2) -> bool:
    status = piper.GetArmStatus()
    joints = piper.GetArmJointMsgs()
    pose = piper.GetArmEndPoseMsgs()
    js = joints.joint_state
    ep = pose.end_pose
    return (
        status.Hz > 0
        or joints.Hz > 0
        or pose.Hz > 0
        or any(
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
        )
    )


def wait_for_real_feedback(piper: C_PiperInterface_V2, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if has_real_feedback(piper):
            return
        time.sleep(0.05)
    raise RuntimeError("No real Piper feedback received; refusing to command motion.")


def enable_all(piper: C_PiperInterface_V2, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    count = 0
    while time.time() < deadline:
        piper.EnableArm(7, 0x02)
        time.sleep(0.02)
        enable_status = list(piper.GetArmEnableStatus())
        if count % 10 == 0:
            print(f"enable status: {enable_status}")
        if enable_status and all(enable_status):
            return True
        count += 1
    return False


def wait_for_mode_ready(
    piper: C_PiperInterface_V2,
    move_mode: int,
    speed: int,
    timeout_s: float,
) -> bool:
    deadline = time.time() + timeout_s
    count = 0
    while time.time() < deadline:
        piper.MotionCtrl_2(0x01, move_mode, speed, 0x00)
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
            and mode_feed == move_mode
            and arm_status == 0x00
            and all(enable_status)
        ):
            return True
        count += 1
    return False


def send_movej_target(
    piper: C_PiperInterface_V2,
    joints: list[float],
    speed: int,
) -> None:
    raw = [int(round(joint * 1000.0)) for joint in joints]
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    piper.JointCtrl(*raw)


def send_movep_target(
    piper: C_PiperInterface_V2,
    pose: list[int],
    speed: int,
) -> None:
    piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
    piper.EndPoseCtrl(*pose)


def max_joint_error(actual: list[float], target: list[float]) -> float:
    return max(abs(actual[idx] - target[idx]) for idx in range(6))


def run_movej(
    piper: C_PiperInterface_V2,
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
        send_movej_target(piper, waypoint, speed)
        if step % max(1, int(rate_hz)) == 0:
            print(f"movej {alpha * 100:5.1f}% joints: {fmt(joints_deg(piper))}")
        if arm_status_code(piper) != 0x00:
            print("[warn] Arm Status changed during MOVE_J.")
            return False
        time.sleep(interval_s)

    deadline = time.time() + settle_timeout_s
    while time.time() < deadline:
        send_movej_target(piper, target, speed)
        actual = joints_deg(piper)
        error = max_joint_error(actual, target)
        print(f"movej settling error={error:.3f} deg joints={fmt(actual)}")
        if error <= tolerance_deg:
            return True
        if arm_status_code(piper) != 0x00:
            print("[warn] Arm Status changed while settling MOVE_J.")
            return False
        time.sleep(interval_s)
    return False


def run_movep_z(
    piper: C_PiperInterface_V2,
    speed: int,
    dz_mm: float,
    duration_s: float,
    rate_hz: float,
    tolerance_mm: float,
    tolerance_deg: float,
) -> bool:
    start_pose = end_pose_raw(piper)
    target_pose = list(start_pose)
    target_pose[2] += int(round(dz_mm * 1000.0))
    print(f"MOVE_P captured start: {pose_mm_deg(start_pose)}")
    print(f"MOVE_P target:         {pose_mm_deg(target_pose)}")

    interval_s = 1.0 / rate_hz
    deadline = time.time() + duration_s
    reached = False
    while time.time() < deadline:
        send_movep_target(piper, target_pose, speed)
        time.sleep(interval_s)
        actual = end_pose_raw(piper)
        xyz_error, rpy_error = pose_error_mm_deg(actual, target_pose)
        status = piper.GetArmStatus()
        enable_status = list(piper.GetArmEnableStatus())
        print(
            "movep: "
            f"enable={enable_status} arm=0x{status.arm_status.arm_status:x} "
            f"motion=0x{status.arm_status.motion_status:x} "
            f"xyz_err={xyz_error:.3f}mm rpy_err={rpy_error:.3f}deg "
            f"pose=[{pose_mm_deg(actual)}]"
        )
        if not all(enable_status) or status.arm_status.arm_status != 0x00:
            return False
        if xyz_error <= tolerance_mm and rpy_error <= tolerance_deg:
            reached = True
            break
    return reached


def print_status(piper: C_PiperInterface_V2, label: str) -> None:
    print(f"\n=== {label} ===")
    print("enable:", piper.GetArmEnableStatus())
    print(piper.GetArmStatus())
    print(piper.GetArmJointMsgs())
    print(piper.GetArmEndPoseMsgs())


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
    parser.add_argument("--movej-speed", type=int, default=10)
    parser.add_argument("--movep-speed", type=int, default=3)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--feedback-timeout", type=float, default=8.0)
    parser.add_argument("--movej-duration", type=float, default=4.0)
    parser.add_argument("--movej-settle-timeout", type=float, default=8.0)
    parser.add_argument("--movej-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--movep-duration", type=float, default=4.0)
    parser.add_argument("--movep-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--movep-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--dz-mm", type=float, default=5.0)
    parser.add_argument(
        "--target-joints-deg",
        type=parse_joint_target,
        help="MOVE_J target. If omitted, hold current joints briefly before switching.",
    )
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("movej_speed", "movep_speed"):
        value = getattr(args, name)
        if value < 0 or value > 100:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 100]")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")

    print("This script verifies MOVE_J -> MOVE_P switching.")
    print("MOVE_J uses official MotionCtrl_2(0x01, 0x01, speed, 0x00).")
    print("MOVE_P uses official MotionCtrl_2(0x01, 0x00, speed, 0x00).")
    print(f"MOVE_P test dz: {args.dz_mm:.3f} mm")
    if args.target_joints_deg is not None:
        print(f"MOVE_J target joints: {fmt(args.target_joints_deg)}")
    else:
        print("MOVE_J target: current joints hold only")
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
    time.sleep(1.0)

    try:
        wait_for_real_feedback(piper, args.feedback_timeout)
        print_status(piper, "initial")
        if not enable_all(piper, args.feedback_timeout):
            print("[warn] Arm did not enable.")
            print_status(piper, "failed enable")
            return 1

        print("\nSelecting MOVE_J...")
        if not wait_for_mode_ready(piper, 0x01, args.movej_speed, args.feedback_timeout):
            print("[warn] MOVE_J mode did not become ready.")
            print_status(piper, "failed MOVE_J ready")
            return 1

        target_joints = args.target_joints_deg or joints_deg(piper)
        if not run_movej(
            piper,
            target_joints,
            args.movej_speed,
            args.movej_duration,
            args.movej_settle_timeout,
            args.rate_hz,
            args.movej_tolerance_deg,
        ):
            print("[warn] MOVE_J phase failed.")
            print_status(piper, "failed MOVE_J")
            return 1

        print_status(piper, "after MOVE_J")
        switch_start = time.time()
        print("\nSwitching immediately to MOVE_P...")
        if not wait_for_mode_ready(piper, 0x00, args.movep_speed, args.feedback_timeout):
            print("[warn] MOVE_P mode did not become ready.")
            print_status(piper, "failed MOVE_P ready")
            return 1
        switch_elapsed = time.time() - switch_start
        print(f"MOVE_J -> MOVE_P ready elapsed: {switch_elapsed:.3f}s")

        if not run_movep_z(
            piper,
            args.movep_speed,
            args.dz_mm,
            args.movep_duration,
            args.rate_hz,
            args.movep_tolerance_mm,
            args.movep_tolerance_deg,
        ):
            print("[warn] MOVE_P Z test failed.")
            print_status(piper, "failed MOVE_P")
            return 1

        print_status(piper, "final")
        print("MOVE_J -> MOVE_P switch test succeeded.")
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
