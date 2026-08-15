#!/usr/bin/env python3
"""Test Piper MOVE_P RPY axes from the current end pose.

This follows the official MOVE_P demo call style:
  MotionCtrl_2(0x01, 0x00, speed, 0x00)
  EndPoseCtrl(X, Y, Z, RX, RY, RZ)

By default it captures the current pose and changes only RY by a small amount.
It does not disable motors automatically.
"""

from __future__ import annotations

import argparse
import sys
import time

from piper_sdk import C_PiperInterface_V2


AXIS_INDEX = {
    "rx": 3,
    "ry": 4,
    "rz": 5,
}

JOINT_LIMITS_DEG = [
    (-150.0, 150.0),
    (0.0, 180.0),
    (-170.0, 0.0),
    (-100.0, 100.0),
    (-70.0, 70.0),
    (-120.0, 120.0),
]


def parse_joint_target(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("expected J1,J2,J3,J4,J5,J6")
    try:
        joints = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("joint values must be numbers") from exc
    for idx, (joint, (lo, hi)) in enumerate(
        zip(joints, JOINT_LIMITS_DEG, strict=True),
        start=1,
    ):
        if joint < lo or joint > hi:
            raise argparse.ArgumentTypeError(f"J{idx}={joint} outside [{lo}, {hi}]")
    return joints


def end_pose_raw(piper: C_PiperInterface_V2) -> list[int]:
    ep = piper.GetArmEndPoseMsgs().end_pose
    return [ep.X_axis, ep.Y_axis, ep.Z_axis, ep.RX_axis, ep.RY_axis, ep.RZ_axis]


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


def fmt_joints(joints: list[float]) -> str:
    return " ".join(f"{joint:8.3f}" for joint in joints)


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
    pose = piper.GetArmEndPoseMsgs()
    joints = piper.GetArmJointMsgs()
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


def wait_for_real_feedback(piper: C_PiperInterface_V2, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if has_real_feedback(piper):
            return
        time.sleep(0.05)
    raise RuntimeError("No Piper feedback received; check CAN and power.")


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


def wait_for_movep_ready(
    piper: C_PiperInterface_V2,
    speed: int,
    timeout_s: float,
) -> bool:
    deadline = time.time() + timeout_s
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
        if count % 10 == 0:
            print(
                "mode ready check: "
                f"ctrl=0x{ctrl_mode:x} mode=0x{mode_feed:x} "
                f"arm=0x{arm_status:x} hz={status.Hz:.1f} "
                f"enable={enable_status}"
            )
        if (
            ctrl_mode == 0x01
            and mode_feed == 0x00
            and arm_status == 0x00
            and all(enable_status)
        ):
            return True
        count += 1
    return False


def wait_for_movej_ready(
    piper: C_PiperInterface_V2,
    speed: int,
    timeout_s: float,
) -> bool:
    deadline = time.time() + timeout_s
    count = 0
    while time.time() < deadline:
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        piper.EnableArm(7, 0x02)
        time.sleep(0.05)
        status = piper.GetArmStatus()
        enable_status = list(piper.GetArmEnableStatus())
        ctrl_mode = ctrl_mode_code(piper)
        mode_feed = mode_feed_code(piper)
        arm_status = arm_status_code(piper)
        if count % 10 == 0:
            print(
                "movej ready check: "
                f"ctrl=0x{ctrl_mode:x} mode=0x{mode_feed:x} "
                f"arm=0x{arm_status:x} hz={status.Hz:.1f} "
                f"enable={enable_status}"
            )
        if (
            ctrl_mode == 0x01
            and mode_feed == 0x01
            and arm_status == 0x00
            and all(enable_status)
        ):
            return True
        count += 1
    return False


def send_movej_for(
    piper: C_PiperInterface_V2,
    target: list[float],
    speed: int,
    duration_s: float,
    rate_hz: float,
) -> bool:
    raw = [int(round(joint * 1000.0)) for joint in target]
    interval_s = 1.0 / rate_hz
    deadline = time.time() + duration_s
    while time.time() < deadline:
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        piper.JointCtrl(*raw)
        time.sleep(interval_s)
        actual = joints_deg(piper)
        error = max(abs(actual[idx] - target[idx]) for idx in range(6))
        enable_status = list(piper.GetArmEnableStatus())
        arm_status = arm_status_code(piper)
        print(
            f"\rmovej start: arm=0x{arm_status:x} err={error:.3f}deg "
            f"joints={fmt_joints(actual)}",
            end="",
            flush=True,
        )
        if not all(enable_status) or arm_status != 0x00:
            print()
            return False
        if error <= 0.8:
            print()
            return True
    print()
    return True


def send_pose_for(
    piper: C_PiperInterface_V2,
    target: list[int],
    speed: int,
    duration_s: float,
    rate_hz: float,
    label: str,
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
                f"\r{label}: arm=0x{arm_status:x} motion=0x{motion_status:x} "
                f"xyz_err={xyz_error:.3f}mm rpy_err={rpy_error:.3f}deg "
                f"joints={fmt_joints(joints_deg(piper))} pose=[{pose_mm_deg(actual)}]",
                end="",
                flush=True,
            )
            if not all(enable_status) or arm_status != 0x00:
                print()
                return False
        count += 1
    print()
    return True


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
    parser.add_argument("--speed", type=int, default=2)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--feedback-timeout", type=float, default=8.0)
    parser.add_argument("--axis", choices=tuple(AXIS_INDEX), default="ry")
    parser.add_argument("--delta-deg", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--movej-start-joints", type=parse_joint_target)
    parser.add_argument("--movej-duration", type=float, default=8.0)
    parser.add_argument(
        "--return-start",
        action="store_true",
        help="after the axis test, move back to the captured start pose",
    )
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.speed <= 100:
        raise ValueError("--speed must be in [0, 100]")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")

    print("Piper MOVE_P RPY axis test")
    print("This captures the current pose, then changes only one RPY axis.")
    print("It follows official MOVE_P style and does not disable motors automatically.")
    print(f"axis={args.axis.upper()} delta={args.delta_deg:.3f} deg speed={args.speed}")
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
        print(f"initial pose: {pose_mm_deg(end_pose_raw(piper))}")
        print(f"initial joints: {fmt_joints(joints_deg(piper))}")

        if not enable_all(piper, args.feedback_timeout):
            print("[warn] Arm did not enable.")
            return 1

        if args.movej_start_joints is not None:
            print("Selecting CAN_CTRL + MOVE_J for start joints...")
            if not wait_for_movej_ready(piper, args.speed, args.feedback_timeout):
                print("[warn] MOVE_J did not become ready.")
                print(piper.GetArmStatus())
                return 1
            print(f"MOVE_J start target: {fmt_joints(args.movej_start_joints)}")
            if not send_movej_for(
                piper,
                args.movej_start_joints,
                args.speed,
                args.movej_duration,
                args.rate_hz,
            ):
                print("[warn] MOVE_J start failed.")
                return 1
            print(f"after MOVE_J pose: {pose_mm_deg(end_pose_raw(piper))}")
            print(f"after MOVE_J joints: {fmt_joints(joints_deg(piper))}")

        print("Selecting CAN_CTRL + MOVE_P...")
        if not wait_for_movep_ready(piper, args.speed, args.feedback_timeout):
            print("[warn] MOVE_P did not become ready. Reset if Arm Status is 0x4.")
            print(piper.GetArmStatus())
            return 1

        start = end_pose_raw(piper)
        target = list(start)
        target[AXIS_INDEX[args.axis]] += int(round(args.delta_deg * 1000.0))

        print(f"captured start: {pose_mm_deg(start)}")
        print(f"target:         {pose_mm_deg(target)}")
        print("Only this value changes:")
        print(
            f"  {args.axis.upper()}: "
            f"{start[AXIS_INDEX[args.axis]] / 1000.0:.3f} -> "
            f"{target[AXIS_INDEX[args.axis]] / 1000.0:.3f} deg"
        )

        if not send_pose_for(
            piper,
            target,
            args.speed,
            args.duration,
            args.rate_hz,
            f"{args.axis.upper()} test",
        ):
            print("[warn] axis test failed")
            return 1

        if args.return_start:
            print("Returning to captured start pose...")
            if not send_pose_for(
                piper,
                start,
                args.speed,
                args.duration,
                args.rate_hz,
                "return",
            ):
                print("[warn] return failed")
                return 1

        print("Done.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Motors were not disabled by this script.")
        return 1
    finally:
        try:
            prompt_before_disable(piper)
        except Exception as exc:
            print(f"[warn] final disable prompt failed: {exc}")


if __name__ == "__main__":
    sys.exit(main())
