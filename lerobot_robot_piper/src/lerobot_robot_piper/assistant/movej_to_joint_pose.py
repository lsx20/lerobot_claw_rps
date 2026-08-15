#!/usr/bin/env python3
"""Move Piper to a joint target with MOVE_J only.

This script never switches to MOVE_P and never disables motors automatically.
It is intended to verify joint-mode motion separately from Cartesian control.
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
        if not lo <= joint <= hi:
            raise argparse.ArgumentTypeError(
                f"J{idx}={joint} is outside [{lo}, {hi}] deg"
            )
    return joints


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


def fmt(values: list[float]) -> str:
    return " ".join(f"{value:9.3f}" for value in values)


def print_status(piper: C_PiperInterface_V2, label: str) -> None:
    print(f"\n=== {label} ===")
    print("enable:", piper.GetArmEnableStatus())
    print(piper.GetArmStatus())
    print(piper.GetArmJointMsgs())
    print(piper.GetArmEndPoseMsgs())


def has_real_feedback(piper: C_PiperInterface_V2) -> bool:
    status = piper.GetArmStatus()
    joints = piper.GetArmJointMsgs()
    pose = piper.GetArmEndPoseMsgs()
    js = joints.joint_state
    ep = pose.end_pose
    has_hz = status.Hz > 0 or joints.Hz > 0 or pose.Hz > 0
    has_joint_data = any(
        value != 0
        for value in (
            js.joint_1,
            js.joint_2,
            js.joint_3,
            js.joint_4,
            js.joint_5,
            js.joint_6,
        )
    )
    has_pose_data = any(
        value != 0
        for value in (
            ep.X_axis,
            ep.Y_axis,
            ep.Z_axis,
            ep.RX_axis,
            ep.RY_axis,
            ep.RZ_axis,
        )
    )
    return has_hz or has_joint_data or has_pose_data


def wait_for_real_feedback(piper: C_PiperInterface_V2, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if has_real_feedback(piper):
            return
        time.sleep(0.05)
    raise RuntimeError(
        "No real Piper feedback received: ArmStatus/JointMsgs/EndPoseMsgs are Hz=0/all-zero. "
        "Refusing to command MOVE_J from fake zero feedback. Check CAN mode, can0 state, "
        "and whether another process owns the arm."
    )


def send_joint_target(
    piper: C_PiperInterface_V2,
    joints: list[float],
    speed: int,
    installation_pos: int,
) -> None:
    values = [int(round(joint * 1000.0)) for joint in joints]
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00, 0, installation_pos)
    piper.JointCtrl(*values)


def enable_all(piper: C_PiperInterface_V2, count: int, interval_s: float) -> bool:
    last_status: list[bool] = []
    for idx in range(count):
        piper.EnableArm(7, 0x02)
        time.sleep(interval_s)
        last_status = list(piper.GetArmEnableStatus())
        if idx % 10 == 0:
            print(f"enable status: {last_status}")
        if last_status and all(last_status):
            return True
    print(f"[warn] enable did not become all True: {last_status}")
    return False


def prepare_movej(
    piper: C_PiperInterface_V2,
    speed: int,
    installation_pos: int,
    count: int,
    interval_s: float,
) -> None:
    for _ in range(count):
        piper.MotionCtrl_1(0x02, 0x06, 0x06)
        time.sleep(interval_s)
    for _ in range(count):
        piper.MotionCtrl_1(0x02, 0x04, 0x02)
        time.sleep(interval_s)
    for _ in range(count):
        piper.MotionCtrl_1(0x02, 0x00, 0x02)
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00, 0, installation_pos)
        time.sleep(interval_s)


def max_joint_error_deg(actual: list[float], target: list[float]) -> float:
    return max(abs(actual[idx] - target[idx]) for idx in range(6))


def hold_current_position(
    piper: C_PiperInterface_V2,
    speed: int,
    installation_pos: int,
    duration_s: float,
    rate_hz: float,
) -> None:
    current = joints_deg(piper)
    print(f"Holding current joints: {fmt(current)}")
    interval_s = 1.0 / rate_hz
    end_t = time.time() + duration_s
    while time.time() < end_t:
        send_joint_target(piper, current, speed, installation_pos)
        time.sleep(interval_s)


def movej_interpolated(
    piper: C_PiperInterface_V2,
    target: list[float],
    speed: int,
    installation_pos: int,
    duration_s: float,
    rate_hz: float,
    settle_timeout_s: float,
    tolerance_deg: float,
) -> bool:
    start = joints_deg(piper)
    print(f"current joints deg: {fmt(start)}")
    print(f"target  joints deg: {fmt(target)}")
    steps = max(1, int(duration_s * rate_hz))
    interval_s = 1.0 / rate_hz
    for step in range(steps + 1):
        alpha = step / steps
        interp = [
            start[idx] + (target[idx] - start[idx]) * alpha
            for idx in range(6)
        ]
        send_joint_target(piper, interp, speed, installation_pos)
        if step % max(1, int(rate_hz)) == 0:
            print(f"movej {alpha * 100:5.1f}% joints: {fmt(joints_deg(piper))}")
        time.sleep(interval_s)

    print("Interpolation finished; waiting for joints to reach the final target...")
    deadline = time.time() + settle_timeout_s
    while time.time() < deadline:
        send_joint_target(piper, target, speed, installation_pos)
        actual = joints_deg(piper)
        error = max_joint_error_deg(actual, target)
        print(f"settling error={error:.3f} deg joints: {fmt(actual)}")
        if error <= tolerance_deg:
            print(f"MOVE_J target reached within {tolerance_deg:.3f} deg.")
            return True
        time.sleep(interval_s)

    actual = joints_deg(piper)
    error = max_joint_error_deg(actual, target)
    print(f"[warn] MOVE_J target was not reached; final error={error:.3f} deg.")
    hold_current_position(piper, speed, installation_pos, 1.0, 20.0)
    return False


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
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--feedback-timeout", type=float, default=8.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--settle-timeout", type=float, default=60.0)
    parser.add_argument("--joint-tolerance-deg", type=float, default=1.0)
    parser.add_argument(
        "--target-joints-deg",
        type=parse_joint_target,
        default=parse_joint_target("0,0,0,0,0,0"),
        help="6 comma-separated MOVE_J target joint degrees.",
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
    if args.feedback_timeout <= 0:
        raise ValueError("--feedback-timeout must be positive")
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.settle_timeout <= 0:
        raise ValueError("--settle-timeout must be positive")
    if args.joint_tolerance_deg <= 0:
        raise ValueError("--joint-tolerance-deg must be positive")

    print("This script will command MOVE_J only:")
    print(f"  target joints deg: {fmt(args.target_joints_deg)}")
    print("Keep the workspace clear and emergency stop reachable.")
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
        prepare_movej(piper, args.speed, args.installation_pos, 50, 0.02)
        enable_all(piper, 80, 0.02)
        wait_for_real_feedback(piper, args.feedback_timeout)
        print_status(piper, "before MOVE_J")
        reached = movej_interpolated(
            piper,
            target=args.target_joints_deg,
            speed=args.speed,
            installation_pos=args.installation_pos,
            duration_s=args.duration,
            rate_hz=args.rate_hz,
            settle_timeout_s=args.settle_timeout,
            tolerance_deg=args.joint_tolerance_deg,
        )
        print_status(piper, "after MOVE_J")
        if not reached:
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Attempting to hold current joint position...")
        try:
            hold_current_position(piper, args.speed, args.installation_pos, 1.0, 20.0)
        except Exception as exc:
            print(f"[warn] could not send hold command: {exc}")
    finally:
        try:
            prompt_before_disable(piper)
        except Exception as exc:
            print(f"[warn] final disable prompt failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
