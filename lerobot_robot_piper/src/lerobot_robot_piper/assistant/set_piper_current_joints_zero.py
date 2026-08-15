#!/usr/bin/env python3
"""Set the current Piper joint readings as the controller zero point.

This writes a zero-point configuration to the arm controller via JointConfig.
It does not command motion and it never disables motors automatically.
"""

from __future__ import annotations

import argparse
import sys
import time

from piper_sdk import C_PiperInterface_V2


def joints_raw(piper: C_PiperInterface_V2) -> list[int]:
    js = piper.GetArmJointMsgs().joint_state
    return [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]


def fmt_raw_and_deg(values: list[int]) -> str:
    return " ".join(f"J{i + 1}={raw} ({raw / 1000.0:.3f} deg)" for i, raw in enumerate(values))


def print_status(piper: C_PiperInterface_V2, label: str) -> None:
    print(f"\n=== {label} ===")
    print("enable:", piper.GetArmEnableStatus())
    print(piper.GetArmStatus())
    print(piper.GetArmJointMsgs())
    print(piper.GetArmEndPoseMsgs())


def wait_for_feedback(piper: C_PiperInterface_V2, timeout_s: float = 5.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status = piper.GetArmStatus()
            joints = piper.GetArmJointMsgs()
            if status.Hz > 0 or joints.Hz > 0 or any(joints_raw(piper)):
                return True
        except Exception:
            pass
        time.sleep(0.05)
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
    parser.add_argument(
        "--joint",
        type=int,
        default=7,
        choices=range(1, 8),
        help="Joint to zero, or 7 for all joints.",
    )
    parser.add_argument(
        "--yes-i-understand",
        action="store_true",
        help="Skip the SETZERO confirmation prompt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    piper = C_PiperInterface_V2(
        args.can,
        judge_flag=False,
        can_auto_init=False,
        dh_is_offset=1,
        start_sdk_fk_cal=True,
    )
    piper.ConnectPort()
    time.sleep(1.0)

    if not wait_for_feedback(piper):
        print("No Piper feedback was received. Check CAN mode and power before setting zero.")
        return 1

    before = joints_raw(piper)
    print_status(piper, "before set-zero")
    print("\nCurrent joint readings that will become zero:")
    print(fmt_raw_and_deg(before))
    print()
    print("This writes controller joint zero configuration.")
    print("After this, current feedback should read near 0 for the selected joint(s).")
    print("Only do this if the arm is physically in the pose you want to define as zero.")

    if not args.yes_i_understand:
        answer = input("Type SETZERO to write current joint position as zero: ").strip()
        if answer != "SETZERO":
            print("Aborted. Zero point was not changed.")
            return 1

    print(f"Writing zero point for joint={args.joint}...")
    for _ in range(20):
        piper.JointConfig(args.joint, 0xAE, 0x00, 500, 0x00)
        time.sleep(0.05)

    time.sleep(1.0)
    after = joints_raw(piper)
    print_status(piper, "after set-zero")
    print("\nJoint readings after set-zero:")
    print(fmt_raw_and_deg(after))
    print("\nIf the values did not change, power-cycle the arm controller and read again.")

    try:
        prompt_before_disable(piper)
    except KeyboardInterrupt:
        print("\nInterrupted. Motors were not disabled by this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
