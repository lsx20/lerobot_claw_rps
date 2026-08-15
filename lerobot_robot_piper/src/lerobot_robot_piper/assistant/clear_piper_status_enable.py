#!/usr/bin/env python3
"""Clear Piper motion/protection state, select a mode, and enable motors.

This script does not send JointCtrl or EndPoseCtrl. It is meant to recover a
clean controller state before running a separate motion test.

Disabling is never automatic. Type uppercase D at the final prompt to disable.
"""

from __future__ import annotations

import argparse
import sys
import time

from piper_sdk import C_PiperInterface_V2


MOVE_MODES = {
    "p": 0x00,
    "j": 0x01,
    "l": 0x02,
}


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


def repeat_motion_ctrl_1(
    piper: C_PiperInterface_V2,
    emergency_stop: int,
    track_ctrl: int,
    teach_ctrl: int,
    count: int,
    interval_s: float,
) -> None:
    for _ in range(count):
        piper.MotionCtrl_1(emergency_stop, track_ctrl, teach_ctrl)
        time.sleep(interval_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0")
    parser.add_argument("--mode", choices=sorted(MOVE_MODES), default="p")
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument(
        "--installation-pos",
        type=lambda value: int(value, 0),
        default=0x01,
        choices=(0x01, 0x02, 0x03),
    )
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--interval", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.speed <= 100:
        raise ValueError("--speed must be in [0, 100]")
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    move_mode = MOVE_MODES[args.mode]
    piper = C_PiperInterface_V2(
        args.can,
        judge_flag=False,
        can_auto_init=False,
        dh_is_offset=1,
        start_sdk_fk_cal=True,
    )
    piper.ConnectPort()
    time.sleep(1)

    print_status(piper, "before")
    print("\nTerminating current motion/teaching execution...")
    repeat_motion_ctrl_1(piper, 0x02, 0x06, 0x06, args.count, args.interval)

    print("Clearing all trajectories and exiting teach mode...")
    repeat_motion_ctrl_1(piper, 0x02, 0x04, 0x02, args.count, args.interval)

    print(f"Selecting CAN_CTRL + MOVE_{args.mode.upper()}...")
    for _ in range(args.count):
        piper.MotionCtrl_1(0x02, 0x00, 0x02)
        piper.MotionCtrl_2(0x01, move_mode, args.speed, 0x00, 0, args.installation_pos)
        time.sleep(args.interval)

    print("Enabling all arm motors...")
    last_enable = []
    for i in range(args.count):
        piper.EnableArm(7, 0x02)
        time.sleep(args.interval)
        last_enable = list(piper.GetArmEnableStatus())
        if i % 10 == 0:
            print(f"enable status: {last_enable}")
        if last_enable and all(last_enable):
            break

    print_status(piper, "after")
    try:
        prompt_before_disable(piper)
    except KeyboardInterrupt:
        print("\nInterrupted. Motors were not disabled by this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
