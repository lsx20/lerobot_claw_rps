#!/usr/bin/env python3
"""Restore Piper to normal CAN feedback/control mode.

Run this when candump shows 0x3A0~0x3A7 instead of the normal 0x2A1~0x2A7
feedback frames. The script sends the SDK master/slave output-arm command and
then checks whether normal feedback appears through piper_sdk.
"""

from __future__ import annotations

import argparse
import time

from piper_sdk import C_PiperInterface_V2


def print_status(piper: C_PiperInterface_V2) -> None:
    print("ArmStatus:")
    print(piper.GetArmStatus())
    print("JointMsgs:")
    print(piper.GetArmJointMsgs())
    print("EndPose:")
    print(piper.GetArmEndPoseMsgs())
    print("EnableStatus:", piper.GetArmEnableStatus())


def has_sdk_feedback(piper: C_PiperInterface_V2) -> bool:
    status = piper.GetArmStatus()
    joints = piper.GetArmJointMsgs()
    pose = piper.GetArmEndPoseMsgs()
    joint_state = joints.joint_state
    end_pose = pose.end_pose
    return (
        status.Hz > 0
        or joints.Hz > 0
        or pose.Hz > 0
        or any(
            value != 0
            for value in (
                joint_state.joint_1,
                joint_state.joint_2,
                joint_state.joint_3,
                joint_state.joint_4,
                joint_state.joint_5,
                joint_state.joint_6,
            )
        )
        or any(
            value != 0
            for value in (
                end_pose.X_axis,
                end_pose.Y_axis,
                end_pose.Z_axis,
                end_pose.RX_axis,
                end_pose.RY_axis,
                end_pose.RZ_axis,
            )
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--wait", type=float, default=3.0)
    args = parser.parse_args()

    piper = C_PiperInterface_V2(args.can, judge_flag=False, can_auto_init=False)
    piper.ConnectPort()
    time.sleep(0.5)

    print("Before restore:")
    print_status(piper)

    print()
    print("Sending MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)...")
    for _ in range(args.repeats):
        piper.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
        time.sleep(args.interval)

    deadline = time.time() + args.wait
    while time.time() < deadline:
        if has_sdk_feedback(piper):
            break
        time.sleep(0.1)

    print()
    print("After restore attempt:")
    print_status(piper)

    print()
    if has_sdk_feedback(piper):
        print("SDK sees normal Piper feedback. Now run test_piper_cartesian.py.")
    else:
        print("SDK still does not see normal 0x2A feedback.")
        print("Power-cycle Piper, then run:")
        print("  candump can0 | grep -E \"2A1|2A2|2A3|2A4|2A5|2A6|2A7\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
