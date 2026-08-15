#!/usr/bin/env python3
"""Pre-enable Piper/RH56F2 without starting an RPS or teleop mode."""

from __future__ import annotations

import argparse
import time

from lerobot_robot_piper.config_piper_rh56f2_follower import PiperRH56F2FollowerConfig
from lerobot_robot_piper.piper_rh56f2_follower import PiperRH56F2Follower


def fmt_enable(robot: PiperRH56F2Follower) -> str:
    if robot.piper is None:
        return "[]"
    return str(list(robot.piper.GetArmEnableStatus()))


def disconnect_without_disable(robot: PiperRH56F2Follower) -> None:
    if robot.piper is not None:
        robot.piper.DisconnectPort()
    if robot.hand is not None:
        robot.hand.disconnect()
    for camera in robot.cameras.values():
        camera.disconnect()
    robot._is_connected = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--speed", type=int, default=8)
    parser.add_argument("--settle", type=float, default=0.5)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    if not args.yes:
        answer = input("Type ENABLE to connect and enable Piper/RH56F2: ").strip()
        if answer != "ENABLE":
            print("Aborted before connecting.")
            return 1

    robot = PiperRH56F2Follower(
        PiperRH56F2FollowerConfig(
            can_port=args.can,
            speed_rate=args.speed,
            hand_port=args.hand_port,
            hand_id=args.hand_id,
            max_ee_delta_mm=None,
            max_ee_delta_deg=None,
            max_hand_delta=None,
        )
    )
    try:
        robot.connect()
        time.sleep(args.settle)
        print(f"Piper enabled: {fmt_enable(robot)}")
        print(f"RH56F2 connected on {args.hand_port}, id={args.hand_id}")
        return 0
    finally:
        if robot.is_connected:
            disconnect_without_disable(robot)
            print("Disconnected without sending DisableArm.")


if __name__ == "__main__":
    raise SystemExit(main())
