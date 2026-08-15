#!/usr/bin/env python3
"""Disable Piper arm motors without moving the robot."""

from __future__ import annotations

import argparse
import time

from lerobot_robot_piper.piper_follower import load_piper_interface_v2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    if not args.yes:
        answer = input("Type DISABLE to disable Piper arm motors: ").strip()
        if answer != "DISABLE":
            print("Aborted before sending DisableArm.")
            return 1

    C_PiperInterface_V2 = load_piper_interface_v2()
    piper = C_PiperInterface_V2(
        args.can,
        judge_flag=False,
        can_auto_init=False,
        dh_is_offset=1,
        start_sdk_fk_cal=True,
    )
    piper.ConnectPort()
    time.sleep(0.2)
    try:
        for _ in range(max(1, args.repeat)):
            piper.DisableArm(7)
            time.sleep(max(0.0, args.interval))
        print(f"Piper enable status after DisableArm: {list(piper.GetArmEnableStatus())}")
        return 0
    finally:
        piper.DisconnectPort()


if __name__ == "__main__":
    raise SystemExit(main())
