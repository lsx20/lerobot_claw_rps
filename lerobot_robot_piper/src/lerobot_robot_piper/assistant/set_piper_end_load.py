#!/usr/bin/env python3
"""Set Piper end-effector load parameter.

This follows the official SDK demo:
  ArmParamEnquiryAndConfig(0, 0, 0, 0xAE, load)

load:
  0 = no load
  1 = half load
  2 = full load
"""

from __future__ import annotations

import argparse
import sys
import time

from piper_sdk import C_PiperInterface_V2


LOAD_NAMES = {
    0: "no load",
    1: "half load",
    2: "full load",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0")
    parser.add_argument("--load", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Set Piper end load to {args.load}: {LOAD_NAMES[args.load]}")
    print("This sends only the official load parameter command; it does not move the arm.")
    if not args.yes:
        answer = input("Type YES to continue: ").strip()
        if answer != "YES":
            print("Aborted.")
            return 1

    piper = C_PiperInterface_V2(args.can, judge_flag=False, can_auto_init=False)
    piper.ConnectPort()
    time.sleep(0.5)

    for _ in range(args.repeat):
        piper.ArmParamEnquiryAndConfig(0, 0, 0, 0xAE, args.load)
        time.sleep(args.interval)

    print("End load command sent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
