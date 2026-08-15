#!/usr/bin/env python3
"""Print Piper joint feedback and simple limit checks periodically.

This is read-only: it does not enable, disable, or command motion.
"""

from __future__ import annotations

import argparse
import math
import signal
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


def joint_values_deg(piper: C_PiperInterface_V2) -> list[float]:
    js = piper.GetArmJointMsgs().joint_state
    return [
        js.joint_1 / 1000.0,
        js.joint_2 / 1000.0,
        js.joint_3 / 1000.0,
        js.joint_4 / 1000.0,
        js.joint_5 / 1000.0,
        js.joint_6 / 1000.0,
    ]


def pose_values(piper: C_PiperInterface_V2) -> list[float]:
    ep = piper.GetArmEndPoseMsgs().end_pose
    return [
        ep.X_axis / 1000.0,
        ep.Y_axis / 1000.0,
        ep.Z_axis / 1000.0,
        ep.RX_axis / 1000.0,
        ep.RY_axis / 1000.0,
        ep.RZ_axis / 1000.0,
    ]


def limit_line(idx: int, value: float, warn_margin_deg: float) -> str:
    lo, hi = JOINT_LIMITS_DEG[idx]
    if value < lo or value > hi:
        state = "OUT"
    elif value - lo <= warn_margin_deg or hi - value <= warn_margin_deg:
        state = "NEAR"
    else:
        state = "OK"
    return f"J{idx}: {value:9.3f} deg  [{lo:7.1f}, {hi:7.1f}]  {state}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0")
    parser.add_argument("--period", type=float, default=5.0)
    parser.add_argument("--warn-margin-deg", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.period <= 0:
        raise ValueError("--period must be positive")
    if args.warn_margin_deg < 0:
        raise ValueError("--warn-margin-deg must be non-negative")

    stop = False

    def handle_sigint(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sigint)

    piper = C_PiperInterface_V2(
        args.can,
        judge_flag=False,
        can_auto_init=False,
        dh_is_offset=1,
        start_sdk_fk_cal=True,
    )
    piper.ConnectPort()
    time.sleep(1)

    print("Read-only Piper joint monitor. Press Ctrl-C to stop.")
    print("No enable/disable/motion commands will be sent.")

    while not stop:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            status = piper.GetArmStatus()
            joints = joint_values_deg(piper)
            pose = pose_values(piper)
            print(f"\n=== {now} ===")
            print(f"enable: {piper.GetArmEnableStatus()}")
            print(status)
            for idx, value in enumerate(joints, start=1):
                print(limit_line(idx, value, args.warn_margin_deg))
            if any(math.isnan(v) for v in pose):
                print("pose: unavailable")
            else:
                print(
                    "pose xyz/rpy:"
                    f" X={pose[0]:8.3f} Y={pose[1]:8.3f} Z={pose[2]:8.3f}"
                    f" RX={pose[3]:8.3f} RY={pose[4]:8.3f} RZ={pose[5]:8.3f}"
                )
        except Exception as exc:
            print(f"[warn] read failed: {exc}")
        sys.stdout.flush()
        for _ in range(int(args.period * 10)):
            if stop:
                break
            time.sleep(0.1)

    print("\nStopped. Motors were not disabled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
