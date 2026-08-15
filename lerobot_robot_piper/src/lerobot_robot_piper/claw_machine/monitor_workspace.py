#!/usr/bin/env python3
"""Read-only workspace monitor for Piper claw-machine runs.

It prints end-effector pose, height, horizontal radius, and running ranges.
No enable, disable, or motion command is sent.
"""

from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import time
from pathlib import Path

from piper_sdk import C_PiperInterface_V2


def read_pose_mm_deg(piper: C_PiperInterface_V2) -> list[float]:
    ep = piper.GetArmEndPoseMsgs().end_pose
    return [
        ep.X_axis / 1000.0,
        ep.Y_axis / 1000.0,
        ep.Z_axis / 1000.0,
        ep.RX_axis / 1000.0,
        ep.RY_axis / 1000.0,
        ep.RZ_axis / 1000.0,
    ]


def read_joints_deg(piper: C_PiperInterface_V2) -> list[float]:
    js = piper.GetArmJointMsgs().joint_state
    return [
        js.joint_1 / 1000.0,
        js.joint_2 / 1000.0,
        js.joint_3 / 1000.0,
        js.joint_4 / 1000.0,
        js.joint_5 / 1000.0,
        js.joint_6 / 1000.0,
    ]


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


def update_range(current: tuple[float, float] | None, value: float) -> tuple[float, float]:
    if current is None:
        return value, value
    return min(current[0], value), max(current[1], value)


def fmt_range(value: tuple[float, float] | None, unit: str) -> str:
    if value is None:
        return "n/a"
    return f"{value[0]:.3f}..{value[1]:.3f}{unit}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0")
    parser.add_argument("--rate-hz", type=float, default=2.0)
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    parser.add_argument("--grab-z", type=float, help="Grab height in mm.")
    parser.add_argument(
        "--grab-z-window-mm",
        type=float,
        default=5.0,
        help="A sample is counted as grab-height data when abs(Z - grab_z) <= this window.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print every N samples.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive")
    if args.grab_z_window_mm < 0:
        raise ValueError("--grab-z-window-mm must be non-negative")

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
    time.sleep(1.0)

    if not has_real_feedback(piper):
        print("No real Piper feedback yet. Check candump/can0 before monitoring.")
        return 1

    csv_file = None
    writer = None
    if args.csv is not None:
        csv_file = args.csv.open("w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "time_s",
                "x_mm",
                "y_mm",
                "z_mm",
                "radius_xy_mm",
                "rx_deg",
                "ry_deg",
                "rz_deg",
                "j1_deg",
                "j2_deg",
                "j3_deg",
                "j4_deg",
                "j5_deg",
                "j6_deg",
                "arm_status",
                "motion_status",
                "enable",
            ]
        )

    print("Read-only Piper workspace monitor. Press Ctrl-C to stop.")
    print("No enable/disable/motion commands will be sent.")
    if args.grab_z is not None:
        print(f"Grab-height radius is tracked when Z is within +/-{args.grab_z_window_mm:.3f} mm of {args.grab_z:.3f} mm.")

    start_t = time.time()
    sample_count = 0
    z_range: tuple[float, float] | None = None
    radius_range: tuple[float, float] | None = None
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None
    grab_radius_range: tuple[float, float] | None = None
    grab_sample_count = 0

    try:
        while not stop:
            now = time.time()
            pose = read_pose_mm_deg(piper)
            joints = read_joints_deg(piper)
            x, y, z, rx, ry, rz = pose
            radius = math.hypot(x, y)
            status = piper.GetArmStatus().arm_status
            arm_status = int(status.arm_status)
            motion_status = int(status.motion_status)
            enable = list(piper.GetArmEnableStatus())

            z_range = update_range(z_range, z)
            radius_range = update_range(radius_range, radius)
            x_range = update_range(x_range, x)
            y_range = update_range(y_range, y)

            near_grab = (
                args.grab_z is not None
                and abs(z - args.grab_z) <= args.grab_z_window_mm
            )
            if near_grab:
                grab_radius_range = update_range(grab_radius_range, radius)
                grab_sample_count += 1

            if writer is not None:
                writer.writerow(
                    [
                        now - start_t,
                        x,
                        y,
                        z,
                        radius,
                        rx,
                        ry,
                        rz,
                        *joints,
                        arm_status,
                        motion_status,
                        enable,
                    ]
                )
                csv_file.flush()

            if sample_count % args.print_every == 0:
                print(
                    f"t={now - start_t:8.2f}s "
                    f"X={x:8.3f} Y={y:8.3f} Z={z:8.3f}mm "
                    f"Rxy={radius:8.3f}mm "
                    f"RPY=({rx:8.3f},{ry:8.3f},{rz:8.3f})deg "
                    f"arm=0x{arm_status:x} motion=0x{motion_status:x} "
                    f"enable={enable}"
                )
                print(
                    "  ranges: "
                    f"X={fmt_range(x_range, 'mm')} "
                    f"Y={fmt_range(y_range, 'mm')} "
                    f"Z={fmt_range(z_range, 'mm')} "
                    f"Rxy={fmt_range(radius_range, 'mm')} "
                    f"grab_Rxy={fmt_range(grab_radius_range, 'mm')} "
                    f"grab_samples={grab_sample_count}"
                )
                sys.stdout.flush()

            sample_count += 1
            time.sleep(1.0 / args.rate_hz)
    finally:
        if csv_file is not None:
            csv_file.close()

    print("\nStopped. Motors were not disabled.")
    print(
        "Final ranges: "
        f"X={fmt_range(x_range, 'mm')} "
        f"Y={fmt_range(y_range, 'mm')} "
        f"Z={fmt_range(z_range, 'mm')} "
        f"Rxy={fmt_range(radius_range, 'mm')} "
        f"grab_Rxy={fmt_range(grab_radius_range, 'mm')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
