#!/usr/bin/env python3
"""Read-only, phase-labeled Piper grasp monitor."""

from __future__ import annotations

import argparse
import csv
import math
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path

from piper_sdk import C_PiperInterface_V2


PHASE_KEYS = {
    "i": "idle",
    "s": "start",
    "h": "hover",
    "d": "descend",
    "c": "close",
    "l": "lift",
    "t": "transfer",
    "o": "open_drop",
    "r": "return",
    "g": "gesture",
}


class RawTerminal:
    def __enter__(self) -> "RawTerminal":
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def read_key() -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None
    return sys.stdin.read(1)


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
    return bool(
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


def status_values(piper: C_PiperInterface_V2) -> tuple[int, int, int, str]:
    status = piper.GetArmStatus().arm_status
    enable = list(piper.GetArmEnableStatus())
    return (
        int(getattr(status, "arm_status", -1)),
        int(getattr(status, "motion_status", -1)),
        int(getattr(status, "ctrl_mode", -1)),
        ",".join(str(value) for value in enable),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--csv", type=Path, default=Path("grasp_monitor.csv"))
    parser.add_argument("--rate-hz", type=float, default=40.0)
    parser.add_argument("--print-every", type=int, default=4)
    return parser.parse_args()


def print_help() -> None:
    print("阶段标记：")
    print("  i idle       s start       h hover       d descend")
    print("  c close      l lift        t transfer    o open/drop")
    print("  r return     g gesture     q quit")
    print("按键只标记当前阶段，不会向 Piper 发送任何控制命令。")


def main() -> int:
    args = parse_args()
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive")

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
        print("没有收到真实 Piper 反馈，请检查 can0 和 Piper 电源。")
        return 1

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    phase = "idle"
    pending_event = ""
    start_time = time.monotonic()
    sample_count = 0

    print("只读抓取过程监测器已启动。")
    print(f"CAN: {args.can}")
    print(f"CSV: {args.csv}")
    print("不会使能、失能或移动机械臂。")
    print_help()

    with args.csv.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "time_s",
                "sample",
                "phase",
                "event",
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
                "ctrl_mode",
                "enable",
            ]
        )
        csv_file.flush()

        try:
            with RawTerminal():
                while not stop:
                    key = read_key()
                    if key is not None:
                        key = key.lower()
                        if key == "q":
                            stop = True
                            continue
                        if key in PHASE_KEYS:
                            phase = PHASE_KEYS[key]
                            pending_event = f"phase:{phase}"
                            print(f"\n阶段切换: {phase}")
                        else:
                            pending_event = f"key:{key!r}"
                            print(f"\n未映射按键: {key!r}")

                    now = time.monotonic()
                    pose = read_pose_mm_deg(piper)
                    joints = read_joints_deg(piper)
                    x, y, z, rx, ry, rz = pose
                    arm_status, motion_status, ctrl_mode, enable = status_values(piper)
                    elapsed = now - start_time
                    radius = math.hypot(x, y)

                    writer.writerow(
                        [
                            f"{elapsed:.6f}",
                            sample_count,
                            phase,
                            pending_event,
                            f"{x:.6f}",
                            f"{y:.6f}",
                            f"{z:.6f}",
                            f"{radius:.6f}",
                            f"{rx:.6f}",
                            f"{ry:.6f}",
                            f"{rz:.6f}",
                            *[f"{joint:.6f}" for joint in joints],
                            arm_status,
                            motion_status,
                            ctrl_mode,
                            enable,
                        ]
                    )
                    csv_file.flush()

                    if sample_count % args.print_every == 0:
                        print(
                            f"\r{elapsed:8.2f}s [{phase:10s}] "
                            f"XYZ=({x:8.2f},{y:8.2f},{z:8.2f}) "
                            f"J=({joints[0]:7.2f},{joints[1]:7.2f},{joints[2]:7.2f},"
                            f"{joints[3]:7.2f},{joints[4]:7.2f},{joints[5]:7.2f}) "
                            f"mode={ctrl_mode} enable={enable}",
                            end="",
                            flush=True,
                        )
                    pending_event = ""
                    sample_count += 1
                    time.sleep(1.0 / args.rate_hz)
        finally:
            print()

    print(f"监测结束，共保存 {sample_count} 条记录到: {args.csv}")
    print("Piper 电机状态未被修改。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
