#!/usr/bin/env python3
"""Read and diagnose a Linux joystick device."""

from __future__ import annotations

import argparse
import os
import select
import struct
import time


EVENT_FORMAT = "IhBB"
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
EVENT_BUTTON = 0x01
EVENT_AXIS = 0x02
EVENT_INIT = 0x80


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/input/js0")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--deadzone", type=float, default=0.18)
    return parser


def normalized(value: int) -> float:
    return max(-1.0, min(1.0, value / 32767.0))


def main() -> int:
    args = build_parser().parse_args()
    if args.duration < 0:
        raise SystemExit("--duration must be non-negative")
    if not 0.0 <= args.deadzone < 1.0:
        raise SystemExit("--deadzone must be in [0, 1)")

    try:
        fd = os.open(args.device, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as exc:
        raise SystemExit(f"无法打开 {args.device}: {exc}") from exc

    axes: dict[int, int] = {}
    buttons: dict[int, int] = {}
    started = time.monotonic()
    last_report = 0.0

    print(f"设备: {args.device}")
    print("请先保持摇杆完全松开，然后分别移动摇杆并按下按键。")
    print("按 Ctrl-C 退出。")
    if args.duration > 0:
        print(f"测试时长: {args.duration:.1f} 秒")
    print()

    try:
        while True:
            if args.duration > 0 and time.monotonic() - started >= args.duration:
                break

            ready, _, _ = select.select([fd], [], [], 0.1)
            if ready:
                while True:
                    try:
                        data = os.read(fd, EVENT_SIZE)
                    except BlockingIOError:
                        break
                    if len(data) != EVENT_SIZE:
                        break

                    _, value, event_type, number = struct.unpack(EVENT_FORMAT, data)
                    is_initial = bool(event_type & EVENT_INIT)
                    event_type &= ~EVENT_INIT

                    if event_type == EVENT_AXIS:
                        axes[number] = value
                        marker = " initial" if is_initial else ""
                        print(
                            f"axis {number}: raw={value:6d} "
                            f"normalized={normalized(value): .3f}{marker}",
                            flush=True,
                        )
                    elif event_type == EVENT_BUTTON:
                        buttons[number] = value
                        state = "pressed" if value else "released"
                        marker = " initial" if is_initial else ""
                        print(f"button {number}: {state}{marker}", flush=True)

            now = time.monotonic()
            if axes and now - last_report >= 1.0:
                last_report = now
                axis_state = " ".join(
                    f"{number}={normalized(value): .3f}"
                    for number, value in sorted(axes.items())
                )
                active = [
                    number
                    for number, value in sorted(axes.items())
                    if abs(normalized(value)) >= args.deadzone
                ]
                print(f"当前轴: {axis_state}")
                if active:
                    print(f"超过死区的轴: {active}")

    except KeyboardInterrupt:
        print("\n测试结束。")
    finally:
        os.close(fd)

    print("\n最终轴状态:")
    if not axes:
        print("没有收到轴事件，请检查设备路径或手柄连接。")
    else:
        for number, value in sorted(axes.items()):
            state = normalized(value)
            status = "CENTER" if abs(state) < args.deadzone else "ACTIVE"
            print(f"  axis {number}: raw={value:6d} normalized={state: .3f} {status}")

    print("\nLeRobot 参数提示:")
    print("  左右移动使用实际对应的轴作为 --gamepad-axis-x")
    print("  前后移动使用实际对应的轴作为 --gamepad-axis-y")
    print("  如果静止轴不接近 0，先不要启动机械臂。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
