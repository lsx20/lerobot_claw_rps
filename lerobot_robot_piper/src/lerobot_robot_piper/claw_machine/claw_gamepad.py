#!/usr/bin/env python3
"""Linux joystick gamepad input for the claw-machine workflow."""

from __future__ import annotations

import os
import select
import struct
import time
from argparse import Namespace
from copy import copy

from piper_sdk import C_PiperInterface_V2

from claw_arm_grasp import run_pick_cycle
from claw_arm_keyboard import (
    apply_joint_step,
    capture_keyboard_reference,
    send_movej_once,
)
from claw_init import (
    arm_status_code,
    end_pose_raw,
    fmt_joints,
    joints_deg,
    pose_mm_deg,
    print_state,
    wait_for_movej_ready,
    wait_for_movep_ready,
)


JS_EVENT_FORMAT = "IhBB"
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80


class LinuxJoystick:
    """Small reader for /dev/input/js* devices."""

    def __init__(self, device: str):
        self.device = device
        self.fd: int | None = None
        self.axes: dict[int, float] = {}
        self.button_presses: set[int] = set()

    def __enter__(self) -> "LinuxJoystick":
        self.fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def read_events(self) -> None:
        if self.fd is None:
            raise RuntimeError("joystick is not open")
        while select.select([self.fd], [], [], 0.0)[0]:
            try:
                data = os.read(self.fd, JS_EVENT_SIZE)
            except BlockingIOError:
                return
            if len(data) != JS_EVENT_SIZE:
                return

            _, value, event_type, number = struct.unpack(JS_EVENT_FORMAT, data)
            event_type &= ~JS_EVENT_INIT
            if event_type == JS_EVENT_AXIS:
                self.axes[number] = max(-1.0, min(1.0, value / 32767.0))
            elif event_type == JS_EVENT_BUTTON and value == 1:
                self.button_presses.add(number)

    def axis(self, number: int, deadzone: float) -> float:
        value = self.axes.get(number, 0.0)
        if abs(value) < deadzone:
            return 0.0
        sign = 1.0 if value > 0.0 else -1.0
        return sign * ((abs(value) - deadzone) / (1.0 - deadzone))

    def pop_button(self, number: int) -> bool:
        if number not in self.button_presses:
            return False
        self.button_presses.remove(number)
        return True


def shaped_axis(value: float, curve: float) -> float:
    if value == 0.0:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    return sign * (abs(value) ** curve)


def velocity_args(args: Namespace, dt_s: float, j1_axis: float = 0.0, reach_axis: float = 0.0) -> Namespace:
    local_args = copy(args)
    local_args.j1_step_deg = args.gamepad_j1_speed_dps * dt_s * abs(j1_axis)
    local_args.reach_step_deg = args.gamepad_reach_speed_dps * dt_s * abs(reach_axis)
    return local_args


def clamp_target_lead(target: list[float], actual: list[float], max_lead_deg: float) -> list[float]:
    if max_lead_deg <= 0.0:
        return target
    limited = list(target)
    for idx, actual_value in enumerate(actual):
        limited[idx] = min(
            max(limited[idx], actual_value - max_lead_deg),
            actual_value + max_lead_deg,
        )
    return limited


def restore_locked_wrist(target: list[float], locked_j4: float, locked_j6: float) -> list[float]:
    locked = list(target)
    locked[3] = locked_j4
    locked[5] = locked_j6
    return locked


def print_gamepad_help(args: Namespace) -> None:
    print("Gamepad control is active.")
    print(f"Device: {args.gamepad_device}")
    print("Left stick X: J1 left/right")
    print("Left stick Y: reach forward/back")
    print(
        "Gamepad speed: "
        f"J1 {args.gamepad_j1_speed_dps:.2f} deg/s, "
        f"reach {args.gamepad_reach_speed_dps:.2f} deg/s, "
        f"curve {args.gamepad_axis_curve:.2f}"
    )
    print(
        "Stop behavior: "
        f"reset target on stick release={args.gamepad_stop_reset}, "
        f"target lead limit={args.gamepad_lead_limit_deg:.2f} deg"
    )
    print("A / button 0: pick cycle")
    print("B / button 1: quit")
    print("X / button 2: reset joint reference")
    print("Y / button 3: print current pose")


def run_gamepad_loop(
    piper: C_PiperInterface_V2,
    hand: object | None,
    args: Namespace,
    start_pose: list[int],
    keyboard_pose: list[int],
) -> None:
    print("Switching to MOVE_J for gamepad control...")
    if not wait_for_movej_ready(piper, args.speed, args.feedback_timeout):
        raise RuntimeError("MOVE_J did not become ready.")

    joint_target, locked_j4, locked_j5, locked_j6 = capture_keyboard_reference(piper)
    drop_pose = list(args.drop) if args.drop is not None else list(start_pose)

    print(f"Captured start pose: {pose_mm_deg(start_pose)}")
    print(f"Keyboard/gamepad pose: {pose_mm_deg(keyboard_pose)}")
    print(f"Gamepad start joints: {fmt_joints(joint_target)}")
    print_gamepad_help(args)

    with LinuxJoystick(args.gamepad_device) as joystick:
        last_loop = time.monotonic()
        last_print = 0.0
        was_moving = False
        while True:
            now = time.monotonic()
            dt_s = min(max(now - last_loop, 0.0), 0.1)
            last_loop = now

            joystick.read_events()

            if joystick.pop_button(1):
                print("\nquit")
                break
            if joystick.pop_button(3):
                print_state(piper)
            if joystick.pop_button(2):
                joint_target, locked_j4, locked_j5, locked_j6 = capture_keyboard_reference(piper)
                print(f"\nreference reset joints: {fmt_joints(joint_target)}")

            if joystick.pop_button(0):
                if not wait_for_movep_ready(piper, args.speed, args.feedback_timeout):
                    print("[warn] MOVE_P did not become ready for pick cycle.")
                    print_state(piper)
                    break
                hover_pose = end_pose_raw(piper)
                ok = run_pick_cycle(piper, hand, args, start_pose, hover_pose, drop_pose)
                if not ok and arm_status_code(piper) != 0x00:
                    print("[warn] arm is in error status; reset before retrying.")
                    break
                if not wait_for_movej_ready(piper, args.speed, args.feedback_timeout):
                    print("[warn] MOVE_J did not become ready after pick cycle.")
                    break
                joint_target, _, locked_j5, _ = capture_keyboard_reference(piper)
                joint_target = restore_locked_wrist(joint_target, locked_j4, locked_j6)
                if not send_movej_once(piper, joint_target, args.speed):
                    print("[warn] failed to restore locked J4/J6 after returning to start.")
                    break
                print(
                    f"cycle {'complete' if ok else 'stopped'}; "
                    f"current joints={fmt_joints(joint_target)} "
                    f"locked J4={locked_j4:.3f} J6={locked_j6:.3f}"
                )

            x_axis = shaped_axis(
                joystick.axis(args.gamepad_axis_x, args.gamepad_deadzone),
                args.gamepad_axis_curve,
            )
            y_axis = shaped_axis(
                joystick.axis(args.gamepad_axis_y, args.gamepad_deadzone),
                args.gamepad_axis_curve,
            )

            moved = False
            labels = []
            phases = []
            if x_axis > 0:
                local_args = velocity_args(args, dt_s, j1_axis=x_axis)
                joint_target, phase = apply_joint_step(
                    joint_target, 2, local_args, locked_j4, locked_j5, locked_j6
                )
                labels.append(f"right {abs(x_axis):.2f}")
                phases.append(phase)
                moved = True
            elif x_axis < 0:
                local_args = velocity_args(args, dt_s, j1_axis=x_axis)
                joint_target, phase = apply_joint_step(
                    joint_target, 3, local_args, locked_j4, locked_j5, locked_j6
                )
                labels.append(f"left {abs(x_axis):.2f}")
                phases.append(phase)
                moved = True

            reach_axis = -y_axis if args.gamepad_invert_y else y_axis
            if reach_axis > 0:
                local_args = velocity_args(args, dt_s, reach_axis=reach_axis)
                joint_target, phase = apply_joint_step(
                    joint_target, 0, local_args, locked_j4, locked_j5, locked_j6
                )
                labels.append(f"forward {abs(reach_axis):.2f}")
                phases.append(phase)
                moved = True
            elif reach_axis < 0:
                local_args = velocity_args(args, dt_s, reach_axis=reach_axis)
                joint_target, phase = apply_joint_step(
                    joint_target, 1, local_args, locked_j4, locked_j5, locked_j6
                )
                labels.append(f"back {abs(reach_axis):.2f}")
                phases.append(phase)
                moved = True

            if moved:
                actual_joints = joints_deg(piper)
                joint_target = clamp_target_lead(
                    joint_target,
                    actual_joints,
                    args.gamepad_lead_limit_deg,
                )
                if now - last_print >= args.gamepad_print_interval:
                    print(
                        f"\rgamepad {'+'.join(labels)} [{'/'.join(phases)}]: "
                        f"joints {fmt_joints(joint_target)}",
                        end="",
                        flush=True,
                    )
                    last_print = now
                if not send_movej_once(piper, joint_target, args.speed):
                    print("[warn] gamepad nudge failed; stop sending commands and reset if needed.")
                    break
                was_moving = True
            elif was_moving and args.gamepad_stop_reset:
                joint_target = joints_deg(piper)
                if not send_movej_once(piper, joint_target, args.speed):
                    print("[warn] gamepad stop failed; reset if needed.")
                    break
                print(f"\rgamepad stop: holding current joints {fmt_joints(joint_target)}", end="", flush=True)
                was_moving = False

            select.select([], [], [], 1.0 / args.rate_hz)
