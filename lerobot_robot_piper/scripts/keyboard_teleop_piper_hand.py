#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import select
import subprocess
import sys
import termios
import time
import tty

from lerobot_robot_piper import PiperRH56F2Follower, PiperRH56F2FollowerConfig
from lerobot_robot_piper.rh56f2_hand import DEFAULT_CLOSED, DEFAULT_OPEN, HAND_LIMITS, HAND_NAMES


ARM_KEYS = [f"joint_{i}.pos" for i in range(1, 7)]
SAVED_CLAW_J4_DEG = -42.53
SAVED_CLAW_J6_DEG = -173.38


HELP = """
Keyboard teleop controls

  q              quit
  p              print current arm + hand state
  m              print Piper mode/status
  1..6           select arm joint
  [ / ]          selected joint -/+ step
  - / =          arm step smaller/larger

  a / d          joint_1 -/+
  s / w          joint_2 -/+
  f / r          joint_3 -/+
  g / t          joint_4 -/+
  h / y          joint_5 -/+
  j / u          joint_6 -/+

  o              open hand
  c              close hand
  z / x          hand open/close one small step

Default mode reads single keys without echo.
Use --line-mode for debugging: type one command then press Enter.

On exit, type D then Enter to disable Piper motors.

Claw mode controls (--claw-mode)

  q              quit
  p              print current arm + hand state
  m              print Piper mode/status
  a / d          joint_1 left/right
  w / s          reach forward/back using joint_2 + stronger joint_3 + joint_5
  - / =          arm step smaller/larger
  o / c          open/close hand
  z / x          hand open/close one small step

Claw mode uses saved joint_4 and joint_6 targets, then keeps those targets fixed.
Joint_6 uses the nearest equivalent angle so it does not chase a full 360 degree turn.
During reach, claw mode uses a stronger joint_3 change.
"""


def check_can_port(can_port: str) -> None:
    result = subprocess.run(
        ["ip", "link", "show", can_port],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"CAN port {can_port} was not found.\n"
            "Check the USB-CAN adapter, then run:\n"
            f"  sudo ip link set {can_port} up type can bitrate 1000000\n"
        )
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    if "UP" not in first_line:
        raise SystemExit(
            f"CAN port {can_port} exists but is not UP.\n"
            "Run these commands, then start this script again:\n"
            f"  sudo ip link set {can_port} down\n"
            f"  sudo ip link set {can_port} up type can bitrate 1000000\n"
            f"  ip link show {can_port}\n"
        )


def check_serial_port(port: str) -> None:
    if not os.path.exists(port):
        raise SystemExit(
            f"Hand serial port {port} was not found.\n"
            "Check the USB/RS485 adapter, then run:\n"
            "  ls /dev/ttyUSB*\n"
        )
    if not os.access(port, os.R_OK | os.W_OK):
        raise SystemExit(
            f"Current user cannot read/write {port}.\n"
            "Temporary fix for this plug-in session:\n"
            f"  sudo chmod a+rw {port}\n\n"
            "Permanent fix:\n"
            "  sudo usermod -aG dialout $USER\n"
            "Then log out and log back in, or reboot."
        )


class RawTerminal:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def read_key(timeout: float = 0.1) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    return sys.stdin.read(1)


def print_state(robot: PiperRH56F2Follower) -> None:
    obs = robot.get_observation()
    arm = " ".join(f"J{i}={obs[f'joint_{i}.pos']:.2f}" for i in range(1, 7))
    hand = " ".join(f"{name}={obs[f'hand.{name}.pos']:.0f}" for name in HAND_NAMES)
    print(f"\nARM  {arm}")
    print(f"HAND {hand}")


def print_piper_status(robot: PiperRH56F2Follower) -> None:
    if robot.piper is None:
        print("\nPiper is not connected")
        return
    print()
    print(robot.piper.GetArmStatus())


def nearest_equivalent_angle(target_deg: float, current_deg: float) -> float:
    return target_deg + round((current_deg - target_deg) / 360.0) * 360.0


def resolve_locked_joints(
    locked_joints: dict[str, float] | None,
    obs: dict[str, float],
) -> dict[str, float]:
    if not locked_joints:
        return {}
    action = dict(locked_joints)
    if "joint_6.pos" in action:
        action["joint_6.pos"] = nearest_equivalent_angle(
            float(action["joint_6.pos"]),
            float(obs["joint_6.pos"]),
        )
    return action


def move_joint(
    robot: PiperRH56F2Follower,
    joint_index: int,
    delta_deg: float,
    locked_joints: dict[str, float] | None = None,
) -> None:
    key = f"joint_{joint_index}.pos"
    obs = robot.get_observation()
    before = float(obs[key])
    target = before + delta_deg
    action = resolve_locked_joints(locked_joints, obs)
    action[key] = target
    robot.send_action(action)
    time.sleep(0.15)
    after = float(robot.get_observation()[key])
    print(f"\r{key}: {before:.1f}->{target:.1f}/{after:.1f} deg", end="", flush=True)


def move_multi_joint(
    robot: PiperRH56F2Follower,
    deltas: dict[int, float],
    label: str,
    locked_joints: dict[str, float] | None = None,
) -> None:
    obs = robot.get_observation()
    action = resolve_locked_joints(locked_joints, obs)
    before = {}
    target = {}
    for joint_index, delta_deg in deltas.items():
        key = f"joint_{joint_index}.pos"
        before[key] = float(obs[key])
        target[key] = before[key] + delta_deg
        action[key] = target[key]

    robot.send_action(action)
    time.sleep(0.15)
    after_obs = robot.get_observation()
    parts = []
    for key in target:
        joint_name = key.split(".")[0].replace("joint_", "J")
        parts.append(f"{joint_name}:{before[key]:.1f}->{target[key]:.1f}/{float(after_obs[key]):.1f}")
    print(f"\r{label}  " + "  ".join(parts), end="", flush=True)


def move_claw_reach(
    robot: PiperRH56F2Follower,
    direction: float,
    arm_step: float,
    args: argparse.Namespace,
    locked_joints: dict[str, float] | None,
) -> None:
    if args.invert_claw_reach:
        direction *= -1.0
    move_multi_joint(
        robot,
        {
            2: direction * arm_step * args.claw_j2_gain,
            3: direction * arm_step * args.claw_j3_gain,
            5: direction * arm_step * args.claw_j5_gain,
        },
        "claw reach",
        locked_joints,
    )


def set_hand(robot: PiperRH56F2Follower, pose: dict[str, float]) -> None:
    robot.send_action({f"hand.{name}.pos": float(pose[name]) for name in HAND_NAMES})
    print("\rhand target sent                                      ", end="", flush=True)


def step_hand(robot: PiperRH56F2Follower, delta: float) -> None:
    obs = robot.get_observation()
    action = {}
    for name in HAND_NAMES:
        current = float(obs[f"hand.{name}.pos"])
        if name == "thumb_swing":
            target = current
        else:
            lo, hi = HAND_LIMITS[name]
            target = min(max(current + delta, lo), hi)
        action[f"hand.{name}.pos"] = target
    robot.send_action(action)
    print("\rhand step sent                                        ", end="", flush=True)


def saved_claw_locked_joints(args: argparse.Namespace) -> dict[str, float]:
    locked_joints = {
        "joint_4.pos": float(args.claw_lock_j4_deg),
        "joint_6.pos": float(args.claw_lock_j6_deg),
    }
    print(
        "\nClaw locked saved pose: "
        f"J4={locked_joints['joint_4.pos']:.3f} deg, "
        f"J6={locked_joints['joint_6.pos']:.3f} deg"
    )
    return locked_joints


def handle_key(
    robot: PiperRH56F2Follower,
    key: str,
    selected_joint: int,
    arm_step: float,
    args: argparse.Namespace,
    locked_joints: dict[str, float] | None,
) -> tuple[bool, int, float]:
    if key == "q":
        print("\nquit requested")
        return False, selected_joint, arm_step
    if key == "p":
        print_state(robot)
        return True, selected_joint, arm_step
    if key == "m":
        print_piper_status(robot)
        return True, selected_joint, arm_step
    if key in "123456":
        selected_joint = int(key)
        print(f"\nselected joint_{selected_joint}, step={arm_step:.2f} deg")
        return True, selected_joint, arm_step
    if key == "-":
        arm_step = max(0.1, arm_step / 2.0)
        print(f"\narm step={arm_step:.2f} deg")
        return True, selected_joint, arm_step
    if key == "=":
        arm_step = min(10.0, arm_step * 2.0)
        print(f"\narm step={arm_step:.2f} deg")
        return True, selected_joint, arm_step

    if args.claw_mode:
        if key == "a":
            move_joint(robot, 1, -arm_step, locked_joints)
            time.sleep(args.command_interval)
            return True, selected_joint, arm_step
        if key == "d":
            move_joint(robot, 1, arm_step, locked_joints)
            time.sleep(args.command_interval)
            return True, selected_joint, arm_step
        if key == "w":
            move_claw_reach(robot, 1.0, arm_step, args, locked_joints)
            time.sleep(args.command_interval)
            return True, selected_joint, arm_step
        if key == "s":
            move_claw_reach(robot, -1.0, arm_step, args, locked_joints)
            time.sleep(args.command_interval)
            return True, selected_joint, arm_step

    joint_deltas = {
        "[": (selected_joint, -arm_step),
        "]": (selected_joint, arm_step),
        "a": (1, -arm_step),
        "d": (1, arm_step),
        "s": (2, -arm_step),
        "w": (2, arm_step),
        "f": (3, -arm_step),
        "r": (3, arm_step),
        "g": (4, -arm_step),
        "t": (4, arm_step),
        "h": (5, -arm_step),
        "y": (5, arm_step),
        "j": (6, -arm_step),
        "u": (6, arm_step),
    }
    if key in joint_deltas:
        if args.claw_mode:
            print("\nclaw mode arm keys: use a/d for joint_1, w/s for reach")
            return True, selected_joint, arm_step
        joint_index, delta = joint_deltas[key]
        move_joint(robot, joint_index, delta)
        time.sleep(args.command_interval)
        return True, selected_joint, arm_step

    if key == "o":
        set_hand(robot, DEFAULT_OPEN)
        print()
        return True, selected_joint, arm_step
    if key == "c":
        set_hand(robot, DEFAULT_CLOSED)
        print()
        return True, selected_joint, arm_step
    if key == "z":
        step_hand(robot, args.hand_step)
        print()
        return True, selected_joint, arm_step
    if key == "x":
        step_hand(robot, -args.hand_step)
        print()
        return True, selected_joint, arm_step

    print(f"unknown key: {key!r}")
    return True, selected_joint, arm_step


def run(args: argparse.Namespace) -> None:
    if args.claw_mode:
        args.speed_rate = min(args.speed_rate, args.claw_speed_rate)
        args.max_arm_delta_deg = max(args.max_arm_delta_deg, args.claw_step_deg)

    cfg = PiperRH56F2FollowerConfig(
        id="piper_rh56f2_keyboard_teleop",
        can_port=args.can_port,
        speed_rate=args.speed_rate,
        max_arm_delta_deg=args.max_arm_delta_deg,
        prompt_before_disable=True,
        clip_joint6_to_sdk_limits=args.clip_joint6_to_sdk_limits,
        hand_port=args.hand_port,
        hand_baudrate=args.hand_baudrate,
        hand_id=args.hand_id,
        hand_speed=args.hand_speed,
        hand_force=args.hand_force,
        max_hand_delta=args.max_hand_delta,
    )

    print(HELP)
    print("Safety:")
    print("  1. Clear space around the arm and hand.")
    print("  2. Keep one hand near the power/emergency stop.")
    print("  3. Start with 1 deg arm steps and do not disable motors unless the arm is supported.")
    confirm = input("Type RUN to connect and enable keyboard teleop: ").strip()
    if confirm != "RUN":
        print("Cancelled.")
        return
    check_can_port(args.can_port)
    check_serial_port(args.hand_port)

    robot = PiperRH56F2Follower(cfg)
    robot.connect()
    print("\nPiper enabled.")
    selected_joint = 1
    arm_step = args.claw_step_deg if args.claw_mode else args.arm_step_deg
    locked_joints = None

    try:
        print_state(robot)
        if args.claw_mode:
            locked_joints = saved_claw_locked_joints(args)
            obs = robot.get_observation()
            claw_origin = {key: float(obs[key]) for key in ARM_KEYS}
            print("\nClaw mode active: A/D=joint_1 left/right, W/S=reach forward/back, J4/J6 locked.")
            print(
                "Claw startup reference: "
                + " ".join(f"J{i}={claw_origin[f'joint_{i}.pos']:.1f}" for i in range(1, 7))
            )
        if args.line_mode:
            print("\nLine mode running. Type one command then Enter. Type q to quit.")
            while True:
                key = input("cmd> ").strip()
                if not key:
                    continue
                keep_running, selected_joint, arm_step = handle_key(
                    robot, key[0], selected_joint, arm_step, args, locked_joints
                )
                if not keep_running:
                    break
        else:
            print("\nTeleop running. Press q to quit.")
            with RawTerminal():
                while True:
                    key = read_key()
                    if key is None:
                        continue
                    keep_running, selected_joint, arm_step = handle_key(
                        robot, key, selected_joint, arm_step, args, locked_joints
                    )
                    if not keep_running:
                        break
    finally:
        print()
        robot.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-baudrate", type=int, default=115200)
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--hand-force", type=int, default=1500)
    parser.add_argument("--speed-rate", type=int, default=15)
    parser.add_argument("--max-arm-delta-deg", type=float, default=2.0)
    parser.add_argument("--max-hand-delta", type=float, default=120.0)
    parser.add_argument("--arm-step-deg", type=float, default=1.0)
    parser.add_argument("--hand-step", type=float, default=80.0)
    parser.add_argument("--command-interval", type=float, default=0.03)
    parser.add_argument("--line-mode", action="store_true")
    parser.add_argument("--claw-mode", action="store_true")
    parser.add_argument("--claw-step-deg", type=float, default=4.05)
    parser.add_argument("--claw-speed-rate", type=int, default=8)
    parser.add_argument("--invert-claw-reach", action="store_true")
    parser.add_argument("--claw-j2-gain", type=float, default=1.0)
    parser.add_argument("--claw-j3-gain", type=float, default=-0.75)
    parser.add_argument("--claw-j5-gain", type=float, default=-0.25)
    parser.add_argument("--claw-lock-j4-deg", type=float, default=SAVED_CLAW_J4_DEG)
    parser.add_argument("--claw-lock-j6-deg", type=float, default=SAVED_CLAW_J6_DEG)
    parser.add_argument("--clip-joint6-to-sdk-limits", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
