#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
from pathlib import Path

from lerobot_robot_piper import PiperRH56F2Follower, PiperRH56F2FollowerConfig
from lerobot_robot_piper.rh56f2_hand import DEFAULT_CLOSED, DEFAULT_OPEN, HAND_NAMES, RH56F2Hand, RH56F2HandConfig


ARM_KEYS = [f"joint_{i}.pos" for i in range(1, 7)]
DEFAULT_WAYPOINTS = Path.home() / "piper_ball_waypoints.json"


def load_waypoints(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_waypoints(path: Path, waypoints: dict) -> None:
    path.write_text(json.dumps(waypoints, ensure_ascii=False, indent=2), encoding="utf-8")


def read_arm_once(can_port: str) -> dict[str, float]:
    from piper_sdk import C_PiperInterface_V2

    piper = C_PiperInterface_V2(can_port)
    piper.ConnectPort()
    time.sleep(0.2)
    js = piper.GetArmJointMsgs().joint_state
    return {
        "joint_1.pos": js.joint_1 / 1000.0,
        "joint_2.pos": js.joint_2 / 1000.0,
        "joint_3.pos": js.joint_3 / 1000.0,
        "joint_4.pos": js.joint_4 / 1000.0,
        "joint_5.pos": js.joint_5 / 1000.0,
        "joint_6.pos": js.joint_6 / 1000.0,
    }


def capture_waypoint(args) -> None:
    waypoints = load_waypoints(args.waypoints)
    arm = read_arm_once(args.can_port)
    print(f"Captured {args.capture}: {arm}")
    waypoints[args.capture] = arm
    save_waypoints(args.waypoints, waypoints)
    print(f"Saved to {args.waypoints}")


def sync_current_joint6(args) -> None:
    waypoints = load_waypoints(args.waypoints)
    if not waypoints:
        raise FileNotFoundError(f"No waypoints found in {args.waypoints}")

    current = read_arm_once(args.can_port)
    joint6 = float(current["joint_6.pos"])
    print(f"Current joint_6.pos: {joint6:.3f} deg")

    changed = []
    for name, pose in waypoints.items():
        if isinstance(pose, dict) and "joint_6.pos" in pose:
            old = float(pose["joint_6.pos"])
            pose["joint_6.pos"] = joint6
            changed.append((name, old, joint6))

    for name, old, new in changed:
        print(f"  {name}: joint_6.pos {old:.3f} -> {new:.3f}")
    save_waypoints(args.waypoints, waypoints)
    print(f"Saved to {args.waypoints}")


def lerp(a: float, b: float, t: float) -> float:
    t = t * t * (3.0 - 2.0 * t)
    return a + (b - a) * t


def move_arm(
    robot: PiperRH56F2Follower,
    target: dict[str, float],
    seconds: float,
    hz: float = 50.0,
    locked_joint6: float | None = None,
) -> None:
    obs = robot.get_observation()
    start = {key: float(obs[key]) for key in ARM_KEYS}
    steps = max(1, int(seconds * hz))
    for i in range(steps):
        t = (i + 1) / steps
        action = {key: lerp(start[key], float(target[key]), t) for key in ARM_KEYS}
        if locked_joint6 is not None:
            action["joint_6.pos"] = locked_joint6
        robot.send_action(action)
        time.sleep(1.0 / hz)


def set_hand(robot: PiperRH56F2Follower, pose: dict[str, float], repeats: int = 20, dt: float = 0.02) -> None:
    action = {f"hand.{name}.pos": float(pose[name]) for name in HAND_NAMES}
    for _ in range(repeats):
        robot.send_action(action)
        time.sleep(dt)


def print_waypoint_help() -> None:
    print("Required waypoints:")
    print("  home       : safe start/end pose")
    print("  pre_grasp  : above the ball")
    print("  grasp      : hand around the ball")
    print("  pre_place  : above the box")
    print("  place      : inside/over the box for release")
    print()
    print("Capture example:")
    print("  python3 scripts/pick_place_ball.py --capture pre_grasp")


def run_pick_place(args) -> None:
    waypoints = load_waypoints(args.waypoints)
    required = ["home", "pre_grasp", "grasp", "pre_place", "place"]
    missing = [name for name in required if name not in waypoints]
    if missing:
        print(f"Missing waypoints: {missing}")
        print_waypoint_help()
        raise SystemExit(2)

    cfg = PiperRH56F2FollowerConfig(
        id="piper_rh56f2_pick_place",
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

    robot = PiperRH56F2Follower(cfg)
    robot.connect()
    try:
        locked_joint6 = None
        if args.lock_joint6:
            locked_joint6 = float(robot.get_observation()["joint_6.pos"])
            print(f"Locking joint_6 at startup value: {locked_joint6:.3f} deg")
            for pose in waypoints.values():
                pose["joint_6.pos"] = locked_joint6

        print("Opening hand...")
        set_hand(robot, DEFAULT_OPEN)

        print("Moving home -> pre_grasp...")
        move_arm(robot, waypoints["home"], args.move_seconds, locked_joint6=locked_joint6)
        move_arm(robot, waypoints["pre_grasp"], args.move_seconds, locked_joint6=locked_joint6)

        print("Descending to grasp...")
        move_arm(robot, waypoints["grasp"], args.approach_seconds, locked_joint6=locked_joint6)

        print("Closing hand on ball...")
        set_hand(robot, DEFAULT_CLOSED, repeats=60)

        print("Lifting...")
        move_arm(robot, waypoints["pre_grasp"], args.approach_seconds, locked_joint6=locked_joint6)

        print("Moving to box...")
        move_arm(robot, waypoints["pre_place"], args.move_seconds, locked_joint6=locked_joint6)
        move_arm(robot, waypoints["place"], args.approach_seconds, locked_joint6=locked_joint6)

        print("Opening hand to release...")
        set_hand(robot, DEFAULT_OPEN, repeats=60)

        print("Retreating...")
        move_arm(robot, waypoints["pre_place"], args.approach_seconds, locked_joint6=locked_joint6)
        move_arm(robot, waypoints["home"], args.move_seconds, locked_joint6=locked_joint6)
    finally:
        robot.disconnect()


def hand_open_close(args) -> None:
    hand = RH56F2Hand(
        RH56F2HandConfig(
            port=args.hand_port,
            baudrate=args.hand_baudrate,
            hand_id=args.hand_id,
            speed=args.hand_speed,
            force=args.hand_force,
        )
    )
    hand.connect()
    try:
        print("Current:", hand.read_positions("angleAct"))
        confirm = input("Type OPEN or CLOSE, or Enter to quit: ").strip().upper()
        if confirm == "OPEN":
            hand.set_angles(DEFAULT_OPEN)
        elif confirm == "CLOSE":
            hand.set_angles(DEFAULT_CLOSED)
        time.sleep(1.0)
        print("Current:", hand.read_positions("angleAct"))
    finally:
        hand.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-baudrate", type=int, default=115200)
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--hand-force", type=int, default=1500)
    parser.add_argument("--speed-rate", type=int, default=25)
    parser.add_argument("--max-arm-delta-deg", type=float, default=5.0)
    parser.add_argument("--max-hand-delta", type=float, default=120.0)
    parser.add_argument("--waypoints", type=Path, default=DEFAULT_WAYPOINTS)
    parser.add_argument("--capture", choices=["home", "pre_grasp", "grasp", "pre_place", "place"])
    parser.add_argument("--sync-current-joint6", action="store_true")
    parser.add_argument("--hand-test", action="store_true")
    parser.add_argument("--lock-joint6", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-joint6-to-sdk-limits", action="store_true")
    parser.add_argument("--move-seconds", type=float, default=4.0)
    parser.add_argument("--approach-seconds", type=float, default=2.0)
    args = parser.parse_args()

    if args.capture:
        capture_waypoint(args)
    elif args.sync_current_joint6:
        sync_current_joint6(args)
    elif args.hand_test:
        hand_open_close(args)
    else:
        run_pick_place(args)


if __name__ == "__main__":
    main()
