#!/usr/bin/env python3
"""Move Piper to claw-machine start, then test a fixed hover target via LeRobot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from lerobot_robot_piper.claw_machine.lerobot_claw import (  # noqa: E402
    ClawMachineController,
    ClawMachineTaskConfig,
    DEFAULT_START_JOINTS,
    DEFAULT_START_POSE,
    fmt_joints,
    pose_from_values,
)
from lerobot_robot_piper.config_piper_rh56f2_follower import (  # noqa: E402
    PiperRH56F2FollowerConfig,
)
from lerobot_robot_piper.piper_rh56f2_follower import PiperRH56F2Follower  # noqa: E402


FIXED_HOVER_XYZ_M = (0.30455, 0.02575, 0.25000)


def parse_xyz(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected X,Y,Z in metres")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("X,Y,Z must be numbers") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--hand-force", type=int, default=1500)
    parser.add_argument(
        "--xyz",
        type=parse_xyz,
        default=list(FIXED_HOVER_XYZ_M),
        help="fixed hover X,Y,Z in metres; no camera is used",
    )
    parser.add_argument("--speed", type=int, default=8)
    parser.add_argument("--rate-hz", type=float, default=40.0)
    parser.add_argument("--start-duration", type=float, default=20.0)
    parser.add_argument("--hover-duration", type=float, default=10.0)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--disconnect-prompt",
        action="store_true",
        help="use PiperRH56F2Follower.disconnect(), which asks whether to disable motors",
    )
    return parser


def make_controller(args: argparse.Namespace) -> tuple[PiperRH56F2Follower, ClawMachineController]:
    robot = PiperRH56F2Follower(
        PiperRH56F2FollowerConfig(
            can_port=args.can,
            speed_rate=args.speed,
            hand_port=args.hand_port,
            hand_id=args.hand_id,
            hand_speed=args.hand_speed,
            hand_force=args.hand_force,
            max_ee_delta_mm=None,
            max_ee_delta_deg=None,
            max_hand_delta=None,
        )
    )
    controller = ClawMachineController(
        robot,
        ClawMachineTaskConfig(
            grab_z=0.0,
            start_pose=pose_from_values(list(DEFAULT_START_POSE)),
            start_joints=list(DEFAULT_START_JOINTS),
            speed_rate=args.speed,
            rate_hz=args.rate_hz,
            start_duration_s=args.start_duration,
            hover_duration_s=args.hover_duration,
        ),
    )
    return robot, controller


def target_pose_from_fraction(
    current_pose: dict[str, float],
    xyz_m: list[float],
    fraction: float,
) -> dict[str, float]:
    target_pose = dict(current_pose)
    requested_mm = {
        "ee.x": xyz_m[0] * 1000.0,
        "ee.y": xyz_m[1] * 1000.0,
        "ee.z": xyz_m[2] * 1000.0,
    }
    for name, requested in requested_mm.items():
        target_pose[name] = current_pose[name] + (requested - current_pose[name]) * fraction
    return target_pose


def print_raw_status(robot: PiperRH56F2Follower, label: str) -> None:
    if robot.piper is None:
        return
    status = robot.piper.GetArmStatus().arm_status
    enable = list(robot.piper.GetArmEnableStatus())
    print(
        f"{label}: ctrl=0x{int(status.ctrl_mode):x} "
        f"mode=0x{int(status.mode_feed):x} "
        f"arm=0x{int(status.arm_status):x} "
        f"enable={enable}"
    )


def disconnect_without_disable_prompt(robot: PiperRH56F2Follower) -> None:
    if robot.piper is not None:
        robot.piper.DisconnectPort()
    if robot.hand is not None:
        robot.hand.disconnect()
    for camera in robot.cameras.values():
        camera.disconnect()
    robot._is_connected = False


def main() -> int:
    args = build_parser().parse_args()
    if not 0 <= args.speed <= 100:
        raise SystemExit("--speed must be between 0 and 100")
    if args.rate_hz <= 0 or args.start_duration <= 0 or args.hover_duration <= 0:
        raise SystemExit("--rate-hz and durations must be positive")
    if not 0.0 < args.fraction <= 1.0:
        raise SystemExit("--fraction must be in (0, 1]")

    print("SAFETY: this uses the same LeRobot follower path as lerobot_claw.py.")
    print("No camera, descent, gripper, or grasp command is issued by this script.")
    print(f"Start joints: {fmt_joints(list(DEFAULT_START_JOINTS))}")
    print(f"Requested fixed hover XYZ(m): {tuple(args.xyz)}")
    print(f"Testing fraction: {args.fraction:.3f}")
    if not args.yes:
        if input("Type START_HOVER to connect and move: ").strip() != "START_HOVER":
            print("Aborted before connecting.")
            return 0

    robot, controller = make_controller(args)
    try:
        robot.connect()
        print_raw_status(robot, "after connect")

        if not controller.move_joints_for(
            list(DEFAULT_START_JOINTS),
            args.speed,
            args.start_duration,
            "start MOVE_J",
        ):
            raise RuntimeError("start MOVE_J failed")
        print_raw_status(robot, "after start MOVE_J")

        current_pose = controller.current_pose()
        target_pose = target_pose_from_fraction(current_pose, args.xyz, args.fraction)
        print(f"current pose: {controller.format_pose(current_pose)}")
        print(f"target pose:  {controller.format_pose(target_pose)}")
        print("Calling controller.move_ee_for(); SDK calls go through PiperRH56F2Follower._send_ee_mm_deg().")

        if not controller.move_ee_for(target_pose, args.hover_duration, "hover MOVE_P"):
            raise RuntimeError("hover MOVE_P failed")
        print_raw_status(robot, "after hover MOVE_P")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Motors were not disabled by this script.")
        return 130
    except Exception as exc:
        print(f"\n[warn] {exc}")
        try:
            print_raw_status(robot, "failure status")
        except Exception:
            pass
        return 1
    finally:
        if robot.is_connected:
            if args.disconnect_prompt:
                robot.disconnect()
            else:
                disconnect_without_disable_prompt(robot)
                print("Disconnected without sending DisableArm.")


if __name__ == "__main__":
    raise SystemExit(main())
