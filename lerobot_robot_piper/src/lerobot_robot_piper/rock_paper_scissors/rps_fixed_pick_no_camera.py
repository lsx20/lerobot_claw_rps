#!/usr/bin/env python3
"""D435i RPS flow with fixed-coordinate Piper/RH56F2 pick."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from lerobot_robot_piper.claw_machine.lerobot_claw import (  # noqa: E402
    ClawMachineController,
    ClawMachineTaskConfig,
    DEFAULT_START_JOINTS,
    DEFAULT_START_POSE,
    fmt_joints,
    parse_joint_degrees,
    pose_from_values,
)
from lerobot_robot_piper.config_piper_rh56f2_follower import (  # noqa: E402
    PiperRH56F2FollowerConfig,
)
from lerobot_robot_piper.piper_rh56f2_follower import PiperRH56F2Follower  # noqa: E402
from lerobot_robot_piper.rock_paper_scissors.rh56f2_rps_demo import (  # noqa: E402
    D455ColorCamera,
    HandGestureRecognizer,
)
from lerobot_robot_piper.rh56f2_hand import DEFAULT_CLOSED, DEFAULT_OPEN  # noqa: E402


GESTURES = ("rock", "paper", "scissors")
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
DISPLAY = {"rock": "Rock", "paper": "Paper", "scissors": "Scissors"}
CAMERA_GESTURES = {"Rock": "rock", "Paper": "paper", "Scissors": "scissors"}
DEFAULT_D435I_SERIAL = "261722071542"
DEFAULT_GESTURE_MODEL = Path(__file__).with_name("gesture_recognizer.task")
FIXED_GRAB_XYZ_M = (0.30455, 0.02575, 0.25000)
DEFAULT_APPROACH_HEIGHT_M = 0.08
DEFAULT_RPS_JOINTS = [-1.391, 53.150, -56.325, 2.544, -17.309, 101.959]

SCISSORS = dict(DEFAULT_CLOSED)
SCISSORS.update(
    {
        "little": 900,
        "ring": 900,
        "middle": 1720,
        "index": 1720,
        "thumb_bend": 1130,
        "thumb_swing": 1700,
    }
)
RPS_HAND_POSES = {
    "rock": dict(DEFAULT_CLOSED),
    "paper": dict(DEFAULT_OPEN),
    "scissors": SCISSORS,
}


def parse_xyz(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected X,Y,Z in metres")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("X,Y,Z must be numbers") from exc


def choose_system_gesture(
    player: str,
    player_win_probability: float,
    tie_probability: float,
    rng: random.Random,
    force_player_win: bool,
) -> str:
    if force_player_win:
        return BEATS[player]
    roll = rng.random()
    if roll < tie_probability:
        return player
    if roll < tie_probability + player_win_probability:
        return BEATS[player]
    return next(gesture for gesture, beaten in BEATS.items() if beaten == player)


def outcome(player: str, system: str) -> str:
    if player == system:
        return "TIE"
    return "PLAYER_WIN" if BEATS[player] == system else "SYSTEM_WIN"


def pose_from_xyz(current_pose: dict[str, float], xyz_m: list[float]) -> dict[str, float]:
    pose = dict(current_pose)
    pose["ee.x"] = xyz_m[0] * 1000.0
    pose["ee.y"] = xyz_m[1] * 1000.0
    pose["ee.z"] = xyz_m[2] * 1000.0
    return pose


def approach_xyz_from_args(args: argparse.Namespace) -> list[float]:
    if args.hover_xyz is not None:
        return list(args.hover_xyz)
    return [
        args.grab_xyz[0],
        args.grab_xyz[1],
        args.grab_xyz[2] + args.approach_height,
    ]


def set_rps_hand_pose(controller: ClawMachineController, gesture: str, delay_s: float) -> None:
    pose = RPS_HAND_POSES[gesture]
    controller.set_hand_speed(controller.config.hand_speed, f"{gesture} hand speed")
    if gesture == "paper":
        controller.set_hand_pose(
            {
                "thumb_swing": pose["thumb_swing"],
                "thumb_bend": pose["thumb_bend"],
            },
            "paper thumb stage",
        )
        time.sleep(delay_s)
    elif gesture == "rock":
        controller.set_hand_pose(
            {
                "little": pose["little"],
                "ring": pose["ring"],
                "middle": pose["middle"],
                "index": pose["index"],
            },
            "rock finger stage",
        )
        time.sleep(delay_s)
    elif gesture == "scissors":
        controller.set_hand_pose(
            {
                "little": pose["little"],
                "ring": pose["ring"],
                "middle": pose["middle"],
                "index": pose["index"],
            },
            "scissors finger stage",
        )
        time.sleep(delay_s)
    controller.set_hand_pose(pose, f"show {gesture}")


def disconnect_without_disable_prompt(robot: PiperRH56F2Follower) -> None:
    if robot.piper is not None:
        robot.piper.DisconnectPort()
    if robot.hand is not None:
        robot.hand.disconnect()
    for camera in robot.cameras.values():
        camera.disconnect()
    robot._is_connected = False


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
            grab_z=args.grab_xyz[2] * 1000.0,
            start_pose=pose_from_values(list(DEFAULT_START_POSE)),
            start_joints=list(DEFAULT_START_JOINTS),
            lift_z=approach_xyz_from_args(args)[2] * 1000.0,
            speed_rate=args.speed,
            rate_hz=args.rate_hz,
            start_duration_s=args.start_duration,
            hover_duration_s=args.hover_duration,
            vertical_duration_s=args.vertical_duration,
            return_duration_s=args.return_duration,
            hand_speed=args.hand_speed,
            pre_grab_open_speed=args.pre_grab_open_speed,
            hand_settle_s=args.hand_settle,
            pre_grab_open_settle_s=args.pre_grab_open_settle,
            auto_position_tolerance_mm=args.position_tolerance_mm,
            auto_rpy_tolerance_deg=args.rpy_tolerance_deg,
            result_gesture=False,
        ),
    )
    return robot, controller


def run_fixed_pick(
    controller: ClawMachineController,
    args: argparse.Namespace,
    rps_joints: list[float],
) -> bool:
    print("Moving to claw-machine start joints before fixed pick.")
    if not controller.move_joints_for(
        list(DEFAULT_START_JOINTS),
        args.speed,
        args.start_duration,
        "start MOVE_J",
    ):
        return False

    approach_xyz = approach_xyz_from_args(args)
    start_pose = controller.current_pose()
    approach_pose = pose_from_xyz(start_pose, approach_xyz)
    grab_pose = pose_from_xyz(approach_pose, args.grab_xyz)
    lift_pose = pose_from_xyz(approach_pose, approach_xyz)

    print(f"approach pose: {controller.format_pose(approach_pose)}")
    print(f"grab pose:     {controller.format_pose(grab_pose)}")
    print(f"lift pose:     {controller.format_pose(lift_pose)}")

    if not controller.move_ee_for(
        approach_pose,
        args.hover_duration,
        "approach MOVE_P",
        require_reached=True,
    ):
        return False

    controller.open_while_descending()
    if args.pre_grab_open_settle > 0:
        if not controller.wait_with_stop(args.pre_grab_open_settle):
            controller.hold_current_position()
            return False

    if not controller.move_ee_for(
        grab_pose,
        args.vertical_duration,
        "descend MOVE_P",
        require_reached=True,
    ):
        return False

    controller.set_hand_force(args.hand_force, "max grasp force")
    if not controller.close_at_grab_adaptive():
        return False
    if args.hand_settle > 0:
        if not controller.wait_with_stop(args.hand_settle):
            controller.hold_current_position()
            return False

    if not controller.move_ee_for(
        lift_pose,
        args.vertical_duration,
        "lift MOVE_P",
        require_reached=True,
    ):
        return False

    if args.check_held:
        held = controller.held_by_force()
        print(f"held_by_force={held}")

    print("Returning to captured RPS arm joints.")
    if not controller.move_joints_for(
        rps_joints,
        args.speed,
        args.return_duration,
        "return RPS MOVE_J",
    ):
        return False

    if args.ready_hand_after_return:
        set_rps_hand_pose(controller, args.ready_hand_after_return, args.hand_stage_delay)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-speed", type=int, default=2500)
    parser.add_argument("--hand-force", type=int, default=1500)
    parser.add_argument("--speed", type=int, default=8)
    parser.add_argument("--rate-hz", type=float, default=40.0)
    parser.add_argument("--start-duration", type=float, default=20.0)
    parser.add_argument("--hover-duration", type=float, default=8.0)
    parser.add_argument("--vertical-duration", type=float, default=4.0)
    parser.add_argument("--return-duration", type=float, default=8.0)
    parser.add_argument("--gesture-serial", default=DEFAULT_D435I_SERIAL)
    parser.add_argument("--gesture-model", type=Path, default=DEFAULT_GESTURE_MODEL)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stable-frames", type=int, default=5)
    parser.add_argument("--min-detection", type=float, default=0.7)
    parser.add_argument("--min-tracking", type=float, default=0.5)
    parser.add_argument("--min-presence", type=float, default=0.5)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--keyboard", action="store_true", help="use keyboard r/p/s instead of D435i")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--print-vision", action="store_true")
    parser.add_argument("--grab-xyz", type=parse_xyz, default=list(FIXED_GRAB_XYZ_M))
    parser.add_argument(
        "--approach-height",
        type=float,
        default=DEFAULT_APPROACH_HEIGHT_M,
        help="metres above --grab-xyz used for the internal safe approach/lift point",
    )
    parser.add_argument(
        "--hover-xyz",
        type=parse_xyz,
        default=None,
        help="advanced override for approach/lift X,Y,Z in metres; default is grab XYZ plus approach height",
    )
    parser.add_argument("--rps-joints", type=parse_joint_degrees, default=list(DEFAULT_RPS_JOINTS))
    parser.add_argument("--player-win-probability", type=float, default=0.50)
    parser.add_argument("--tie-probability", type=float, default=0.0)
    parser.add_argument("--force-player-win", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hand-stage-delay", type=float, default=0.15)
    parser.add_argument("--pre-grab-open-speed", type=int, default=2500)
    parser.add_argument("--pre-grab-open-settle", type=float, default=1.0)
    parser.add_argument("--hand-settle", type=float, default=1.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--rpy-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--check-held", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ready-hand-after-return",
        choices=("rock", "paper", "scissors"),
        default=None,
        help="optional hand gesture after returning; default keeps the grasp closed",
    )
    parser.add_argument("--one-shot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--disconnect-prompt",
        action="store_true",
        help="use follower disconnect prompt; default disconnects without DisableArm",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.speed <= 100:
        raise SystemExit("--speed must be between 0 and 100")
    for name in ("rate_hz", "start_duration", "hover_duration", "vertical_duration", "return_duration"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("--width, --height, and --fps must be positive")
    if args.stable_frames <= 0:
        raise SystemExit("--stable-frames must be positive")
    if not 0.0 <= args.player_win_probability <= 1.0:
        raise SystemExit("--player-win-probability must be in [0, 1]")
    if not 0.0 <= args.tie_probability < 1.0:
        raise SystemExit("--tie-probability must be in [0, 1)")
    if args.player_win_probability + args.tie_probability > 1.0:
        raise SystemExit("--player-win-probability + --tie-probability must be <= 1")
    if args.approach_height < 0:
        raise SystemExit("--approach-height must be non-negative")


def read_player_gesture() -> str | None:
    value = input("Player gesture [r/p/s/q]: ").strip().lower()
    aliases = {
        "r": "rock",
        "rock": "rock",
        "p": "paper",
        "paper": "paper",
        "s": "scissors",
        "scissors": "scissors",
        "q": None,
        "quit": None,
        "exit": None,
    }
    if value not in aliases:
        print("Ignored. Use r, p, s, or q.")
        return ""
    return aliases[value]


def wait_for_camera_gesture(
    camera: D455ColorCamera,
    recognizer: HandGestureRecognizer,
    args: argparse.Namespace,
) -> str | None:
    stable_gesture = ""
    stable_count = 0
    last_print_t = 0.0
    window = "D435i RPS - q to quit"
    if not args.no_window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        frame = camera.read()
        if frame is None:
            continue
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)
        annotated, gesture_name = recognizer.get_gesture(frame)
        player = CAMERA_GESTURES.get(gesture_name, "")
        if player and player == stable_gesture:
            stable_count += 1
        elif player:
            stable_gesture = player
            stable_count = 1
        else:
            stable_gesture = ""
            stable_count = 0

        if args.print_vision and time.monotonic() - last_print_t >= 1.0:
            print(
                "vision: "
                f"raw={recognizer.last_category} "
                f"score={recognizer.last_score:.2f} "
                f"hands={recognizer.last_hand_count} "
                f"fallback={recognizer.last_fallback} "
                f"mapped={gesture_name} "
                f"stable={stable_gesture or 'none'}:{stable_count}/{args.stable_frames}"
            )
            last_print_t = time.monotonic()

        if not args.no_window:
            cv2.putText(
                annotated,
                f"gesture={gesture_name} stable={stable_count}/{args.stable_frames}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return None

        if stable_gesture and stable_count >= args.stable_frames:
            print(f"Detected player gesture: {DISPLAY[stable_gesture]}")
            return stable_gesture


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

    print("SAFETY: D435i is used only for RPS; D405/ball camera is not used.")
    print("Player win triggers fixed-coordinate pick and return to configured RPS pose.")
    print(f"gesture serial: {args.gesture_serial or 'any RealSense color camera'}")
    print(f"grab XYZ(m):     {tuple(args.grab_xyz)}")
    print(f"approach XYZ(m): {tuple(approach_xyz_from_args(args))}")
    print(f"RPS joints: {fmt_joints(args.rps_joints)}")
    print(f"claw start joints: {fmt_joints(list(DEFAULT_START_JOINTS))}")
    if not args.yes:
        if input("Type RPS_PICK to connect and enable Piper/RH56F2: ").strip() != "RPS_PICK":
            print("Aborted before connecting.")
            return 0

    rng = random.Random(args.seed)
    robot, controller = make_controller(args)
    camera = None
    recognizer = None
    try:
        if not args.keyboard:
            camera = D455ColorCamera(args.gesture_serial, args.width, args.height, args.fps)
            camera.start()
            recognizer = HandGestureRecognizer(
                args.gesture_model,
                args.min_detection,
                args.min_tracking,
                args.min_presence,
                args.min_score,
            )
        robot.connect()
        print_raw_status(robot, "after connect")
        rps_joints = list(args.rps_joints)
        print("Moving to configured RPS arm joints.")
        if not controller.move_joints_for(
            rps_joints,
            args.speed,
            args.return_duration,
            "RPS ready MOVE_J",
        ):
            raise RuntimeError("RPS ready MOVE_J failed")
        set_rps_hand_pose(controller, "paper", args.hand_stage_delay)

        while True:
            if args.keyboard:
                player = read_player_gesture()
            else:
                print("Show rock/paper/scissors to D435i.")
                player = wait_for_camera_gesture(camera, recognizer, args)
            if player is None:
                break
            if player == "":
                continue

            system = choose_system_gesture(
                player,
                args.player_win_probability,
                args.tie_probability,
                rng,
                args.force_player_win,
            )
            set_rps_hand_pose(controller, system, args.hand_stage_delay)
            result = outcome(player, system)
            print(
                f"player={DISPLAY[player]} system={DISPLAY[system]} result={result}"
            )
            if result == "PLAYER_WIN":
                if not args.yes:
                    answer = input("Player won. Press Enter to pick, or type skip: ").strip().lower()
                    if answer == "skip":
                        continue
                ok = run_fixed_pick(controller, args, rps_joints)
                print(f"fixed pick {'complete' if ok else 'failed'}")
                if args.one_shot:
                    return 0 if ok else 1
            else:
                print("Player did not win; fixed pick is not triggered.")
            if args.one_shot:
                return 0
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
        if recognizer is not None:
            recognizer.close()
        if camera is not None:
            camera.stop()
        cv2.destroyAllWindows()
        if robot.is_connected:
            if args.disconnect_prompt:
                robot.disconnect()
            else:
                disconnect_without_disable_prompt(robot)
                print("Disconnected without sending DisableArm.")


if __name__ == "__main__":
    raise SystemExit(main())
