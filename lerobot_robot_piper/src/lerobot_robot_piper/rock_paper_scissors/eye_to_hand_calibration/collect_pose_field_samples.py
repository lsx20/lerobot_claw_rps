#!/usr/bin/env python3
"""Collect nearest-neighbor pose-field samples: base X,Y -> RX,RY,RZ."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
RPS_DIR = THIS_DIR.parent
PACKAGE_DIR = RPS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(RPS_DIR) not in sys.path:
    sys.path.insert(0, str(RPS_DIR))
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from ball_tactile_classifier.common import BALL_READY_OPEN, BALL_SAFE_CLOSED, FINGER_NAMES
from common import read_piper_pose
from rh56f2_hand import RH56F2Hand, RH56F2HandConfig


FIELDS = [
    "unix_time",
    "base_x_m",
    "base_y_m",
    "base_z_m",
    "base_rx_deg",
    "base_ry_deg",
    "base_rz_deg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--output", type=Path, default=Path("pose_field_samples.csv"))
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-baudrate", type=int, default=115200)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--hand-force", type=int, default=600)
    parser.add_argument("--hand-settle", type=float, default=1.0)
    parser.add_argument(
        "--hand-pose",
        choices=("ball_safe_closed", "ball_ready_open"),
        default="ball_ready_open",
        help="RH56F2 pose to command before sampling. Default matches tactile classification pre-grasp pose.",
    )
    parser.add_argument("--no-hand-pose", action="store_true", help="do not connect RH56F2 or command hand pose")
    return parser.parse_args()


def require_can_up(can_name: str) -> None:
    operstate_path = Path("/sys/class/net") / can_name / "operstate"
    if not operstate_path.exists():
        raise SystemExit(f"CAN interface {can_name!r} does not exist. Check USB-CAN connection.")
    state = operstate_path.read_text().strip()
    if state != "up":
        raise SystemExit(
            f"CAN interface {can_name!r} is {state.upper()}, so Piper pose cannot be read.\n"
            f"Bring it up first:\n"
            f"  sudo ip link set {can_name} down\n"
            f"  sudo ip link set {can_name} up type can bitrate 1000000"
        )


def count_existing_samples(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def selected_hand_pose(args: argparse.Namespace) -> dict[str, float]:
    if args.hand_pose == "ball_safe_closed":
        return BALL_SAFE_CLOSED
    return BALL_READY_OPEN


def format_hand_pose(pose: dict[str, float]) -> str:
    return ", ".join(f"{name}={pose[name]:.0f}" for name in FINGER_NAMES)


def connect_and_command_hand(args: argparse.Namespace) -> RH56F2Hand | None:
    if args.no_hand_pose:
        print("RH56F2 hand pose skipped by --no-hand-pose")
        return None
    hand = RH56F2Hand(
        RH56F2HandConfig(
            port=args.hand_port,
            baudrate=args.hand_baudrate,
            hand_id=args.hand_id,
            speed=args.hand_speed,
            force=args.hand_force,
            mode=0,
        )
    )
    pose = selected_hand_pose(args)
    print(f"Setting RH56F2 hand pose: {args.hand_pose} port={args.hand_port} id={args.hand_id}")
    print(f"pose: {format_hand_pose(pose)}")
    hand.connect()
    accepted = hand.set_angles(pose)
    print(f"hand command accepted={accepted} ack={hand.last_write_ack}")
    if args.hand_settle > 0:
        time.sleep(args.hand_settle)
    return hand


def main() -> int:
    args = parse_args()
    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError as exc:
        raise SystemExit("piper_sdk is required") from exc

    print("SAFETY: pose-field collection is read-only for Piper.")
    print("It does not enable, disable, or move the arm.")
    print("By default it commands RH56F2 to the tactile-classifier pre-grasp pose.")
    print("Move/teach the gripper to a good tabletop pose, then press Enter to save.")
    if input("Type READONLY to continue: ").strip() != "READONLY":
        print("Cancelled.")
        return 0

    require_can_up(args.can)
    piper = C_PiperInterface_V2(args.can, judge_flag=False, can_auto_init=False, dh_is_offset=1, start_sdk_fk_cal=True)
    piper.ConnectPort()
    time.sleep(1.0)
    hand = connect_and_command_hand(args)
    output_exists = args.output.exists() and args.output.stat().st_size > 0
    sample_count = count_existing_samples(args.output)
    try:
        with args.output.open("a", newline="") as sample_file:
            writer = csv.writer(sample_file)
            if not output_exists:
                writer.writerow(FIELDS)
            while True:
                pose = read_piper_pose(piper)
                print(
                    f"current pose: X={pose[0]:.4f} Y={pose[1]:.4f} Z={pose[2]:.4f} "
                    f"RX={pose[3]:.2f} RY={pose[4]:.2f} RZ={pose[5]:.2f}"
                )
                answer = input("Enter=save, p=print again, g=pregrasp pose, q=quit: ").strip().lower()
                if answer == "q":
                    break
                if answer == "p":
                    continue
                if answer == "g":
                    if hand is None:
                        print("Hand is not connected; run without --no-hand-pose to use g:hand pose.")
                    else:
                        accepted = hand.set_angles(selected_hand_pose(args))
                        print(f"hand command accepted={accepted} ack={hand.last_write_ack}")
                    continue
                writer.writerow([f"{time.time():.6f}", *[f"{value:.6f}" for value in pose]])
                sample_file.flush()
                sample_count += 1
                print(f"saved #{sample_count}")
    finally:
        try:
            piper.DisconnectPort()
        except Exception:
            pass
        if hand is not None:
            try:
                hand.disconnect()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
