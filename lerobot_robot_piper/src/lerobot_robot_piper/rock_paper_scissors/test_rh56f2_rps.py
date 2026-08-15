#!/usr/bin/env python3
"""Smoke test for RH56F2 rock-paper-scissors poses.

Keep the arm still and make sure the hand has free space before running:

    python3 test_rh56f2_rps.py --port /dev/ttyUSB0 --cycle
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rh56f2_hand import DEFAULT_CLOSED, DEFAULT_OPEN, RH56F2Hand, RH56F2HandConfig


PAPER = dict(DEFAULT_OPEN)
ROCK = dict(DEFAULT_CLOSED)
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

POSES = {
    "rock": ROCK,
    "paper": PAPER,
    "scissors": SCISSORS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--speed", type=int, default=800)
    parser.add_argument("--force", type=int, default=1500)
    parser.add_argument(
        "--gesture",
        choices=sorted(POSES),
        default="paper",
        help="Gesture to command when --cycle is not used.",
    )
    parser.add_argument("--cycle", action="store_true", help="Loop through paper, rock, scissors.")
    parser.add_argument("--delay", type=float, default=2.5)
    parser.add_argument(
        "--stage-delay",
        type=float,
        default=0.12,
        help="Seconds between staged finger/thumb commands inside one gesture.",
    )
    return parser.parse_args()


def safe_set_pose(hand: RH56F2Hand, name: str, delay: float = 0.18) -> None:
    """Stage thumb/finger motion to reduce self-collision risk."""
    pose = POSES[name]
    if name == "paper":
        hand.set_angles({"thumb_swing": PAPER["thumb_swing"], "thumb_bend": PAPER["thumb_bend"]})
        time.sleep(delay)
        hand.set_angles(PAPER)
    elif name == "rock":
        hand.set_angles(
            {
                "little": ROCK["little"],
                "ring": ROCK["ring"],
                "middle": ROCK["middle"],
                "index": ROCK["index"],
            }
        )
        time.sleep(delay)
        hand.set_angles(ROCK)
    elif name == "scissors":
        hand.set_angles(
            {
                "little": SCISSORS["little"],
                "ring": SCISSORS["ring"],
                "middle": SCISSORS["middle"],
                "index": SCISSORS["index"],
            }
        )
        time.sleep(delay)
        hand.set_angles(SCISSORS)
    else:
        hand.set_angles(pose)


def main() -> int:
    args = parse_args()
    hand = RH56F2Hand(
        RH56F2HandConfig(
            port=args.port,
            baudrate=args.baudrate,
            hand_id=args.hand_id,
            speed=args.speed,
            force=args.force,
        )
    )

    try:
        print(f"Connecting RH56F2 on {args.port}, id={args.hand_id}...")
        hand.connect()
        print("Connected. Current angleAct:")
        print(hand.read_positions("angleAct"))

        if args.cycle:
            while True:
                for name in ("paper", "rock", "scissors"):
                    print(f"Command: {name}")
                    safe_set_pose(hand, name, args.stage_delay)
                    time.sleep(args.delay)
        else:
            print(f"Command: {args.gesture}")
            safe_set_pose(hand, args.gesture, args.stage_delay)
            time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            safe_set_pose(hand, "paper", args.stage_delay)
            time.sleep(0.5)
        except Exception:
            pass
        hand.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
