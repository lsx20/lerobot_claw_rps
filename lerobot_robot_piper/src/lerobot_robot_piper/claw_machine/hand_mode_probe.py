#!/usr/bin/env python3
"""Standalone RH56F2 hand mode test; it does not connect to Piper/CAN."""

from __future__ import annotations

import argparse
import time

try:
    from ..rh56f2_hand import DEFAULT_OPEN, HAND_NAMES, RH56F2Hand, RH56F2HandConfig
except ImportError:
    from lerobot_robot_piper.rh56f2_hand import (
        DEFAULT_OPEN,
        HAND_NAMES,
        RH56F2Hand,
        RH56F2HandConfig,
    )

TEST_CLOSE = {
    "little": 1400,
    "ring": 1400,
    "middle": 1450,
    "index": 1450,
    "thumb_bend": 1400,
    "thumb_swing": 1050,
}


def values(hand: RH56F2Hand, key: str) -> str:
    data = hand.read_positions(key)
    return " ".join(f"{name}={data[name]:.0f}" for name in HAND_NAMES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--mode", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--speed", type=int, default=300)
    parser.add_argument("--force", type=int, default=300)
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    hand = RH56F2Hand(
        RH56F2HandConfig(
            port=args.hand_port,
            hand_id=args.hand_id,
            speed=args.speed,
            force=args.force,
            mode=0,
        )
    )
    hand.connect()
    try:
        print(f"RH56F2 standalone mode test: mode={args.mode}")
        for key, values_by_name in (
            ("mode", {name: args.mode for name in HAND_NAMES}),
            ("speedSet", {name: args.speed for name in HAND_NAMES}),
            ("forceSet", {name: args.force for name in HAND_NAMES}),
        ):
            accepted = hand.write_positions(key, values_by_name)
            print(f"write {key}: accepted={accepted} ack={hand.last_write_ack}")
        print("mode:  " + values(hand, "mode"))
        accepted = hand.set_angles(DEFAULT_OPEN)
        print(f"write angle(open): accepted={accepted} ack={hand.last_write_ack}")
        time.sleep(1.0)
        print("open:  " + values(hand, "angleAct"))
        accepted = hand.set_angles(TEST_CLOSE)
        print(f"write angle(close): accepted={accepted} ack={hand.last_write_ack}")

        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            print(
                "state: "
                + values(hand, "angleAct")
                + " | force: "
                + values(hand, "forceAct")
            )
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("restoring hand mode 0 and opening hand")
        try:
            hand.write_positions("mode", {name: 0 for name in HAND_NAMES})
            hand.set_angles(DEFAULT_OPEN)
        finally:
            hand.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
