#!/usr/bin/env python3
"""Open the RH56F2 dexterous hand only; no Piper/CAN commands."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from rh56f2_hand import DEFAULT_OPEN, HAND_NAMES, RH56F2Hand, RH56F2HandConfig  # noqa: E402


def fmt(values: dict[str, float]) -> str:
    return " ".join(f"{name}={values.get(name, 0.0):.0f}" for name in HAND_NAMES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--speed", type=int, default=2500)
    parser.add_argument("--force", type=int, default=1500)
    parser.add_argument("--settle", type=float, default=2.0)
    args = parser.parse_args()

    print("SAFETY: opening RH56F2 only. No Piper/CAN connection or arm motion.")
    print(f"port={args.hand_port} id={args.hand_id} speed={args.speed} force={args.force}")
    hand = RH56F2Hand(
        RH56F2HandConfig(
            port=args.hand_port,
            baudrate=args.baudrate,
            hand_id=args.hand_id,
            speed=args.speed,
            force=args.force,
            mode=0,
        )
    )
    hand.connect()
    try:
        accepted = hand.set_angles(DEFAULT_OPEN)
        print(f"open command accepted={accepted} ack={hand.last_write_ack}")
        time.sleep(args.settle)
        try:
            print("angleAct: " + fmt(hand.read_positions("angleAct")))
        except Exception as exc:
            print(f"[warn] angle read failed: {exc}")
    finally:
        hand.disconnect()
    print("RH56F2 disconnected; commanded open pose remains set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
