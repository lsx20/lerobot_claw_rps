#!/usr/bin/env python3
"""Probe RH56F2 readable feedback registers.

This script does not command finger angles. It reads actual angle, force,
status, and temperature registers so you can touch each finger and see whether
the hand exposes usable contact feedback.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rh56f2_hand import DEFAULT_OPEN, HAND_NAMES, RH56F2Hand, RH56F2HandConfig


def fmt_values(values: dict[str, float]) -> str:
    return " ".join(f"{name}={values.get(name, 0.0):7.1f}" for name in HAND_NAMES)


def diff_values(values: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {name: values.get(name, 0.0) - baseline.get(name, 0.0) for name in HAND_NAMES}


def fmt_ranked_abs(values: dict[str, float]) -> str:
    ranked = sorted(
        ((name, abs(values.get(name, 0.0))) for name in HAND_NAMES),
        key=lambda item: item[1],
        reverse=True,
    )
    return " ".join(f"{name}={value:.1f}" for name, value in ranked)


def read_optional(hand: RH56F2Hand, key: str) -> dict[str, float] | None:
    try:
        return hand.read_positions(key)
    except Exception as exc:
        print(f"[warn] read {key} failed: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--open-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-settle", type=float, default=2.0)
    args = parser.parse_args()

    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")

    hand = RH56F2Hand(
        RH56F2HandConfig(
            port=args.hand_port,
            baudrate=args.baudrate,
            hand_id=args.hand_id,
        )
    )

    print("RH56F2 feedback probe")
    print("Touch/press one fingertip at a time and watch forceAct delta.")
    print("The hand opens first by default, then the script reads feedback.")
    print(f"port={args.hand_port} id={args.hand_id} rate={args.rate_hz:.1f}Hz")

    hand.connect()
    try:
        if args.open_first:
            print("Opening hand for feedback probe...")
            hand.set_angles(DEFAULT_OPEN)
            time.sleep(args.open_settle)

        time.sleep(0.2)
        baseline_force = read_optional(hand, "forceAct") or {name: 0.0 for name in HAND_NAMES}
        print(f"force baseline: {fmt_values(baseline_force)}")

        deadline = time.time() + args.duration
        interval = 1.0 / args.rate_hz
        while time.time() < deadline:
            angle = read_optional(hand, "angleAct")
            force = read_optional(hand, "forceAct")
            status = read_optional(hand, "statusCode")
            temp = read_optional(hand, "temp")

            print()
            if angle is not None:
                print(f"angleAct:       {fmt_values(angle)}")
            if force is not None:
                print(f"forceAct:       {fmt_values(force)}")
                print(f"forceAct delta: {fmt_values(diff_values(force, baseline_force))}")
                print(f"forceAbs rank:  {fmt_ranked_abs(force)}")
            if status is not None:
                print(f"statusCode:     {fmt_values(status)}")
            if temp is not None:
                print(f"temp:           {fmt_values(temp)}")
            time.sleep(interval)
    finally:
        hand.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
