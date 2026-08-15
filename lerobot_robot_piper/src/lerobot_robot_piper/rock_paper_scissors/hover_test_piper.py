#!/usr/bin/env python3
"""Move Piper to a ball-hover pose only, with keyboard pause/quit controls."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


CLAW_MACHINE_DIR = Path(__file__).resolve().parents[1] / "claw_machine"
if str(CLAW_MACHINE_DIR) not in sys.path:
    sys.path.insert(0, str(CLAW_MACHINE_DIR))

from claw_init import (  # noqa: E402
    RawTerminal,
    connect_piper,
    end_pose_raw,
    enable_all,
    pose_mm_deg,
    read_key,
    wait_for_movep_ready,
    wait_for_real_feedback,
)


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
    parser.add_argument(
        "--xyz",
        type=parse_xyz,
        default=[0.278, 0.002, 0.237],
        help="hover target X,Y,Z in metres; default is 12 cm above the observed ball",
    )
    parser.add_argument("--speed", type=int, default=3)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--feedback-timeout", type=float, default=10.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 <= args.speed <= 100:
        raise SystemExit("--speed must be between 0 and 100")
    if args.rate_hz <= 0 or args.duration <= 0:
        raise SystemExit("--rate-hz and --duration must be positive")

    print("SAFETY: hover-only test; no camera, descent, or gripper command.")
    print("The arm will move to the requested XYZ and keep its current orientation.")
    print("SPACE = pause/hold, SPACE again = resume, q = stop program, Ctrl+C = abort.")
    print("The physical emergency-stop button remains the primary emergency control.")
    print(f"Requested hover XYZ (m): {tuple(args.xyz)}")
    if input("Type MOVE to enable Piper and start: ").strip() != "MOVE":
        print("Aborted before connecting/moving.")
        return 0

    piper = connect_piper(args)
    paused = False
    sent_any_command = False
    try:
        wait_for_real_feedback(piper, args.feedback_timeout)
        start = end_pose_raw(piper)
        target = [
            int(round(args.xyz[0] * 1_000_000.0)),
            int(round(args.xyz[1] * 1_000_000.0)),
            int(round(args.xyz[2] * 1_000_000.0)),
            start[3],
            start[4],
            start[5],
        ]
        print(f"Current pose: {pose_mm_deg(start)}")
        print(f"Hover target: {pose_mm_deg(target)}")

        if not enable_all(piper, args.feedback_timeout):
            raise RuntimeError("Piper did not enable; no motion command was sent.")
        if not wait_for_movep_ready(piper, args.speed, args.feedback_timeout):
            raise RuntimeError("MOVE_P mode was not ready; no target command was sent.")

        deadline = time.monotonic() + args.duration
        interval = 1.0 / args.rate_hz
        with RawTerminal():
            while time.monotonic() < deadline:
                key = read_key(0.0)
                if key == " ":
                    paused = not paused
                    print("\nPAUSED: holding current pose." if paused else "\nRESUMED: moving toward hover target.")
                elif key in {"q", "Q"}:
                    print("\nStopped by q; no gripper or descent command was issued.")
                    break

                if paused:
                    hold = end_pose_raw(piper)
                    piper.MotionCtrl_2(0x01, 0x00, args.speed, 0x00)
                    piper.EndPoseCtrl(*hold)
                else:
                    piper.MotionCtrl_2(0x01, 0x00, args.speed, 0x00)
                    piper.EndPoseCtrl(*target)
                    sent_any_command = True
                time.sleep(interval)
        print(f"Final pose: {pose_mm_deg(end_pose_raw(piper))}")
    except KeyboardInterrupt:
        print("\nCtrl+C received; stopped sending motion commands.")
    finally:
        try:
            piper.DisconnectPort()
        except Exception as exc:
            print(f"[warn] disconnect failed: {exc}")
    if sent_any_command:
        print("Motors remain enabled; support the arm before any manual handling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
