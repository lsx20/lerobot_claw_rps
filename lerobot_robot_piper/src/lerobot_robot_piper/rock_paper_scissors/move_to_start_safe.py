#!/usr/bin/env python3
"""Safely move Piper to the claw-machine start pose only."""

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
    fmt_joints,
    joints_deg,
    pose_mm_deg,
    read_key,
    wait_for_movep_ready,
    wait_for_movej_ready,
    wait_for_real_feedback,
)
from claw_arm_keyboard import send_movej_once  # noqa: E402
from lerobot_claw import DEFAULT_START_JOINTS, DEFAULT_START_POSE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--speed", type=int, default=3)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--feedback-timeout", type=float, default=10.0)
    args = parser.parse_args()

    if not 0 <= args.speed <= 100:
        raise SystemExit("--speed must be between 0 and 100")
    if args.rate_hz <= 0 or args.duration <= 0:
        raise SystemExit("--rate-hz and --duration must be positive")

    target_joints = list(DEFAULT_START_JOINTS)
    print("SAFETY: this script only moves Piper to the configured start pose.")
    print("No camera, descent, gripper, or object-grasp command is issued.")
    print("SPACE = pause/hold, SPACE again = resume, q = stop, Ctrl+C = abort.")
    print("Use the physical emergency-stop button for a real emergency.")
    print(f"Latest configured start pose: {tuple(DEFAULT_START_POSE)}")
    print(f"Latest configured start joints: {fmt_joints(target_joints)}")
    if input("Type START to enable Piper and move: ").strip() != "START":
        print("Aborted before connecting/moving.")
        return 0

    piper = connect_piper(args)
    paused = False
    try:
        wait_for_real_feedback(piper, args.feedback_timeout)
        print(f"Current pose: {pose_mm_deg(end_pose_raw(piper))}")
        if not enable_all(piper, args.feedback_timeout):
            raise RuntimeError("Piper did not enable; no motion command was sent.")
        if not wait_for_movej_ready(piper, args.speed, args.feedback_timeout):
            raise RuntimeError("MOVE_J mode was not ready; no target command was sent.")

        deadline = time.monotonic() + args.duration
        interval = 1.0 / args.rate_hz
        with RawTerminal():
            while time.monotonic() < deadline:
                key = read_key(0.0)
                if key == " ":
                    paused = not paused
                    print("\nPAUSED: holding current joints." if paused else "\nRESUMED: moving to start joints.")
                elif key in {"q", "Q"}:
                    print("\nStopped by q; start move may be incomplete.")
                    break

                command_joints = joints_deg(piper) if paused else target_joints
                if not send_movej_once(piper, command_joints, args.speed):
                    raise RuntimeError("MOVE_J command failed; stop and inspect Piper status.")
                time.sleep(interval)
        print(f"Final pose: {pose_mm_deg(end_pose_raw(piper))}")
        print("Now the arm is ready for the separate D405 ball-detection step.")
    except KeyboardInterrupt:
        print("\nCtrl+C received; stopped sending motion commands.")
    finally:
        try:
            piper.DisconnectPort()
        except Exception as exc:
            print(f"[warn] disconnect failed: {exc}")
    print("Motors remain enabled; support the arm before manual handling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
