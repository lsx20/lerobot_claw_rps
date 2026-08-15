#!/usr/bin/env python3
"""Collect empty/missed-grasp tactile samples for RH56F2 ball handling.

These samples are saved separately from A/B/C samples. Use them to study or
train a "not held" detector; do not mix them into the A/B/C model directly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from . import collect_lift_samples as lift
    from .common import now_trial_id
except ImportError:  # Allow: python3 collect_miss_samples.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import collect_lift_samples as lift  # type: ignore
    from common import now_trial_id  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--hand-force", type=int, default=600)
    parser.add_argument("--hand-read-retries", type=int, default=5)
    parser.add_argument("--hand-read-retry-delay", type=float, default=0.03)
    parser.add_argument(
        "--tactile-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read SDK touchData normal/tangential/proximity fields in each frame.",
    )
    parser.add_argument(
        "--tactile-required",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail a frame if touchData cannot be read.",
    )
    parser.add_argument("--speed", type=int, default=8, help="Piper MOVE_P speed.")
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--feedback-timeout", type=float, default=8.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--rpy-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--movep-retries", type=int, default=2)
    parser.add_argument(
        "--grab-pose",
        type=lift.parse_pose_mm_deg,
        required=True,
        help="Lower grasp X,Y,Z,RX,RY,RZ in mm/deg.",
    )
    parser.add_argument(
        "--drop-pose",
        type=lift.parse_pose_mm_deg,
        default=None,
        help="Optional release X,Y,Z,RX,RY,RZ in mm/deg. Default returns to grasp pose.",
    )
    parser.add_argument("--grab-duration", type=float, default=5.0)
    parser.add_argument("--label", default="NONE", help="Label written to the CSV.")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("miss_samples.csv"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--contact-threshold", type=float, default=70.0)
    parser.add_argument("--max-force-delta", type=float, default=900.0)
    parser.add_argument("--step-settle", type=float, default=0.06)
    parser.add_argument(
        "--close-mode",
        choices=("fixed", "contact_stop"),
        default="fixed",
        help="fixed closes fully; contact_stop freezes fingers after contact.",
    )
    parser.add_argument("--hold-after-contact", type=float, default=1.0)
    parser.add_argument("--max-close-duration", type=float, default=3.0)
    parser.add_argument("--lift-height-mm", type=float, default=30.0)
    parser.add_argument("--lift-duration", type=float, default=3.0)
    parser.add_argument("--lower-duration", type=float, default=3.0)
    parser.add_argument("--hover-duration", type=float, default=2.0)
    parser.add_argument("--hover-rate-hz", type=float, default=10.0)
    parser.add_argument("--lift-settle", type=float, default=0.2)
    parser.add_argument("--open-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-settle", type=float, default=1.2)
    parser.add_argument("--open-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--between-repeat", type=float, default=1.0)
    parser.add_argument("--notes", default="miss_grasp empty_or_not_held")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    for name in (
        "rate_hz",
        "hover_rate_hz",
        "grab_duration",
        "lift_duration",
        "lower_duration",
        "hover_duration",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.lift_height_mm <= 0:
        raise SystemExit("--lift-height-mm must be positive")
    if not 0 <= args.speed <= 100:
        raise SystemExit("--speed must be between 0 and 100")
    if args.movep_retries <= 0:
        raise SystemExit("--movep-retries must be positive")
    if args.hand_read_retries <= 0 or args.hand_read_retry_delay < 0:
        raise SystemExit("--hand-read-retries must be positive and retry delay must be >= 0")
    if args.max_close_duration <= 0 or args.hold_after_contact < 0:
        raise SystemExit("--max-close-duration must be positive and --hold-after-contact must be >= 0")


def main() -> int:
    args = parse_args()
    validate_args(args)

    # collect_lift_samples.collect_one_trial expects these fields.
    args.grab_xyz = None
    args.squeeze_test = False
    args.squeeze_delta = 40.0
    args.squeeze_pre_duration = 0.4
    args.squeeze_duration = 3.0
    args.squeeze_rate_hz = 20.0
    args.squeeze_max_force_delta = 900.0
    args.squeeze_baseline_hover_samples = 10
    args.squeeze_middle_touch_threshold = 80.0
    args.squeeze_middle_seek_step = 15.0
    args.squeeze_middle_seek_max_delta = 160.0
    args.squeeze_middle_seek_settle = 0.08
    args.min_lift_contacts = 999.0
    args.force_lift = True
    args.save_skipped = True
    args.predict_model = None

    print("RH56F2 missed/empty grasp sample collection")
    print("Use this with no ball in the grasp, or with a deliberately missed grasp.")
    print("Samples are saved separately and should not be mixed into A/B/C training.")
    print(f"grab pose: {lift.pose_mm_deg(args.grab_pose)}")
    if args.drop_pose is not None:
        print(f"drop pose: {lift.pose_mm_deg(args.drop_pose)}")
    print(f"can={args.can} hand={args.hand_port} label={args.label} output={args.output}")
    if not args.yes:
        confirm = input(
            f"Type MISS_LIFT to collect {args.repeats} missed/empty grasp sample(s): "
        ).strip()
        if confirm != "MISS_LIFT":
            print("Aborted before connecting.")
            return 0

    piper = None
    hand = lift.RH56F2Hand(
        lift.RH56F2HandConfig(
            port=args.hand_port,
            baudrate=args.baudrate,
            hand_id=args.hand_id,
            speed=args.hand_speed,
            force=args.hand_force,
        )
    )
    trial_id = now_trial_id()
    try:
        piper = lift.connect_piper(args)
        lift.wait_for_real_feedback(piper, args.feedback_timeout)
        if not lift.enable_all(piper, args.feedback_timeout):
            raise RuntimeError("Piper did not enable; no motion command was sent.")
        if not lift.wait_for_movep_ready(piper, args.speed, args.feedback_timeout):
            if not lift.recover_movep_control(piper, args, "initial MOVE_P ready"):
                raise RuntimeError("MOVE_P mode was not ready; no target command was sent.")
        hand.connect()

        for repeat in range(args.repeats):
            print("\nMake sure this trial is a missed/empty grasp before it starts.")
            if not lift.collect_one_trial(piper, hand, args.label, trial_id, repeat, args, None):
                print("[warn] stopping repeats because the arm did not return to the lower pose.")
                break
            if repeat + 1 < args.repeats:
                print("Keep the grasp area empty, or reset the deliberate miss, before the next trial.")
                time.sleep(args.between_repeat)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            hand.disconnect()
        except Exception:
            pass
        if piper is not None:
            try:
                piper.DisconnectPort()
                print("Disconnected Piper without sending DisableArm.")
            except Exception as exc:
                print(f"[warn] Piper disconnect failed: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
