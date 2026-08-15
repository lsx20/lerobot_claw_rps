#!/usr/bin/env python3
"""Collect RH56F2 repeated-touch samples for three-ball classification."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from .common import (
        BALL_READY_OPEN,
        BALL_SAFE_CLOSED,
        FINGER_NAMES,
        TouchFrame,
        TouchTrial,
        append_feature_row,
        extract_features,
        now_trial_id,
    )
except ImportError:  # Allow: python3 collect_samples.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (  # type: ignore
        BALL_READY_OPEN,
        BALL_SAFE_CLOSED,
        FINGER_NAMES,
        TouchFrame,
        TouchTrial,
        append_feature_row,
        extract_features,
        now_trial_id,
    )


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rh56f2_hand import RH56F2Hand, RH56F2HandConfig  # noqa: E402


PHASES = [
    ("little_thumb", ("little", "thumb_swing"), 0.00),
    ("ring_thumb", ("ring", "thumb_bend"), 0.15),
    ("middle_index", ("middle", "index"), 0.30),
]
STEP_BY_NAME = {
    "little": 60.0,
    "ring": 55.0,
    "middle": 45.0,
    "index": 45.0,
    "thumb_bend": 30.0,
    "thumb_swing": 55.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--hand-force", type=int, default=600)
    parser.add_argument("--label", required=True, help="Ball class label, for example A/B/C or small/mid/large.")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("samples.csv"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--contact-threshold", type=float, default=120.0)
    parser.add_argument("--max-force-delta", type=float, default=800.0)
    parser.add_argument("--settle", type=float, default=0.7)
    parser.add_argument("--step-settle", type=float, default=0.06)
    parser.add_argument("--hold-after-contact", type=float, default=0.5)
    parser.add_argument("--weight-g", type=float, default=None, help="Optional measured mass for this labelled ball.")
    parser.add_argument(
        "--lift-force-delta",
        type=float,
        default=None,
        help="Optional vertical force/torque delta from an external wrist or arm signal.",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--no-open-at-end", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the safety confirmation prompt.")
    return parser.parse_args()


def read_feedback(hand: RH56F2Hand) -> tuple[dict[str, float], dict[str, float]]:
    return hand.read_positions("angleAct"), hand.read_positions("forceAct")


def force_delta(force: dict[str, float], baseline: dict[str, float], name: str) -> float:
    return abs(float(force.get(name, 0.0)) - float(baseline.get(name, 0.0)))


def collect_one_trial(
    hand: RH56F2Hand,
    label: str,
    trial_id: str,
    repeat_index: int,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    print(f"\n[{repeat_index + 1}/{args.repeats}] open hand and settle")
    hand.write_positions("speedSet", {name: float(args.hand_speed) for name in FINGER_NAMES})
    hand.write_positions("forceSet", {name: float(args.hand_force) for name in FINGER_NAMES})
    hand.set_angles(BALL_READY_OPEN)
    time.sleep(args.settle)

    baseline_angles, baseline_forces = read_feedback(hand)
    trial = TouchTrial(
        label=label,
        trial_id=trial_id,
        repeat_index=repeat_index,
        baseline_angles=baseline_angles,
        baseline_forces=baseline_forces,
        contact_threshold=args.contact_threshold,
        lift_force_delta=args.lift_force_delta,
        weight_g=args.weight_g,
        notes=args.notes,
    )

    current_target = dict(baseline_angles)
    contacted: set[str] = set()
    active_phases: set[str] = set()
    started = time.monotonic()
    hold_deadline: float | None = None

    while True:
        elapsed = time.monotonic() - started
        action: dict[str, float] = {}
        for phase_name, names, offset_s in PHASES:
            if elapsed < offset_s:
                continue
            active_phases.add(phase_name)
            for name in names:
                if name in contacted:
                    continue
                goal = BALL_SAFE_CLOSED[name]
                if current_target[name] > goal:
                    current_target[name] = max(goal, current_target[name] - STEP_BY_NAME[name])
                    action[name] = current_target[name]

        if action:
            hand.set_angles(action)
        time.sleep(args.step_settle)
        angles, forces = read_feedback(hand)
        frame = TouchFrame(time.monotonic() - started, angles, forces)
        trial.frames.append(frame)

        force_text = []
        for name in FINGER_NAMES:
            delta = force_delta(forces, baseline_forces, name)
            force_text.append(f"{name}={delta:.0f}")
            if delta >= args.contact_threshold:
                contacted.add(name)
        print(
            "\rcontact "
            f"{len(contacted)}/6 "
            + " ".join(force_text),
            end="",
            flush=True,
        )

        if max(force_delta(forces, baseline_forces, name) for name in FINGER_NAMES) >= args.max_force_delta:
            print("\nmax force reached; stopping this touch.")
            break

        all_goals_reached = all(current_target[name] <= BALL_SAFE_CLOSED[name] for name in FINGER_NAMES)
        enough_contacts = len(contacted) >= 3 and any(name.startswith("thumb") for name in contacted)
        if enough_contacts and hold_deadline is None:
            hold_deadline = time.monotonic() + args.hold_after_contact
        if hold_deadline is not None and time.monotonic() >= hold_deadline:
            break
        if all_goals_reached:
            break

    print()
    row = extract_features(trial)
    if float(row["active_contact_count"]) == 0.0:
        print(
            "[warn] no contact passed the threshold; "
            "check ball placement or lower --contact-threshold."
        )
    append_feature_row(args.output, row)
    print(
        "saved: "
        f"label={label} "
        f"size_closure_mean={float(row['size_closure_mean']):.1f} "
        f"active={float(row['active_contact_count']):.0f} "
        f"force_sum={float(row['final_force_delta_sum']):.1f}"
    )
    return row


def main() -> int:
    args = parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.contact_threshold < 0 or args.max_force_delta <= 0:
        raise SystemExit("--contact-threshold must be >= 0 and --max-force-delta must be > 0")

    print("RH56F2 tactile ball sample collection")
    print("Keep the arm still, place one ball in the hand, and keep fingers clear.")
    print(f"port={args.hand_port} id={args.hand_id} output={args.output}")
    if not args.yes:
        confirm = input("Type BALL_TOUCH to connect and start: ").strip()
        if confirm != "BALL_TOUCH":
            print("Aborted before connecting.")
            return 0

    hand = RH56F2Hand(
        RH56F2HandConfig(
            port=args.hand_port,
            baudrate=args.baudrate,
            hand_id=args.hand_id,
            speed=args.hand_speed,
            force=args.hand_force,
        )
    )
    trial_id = now_trial_id()
    hand.connect()
    try:
        for repeat in range(args.repeats):
            collect_one_trial(hand, args.label, trial_id, repeat, args)
            if repeat + 1 < args.repeats:
                print("Reset the ball if it moved; next touch starts soon.")
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if not args.no_open_at_end:
            try:
                hand.set_angles(BALL_READY_OPEN)
                time.sleep(0.5)
            except Exception:
                pass
        hand.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
