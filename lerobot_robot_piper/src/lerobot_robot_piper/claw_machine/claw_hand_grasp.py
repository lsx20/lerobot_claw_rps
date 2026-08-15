#!/usr/bin/env python3
"""Hand-side grasp actions for the claw-machine workflow."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from rh56f2_hand import DEFAULT_CLOSED, DEFAULT_OPEN
except Exception:  # pragma: no cover - arm-only mode.
    DEFAULT_CLOSED = {}
    DEFAULT_OPEN = {}


BALL_READY_OPEN = dict(DEFAULT_OPEN)
BALL_READY_OPEN.update(
    {
        # Keep the hand partially open so the four fingers stay slightly bent
        # instead of fully splayed before the descent starts.
        "little": 1600,
        "ring": 1600,
        "middle": 1600,
        "index": 1600,
        "thumb_bend": 1470,
        "thumb_swing": 1050,
    }
)

BALL_CLOSED = dict(DEFAULT_CLOSED)
BALL_CLOSED.update(
    {
        # At grab height, curl the thumb and four fingers while keeping the
        # thumb swung inward around the ball.
        "thumb_bend": 1060,
        "thumb_swing": 900,
    }
)

THUMB_GESTURE = dict(DEFAULT_CLOSED)
THUMB_GESTURE.update(
    {
        # Keep four fingers folded and straighten the thumb; J6 rotation
        # decides whether this reads as thumbs-up or thumbs-down.
        "thumb_bend": 1500,
        "thumb_swing": 1800,
    }
)


def set_hand(hand: object | None, pose: dict[str, float], label: str) -> bool:
    if hand is None:
        print(f"{label}: hand disabled; skipped")
        return True
    try:
        hand.set_angles(pose)
    except Exception as exc:
        print(f"[warn] {label}: hand command failed: {exc}")
        return False
    print(f"{label}: hand command sent")
    return True


def set_hand_speed(hand: object | None, speed: int | None, label: str) -> bool:
    if hand is None or speed is None:
        return True
    try:
        hand.write_positions("speedSet", {name: speed for name in BALL_READY_OPEN})
    except Exception as exc:
        print(f"[warn] {label}: hand speed command failed: {exc}")
        return False
    print(f"{label}: hand speed set to {speed}")
    return True


def set_hand_async(hand: object | None, pose: dict[str, float], label: str) -> None:
    if hand is None:
        print(f"{label}: hand disabled; skipped")
        return

    def worker() -> None:
        set_hand(hand, pose, label)

    threading.Thread(target=worker, daemon=True).start()


def open_while_descending(hand: object | None, speed: int | None = None) -> None:
    def worker() -> None:
        set_hand_speed(hand, speed, "fast open while descending")
        set_hand(hand, BALL_READY_OPEN, "partially open while descending")

    if hand is None:
        print("partially open while descending: hand disabled; skipped")
        return
    threading.Thread(target=worker, daemon=True).start()


def close_at_grab(hand: object | None, speed: int | None = None) -> bool:
    set_hand_speed(hand, speed, "restore close speed")
    return set_hand(hand, BALL_CLOSED, "close ball grasp")


def object_held_by_force(
    hand: object | None,
    threshold: float,
    required_names: list[str],
    alternate_required_names: list[str],
    duration_s: float,
    rate_hz: float,
    required_samples: int,
) -> bool:
    if hand is None:
        print("held check: hand disabled; assume empty")
        return False

    deadline = time.time() + duration_s
    interval_s = 1.0 / rate_hz
    consecutive = 0
    best_consecutive = 0
    groups = [required_names]
    if alternate_required_names:
        groups.append(alternate_required_names)
    tracked_names = sorted({name for group in groups for name in group})
    last_active = {name: 0.0 for name in tracked_names}
    best_group = ""

    while time.time() < deadline:
        try:
            values = hand.read_positions("forceAct")
        except Exception as exc:
            print(f"[warn] held check: forceAct read failed: {exc}; assume empty")
            return False

        last_active = {
            name: abs(values.get(name, 0.0))
            for name in tracked_names
        }
        active_group = next(
            (
                group
                for group in groups
                if all(last_active.get(name, 0.0) >= threshold for name in group)
            ),
            None,
        )
        if active_group is not None:
            consecutive += 1
            best_consecutive = max(best_consecutive, consecutive)
            best_group = ",".join(active_group)
        else:
            consecutive = 0
        time.sleep(interval_s)

    held = best_consecutive >= required_samples
    force_text = " ".join(f"{name}={value:.1f}" for name, value in last_active.items())
    print(
        f"held check: {force_text}, threshold={threshold:.1f}, "
        f"best_consecutive={best_consecutive}/{required_samples}, "
        f"best_group={best_group or 'none'}, held={held}"
    )
    return held


def show_thumb_gesture(
    hand: object | None,
    speed: int | None = None,
) -> bool:
    set_hand_speed(hand, speed, "thumb gesture speed")
    return set_hand(hand, THUMB_GESTURE, "thumb gesture")


def open_at_drop(hand: object | None) -> bool:
    return set_hand(hand, DEFAULT_OPEN, "open")


def close_while_returning(hand: object | None) -> None:
    set_hand_async(hand, DEFAULT_CLOSED, "close while returning")
