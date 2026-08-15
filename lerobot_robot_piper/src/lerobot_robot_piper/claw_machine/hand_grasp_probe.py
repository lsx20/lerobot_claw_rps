#!/usr/bin/env python3
"""Step RH56F2 finger closure while recording feedback to CSV."""

from __future__ import annotations

import argparse
import csv
import select
import signal
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rh56f2_hand import DEFAULT_CLOSED, DEFAULT_OPEN, HAND_NAMES, RH56F2Hand, RH56F2HandConfig


class RawTerminal:
    def __enter__(self) -> "RawTerminal":
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def read_key() -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None
    return sys.stdin.read(1).lower()


def fmt(values: dict[str, float]) -> str:
    return " ".join(f"{name}={values.get(name, 0.0):7.1f}" for name in HAND_NAMES)


def allocate_csv_path(requested: Path) -> Path:
    """Return a new path without overwriting an earlier trial."""
    if not requested.exists():
        return requested
    for index in range(1, 10000):
        candidate = requested.with_name(
            f"{requested.stem}_{index:03d}{requested.suffix}"
        )
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate a new CSV path from {requested}")


def append_trial_log(log_path: Path, csv_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="") as log_file:
        writer = csv.writer(log_file)
        if needs_header:
            writer.writerow(["timestamp", "csv", "result"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), str(csv_path), "pending"])


def step_toward_closed(
    target: dict[str, float],
    front_step: int,
    little_step: int,
    ring_step: int,
    thumb_bend_step: int,
    thumb_swing: int,
    closed_targets: dict[str, float],
) -> dict[str, float]:
    steps = {
        "little": little_step,
        "ring": ring_step,
        "middle": front_step,
        "index": front_step,
        "thumb_bend": thumb_bend_step,
        "thumb_swing": 0,
    }
    next_target = {
        name: max(float(closed_targets[name]), value - steps[name])
        for name, value in target.items()
    }
    next_target["thumb_swing"] = float(thumb_swing)
    return next_target


def step_toward_open(
    target: dict[str, float],
    front_step: int,
    little_step: int,
    ring_step: int,
    thumb_bend_step: int,
    thumb_swing: int,
) -> dict[str, float]:
    steps = {
        "little": little_step,
        "ring": ring_step,
        "middle": front_step,
        "index": front_step,
        "thumb_bend": thumb_bend_step,
        "thumb_swing": 0,
    }
    next_target = {
        name: min(float(DEFAULT_OPEN[name]), value + steps[name])
        for name, value in target.items()
    }
    next_target["thumb_swing"] = float(thumb_swing)
    return next_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--hand-speed", type=int, default=300)
    parser.add_argument("--hand-force", type=int, default=800)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--rear-step", type=int, default=30)
    parser.add_argument("--little-step", type=int)
    parser.add_argument("--ring-step", type=int)
    parser.add_argument("--thumb-bend-step", type=int)
    parser.add_argument("--rear-closed", type=int, default=1100)
    parser.add_argument("--little-closed", type=int, default=1200)
    parser.add_argument("--ring-closed", type=int, default=1220)
    parser.add_argument("--front-closed", type=int, default=1350)
    parser.add_argument(
        "--thumb-bend-closed",
        type=int,
        default=1300,
        help="thumb bend target for moderate inward contact",
    )
    parser.add_argument("--thumb-swing", type=int, default=1200)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--settle", type=float, default=0.4)
    parser.add_argument("--detect-threshold", type=float, default=300.0)
    parser.add_argument("--detect-count", type=int, default=3)
    parser.add_argument("--detect-duration", type=float, default=5.0)
    parser.add_argument("--csv", type=Path, default=Path("hand_grasp_probe.csv"))
    parser.add_argument("--trial-log", type=Path, default=Path("grasp_trial_log.csv"))
    parser.add_argument(
        "--open-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.csv = allocate_csv_path(args.csv)
    append_trial_log(args.trial_log, args.csv)
    if args.step <= 0:
        raise ValueError("--step must be positive")
    if args.rear_step <= 0:
        raise ValueError("--rear-step must be positive")
    if args.little_step is None:
        args.little_step = args.rear_step
    if args.ring_step is None:
        args.ring_step = args.rear_step
    if args.little_step <= 0 or args.ring_step <= 0:
        raise ValueError("--little-step and --ring-step must be positive")
    if args.thumb_bend_step is None:
        args.thumb_bend_step = args.step
    if args.thumb_bend_step < 0:
        raise ValueError("--thumb-bend-step must be non-negative")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")
    if args.settle < 0:
        raise ValueError("--settle must be non-negative")
    if args.detect_threshold < 0:
        raise ValueError("--detect-threshold must be non-negative")
    if not 1 <= args.detect_count <= len(HAND_NAMES):
        raise ValueError(f"--detect-count must be in [1, {len(HAND_NAMES)}]")
    if args.detect_duration <= 0:
        raise ValueError("--detect-duration must be positive")
    if not 450 <= args.thumb_swing <= 1800:
        raise ValueError("--thumb-swing must be in [450, 1800]")
    if not 850 <= args.rear_closed <= 1800:
        raise ValueError("--rear-closed must be in [850, 1800]")
    if not 850 <= args.little_closed <= 1800:
        raise ValueError("--little-closed must be in [850, 1800]")
    if not 850 <= args.ring_closed <= 1800:
        raise ValueError("--ring-closed must be in [850, 1800]")
    if not 850 <= args.front_closed <= 1800:
        raise ValueError("--front-closed must be in [850, 1800]")
    if not 1050 <= args.thumb_bend_closed <= 1500:
        raise ValueError("--thumb-bend-closed must be in [1050, 1500]")

    stop = False

    def handle_sigint(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sigint)

    hand = RH56F2Hand(
        RH56F2HandConfig(
            port=args.hand_port,
            baudrate=args.baudrate,
            hand_id=args.hand_id,
            speed=args.hand_speed,
            force=args.hand_force,
        )
    )
    target = dict(DEFAULT_OPEN)
    target["thumb_swing"] = float(args.thumb_swing)
    closed_targets = dict(DEFAULT_CLOSED)
    closed_targets.update(
        {
            "little": float(args.little_closed),
            "ring": float(args.ring_closed),
            "middle": float(args.front_closed),
            "index": float(args.front_closed),
            "thumb_bend": float(args.thumb_bend_closed),
        }
    )
    action = "initial_open"
    auto_closing = False
    next_auto_step = 0.0
    sample = 0
    start_time = time.monotonic()
    baseline_force: dict[str, float] | None = None
    detection_armed = False
    grasp_active_since: float | None = None
    grasp_prediction = "pending"

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    hand.connect()
    try:
        hand.set_angles(target)
        time.sleep(1.0)
        baseline_force = hand.read_positions("forceAct")

        print("RH56F2 step grasp probe")
        print(f"serial={args.hand_port} speed={args.hand_speed} force_limit={args.hand_force}")
        print(
            f"front_step={args.step} little_step={args.little_step} "
            f"ring_step={args.ring_step} "
            f"thumb_bend_step={args.thumb_bend_step} "
            f"front_closed={args.front_closed} little_closed={args.little_closed} "
            f"ring_closed={args.ring_closed} "
            f"thumb_bend_closed={args.thumb_bend_closed} thumb_swing={args.thumb_swing} "
            f"settle={args.settle:.2f}s csv={args.csv}"
        )
        print("手爪已张开，请把球放在手指之间。")
        print("按键：c=闭合一步，o=张开一步，a=自动闭合，p=打印，q=退出")
        print("按 a 后会自动小步闭合到目标，thumb_swing 保持固定。")
        print(
            f"抓取判断：六指任意 {args.detect_count} 指 forceAct >= "
            f"{args.detect_threshold:.0f}，连续 {args.detect_duration:.1f}s。"
        )
        print(f"空载 forceAct: {fmt(baseline_force)}")

        with args.csv.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "time_s",
                    "sample",
                    "action",
                    "target_little",
                    "target_ring",
                    "target_middle",
                    "target_index",
                    "target_thumb_bend",
                    "target_thumb_swing",
                    "angle_little",
                    "angle_ring",
                    "angle_middle",
                    "angle_index",
                    "angle_thumb_bend",
                    "angle_thumb_swing",
                    "force_little",
                    "force_ring",
                    "force_middle",
                    "force_index",
                    "force_thumb_bend",
                    "force_thumb_swing",
                    "force_delta_little",
                    "force_delta_ring",
                    "force_delta_middle",
                    "force_delta_index",
                    "force_delta_thumb_bend",
                    "force_delta_thumb_swing",
                    "status_little",
                    "status_ring",
                    "status_middle",
                    "status_index",
                    "status_thumb_bend",
                    "status_thumb_swing",
                    "temp_little",
                    "temp_ring",
                    "temp_middle",
                    "temp_index",
                    "temp_thumb_bend",
                    "temp_thumb_swing",
                    "active_force_count",
                    "grasp_continuous_s",
                    "grasp_prediction",
                ]
            )
            csv_file.flush()

            with RawTerminal():
                while not stop:
                    key = read_key()
                    if key == "q":
                        stop = True
                    elif key == "c":
                        auto_closing = False
                        target = step_toward_closed(
                            target,
                            args.step,
                            args.little_step,
                            args.ring_step,
                            args.thumb_bend_step,
                            args.thumb_swing,
                            closed_targets,
                        )
                        hand.set_angles(target)
                        action = f"close_step_{args.step}"
                        print(f"\n闭合一步 target: {fmt(target)}")
                        if args.settle:
                            time.sleep(args.settle)
                    elif key == "o":
                        auto_closing = False
                        detection_armed = False
                        grasp_active_since = None
                        grasp_prediction = "pending"
                        target = step_toward_open(
                            target,
                            args.step,
                            args.little_step,
                            args.ring_step,
                            args.thumb_bend_step,
                            args.thumb_swing,
                        )
                        hand.set_angles(target)
                        action = f"open_step_{args.step}"
                        print(f"\n张开一步 target: {fmt(target)}")
                        if args.settle:
                            time.sleep(args.settle)
                    elif key == "a":
                        auto_closing = True
                        next_auto_step = 0.0
                        detection_armed = False
                        grasp_active_since = None
                        grasp_prediction = "pending"
                        action = "auto_close_start"
                    elif key == "p":
                        angle = hand.read_positions("angleAct")
                        force = hand.read_positions("forceAct")
                        print(f"\nangleAct: {fmt(angle)}")
                        print(f"forceAct: {fmt(force)}")

                    now = time.monotonic()
                    close_names = [
                        "little",
                        "ring",
                        "middle",
                        "index",
                        "thumb_bend",
                    ]
                    if auto_closing and now >= next_auto_step:
                        if all(target[name] <= closed_targets[name] for name in close_names):
                            auto_closing = False
                            action = "auto_close_complete"
                            detection_armed = True
                            grasp_active_since = None
                            grasp_prediction = "checking"
                            print("\n自动闭合完成，已到达设定目标。")
                        else:
                            target = step_toward_closed(
                                target,
                                args.step,
                                args.little_step,
                                args.ring_step,
                                args.thumb_bend_step,
                                args.thumb_swing,
                                closed_targets,
                            )
                            hand.set_angles(target)
                            action = f"auto_close_step_{args.step}"
                            print(f"\n自动闭合 target: {fmt(target)}")
                            next_auto_step = now + args.settle

                    angle = hand.read_positions("angleAct")
                    force = hand.read_positions("forceAct")
                    status = hand.read_positions("statusCode")
                    temp = hand.read_positions("temp")
                    delta = {
                        name: force[name] - (baseline_force or {}).get(name, 0.0)
                        for name in HAND_NAMES
                    }
                    elapsed = time.monotonic() - start_time
                    active_names = [
                        name
                        for name in HAND_NAMES
                        if abs(force[name]) >= args.detect_threshold
                    ]
                    continuous_s = 0.0
                    if detection_armed and len(active_names) >= args.detect_count:
                        if grasp_active_since is None:
                            grasp_active_since = time.monotonic()
                        continuous_s = time.monotonic() - grasp_active_since
                        if continuous_s >= args.detect_duration:
                            if grasp_prediction != "grasped":
                                print(
                                    f"\n抓取判断=成功：{','.join(active_names)} "
                                    f"同时超过 {args.detect_threshold:.0f}，"
                                    f"持续 {continuous_s:.1f}s。"
                                )
                            grasp_prediction = "grasped"
                        elif grasp_prediction != "grasped":
                            grasp_prediction = "checking"
                    elif detection_armed:
                        grasp_active_since = None
                        continuous_s = 0.0
                        if grasp_prediction != "grasped":
                            grasp_prediction = "checking"
                    writer.writerow(
                        [
                            f"{elapsed:.6f}",
                            sample,
                            action,
                            *[target[name] for name in HAND_NAMES],
                            *[angle[name] for name in HAND_NAMES],
                            *[force[name] for name in HAND_NAMES],
                            *[delta[name] for name in HAND_NAMES],
                            *[status[name] for name in HAND_NAMES],
                            *[temp[name] for name in HAND_NAMES],
                            len(active_names),
                            f"{continuous_s:.3f}",
                            grasp_prediction,
                        ]
                    )
                    csv_file.flush()
                    print(
                        f"\r{elapsed:7.2f}s action={action:18s} "
                        f"active={len(active_names)}/{args.detect_count} "
                        f"hold={continuous_s:4.1f}s "
                        f"prediction={grasp_prediction:8s} force={fmt(force)}",
                        end="",
                        flush=True,
                    )
                    sample += 1
                    time.sleep(1.0 / args.rate_hz)
    finally:
        if args.open_on_exit:
            try:
                exit_target = dict(DEFAULT_OPEN)
                exit_target["thumb_swing"] = float(args.thumb_swing)
                hand.set_angles(exit_target)
                print("\n退出前已张开手爪。")
            except Exception as exc:
                print(f"\n[warn] exit open failed: {exc}")
        hand.disconnect()

    print(f"记录完成：{args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
