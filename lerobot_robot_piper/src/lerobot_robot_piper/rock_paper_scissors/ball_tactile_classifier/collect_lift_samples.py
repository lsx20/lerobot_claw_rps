#!/usr/bin/env python3
"""Collect tactile samples while lifting from a current or configured Piper pose.

By default, move the arm to the lower grasp/drop pose before starting. Or pass
--grab-xyz/--grab-pose and the script will enable Piper, move to that grasp
pose, close the hand, lift, hover, collect tactile data, then lower. This script
uses the same direct Piper MOVE_P helper as the claw-machine scripts:

    wait_for_movep_ready() -> send_movep_for()

Only ee.z is changed during lift/lower. X/Y/RX/RY/RZ stay at the selected lower
grasp pose.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from .common import (
        BALL_READY_OPEN,
        BALL_SAFE_CLOSED,
        CLOSE_PHASES,
        CLOSE_STEP_BY_NAME,
        CORE_GRASP_FINGERS,
        FINGER_NAMES,
        THUMB_NAMES,
        TouchFrame,
        TouchTrial,
        append_feature_row,
        extract_features,
        force_delta,
        load_model,
        now_trial_id,
        predict_row,
    )
    from .visualize_live import render_dashboard_from_csv
except ImportError:  # Allow: python3 collect_lift_samples.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (  # type: ignore
        BALL_READY_OPEN,
        BALL_SAFE_CLOSED,
        CLOSE_PHASES,
        CLOSE_STEP_BY_NAME,
        CORE_GRASP_FINGERS,
        FINGER_NAMES,
        THUMB_NAMES,
        TouchFrame,
        TouchTrial,
        append_feature_row,
        extract_features,
        force_delta,
        load_model,
        now_trial_id,
        predict_row,
    )
    from visualize_live import render_dashboard_from_csv  # type: ignore


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

CLAW_MACHINE_DIR = PACKAGE_ROOT / "lerobot_robot_piper" / "claw_machine"
if str(CLAW_MACHINE_DIR) not in sys.path:
    sys.path.insert(0, str(CLAW_MACHINE_DIR))

from claw_init import (  # noqa: E402
    connect_piper,
    enable_all,
    end_pose_raw,
    pose_mm_deg,
    send_movep_for,
    wait_for_movep_ready,
    wait_for_real_feedback,
)
from lerobot_robot_piper.rh56f2_hand import RH56F2Hand, RH56F2HandConfig  # noqa: E402


def parse_xyz_m(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected X,Y,Z in metres")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("X,Y,Z must be numbers") from exc


def parse_pose_mm_deg(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("expected X,Y,Z,RX,RY,RZ in mm/deg")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pose values must be numbers") from exc
    return [int(round(item * 1000.0)) for item in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--hand-force", type=int, default=600)
    parser.add_argument("--hand-read-retries", type=int, default=3)
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
        help="Fail a frame if touchData cannot be read. Default keeps forceAct sampling alive.",
    )
    parser.add_argument("--speed", type=int, default=8, help="Piper MOVE_P speed.")
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--feedback-timeout", type=float, default=8.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--rpy-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--movep-retries", type=int, default=2)
    parser.add_argument(
        "--grab-xyz",
        type=parse_xyz_m,
        default=None,
        help="Optional lower grasp X,Y,Z in metres. Keeps current RX/RY/RZ.",
    )
    parser.add_argument(
        "--grab-pose",
        type=parse_pose_mm_deg,
        default=None,
        help="Optional lower grasp X,Y,Z,RX,RY,RZ in mm/deg.",
    )
    parser.add_argument(
        "--drop-pose",
        type=parse_pose_mm_deg,
        default=None,
        help="Optional release X,Y,Z,RX,RY,RZ in mm/deg. Default returns to the grasp pose.",
    )
    parser.add_argument("--grab-duration", type=float, default=5.0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("samples.csv"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--contact-threshold", type=float, default=120.0)
    parser.add_argument("--max-force-delta", type=float, default=900.0)
    parser.add_argument("--step-settle", type=float, default=0.06)
    parser.add_argument(
        "--close-mode",
        choices=("fixed", "contact_stop"),
        default="fixed",
        help="fixed closes to the same target each time; contact_stop freezes contacted fingers.",
    )
    parser.add_argument("--hold-after-contact", type=float, default=1.0)
    parser.add_argument(
        "--max-close-duration",
        type=float,
        default=3.0,
        help="Maximum seconds spent closing before lift decision.",
    )
    parser.add_argument("--lift-height-mm", type=float, default=50.0)
    parser.add_argument("--lift-duration", type=float, default=3.0)
    parser.add_argument("--lower-duration", type=float, default=3.0)
    parser.add_argument("--hover-duration", type=float, default=5.0)
    parser.add_argument("--hover-rate-hz", type=float, default=10.0)
    parser.add_argument("--lift-settle", type=float, default=0.2)
    parser.add_argument("--squeeze-test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--squeeze-delta",
        type=float,
        default=40.0,
        help="Extra close angle units for index/middle/thumb during squeeze test.",
    )
    parser.add_argument(
        "--squeeze-pre-duration",
        type=float,
        default=0.4,
        help="Seconds to sample after middle contact but before sending the squeeze command.",
    )
    parser.add_argument("--squeeze-duration", type=float, default=3.0)
    parser.add_argument("--squeeze-rate-hz", type=float, default=20.0)
    parser.add_argument(
        "--squeeze-max-force-delta",
        type=float,
        default=900.0,
        help="Stop squeeze test early if any core finger force rises by this amount.",
    )
    parser.add_argument(
        "--squeeze-baseline-hover-samples",
        type=int,
        default=10,
        help="Use the last N hover frames as the squeeze force/angle baseline.",
    )
    parser.add_argument(
        "--squeeze-middle-touch-threshold",
        type=float,
        default=80.0,
        help="Middle-finger force delta from the original grasp baseline required before squeezing.",
    )
    parser.add_argument("--squeeze-middle-seek-step", type=float, default=15.0)
    parser.add_argument("--squeeze-middle-seek-max-delta", type=float, default=160.0)
    parser.add_argument("--squeeze-middle-seek-settle", type=float, default=0.08)
    parser.add_argument("--min-lift-contacts", type=float, default=2.0)
    parser.add_argument("--force-lift", action="store_true", help="Lift even if contact count is weak.")
    parser.add_argument("--open-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-settle", type=float, default=1.2)
    parser.add_argument("--open-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--between-repeat", type=float, default=1.0)
    parser.add_argument("--notes", default="current pose direct MOVEP lift-hover sample")
    parser.add_argument("--predict-model", type=Path, default=None)
    parser.add_argument("--visual-output", type=Path, default=Path(__file__).with_name("live_dashboard.html"))
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-skipped",
        action="store_true",
        help="Save weak/no-contact rows even when lift is skipped.",
    )
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def selected_lower_pose(piper: object, args: argparse.Namespace) -> list[int]:
    current = end_pose_raw(piper)
    if args.grab_pose is not None:
        return list(args.grab_pose)
    if args.grab_xyz is not None:
        pose = list(current)
        pose[0] = int(round(args.grab_xyz[0] * 1_000_000.0))
        pose[1] = int(round(args.grab_xyz[1] * 1_000_000.0))
        pose[2] = int(round(args.grab_xyz[2] * 1_000_000.0))
        return pose
    return current


def recover_movep_control(piper: object, args: argparse.Namespace, label: str) -> bool:
    print(f"[warn] {label}: re-entering MOVE_P after failed command.")
    for _ in range(10):
        piper.MotionCtrl_1(0x02, 0x04, 0x02)
        time.sleep(0.02)
    for _ in range(10):
        piper.MotionCtrl_1(0x02, 0x00, 0x02)
        piper.MotionCtrl_2(0x01, 0x00, args.speed, 0x00)
        time.sleep(0.02)
    if not enable_all(piper, args.feedback_timeout):
        print(f"[warn] {label}: enable_all failed during retry.")
        return False
    return wait_for_movep_ready(piper, args.speed, args.feedback_timeout)


def send_movep_checked(
    piper: object,
    target: list[int],
    args: argparse.Namespace,
    duration_s: float,
    label: str,
    require_reached: bool = False,
) -> bool:
    attempts = max(1, args.movep_retries)
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            if not recover_movep_control(piper, args, f"{label} retry {attempt}/{attempts}"):
                continue
        elif not wait_for_movep_ready(piper, args.speed, args.feedback_timeout):
            if not recover_movep_control(piper, args, f"{label} initial ready"):
                continue

        ok = send_movep_for(
            piper,
            target,
            args.speed,
            duration_s,
            args.rate_hz,
            label if attempt == 1 else f"{label} retry {attempt}",
            args.position_tolerance_mm,
            args.rpy_tolerance_deg,
            require_reached=require_reached,
        )
        if ok:
            return True
    return False


def configure_hand(hand: RH56F2Hand, speed: int, force: int) -> None:
    hand.write_positions("speedSet", {name: float(speed) for name in FINGER_NAMES})
    hand.write_positions("forceSet", {name: float(force) for name in FINGER_NAMES})
    print(f"hand limits: speed={speed} force={force}")


def set_hand_pose(hand: RH56F2Hand, pose: dict[str, float], label: str) -> None:
    hand.set_angles(pose)
    print(f"{label}: hand command sent")


def read_touch_frame(hand: RH56F2Hand, started_at: float, args: argparse.Namespace) -> TouchFrame:
    last_exc: Exception | None = None
    for _ in range(args.hand_read_retries):
        try:
            angles = hand.read_positions("angleAct")
            forces = hand.read_positions("forceAct")
            tactile = None
            if getattr(args, "tactile_data", True):
                try:
                    tactile = hand.read_touch_data()
                except Exception as exc:
                    if getattr(args, "tactile_required", False):
                        raise
                    last_exc = exc
            return TouchFrame(time.monotonic() - started_at, angles, forces, tactile)
        except Exception as exc:
            last_exc = exc
            time.sleep(args.hand_read_retry_delay)
    raise RuntimeError(f"RH56F2 feedback read failed after retries: {last_exc}")


def refresh_sample_dashboard(args: argparse.Namespace) -> None:
    if not getattr(args, "visualize", True):
        return
    try:
        render_dashboard_from_csv(args.output, args.visual_output, last=30, reference_samples=args.output)
        print(f"visual dashboard updated: {args.visual_output}")
    except Exception as exc:
        print(f"[warn] sample dashboard refresh failed: {exc}")


def average_frame_values(frames: list[TouchFrame], attr: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in FINGER_NAMES:
        values = [float(getattr(frame, attr).get(name, 0.0)) for frame in frames]
        result[name] = sum(values) / len(values) if values else 0.0
    return result


def stable_hover_baseline(trial: TouchTrial, args: argparse.Namespace) -> tuple[dict[str, float], dict[str, float], str]:
    count = max(1, int(args.squeeze_baseline_hover_samples))
    frames = trial.hover_frames[-count:]
    if frames:
        return average_frame_values(frames, "angles"), average_frame_values(frames, "forces"), f"hover_last_{len(frames)}"
    return dict(trial.baseline_angles), dict(trial.baseline_forces), "trial_baseline"


def middle_contact_delta(frame: TouchFrame, trial: TouchTrial) -> float:
    return force_delta(frame.forces, trial.baseline_forces, "middle")


def close_and_record(
    hand: RH56F2Hand,
    trial: TouchTrial,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    current_target = dict(trial.baseline_angles)
    contacted: set[str] = set()
    started_at = time.monotonic()
    hold_deadline: float | None = None

    while True:
        elapsed_s = time.monotonic() - started_at
        action: dict[str, float] = {}
        for _, names, offset_s in CLOSE_PHASES:
            if elapsed_s < offset_s:
                continue
            for name in names:
                if args.close_mode == "contact_stop" and name in contacted:
                    continue
                goal = BALL_SAFE_CLOSED[name]
                if current_target[name] > goal:
                    current_target[name] = max(goal, current_target[name] - CLOSE_STEP_BY_NAME[name])
                    action[name] = current_target[name]

        if action:
            hand.set_angles(action)
        time.sleep(args.step_settle)

        frame = read_touch_frame(hand, started_at, args)
        trial.frames.append(frame)
        force_text = []
        for name in FINGER_NAMES:
            delta = force_delta(frame.forces, trial.baseline_forces, name)
            force_text.append(f"{name}={delta:.0f}")
            if delta >= args.contact_threshold:
                contacted.add(name)
        print(
            "\rclose contact "
            f"{len(contacted)}/6 "
            + " ".join(force_text),
            end="",
            flush=True,
        )

        max_delta = max(force_delta(frame.forces, trial.baseline_forces, name) for name in FINGER_NAMES)
        if max_delta >= args.max_force_delta:
            print("\nmax close force reached; stopping close.")
            break

        all_goals_reached = all(current_target[name] <= BALL_SAFE_CLOSED[name] for name in FINGER_NAMES)
        if args.close_mode == "contact_stop":
            enough_contacts = len(contacted) >= args.min_lift_contacts
            if enough_contacts and hold_deadline is None:
                hold_deadline = time.monotonic() + args.hold_after_contact
            if hold_deadline is not None and time.monotonic() >= hold_deadline:
                break
        if elapsed_s >= args.max_close_duration:
            print(f"\nmax close duration reached: {args.max_close_duration:.2f}s")
            break
        if all_goals_reached:
            break

    print()
    return extract_features(trial)


def collect_hover(
    hand: RH56F2Hand,
    trial: TouchTrial,
    args: argparse.Namespace,
) -> None:
    started_at = time.monotonic()
    interval_s = 1.0 / args.hover_rate_hz
    deadline = started_at + args.hover_duration
    missed = 0
    while time.monotonic() < deadline:
        try:
            frame = read_touch_frame(hand, started_at, args)
        except RuntimeError as exc:
            missed += 1
            print(f"\r[warn] hover read missed={missed}: {exc}", end="", flush=True)
            time.sleep(interval_s)
            continue
        trial.hover_frames.append(frame)
        thumb_delta = sum(force_delta(frame.forces, trial.baseline_forces, name) for name in THUMB_NAMES)
        total_delta = sum(force_delta(frame.forces, trial.baseline_forces, name) for name in FINGER_NAMES)
        print(
            f"\rhover samples={len(trial.hover_frames)} missed={missed} "
            f"thumb_delta={thumb_delta:.0f} force_sum={total_delta:.0f}",
            end="",
            flush=True,
        )
        time.sleep(interval_s)
    print()


def collect_squeeze(
    hand: RH56F2Hand,
    trial: TouchTrial,
    args: argparse.Namespace,
) -> bool:
    trial.squeeze_hover_tail_count = max(0, int(args.squeeze_baseline_hover_samples))
    baseline_angles, baseline_forces, baseline_source = stable_hover_baseline(trial, args)
    trial.squeeze_baseline_angles = baseline_angles
    trial.squeeze_baseline_forces = baseline_forces
    trial.squeeze_baseline_source = baseline_source

    seek_started_at = time.monotonic()
    current_frame = read_touch_frame(hand, seek_started_at, args)
    middle_delta = middle_contact_delta(current_frame, trial)
    middle_target = float(current_frame.angles.get("middle", BALL_SAFE_CLOSED["middle"]))
    min_middle_target = max(
        BALL_SAFE_CLOSED["middle"] - args.squeeze_middle_seek_max_delta,
        middle_target - args.squeeze_middle_seek_max_delta,
    )

    while middle_delta < args.squeeze_middle_touch_threshold and middle_target > min_middle_target:
        middle_target = max(min_middle_target, middle_target - args.squeeze_middle_seek_step)
        hand.set_angles({"middle": middle_target})
        trial.squeeze_middle_seek_steps += 1
        time.sleep(args.squeeze_middle_seek_settle)
        current_frame = read_touch_frame(hand, seek_started_at, args)
        middle_delta = middle_contact_delta(current_frame, trial)
        print(
            "\rmiddle seek "
            f"steps={trial.squeeze_middle_seek_steps} "
            f"middle_contact_delta={middle_delta:.0f}/{args.squeeze_middle_touch_threshold:.0f}",
            end="",
            flush=True,
        )

    trial.squeeze_middle_contact_delta = middle_delta
    if middle_delta < args.squeeze_middle_touch_threshold:
        trial.squeeze_middle_ready = False
        print(
            "\n[warn] middle finger did not reach squeeze touch threshold; "
            "skipping effective squeeze."
        )
        return False
    trial.squeeze_middle_ready = True
    if trial.squeeze_middle_seek_steps:
        print()
    print(
        "middle ready: "
        f"contact_delta={middle_delta:.0f} "
        f"baseline={baseline_source}"
    )

    started_at = time.monotonic()
    interval_s = 1.0 / args.squeeze_rate_hz

    pre_deadline = started_at + args.squeeze_pre_duration
    missed = 0
    while time.monotonic() < pre_deadline:
        try:
            frame = read_touch_frame(hand, started_at, args)
        except RuntimeError as exc:
            missed += 1
            print(f"\r[warn] squeeze pre-read missed={missed}: {exc}", end="", flush=True)
            time.sleep(interval_s)
            continue
        trial.squeeze_frames.append(frame)
        time.sleep(interval_s)

    target = {}
    pre_frame = trial.squeeze_frames[-1] if trial.squeeze_frames else read_touch_frame(hand, started_at, args)
    for name in CORE_GRASP_FINGERS:
        current = float(pre_frame.angles.get(name, BALL_SAFE_CLOSED[name]))
        target[name] = max(BALL_SAFE_CLOSED[name] - args.squeeze_delta, current - args.squeeze_delta)
    hand.set_angles(target)
    trial.squeeze_command_sample_index = len(trial.squeeze_frames)
    print(
        "squeeze target: "
        + " ".join(f"{name}={target[name]:.0f}" for name in CORE_GRASP_FINGERS)
    )

    deadline = time.monotonic() + args.squeeze_duration
    while time.monotonic() < deadline:
        try:
            frame = read_touch_frame(hand, started_at, args)
        except RuntimeError as exc:
            missed += 1
            print(f"\r[warn] squeeze read missed={missed}: {exc}", end="", flush=True)
            time.sleep(interval_s)
            continue
        trial.squeeze_frames.append(frame)
        core_deltas = [
            force_delta(frame.forces, trial.squeeze_baseline_forces, name)
            for name in CORE_GRASP_FINGERS
        ]
        thumb_delta = sum(
            force_delta(frame.forces, trial.squeeze_baseline_forces, name)
            for name in THUMB_NAMES
        )
        total_delta = sum(core_deltas)
        print(
            f"\rsqueeze samples={len(trial.squeeze_frames)} missed={missed} "
            f"middle_delta={core_deltas[1]:.0f} thumb_delta={thumb_delta:.0f} core_sum={total_delta:.0f}",
            end="",
            flush=True,
        )
        if max(core_deltas, default=0.0) >= args.squeeze_max_force_delta:
            print("\nsqueeze max force reached; stopping squeeze.")
            break
        time.sleep(interval_s)
    print()
    return True


def collect_one_trial(
    piper: object,
    hand: RH56F2Hand,
    label: str,
    trial_id: str,
    repeat_index: int,
    args: argparse.Namespace,
    model: dict[str, object] | None,
) -> bool:
    print(f"\n[{repeat_index + 1}/{args.repeats}] current-pose direct MOVEP lift sample")
    configure_hand(hand, args.hand_speed, args.hand_force)
    if args.open_first:
        set_hand_pose(hand, BALL_READY_OPEN, "open before grasp")
        time.sleep(args.open_settle)

    lower_pose = selected_lower_pose(piper, args)
    if args.grab_pose is not None or args.grab_xyz is not None:
        print(f"moving to lower grasp pose: {pose_mm_deg(lower_pose)}")
        if not send_movep_checked(
            piper,
            lower_pose,
            args,
            args.grab_duration,
            "grasp pose MOVE_P",
            require_reached=False,
        ):
            raise RuntimeError("grasp pose MOVE_P failed")
        time.sleep(0.2)

    baseline_started = time.monotonic()
    baseline_frame = read_touch_frame(hand, baseline_started, args)
    lower_pose = selected_lower_pose(piper, args)
    lift_pose = list(lower_pose)
    lift_pose[2] += int(round(args.lift_height_mm * 1000.0))
    drop_pose = list(args.drop_pose) if args.drop_pose is not None else lower_pose
    print(f"lower pose: {pose_mm_deg(lower_pose)}")
    print(f"lift pose:  {pose_mm_deg(lift_pose)}")
    if args.drop_pose is not None:
        print(f"drop pose:  {pose_mm_deg(drop_pose)}")

    trial = TouchTrial(
        label=label,
        trial_id=trial_id,
        repeat_index=repeat_index,
        baseline_angles=baseline_frame.angles,
        baseline_forces=baseline_frame.forces,
        contact_threshold=args.contact_threshold,
        notes=args.notes,
    )
    row = close_and_record(hand, trial, args)
    active_contacts = float(row["active_contact_count"])
    if active_contacts == 0.0:
        print("[warn] no contact passed the threshold; not enough tactile information.")

    lift_skipped = False
    if active_contacts < args.min_lift_contacts and not args.force_lift:
        lift_skipped = True
        print(
            f"[warn] active_contact_count={active_contacts:.0f} < "
            f"{args.min_lift_contacts:g}; skipping lift. Use --force-lift to override."
        )
    else:
        if not send_movep_checked(
            piper,
            lift_pose,
            args,
            args.lift_duration,
            "lift MOVE_P",
            require_reached=False,
        ):
            raise RuntimeError("lift MOVE_P failed")
        if args.lift_settle > 0:
            time.sleep(args.lift_settle)
        collect_hover(hand, trial, args)
        if args.squeeze_test:
            collect_squeeze(hand, trial, args)
        if not send_movep_checked(
            piper,
            drop_pose,
            args,
            args.lower_duration,
            "drop MOVE_P" if args.drop_pose is not None else "lower MOVE_P",
            require_reached=False,
        ):
            row = extract_features(trial)
            append_feature_row(args.output, row)
            refresh_sample_dashboard(args)
            print(
                "[warn] drop/lower MOVE_P failed after retries; sample saved, "
                "hand left closed and arm may still be at lift pose."
            )
            return False

    if args.open_at_end:
        set_hand_pose(
            hand,
            BALL_READY_OPEN,
            "open at drop pose" if args.drop_pose is not None else "open at lower pose",
        )
        time.sleep(args.open_settle)

    row = extract_features(trial)
    if lift_skipped and not args.save_skipped:
        print("not saved: lift was skipped because contact was too weak.")
        return True
    append_feature_row(args.output, row)
    refresh_sample_dashboard(args)
    print(
        "saved: "
        f"label={label} "
        f"active={float(row['active_contact_count']):.0f} "
        f"force_sum={float(row['final_force_delta_sum']):.1f} "
        f"hover_samples={row['hover_sample_count'] or 0} "
        f"hover_thumb_mean={row['hover_thumb_force_delta_mean'] or '-'}"
    )
    if model is not None:
        result = predict_row(row, model)
        print(
            f"prediction={result['label']} "
            f"confidence={float(result['confidence']):.2f} "
            f"distance={float(result['distance']):.2f}"
        )
    return True


def main() -> int:
    args = parse_args()
    if args.label.upper() == "C" and args.squeeze_test:
        print("[warn] label C does not need squeeze; disabling --squeeze-test to avoid dropping it.")
        args.squeeze_test = False
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    for name in ("rate_hz", "hover_rate_hz", "grab_duration", "lift_duration", "lower_duration", "hover_duration"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.lift_height_mm <= 0:
        raise SystemExit("--lift-height-mm must be positive")
    if not 0 <= args.speed <= 100:
        raise SystemExit("--speed must be between 0 and 100")
    if args.movep_retries <= 0:
        raise SystemExit("--movep-retries must be positive")
    if args.hand_read_retries <= 0 or args.hand_read_retry_delay < 0:
        raise SystemExit("--hand-read-retries must be positive and --hand-read-retry-delay must be >= 0")
    if args.max_close_duration <= 0 or args.hold_after_contact < 0:
        raise SystemExit("--max-close-duration must be positive and --hold-after-contact must be >= 0")
    if args.squeeze_delta <= 0 or args.squeeze_pre_duration < 0 or args.squeeze_duration <= 0 or args.squeeze_rate_hz <= 0:
        raise SystemExit("--squeeze-delta, --squeeze-duration, and --squeeze-rate-hz must be positive; --squeeze-pre-duration must be >= 0")
    if args.squeeze_max_force_delta <= 0:
        raise SystemExit("--squeeze-max-force-delta must be positive")
    if args.squeeze_baseline_hover_samples <= 0:
        raise SystemExit("--squeeze-baseline-hover-samples must be positive")
    if (
        args.squeeze_middle_touch_threshold <= 0
        or args.squeeze_middle_seek_step <= 0
        or args.squeeze_middle_seek_max_delta <= 0
        or args.squeeze_middle_seek_settle < 0
    ):
        raise SystemExit("middle squeeze seek parameters must be positive")

    if args.grab_xyz is not None and args.grab_pose is not None:
        raise SystemExit("Use only one of --grab-xyz or --grab-pose")

    print("RH56F2 current-pose lift/hover sample collection")
    if args.grab_xyz is None and args.grab_pose is None:
        print("Move the arm to the lower grasp/drop pose before starting.")
    else:
        print("The script will move to the configured lower grasp pose before closing.")
    print("This script uses direct Piper MOVE_P: wait_for_movep_ready() -> send_movep_for().")
    print("Lift/lower only changes ee.z around the selected lower grasp pose.")
    if args.grab_xyz is not None:
        print(f"grab XYZ(m): {tuple(args.grab_xyz)} with current orientation")
    if args.grab_pose is not None:
        print(f"grab pose: {pose_mm_deg(args.grab_pose)}")
    if args.drop_pose is not None:
        print(f"drop pose: {pose_mm_deg(args.drop_pose)}")
    print(f"can={args.can} hand={args.hand_port} label={args.label} output={args.output}")
    if not args.yes:
        confirm = input("Type BALL_LIFT to connect, close, lift, hover, lower, and open: ").strip()
        if confirm != "BALL_LIFT":
            print("Aborted before connecting.")
            return 0

    model = load_model(args.predict_model) if args.predict_model is not None else None
    piper = None
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
    try:
        piper = connect_piper(args)
        wait_for_real_feedback(piper, args.feedback_timeout)
        if not enable_all(piper, args.feedback_timeout):
            raise RuntimeError("Piper did not enable; no motion command was sent.")
        if not wait_for_movep_ready(piper, args.speed, args.feedback_timeout):
            if not recover_movep_control(piper, args, "initial MOVE_P ready"):
                raise RuntimeError("MOVE_P mode was not ready; no target command was sent.")
        hand.connect()

        for repeat in range(args.repeats):
            if not collect_one_trial(piper, hand, args.label, trial_id, repeat, args, model):
                print("[warn] stopping repeats because the arm did not return to the lower pose.")
                break
            if repeat + 1 < args.repeats:
                print("Reset/place the ball if needed before the next lift.")
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
