#!/usr/bin/env python3
"""Ball classification adapter for the claw-machine pick cycle.

This module assumes the arm has already grasped and lifted the object. It only
reads RH56F2 tactile feedback and optionally performs the A/B squeeze check.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from .common import (
    BALL_SAFE_CLOSED,
    CORE_GRASP_FINGERS,
    FINGER_NAMES,
    THUMB_NAMES,
    TouchFrame,
    TouchTrial,
    ab_shape_decision,
    append_feature_row,
    average_series,
    detect_missed_grasp,
    extract_features,
    force_delta,
    load_model,
    normalized_series_distance,
    now_trial_id,
    parse_series,
    predict_row,
    read_feature_rows,
)
from .visualize_live import (
    render_dashboard_from_csv,
    render_dashboard_preview_from_csv,
    render_pending_dashboard_from_csv,
)


DEFAULT_MODEL = Path(__file__).with_name("model_with_newB.json")
DEFAULT_OUTPUT = Path(__file__).with_name("claw_predictions.csv")
DEFAULT_DASHBOARD = Path(__file__).with_name("live_dashboard.html")
DEFAULT_REFERENCE_SAMPLES = Path(__file__).with_name("samples_with_newB.csv")


@dataclass
class BallClassifierConfig:
    model: Path = DEFAULT_MODEL
    output: Path = DEFAULT_OUTPUT
    visual_reference_samples: Path = DEFAULT_REFERENCE_SAMPLES
    contact_threshold: float = 70.0
    min_active_contacts: float | None = None
    min_hover_samples: float | None = None
    hover_duration: float = 1.5
    hover_rate_hz: float = 10.0
    hand_read_retries: int = 5
    hand_read_retry_delay: float = 0.03
    tactile_data: bool = True
    tactile_required: bool = False
    squeeze_delta: float = 40.0
    squeeze_pre_duration: float = 0.4
    squeeze_duration: float = 3.0
    squeeze_rate_hz: float = 20.0
    squeeze_max_force_delta: float = 900.0
    squeeze_baseline_hover_samples: int = 10
    squeeze_middle_touch_threshold: float = 80.0
    squeeze_middle_seek_step: float = 15.0
    squeeze_middle_seek_max_delta: float = 160.0
    squeeze_middle_seek_settle: float = 0.08
    squeeze_test: bool = False
    ab_squeeze_test: bool = False
    ab_squeeze_threshold: float = 190.0
    ab_squeeze_a_standard: float = 238.0
    ab_squeeze_b_standard: float = 142.5
    ab_squeeze_mode: str = "friction"
    ab_squeeze_reference_samples: Path = DEFAULT_REFERENCE_SAMPLES
    low_confidence_c_squeeze_threshold: float = 0.0
    ab_friction_finger: str = "middle"
    ab_friction_feature: str = "last"
    ab_friction_threshold: float = 0.1464
    ab_friction_a_direction: str = ">="
    ab_shape_late_slope_threshold: float = -5.0
    ab_shape_rebound_threshold: float = 15.0
    ab_shape_peak_pos_threshold: float = 0.5
    ab_shape_min_a_score: int = 2
    ab_proximity_assist: bool = True
    ab_proximity_index_force_threshold: float = 70.0
    ab_proximity_thumb_threshold: float = 169619.0
    ab_proximity_a_direction: str = "<="
    ab_proximity_min_samples: float = 5.0
    bc_proximity_assist: bool = True
    bc_proximity_thumb_threshold: float = 180000.0
    bc_proximity_middle_threshold: float = 100000.0
    notes: str = "claw_machine"


def _number(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def quality_number(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        return 0.0
    return float(value)


def squeeze_ab_label(middle_force: float, args: argparse.Namespace) -> tuple[str, float]:
    label = "A" if middle_force >= args.ab_squeeze_threshold else "B"
    half_gap = abs(args.ab_squeeze_a_standard - args.ab_squeeze_b_standard) / 2.0
    confidence = min(1.0, abs(middle_force - args.ab_squeeze_threshold) / max(1.0, half_gap))
    return label, confidence


def squeeze_series_from_row(row: dict[str, object]) -> tuple[str, list[float]]:
    middle = parse_series(str(row.get("squeeze_middle_force_delta_series") or ""))
    if middle:
        return "middle", middle
    total = parse_series(str(row.get("squeeze_force_delta_sum_series") or ""))
    return "core_sum", total


def _friction_series_from_row(row: dict[str, object], finger: str) -> list[float]:
    return parse_series(str(row.get(f"tactile_friction_{finger}_series") or ""))


def _friction_feature_value(series: list[float], feature: str) -> float:
    if not series:
        return 0.0
    if feature == "mean":
        return sum(series) / len(series)
    if feature == "max":
        return max(series)
    if feature == "late_slope":
        late = series[len(series) // 2 :]
        return late[-1] - late[0] if len(late) >= 2 else 0.0
    return series[-1]


def friction_ab_label(row: dict[str, object], args: argparse.Namespace) -> tuple[str, float, float, list[float]]:
    series = _friction_series_from_row(row, args.ab_friction_finger)
    if not series:
        return "", 0.0, 0.0, []
    value = _friction_feature_value(series, args.ab_friction_feature)
    if args.ab_friction_a_direction == ">=":
        label = "A" if value >= args.ab_friction_threshold else "B"
    else:
        label = "A" if value <= args.ab_friction_threshold else "B"
    confidence = min(
        1.0,
        abs(value - args.ab_friction_threshold) / max(0.05, abs(args.ab_friction_threshold)),
    )
    return label, confidence, value, series


def init_ab_proximity_fields(row: dict[str, object], args: argparse.Namespace) -> None:
    row["ab_proximity_assist_triggered"] = 0.0
    row["ab_proximity_index_force_threshold"] = float(args.ab_proximity_index_force_threshold)
    row["ab_proximity_thumb_threshold"] = float(args.ab_proximity_thumb_threshold)
    row["ab_proximity_a_direction"] = args.ab_proximity_a_direction
    row["ab_proximity_thumb_mean"] = row.get("hover_proximity_thumb_mean", "")
    row["ab_proximity_index_contacted"] = row.get("contacted_index", "")
    row["ab_proximity_index_final_force_delta"] = row.get("final_force_delta_index", "")
    row["ab_proximity_index_hover_force_delta_mean"] = row.get("hover_force_delta_index_mean", "")
    row["ab_proximity_label"] = ""
    row["bc_proximity_assist_triggered"] = 0.0
    row["bc_proximity_thumb_threshold"] = float(args.bc_proximity_thumb_threshold)
    row["bc_proximity_middle_threshold"] = float(args.bc_proximity_middle_threshold)
    row["bc_proximity_thumb_mean"] = row.get("hover_proximity_thumb_mean", "")
    row["bc_proximity_middle_mean"] = row.get("hover_proximity_middle_mean", "")
    row["bc_proximity_index_mean"] = row.get("hover_proximity_index_mean", "")
    row["bc_proximity_label"] = ""


def _proximity_ab_label(
    row: dict[str, object],
    stage1_label: str,
    args: argparse.Namespace,
) -> tuple[str, float, dict[str, float]]:
    if not args.ab_proximity_assist or stage1_label not in {"A", "B"}:
        return "", 0.0, {}

    sample_count = quality_number(row, "hover_proximity_sample_count")
    thumb_mean = quality_number(row, "hover_proximity_thumb_mean")
    index_contacted = quality_number(row, "contacted_index")
    index_final = quality_number(row, "final_force_delta_index")
    index_hover = quality_number(row, "hover_force_delta_index_mean")
    low_index = (
        index_contacted < 0.5
        or index_final <= args.ab_proximity_index_force_threshold
        or index_hover <= args.ab_proximity_index_force_threshold
    )
    has_proximity = sample_count >= args.ab_proximity_min_samples and thumb_mean > 0.0
    stats = {
        "sample_count": sample_count,
        "thumb_mean": thumb_mean,
        "index_contacted": index_contacted,
        "index_final": index_final,
        "index_hover": index_hover,
        "low_index": float(low_index),
        "has_proximity": float(has_proximity),
    }
    if not low_index or not has_proximity:
        return "", 0.0, stats

    if args.ab_proximity_a_direction == ">=":
        label = "A" if thumb_mean >= args.ab_proximity_thumb_threshold else "B"
    else:
        label = "A" if thumb_mean <= args.ab_proximity_thumb_threshold else "B"
    confidence = min(
        1.0,
        abs(thumb_mean - args.ab_proximity_thumb_threshold)
        / max(1.0, args.ab_proximity_thumb_threshold * 0.5),
    )
    return label, confidence, stats


def _proximity_c_to_b_label(
    row: dict[str, object],
    stage1_label: str,
    args: argparse.Namespace,
) -> tuple[str, float, dict[str, float]]:
    if not args.bc_proximity_assist or stage1_label != "C":
        return "", 0.0, {}

    sample_count = quality_number(row, "hover_proximity_sample_count")
    thumb_mean = quality_number(row, "hover_proximity_thumb_mean")
    middle_mean = quality_number(row, "hover_proximity_middle_mean")
    index_mean = quality_number(row, "hover_proximity_index_mean")
    stats = {
        "sample_count": sample_count,
        "thumb_mean": thumb_mean,
        "middle_mean": middle_mean,
        "index_mean": index_mean,
    }
    if sample_count < args.ab_proximity_min_samples:
        return "", 0.0, stats
    if thumb_mean < args.bc_proximity_thumb_threshold:
        return "", 0.0, stats
    if middle_mean < args.bc_proximity_middle_threshold:
        return "", 0.0, stats

    thumb_margin = (thumb_mean - args.bc_proximity_thumb_threshold) / max(1.0, args.bc_proximity_thumb_threshold)
    middle_margin = (middle_mean - args.bc_proximity_middle_threshold) / max(1.0, args.bc_proximity_middle_threshold)
    confidence = min(1.0, max(0.15, (thumb_margin + middle_margin) / 2.0))
    return "B", confidence, stats


def apply_proximity_assist(
    row: dict[str, object],
    stage1_label: str,
    stage1_confidence: float,
    stage1_distance: float,
    args: argparse.Namespace,
) -> None:
    ab_label, ab_confidence, ab_stats = _proximity_ab_label(row, stage1_label, args)
    if ab_stats:
        row["ab_proximity_thumb_mean"] = ab_stats["thumb_mean"]
        row["ab_proximity_index_contacted"] = ab_stats["index_contacted"]
        row["ab_proximity_index_final_force_delta"] = ab_stats["index_final"]
        row["ab_proximity_index_hover_force_delta_mean"] = ab_stats["index_hover"]
    if ab_label:
        row["prediction_status"] = "ab_proximity_assist"
        row["predicted_label"] = ab_label
        row["prediction_confidence"] = ab_confidence
        row["prediction_distance"] = ""
        row["ab_proximity_assist_triggered"] = 1.0
        row["ab_proximity_label"] = ab_label
        print(
            "ab_proximity_assist "
            f"index_contacted={row.get('ab_proximity_index_contacted')} "
            f"index_final={row.get('ab_proximity_index_final_force_delta')} "
            f"index_hover={row.get('ab_proximity_index_hover_force_delta_mean')} "
            f"thumb_mean={row.get('ab_proximity_thumb_mean')} "
            f"rule: A if thumb_mean {args.ab_proximity_a_direction} "
            f"{args.ab_proximity_thumb_threshold:.1f} else B "
            f"final_prediction={ab_label} "
            f"confidence={ab_confidence:.2f}"
        )
        return

    bc_label, bc_confidence, bc_stats = _proximity_c_to_b_label(row, stage1_label, args)
    if bc_stats:
        row["bc_proximity_thumb_mean"] = bc_stats["thumb_mean"]
        row["bc_proximity_middle_mean"] = bc_stats["middle_mean"]
        row["bc_proximity_index_mean"] = bc_stats["index_mean"]
    if bc_label:
        row["prediction_status"] = "bc_proximity_assist"
        row["predicted_label"] = bc_label
        row["prediction_confidence"] = bc_confidence
        row["prediction_distance"] = ""
        row["bc_proximity_assist_triggered"] = 1.0
        row["bc_proximity_label"] = bc_label
        print(
            "bc_proximity_assist "
            f"thumb_mean={row.get('bc_proximity_thumb_mean')} "
            f"middle_mean={row.get('bc_proximity_middle_mean')} "
            f"index_mean={row.get('bc_proximity_index_mean')} "
            f"rule: B if thumb_mean >= {args.bc_proximity_thumb_threshold:.1f} "
            f"and middle_mean >= {args.bc_proximity_middle_threshold:.1f} "
            f"final_prediction={bc_label} "
            f"confidence={bc_confidence:.2f}"
        )
        return

    row["prediction_status"] = "ok"
    row["predicted_label"] = stage1_label
    row["prediction_confidence"] = stage1_confidence
    row["prediction_distance"] = stage1_distance


def load_ab_squeeze_references(path: Path) -> dict[str, object]:
    references: dict[str, object] = {"series_key": "middle", "A": [], "B": []}
    if not path.exists():
        return references
    rows = read_feature_rows(path)
    grouped: dict[str, list[list[float]]] = {"A": [], "B": []}
    for row in rows:
        notes = (row.get("notes") or "").lower()
        if "void" in notes or "bad_grasp" in notes:
            continue
        label = row.get("label", "")
        if label not in grouped:
            continue
        series = parse_series(row.get("squeeze_middle_force_delta_series"))
        if not series:
            series = parse_series(row.get("squeeze_force_delta_sum_series"))
        if series:
            grouped[label].append(series)
    references["A"] = average_series(grouped["A"])
    references["B"] = average_series(grouped["B"])
    return references


def squeeze_ab_curve_label(
    row: dict[str, object],
    references: dict[str, object],
    args: argparse.Namespace,
) -> tuple[str, float, dict[str, float]]:
    _, current = squeeze_series_from_row(row)
    ref_a = list(references.get("A") or [])
    ref_b = list(references.get("B") or [])
    if not current or not ref_a or not ref_b:
        return "", 0.0, {"A": 0.0, "B": 0.0}
    dist_a = normalized_series_distance(current, ref_a)
    dist_b = normalized_series_distance(current, ref_b)
    label = "A" if dist_a <= dist_b else "B"
    confidence = abs(dist_a - dist_b) / max(dist_a, dist_b, 1e-9)
    return label, confidence, {"A": dist_a, "B": dist_b}


def _read_touch_frame(hand: object, started_at: float, args: argparse.Namespace) -> TouchFrame:
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


def _average_frame_values(frames: list[TouchFrame], attr: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in FINGER_NAMES:
        values = [float(getattr(frame, attr).get(name, 0.0)) for frame in frames]
        result[name] = sum(values) / len(values) if values else 0.0
    return result


def _stable_hover_baseline(
    trial: TouchTrial,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, float], str]:
    count = max(1, int(args.squeeze_baseline_hover_samples))
    frames = trial.hover_frames[-count:]
    if frames:
        return (
            _average_frame_values(frames, "angles"),
            _average_frame_values(frames, "forces"),
            f"hover_last_{len(frames)}",
        )
    return dict(trial.baseline_angles), dict(trial.baseline_forces), "trial_baseline"


def _middle_contact_delta(frame: TouchFrame, trial: TouchTrial) -> float:
    return force_delta(frame.forces, trial.baseline_forces, "middle")


def _collect_hover(hand: object, trial: TouchTrial, args: argparse.Namespace) -> None:
    started_at = time.monotonic()
    interval_s = 1.0 / args.hover_rate_hz
    deadline = started_at + args.hover_duration
    missed = 0
    while time.monotonic() < deadline:
        try:
            frame = _read_touch_frame(hand, started_at, args)
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


def _collect_squeeze(hand: object, trial: TouchTrial, args: argparse.Namespace) -> bool:
    trial.squeeze_hover_tail_count = max(0, int(args.squeeze_baseline_hover_samples))
    baseline_angles, baseline_forces, baseline_source = _stable_hover_baseline(trial, args)
    trial.squeeze_baseline_angles = baseline_angles
    trial.squeeze_baseline_forces = baseline_forces
    trial.squeeze_baseline_source = baseline_source

    seek_started_at = time.monotonic()
    current_frame = _read_touch_frame(hand, seek_started_at, args)
    middle_delta = _middle_contact_delta(current_frame, trial)
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
        current_frame = _read_touch_frame(hand, seek_started_at, args)
        middle_delta = _middle_contact_delta(current_frame, trial)
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
            frame = _read_touch_frame(hand, started_at, args)
        except RuntimeError as exc:
            missed += 1
            print(f"\r[warn] squeeze pre-read missed={missed}: {exc}", end="", flush=True)
            time.sleep(interval_s)
            continue
        trial.squeeze_frames.append(frame)
        time.sleep(interval_s)

    target = {}
    pre_frame = trial.squeeze_frames[-1] if trial.squeeze_frames else _read_touch_frame(hand, started_at, args)
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
            frame = _read_touch_frame(hand, started_at, args)
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


def _lift_args(config: BallClassifierConfig) -> argparse.Namespace:
    return argparse.Namespace(
        hover_duration=config.hover_duration,
        hover_rate_hz=config.hover_rate_hz,
        hand_read_retries=config.hand_read_retries,
        hand_read_retry_delay=config.hand_read_retry_delay,
        tactile_data=config.tactile_data,
        tactile_required=config.tactile_required,
        squeeze_delta=config.squeeze_delta,
        squeeze_pre_duration=config.squeeze_pre_duration,
        squeeze_duration=config.squeeze_duration,
        squeeze_rate_hz=config.squeeze_rate_hz,
        squeeze_max_force_delta=config.squeeze_max_force_delta,
        squeeze_baseline_hover_samples=config.squeeze_baseline_hover_samples,
        squeeze_middle_touch_threshold=config.squeeze_middle_touch_threshold,
        squeeze_middle_seek_step=config.squeeze_middle_seek_step,
        squeeze_middle_seek_max_delta=config.squeeze_middle_seek_max_delta,
        squeeze_middle_seek_settle=config.squeeze_middle_seek_settle,
        squeeze_test=config.squeeze_test,
        ab_squeeze_test=config.ab_squeeze_test,
        ab_squeeze_threshold=config.ab_squeeze_threshold,
        ab_squeeze_a_standard=config.ab_squeeze_a_standard,
        ab_squeeze_b_standard=config.ab_squeeze_b_standard,
        ab_squeeze_mode=config.ab_squeeze_mode,
        low_confidence_c_squeeze_threshold=config.low_confidence_c_squeeze_threshold,
        ab_friction_finger=config.ab_friction_finger,
        ab_friction_feature=config.ab_friction_feature,
        ab_friction_threshold=config.ab_friction_threshold,
        ab_friction_a_direction=config.ab_friction_a_direction,
        ab_shape_late_slope_threshold=config.ab_shape_late_slope_threshold,
        ab_shape_rebound_threshold=config.ab_shape_rebound_threshold,
        ab_shape_peak_pos_threshold=config.ab_shape_peak_pos_threshold,
        ab_shape_min_a_score=config.ab_shape_min_a_score,
        ab_proximity_assist=config.ab_proximity_assist,
        ab_proximity_index_force_threshold=config.ab_proximity_index_force_threshold,
        ab_proximity_thumb_threshold=config.ab_proximity_thumb_threshold,
        ab_proximity_a_direction=config.ab_proximity_a_direction,
        ab_proximity_min_samples=config.ab_proximity_min_samples,
        bc_proximity_assist=config.bc_proximity_assist,
        bc_proximity_thumb_threshold=config.bc_proximity_thumb_threshold,
        bc_proximity_middle_threshold=config.bc_proximity_middle_threshold,
    )


class HeldBallClassifier:
    """Two-stage classifier for an object already held by the claw machine."""

    def __init__(self, config: BallClassifierConfig):
        self.config = config
        self.model = load_model(config.model)
        self.min_active_contacts = (
            float(self.model.get("min_active_contacts", 0.0))
            if config.min_active_contacts is None
            else float(config.min_active_contacts)
        )
        self.min_hover_samples = (
            float(self.model.get("min_hover_samples", 0.0))
            if config.min_hover_samples is None
            else float(config.min_hover_samples)
        )
        self.ab_squeeze_references = load_ab_squeeze_references(config.ab_squeeze_reference_samples)

    def begin_trial(self, hand: object, repeat_index: int = 0) -> TouchTrial:
        args = _lift_args(self.config)
        baseline = _read_touch_frame(hand, 0.0, args)
        try:
            render_pending_dashboard_from_csv(
                self.config.output,
                DEFAULT_DASHBOARD,
                last=20,
                reference_samples=self.config.visual_reference_samples,
                notes="claw grasp in progress",
            )
        except Exception as exc:
            print(f"[warn] pending live dashboard refresh failed: {exc}")
        return TouchTrial(
            label="unknown",
            trial_id=now_trial_id(),
            repeat_index=repeat_index,
            baseline_angles=baseline.angles,
            baseline_forces=baseline.forces,
            contact_threshold=self.config.contact_threshold,
            notes=self.config.notes,
        )

    def record_grasp_frame(
        self,
        hand: object,
        trial: TouchTrial,
        started_at: float = 0.0,
    ) -> TouchFrame:
        args = _lift_args(self.config)
        frame = _read_touch_frame(hand, started_at, args)
        trial.frames.append(frame)
        return frame

    def record_observation_frame(
        self,
        trial: TouchTrial,
        observation: dict[str, object],
        started_at: float,
    ) -> None:
        angles = {
            name: float(observation.get(f"hand.{name}.pos", trial.baseline_angles.get(name, 0.0)))
            for name in FINGER_NAMES
        }
        forces = {
            name: float(observation.get(f"hand.{name}.force", trial.baseline_forces.get(name, 0.0)))
            for name in FINGER_NAMES
        }
        trial.frames.append(TouchFrame(time.monotonic() - started_at, angles, forces))

    def refresh_decision_dashboard(self, row: dict[str, object]) -> None:
        try:
            render_dashboard_preview_from_csv(
                self.config.output,
                DEFAULT_DASHBOARD,
                row,
                last=20,
                reference_samples=self.config.visual_reference_samples,
            )
        except Exception as exc:
            print(f"[warn] decision dashboard refresh failed: {exc}")

    def classify_held(self, hand: object, trial: TouchTrial) -> dict[str, object]:
        args = _lift_args(self.config)
        if not trial.frames:
            self.record_grasp_frame(hand, trial)

        _collect_hover(hand, trial, args)
        row = extract_features(trial)

        active_contacts = _number(row.get("active_contact_count"))
        hover_samples = _number(row.get("hover_sample_count"))
        if active_contacts < self.min_active_contacts or hover_samples < self.min_hover_samples:
            row["prediction_status"] = "skipped_quality"
            row["predicted_label"] = ""
            row["prediction_confidence"] = ""
            row["prediction_distance"] = ""
            self.refresh_decision_dashboard(row)
            append_feature_row(self.config.output, row)
            print(
                "ball_type=unknown "
                f"status=skipped_quality active={active_contacts:.0f}/{self.min_active_contacts:g} "
                f"hover_samples={hover_samples:.0f}/{self.min_hover_samples:g}"
            )
            return row

        miss_hit, miss_stats = detect_missed_grasp(row)
        row["miss_grasp_active_contact_count"] = miss_stats["active_contact_count"]
        row["miss_grasp_size_closure_mean"] = miss_stats["size_closure_mean"]
        row["miss_grasp_size_contact_angle_mean"] = miss_stats["size_contact_angle_mean"]
        row["miss_grasp_hover_sample_count"] = miss_stats["hover_sample_count"]
        row["miss_grasp_hover_thumb_force_delta_mean"] = miss_stats["hover_thumb_force_delta_mean"]
        row["miss_grasp_hover_force_delta_sum_mean"] = miss_stats["hover_force_delta_sum_mean"]
        row["miss_grasp_final_force_delta_sum"] = miss_stats["final_force_delta_sum"]
        row["miss_grasp_low_hover_after_lift"] = miss_stats["low_hover_after_lift"]
        row["miss_grasp_deep_empty_close"] = miss_stats["deep_empty_close"]
        row["ab_squeeze_triggered"] = 0.0
        row["ab_squeeze_threshold"] = float(args.ab_squeeze_threshold)
        row["ab_squeeze_a_standard"] = float(args.ab_squeeze_a_standard)
        row["ab_squeeze_b_standard"] = float(args.ab_squeeze_b_standard)
        row["ab_squeeze_middle_force_delta_max"] = ""
        row["ab_squeeze_curve_distance_a"] = ""
        row["ab_squeeze_curve_distance_b"] = ""
        row["ab_shape_late_slope"] = ""
        row["ab_shape_rebound"] = ""
        row["ab_shape_peak_pos"] = ""
        row["ab_shape_score_a"] = ""
        row["ab_friction_finger"] = args.ab_friction_finger
        row["ab_friction_feature"] = args.ab_friction_feature
        row["ab_friction_value"] = ""
        row["ab_friction_threshold"] = float(args.ab_friction_threshold)
        row["ab_friction_a_direction"] = args.ab_friction_a_direction
        init_ab_proximity_fields(row, args)
        if miss_hit:
            row["prediction_status"] = "miss_grasp"
            row["predicted_label"] = "NONE"
            row["prediction_confidence"] = 1.0
            row["prediction_distance"] = ""
            row["stage1_predicted_label"] = ""
            row["stage1_prediction_confidence"] = ""
            row["stage1_prediction_distance"] = ""
            self.refresh_decision_dashboard(row)
            append_feature_row(self.config.output, row)
            print(
                "ball_type=NONE "
                "stage1=- "
                "confidence=1.00 "
                "status=miss_grasp "
                f"active={miss_stats['active_contact_count']:.0f} "
                f"closure={miss_stats['size_closure_mean']:.1f} "
                f"angle={miss_stats['size_contact_angle_mean']:.1f} "
                f"hover={miss_stats['hover_sample_count']:.0f} "
                f"hover_sum={miss_stats['hover_force_delta_sum_mean']:.1f} "
                f"thumb={miss_stats['hover_thumb_force_delta_mean']:.1f} "
                f"force_sum={miss_stats['final_force_delta_sum']:.1f} "
                f"low_hover={miss_stats['low_hover_after_lift']:.0f}"
            )
            return row

        stage1 = predict_row(row, self.model)
        stage1_label = str(stage1["label"])
        row["stage1_predicted_label"] = stage1_label
        row["stage1_prediction_confidence"] = float(stage1["confidence"])
        row["stage1_prediction_distance"] = float(stage1["distance"])
        print(
            f"stage1_prediction={stage1_label} "
            f"confidence={float(stage1['confidence']):.2f} "
            f"distance={float(stage1['distance']):.2f}"
        )

        low_confidence_c = (
            stage1_label == "C"
            and float(stage1["confidence"]) < args.low_confidence_c_squeeze_threshold
        )
        squeeze_triggered = (
            args.squeeze_test
            or (args.ab_squeeze_test and stage1_label in {"A", "B"})
            or low_confidence_c
        )
        if squeeze_triggered:
            if stage1_label in {"A", "B"} and args.ab_squeeze_test:
                reason = "A/B hardness check"
            elif low_confidence_c:
                reason = (
                    "low-confidence C fallback "
                    f"({float(stage1['confidence']):.2f} < {args.low_confidence_c_squeeze_threshold:.2f})"
                )
            else:
                reason = "forced squeeze test"
            print(f"squeeze test triggered: {reason}")
            _collect_squeeze(hand, trial, args)
            row = extract_features(trial)
            row["stage1_predicted_label"] = stage1_label
            row["stage1_prediction_confidence"] = float(stage1["confidence"])
            row["stage1_prediction_distance"] = float(stage1["distance"])
            row["ab_squeeze_triggered"] = 1.0
            row["ab_squeeze_threshold"] = float(args.ab_squeeze_threshold)
            row["ab_squeeze_a_standard"] = float(args.ab_squeeze_a_standard)
            row["ab_squeeze_b_standard"] = float(args.ab_squeeze_b_standard)
            row["ab_squeeze_middle_force_delta_max"] = quality_number(row, "squeeze_middle_force_delta_max")
            row["ab_friction_finger"] = args.ab_friction_finger
            row["ab_friction_feature"] = args.ab_friction_feature
            row["ab_friction_value"] = ""
            row["ab_friction_threshold"] = float(args.ab_friction_threshold)
            row["ab_friction_a_direction"] = args.ab_friction_a_direction
            init_ab_proximity_fields(row, args)
            if (stage1_label in {"A", "B"} and args.ab_squeeze_test) or low_confidence_c:
                middle_force = quality_number(row, "squeeze_middle_force_delta_max")
                middle_ready = quality_number(row, "squeeze_middle_ready") >= 0.5
                row["ab_squeeze_curve_distance_a"] = ""
                row["ab_squeeze_curve_distance_b"] = ""
                row["ab_shape_late_slope"] = ""
                row["ab_shape_rebound"] = ""
                row["ab_shape_peak_pos"] = ""
                row["ab_shape_score_a"] = ""
                if not middle_ready:
                    final_label = stage1_label
                    ab_confidence = float(stage1["confidence"])
                    row["prediction_status"] = (
                        "low_confidence_c_squeeze_no_middle_contact"
                        if low_confidence_c
                        else "ab_squeeze_no_middle_contact"
                    )
                    print("ab_squeeze skipped: middle finger did not confirm contact.")
                elif args.ab_squeeze_mode == "friction":
                    final_label, ab_confidence, friction_value, friction_series = friction_ab_label(row, args)
                    row["ab_friction_value"] = friction_value if friction_series else ""
                    if not final_label:
                        final_label = stage1_label
                        ab_confidence = float(stage1["confidence"])
                        row["prediction_status"] = "ab_friction_no_touchdata"
                        print("ab_friction skipped: no touchData friction series.")
                elif args.ab_squeeze_mode == "shape":
                    series_name, series = squeeze_series_from_row(row)
                    if series_name == "middle" and series:
                        final_label, ab_confidence, shape_metrics = ab_shape_decision(
                            series,
                            late_slope_threshold=args.ab_shape_late_slope_threshold,
                            rebound_threshold=args.ab_shape_rebound_threshold,
                            peak_pos_threshold=args.ab_shape_peak_pos_threshold,
                            min_a_score=args.ab_shape_min_a_score,
                        )
                        row["ab_shape_late_slope"] = shape_metrics["curve_late_slope"]
                        row["ab_shape_rebound"] = shape_metrics["curve_rebound"]
                        row["ab_shape_peak_pos"] = shape_metrics["curve_peak_pos"]
                        row["ab_shape_score_a"] = shape_metrics["ab_shape_score_a"]
                    else:
                        final_label, ab_confidence = squeeze_ab_label(middle_force, args)
                        row["prediction_status"] = "ab_squeeze_no_middle_series"
                        print("ab_squeeze shape fallback: no middle squeeze curve, using threshold.")
                elif args.ab_squeeze_mode == "curve":
                    final_label, ab_confidence, ab_distances = squeeze_ab_curve_label(
                        row,
                        self.ab_squeeze_references,
                        args,
                    )
                    row["ab_squeeze_curve_distance_a"] = ab_distances["A"]
                    row["ab_squeeze_curve_distance_b"] = ab_distances["B"]
                    if not final_label:
                        final_label = stage1_label
                        ab_confidence = float(stage1["confidence"])
                        row["prediction_status"] = "ab_squeeze_no_reference"
                        print("ab_squeeze skipped: no labelled squeeze reference curves yet.")
                else:
                    final_label, ab_confidence = squeeze_ab_label(middle_force, args)
                if row.get("prediction_status") not in {"ab_squeeze_no_reference", "ab_squeeze_no_middle_series", "ab_friction_no_touchdata"}:
                    if row.get("prediction_status") not in {
                        "ab_squeeze_no_middle_contact",
                        "low_confidence_c_squeeze_no_middle_contact",
                    }:
                        if low_confidence_c:
                            row["prediction_status"] = (
                                "low_confidence_c_friction"
                                if args.ab_squeeze_mode == "friction"
                                else "low_confidence_c_squeeze"
                            )
                        else:
                            row["prediction_status"] = "ab_friction" if args.ab_squeeze_mode == "friction" else "ab_squeeze"
                row["predicted_label"] = final_label
                row["prediction_confidence"] = ab_confidence
                row["prediction_distance"] = ""
                print(
                    f"ab_squeeze middle_force={middle_force:.1f} "
                    f"mode={args.ab_squeeze_mode} "
                    f"final_prediction={final_label} "
                    f"confidence={ab_confidence:.2f}"
                )
                if args.ab_squeeze_mode == "friction":
                    print(
                        "ab_friction "
                        f"{row.get('ab_friction_finger')}.{row.get('ab_friction_feature')}="
                        f"{row.get('ab_friction_value') or '-'} "
                        f"rule: A if value {row.get('ab_friction_a_direction')} "
                        f"{row.get('ab_friction_threshold')} else B"
                    )
                if args.ab_squeeze_mode == "shape":
                    print(
                        "ab_shape "
                        f"late_slope={row.get('ab_shape_late_slope') or '-'} "
                        f"rebound={row.get('ab_shape_rebound') or '-'} "
                        f"peak_pos={row.get('ab_shape_peak_pos') or '-'} "
                        f"score_A={row.get('ab_shape_score_a') or '-'}"
                    )
            else:
                apply_proximity_assist(
                    row,
                    stage1_label,
                    float(stage1["confidence"]),
                    float(stage1["distance"]),
                    args,
                )
        else:
            apply_proximity_assist(
                row,
                stage1_label,
                float(stage1["confidence"]),
                float(stage1["distance"]),
                args,
            )

        self.refresh_decision_dashboard(row)
        append_feature_row(self.config.output, row)

        friction_text = ""
        if row.get("ab_squeeze_triggered") == 1.0:
            value = row.get("ab_friction_value")
            friction_text = (
                f" friction_{self.config.ab_friction_finger}_{self.config.ab_friction_feature}="
                f"{value if value != '' else '-'} threshold={self.config.ab_friction_threshold:g}"
            )
        print(
            f"ball_type={row.get('predicted_label')} "
            f"stage1={stage1_label} "
            f"confidence={float(row.get('prediction_confidence') or 0.0):.2f} "
            f"status={row.get('prediction_status')} "
            f"distance={row.get('prediction_distance') or '-'}"
            f"{friction_text}"
        )
        return row
