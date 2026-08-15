#!/usr/bin/env python3
"""Lift, hover, collect RH56F2 tactile data, then classify one live ball."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from .common import (
        BALL_READY_OPEN,
        TouchTrial,
        ab_shape_decision,
        append_feature_row,
        average_series,
        detect_missed_grasp,
        extract_features,
        load_model,
        normalized_series_distance,
        numeric_or_none,
        now_trial_id,
        parse_series,
        read_feature_rows,
        predict_row,
    )
    from . import collect_lift_samples as lift
    from .visualize_live import render_dashboard_from_csv, render_dashboard_preview_from_csv, render_pending_dashboard_from_csv
except ImportError:  # Allow: python3 predict_live.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (  # type: ignore
        BALL_READY_OPEN,
        TouchTrial,
        ab_shape_decision,
        append_feature_row,
        average_series,
        detect_missed_grasp,
        extract_features,
        load_model,
        normalized_series_distance,
        numeric_or_none,
        now_trial_id,
        parse_series,
        read_feature_rows,
        predict_row,
    )
    import collect_lift_samples as lift  # type: ignore
    from visualize_live import render_dashboard_from_csv, render_dashboard_preview_from_csv, render_pending_dashboard_from_csv  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("model.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("live_predictions.csv"))
    parser.add_argument("--visual-output", type=Path, default=Path(__file__).with_name("live_dashboard.html"))
    parser.add_argument("--visual-reference-samples", type=Path, default=Path(__file__).with_name("samples.csv"))
    parser.add_argument("--visual-last", type=int, default=20)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True)
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
        help="Fail a frame if touchData cannot be read. Default keeps forceAct prediction alive.",
    )
    parser.add_argument("--speed", type=int, default=8, help="Piper MOVE_P speed.")
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--feedback-timeout", type=float, default=8.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--rpy-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--movep-retries", type=int, default=3)
    parser.add_argument(
        "--grab-xyz",
        type=lift.parse_xyz_m,
        default=None,
        help="Optional lower grasp X,Y,Z in metres. Keeps current RX/RY/RZ.",
    )
    parser.add_argument(
        "--grab-pose",
        type=lift.parse_pose_mm_deg,
        default=None,
        help="Optional lower grasp X,Y,Z,RX,RY,RZ in mm/deg.",
    )
    parser.add_argument(
        "--drop-pose",
        type=lift.parse_pose_mm_deg,
        default=None,
        help="Optional release X,Y,Z,RX,RY,RZ in mm/deg. Default returns to the grasp pose.",
    )
    parser.add_argument("--grab-duration", type=float, default=5.0)
    parser.add_argument("--contact-threshold", type=float, default=70.0)
    parser.add_argument("--max-force-delta", type=float, default=900.0)
    parser.add_argument("--step-settle", type=float, default=0.06)
    parser.add_argument(
        "--close-mode",
        choices=("fixed", "contact_stop"),
        default="fixed",
        help="fixed closes to the same target each time; contact_stop freezes contacted fingers.",
    )
    parser.add_argument("--hold-after-contact", type=float, default=1.0)
    parser.add_argument("--max-close-duration", type=float, default=3.0)
    parser.add_argument("--lift-height-mm", type=float, default=30.0)
    parser.add_argument("--lift-duration", type=float, default=3.0)
    parser.add_argument("--lower-duration", type=float, default=5.0)
    parser.add_argument("--hover-duration", type=float, default=5.0)
    parser.add_argument("--hover-rate-hz", type=float, default=10.0)
    parser.add_argument("--lift-settle", type=float, default=0.2)
    parser.add_argument("--squeeze-test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--ab-squeeze-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After stage-1 predicts A/B, run squeeze test and override A/B by hardness.",
    )
    parser.add_argument("--ab-squeeze-threshold", type=float, default=190.0)
    parser.add_argument("--ab-squeeze-a-standard", type=float, default=238.0)
    parser.add_argument("--ab-squeeze-b-standard", type=float, default=142.5)
    parser.add_argument(
        "--ab-squeeze-mode",
        choices=("friction", "shape", "curve", "threshold"),
        default="friction",
        help="A/B squeeze decision mode. friction uses SDK touchData tangential/normal after squeeze.",
    )
    parser.add_argument(
        "--ab-friction-finger",
        choices=("index", "middle", "thumb"),
        default="middle",
        help="Finger used by friction A/B mode.",
    )
    parser.add_argument(
        "--ab-friction-feature",
        choices=("last", "mean", "max", "late_slope"),
        default="last",
        help="Feature used by friction A/B mode.",
    )
    parser.add_argument(
        "--ab-friction-threshold",
        type=float,
        default=0.1464,
        help="Default from filtered A/B leave-one-out test: A if middle friction last >= 0.1464 else B.",
    )
    parser.add_argument(
        "--ab-friction-a-direction",
        choices=(">=", "<="),
        default=">=",
        help="Direction for friction A/B mode: A if feature crosses this threshold, else B.",
    )
    parser.add_argument(
        "--ab-proximity-assist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When stage-1 is A/B and the index finger barely participates, use thumb hover proximity as an A/B assist.",
    )
    parser.add_argument(
        "--ab-proximity-index-force-threshold",
        type=float,
        default=70.0,
        help="Raw forceAct delta threshold for the low-index-contact A/B proximity assist. 70 means about 0.70 N.",
    )
    parser.add_argument(
        "--ab-proximity-thumb-threshold",
        type=float,
        default=169619.0,
        help="Thumb hover proximity threshold from current A/B proximity samples.",
    )
    parser.add_argument(
        "--ab-proximity-a-direction",
        choices=(">=", "<="),
        default="<=",
        help="Direction for thumb proximity A/B assist: A if thumb proximity crosses this threshold, else B.",
    )
    parser.add_argument(
        "--ab-proximity-min-samples",
        type=float,
        default=5.0,
        help="Minimum hover proximity samples required before using the A/B proximity assist.",
    )
    parser.add_argument(
        "--bc-proximity-assist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When stage-1 is C but thumb/middle hover proximity looks like B, override C to B.",
    )
    parser.add_argument(
        "--bc-proximity-thumb-threshold",
        type=float,
        default=180000.0,
        help="Thumb hover proximity threshold for the C->B proximity assist.",
    )
    parser.add_argument(
        "--bc-proximity-middle-threshold",
        type=float,
        default=100000.0,
        help="Middle hover proximity threshold for the C->B proximity assist.",
    )
    parser.add_argument("--ab-shape-late-slope-threshold", type=float, default=-5.0)
    parser.add_argument("--ab-shape-rebound-threshold", type=float, default=15.0)
    parser.add_argument("--ab-shape-peak-pos-threshold", type=float, default=0.5)
    parser.add_argument("--ab-shape-min-a-score", type=int, default=2)
    parser.add_argument(
        "--ab-squeeze-reference-samples",
        type=Path,
        default=Path(__file__).with_name("samples.csv"),
    )
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
    parser.add_argument("--min-lift-contacts", type=float, default=1.0)
    parser.add_argument(
        "--force-lift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lift by default because the trained live model uses hover-pressure features.",
    )
    parser.add_argument("--open-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-settle", type=float, default=1.2)
    parser.add_argument(
        "--drop-open-settle",
        type=float,
        default=0.0,
        help="Seconds to wait after sending the final open-hand command at the drop/lower pose.",
    )
    parser.add_argument("--open-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--between-repeat", type=float, default=1.0)
    parser.add_argument(
        "--same-ball-vote",
        action="store_true",
        help="Treat repeated grasps as the same ball and print a majority vote. Default treats each grasp independently.",
    )
    parser.add_argument(
        "--expected-labels",
        default="",
        help="Optional sequence labels, e.g. ACBC. When set, results are scored per grasp.",
    )
    parser.add_argument("--notes", default="live lift-hover prediction")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.grab_xyz is not None and args.grab_pose is not None:
        raise SystemExit("Use only one of --grab-xyz or --grab-pose")
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
    if args.open_settle < 0 or args.drop_open_settle < 0:
        raise SystemExit("--open-settle and --drop-open-settle must be >= 0")
    if args.squeeze_baseline_hover_samples <= 0:
        raise SystemExit("--squeeze-baseline-hover-samples must be positive")
    if (
        args.squeeze_middle_touch_threshold <= 0
        or args.squeeze_middle_seek_step <= 0
        or args.squeeze_middle_seek_max_delta <= 0
        or args.squeeze_middle_seek_settle < 0
    ):
        raise SystemExit("middle squeeze seek parameters must be positive")
    if args.ab_squeeze_a_standard == args.ab_squeeze_b_standard:
        raise SystemExit("--ab-squeeze-a-standard and --ab-squeeze-b-standard must differ")
    if not 0 <= args.ab_shape_peak_pos_threshold <= 1:
        raise SystemExit("--ab-shape-peak-pos-threshold must be between 0 and 1")
    if not 0 <= args.ab_shape_min_a_score <= 3:
        raise SystemExit("--ab-shape-min-a-score must be between 0 and 3")
    if args.ab_proximity_index_force_threshold < 0 or args.ab_proximity_thumb_threshold <= 0:
        raise SystemExit("--ab-proximity-index-force-threshold must be >= 0 and --ab-proximity-thumb-threshold must be positive")
    if args.ab_proximity_min_samples < 0:
        raise SystemExit("--ab-proximity-min-samples must be >= 0")
    if args.bc_proximity_thumb_threshold <= 0 or args.bc_proximity_middle_threshold <= 0:
        raise SystemExit("--bc-proximity-thumb-threshold and --bc-proximity-middle-threshold must be positive")
    if args.repeats <= 0 or args.between_repeat < 0:
        raise SystemExit("--repeats must be positive and --between-repeat must be >= 0")
    if args.visual_last <= 0:
        raise SystemExit("--visual-last must be positive")
    if args.expected_labels:
        labels = [label.strip() for label in args.expected_labels.replace(",", "") if label.strip()]
        if len(labels) != args.repeats:
            raise SystemExit("--expected-labels length must match --repeats")
        args.expected_labels = labels


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


def friction_series_from_row(row: dict[str, object], finger: str) -> list[float]:
    return parse_series(str(row.get(f"tactile_friction_{finger}_series") or ""))


def friction_feature_value(series: list[float], feature: str) -> float:
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
    series = friction_series_from_row(row, args.ab_friction_finger)
    if not series:
        return "", 0.0, 0.0, []
    value = friction_feature_value(series, args.ab_friction_feature)
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


def proximity_ab_label(
    row: dict[str, object],
    stage1_label: str,
    args: argparse.Namespace,
) -> tuple[str, float, dict[str, float]]:
    """Use thumb proximity only in the low-index-contact A/B posture."""
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


def proximity_c_to_b_label(
    row: dict[str, object],
    stage1_label: str,
    args: argparse.Namespace,
) -> tuple[str, float, dict[str, float]]:
    """Catch B grasps that stage-1 calls C but have non-C thumb/middle proximity."""
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
    ab_label, ab_confidence, ab_stats = proximity_ab_label(row, stage1_label, args)
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

    bc_label, bc_confidence, bc_stats = proximity_c_to_b_label(row, stage1_label, args)
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


def collect_live_row(
    piper: object,
    hand: lift.RH56F2Hand,
    args: argparse.Namespace,
    trial_id: str,
    repeat_index: int,
    model: dict[str, object],
    min_active_contacts: float,
    min_hover_samples: float,
    ab_squeeze_references: dict[str, object],
) -> tuple[dict[str, object], bool]:
    print(f"\n[{repeat_index + 1}/{args.repeats}] live direct MOVEP lift-hover prediction")
    if args.visualize:
        try:
            render_pending_dashboard_from_csv(
                args.output,
                args.visual_output,
                args.visual_last,
                args.visual_reference_samples,
                notes="live grasp in progress",
            )
        except Exception as exc:
            print(f"[warn] pending dashboard refresh failed: {exc}")
    lift.configure_hand(hand, args.hand_speed, args.hand_force)
    if args.open_first:
        lift.set_hand_pose(hand, BALL_READY_OPEN, "open before grasp")
        time.sleep(args.open_settle)

    lower_pose = lift.selected_lower_pose(piper, args)
    if args.grab_pose is not None or args.grab_xyz is not None:
        print(f"moving to lower grasp pose: {lift.pose_mm_deg(lower_pose)}")
        if not lift.send_movep_checked(
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
    baseline_frame = lift.read_touch_frame(hand, baseline_started, args)
    lower_pose = lift.selected_lower_pose(piper, args)
    lift_pose = list(lower_pose)
    lift_pose[2] += int(round(args.lift_height_mm * 1000.0))
    drop_pose = list(args.drop_pose) if args.drop_pose is not None else lower_pose
    print(f"lower pose: {lift.pose_mm_deg(lower_pose)}")
    print(f"lift pose:  {lift.pose_mm_deg(lift_pose)}")
    if args.drop_pose is not None:
        print(f"drop pose:  {lift.pose_mm_deg(drop_pose)}")

    trial = TouchTrial(
        label="unknown",
        trial_id=trial_id,
        repeat_index=repeat_index,
        baseline_angles=baseline_frame.angles,
        baseline_forces=baseline_frame.forces,
        contact_threshold=args.contact_threshold,
        notes=args.notes,
    )
    row = lift.close_and_record(hand, trial, args)
    active_contacts = quality_number(row, "active_contact_count")
    if active_contacts == 0.0:
        print("[warn] no contact passed the threshold; continuing to lift/hover for live check.")
    elif active_contacts < args.min_lift_contacts:
        print(
            f"[warn] active_contact_count={active_contacts:.0f} < "
            f"{args.min_lift_contacts:g}; continuing to lift/hover for live check."
        )

    if active_contacts < args.min_lift_contacts and not args.force_lift:
        print("[warn] lift disabled by --no-force-lift; no live prediction will be made.")
        if args.open_at_end:
            lift.set_hand_pose(hand, BALL_READY_OPEN, "open at lower pose")
            time.sleep(args.open_settle)
        return extract_features(trial), True

    if not lift.send_movep_checked(
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
    lift.collect_hover(hand, trial, args)

    row = extract_features(trial)
    active_contacts = quality_number(row, "active_contact_count")
    hover_samples = quality_number(row, "hover_sample_count")
    if active_contacts < min_active_contacts or hover_samples < min_hover_samples:
        row["prediction_status"] = "skipped_quality"
        row["predicted_label"] = ""
        row["prediction_confidence"] = ""
        row["prediction_distance"] = ""
        print(
            "prediction skipped: "
            f"sample quality active={active_contacts:.0f}/{min_active_contacts:g}, "
            f"hover_samples={hover_samples:.0f}/{min_hover_samples:g}"
        )
    else:
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
            print(
                "prediction=NONE "
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
        else:
            stage1 = predict_row(row, model)
            stage1_label = str(stage1["label"])
            row["stage1_predicted_label"] = stage1_label
            row["stage1_prediction_confidence"] = float(stage1["confidence"])
            row["stage1_prediction_distance"] = float(stage1["distance"])
            print(
                f"stage1_prediction={stage1_label} "
                f"confidence={float(stage1['confidence']):.2f} "
                f"distance={float(stage1['distance']):.2f}"
            )

            squeeze_triggered = args.squeeze_test or (args.ab_squeeze_test and stage1_label in {"A", "B"})
            if squeeze_triggered:
                reason = "A/B hardness check" if stage1_label in {"A", "B"} and args.ab_squeeze_test else "forced squeeze test"
                print(f"squeeze test triggered: {reason}")
                lift.collect_squeeze(hand, trial, args)
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
                if stage1_label in {"A", "B"} and args.ab_squeeze_test:
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
                        row["prediction_status"] = "ab_squeeze_no_middle_contact"
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
                            ab_squeeze_references,
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
                        if row.get("prediction_status") != "ab_squeeze_no_middle_contact":
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

    if args.visualize:
        try:
            render_dashboard_preview_from_csv(
                args.output,
                args.visual_output,
                row,
                args.visual_last,
                args.visual_reference_samples,
            )
            print(f"visual dashboard decision preview: {args.visual_output}")
        except Exception as exc:
            print(f"[warn] decision dashboard refresh failed: {exc}")

    returned_lower = lift.send_movep_checked(
        piper,
        drop_pose,
        args,
        args.lower_duration,
        "drop MOVE_P" if args.drop_pose is not None else "lower MOVE_P",
        require_reached=False,
    )
    if not returned_lower:
        print("[warn] drop/lower MOVE_P failed after retries; hand is left closed to avoid dropping the ball.")
    elif args.open_at_end:
        lift.set_hand_pose(
            hand,
            BALL_READY_OPEN,
            "open at drop pose" if args.drop_pose is not None else "open at lower pose",
        )
        if args.drop_open_settle > 0:
            time.sleep(args.drop_open_settle)

    return row, returned_lower


def main() -> int:
    args = parse_args()
    validate_args(args)
    model = load_model(args.model)
    min_active_contacts = float(model.get("min_active_contacts", 0.0))
    min_hover_samples = float(model.get("min_hover_samples", 0.0))
    min_squeeze_samples = float(model.get("min_squeeze_samples", 0.0))
    ab_squeeze_references = load_ab_squeeze_references(args.ab_squeeze_reference_samples)

    print("RH56F2 live ball prediction with Piper lift/hover")
    print("This command moves the arm: grasp pose -> lift pose -> drop/lower pose.")
    print(f"model={args.model} output={args.output}")
    if args.visualize:
        print(f"visual dashboard={args.visual_output}")
    if args.ab_squeeze_test:
        ref_a = list(ab_squeeze_references.get("A") or [])
        ref_b = list(ab_squeeze_references.get("B") or [])
        print(
            f"ab squeeze mode={args.ab_squeeze_mode} "
            f"reference={args.ab_squeeze_reference_samples} "
            f"series={ab_squeeze_references.get('series_key') or '-'} "
            f"A_ref_len={len(ref_a)} B_ref_len={len(ref_b)}"
        )
    if args.grab_pose is not None:
        print(f"grab pose: {lift.pose_mm_deg(args.grab_pose)}")
    if args.drop_pose is not None:
        print(f"drop pose: {lift.pose_mm_deg(args.drop_pose)}")
    if args.expected_labels:
        print("sequence mode expected labels: " + "".join(args.expected_labels))
    if not args.yes:
        confirm = input(
            f"Type BALL_PREDICT_LIFT to run {args.repeats} lift-hover prediction(s): "
        ).strip()
        if confirm != "BALL_PREDICT_LIFT":
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
    try:
        piper = lift.connect_piper(args)
        lift.wait_for_real_feedback(piper, args.feedback_timeout)
        if not lift.enable_all(piper, args.feedback_timeout):
            raise RuntimeError("Piper did not enable; no motion command was sent.")
        if not lift.wait_for_movep_ready(piper, args.speed, args.feedback_timeout):
            if not lift.recover_movep_control(piper, args, "initial MOVE_P ready"):
                raise RuntimeError("MOVE_P mode was not ready; no target command was sent.")
        hand.connect()

        trial_id = now_trial_id()
        predictions: list[dict[str, object]] = []
        scored: list[tuple[str, str]] = []
        for repeat in range(args.repeats):
            row, returned_lower = collect_live_row(
                piper,
                hand,
                args,
                trial_id,
                repeat,
                model,
                min_active_contacts,
                min_hover_samples,
                ab_squeeze_references,
            )
            active_contacts = quality_number(row, "active_contact_count")
            hover_samples = quality_number(row, "hover_sample_count")
            predicted_label = str(row.get("predicted_label") or "")
            prediction_confidence = quality_number(row, "prediction_confidence")
            if not predicted_label:
                row["prediction_status"] = "skipped_quality"
            else:
                result = {
                    "label": predicted_label,
                    "confidence": prediction_confidence,
                }
                predictions.append(result)
                expected = args.expected_labels[repeat] if args.expected_labels else ""
                if expected:
                    scored.append((expected, predicted_label))
                print(
                    (f"expected={expected} " if expected else "")
                    +
                    f"prediction={predicted_label} "
                    f"confidence={prediction_confidence:.2f} "
                    f"status={row.get('prediction_status') or '-'}"
                )
            append_feature_row(args.output, row)
            print(
                "live sample saved: "
                f"active={active_contacts:.0f} "
                f"hover_samples={hover_samples:.0f} "
                f"hover_thumb_mean={row.get('hover_thumb_force_delta_mean') or '-'}"
            )
            if args.visualize:
                render_dashboard_from_csv(
                    args.output,
                    args.visual_output,
                    args.visual_last,
                    args.visual_reference_samples,
                )
                print(f"visual dashboard updated: {args.visual_output}")
            if not returned_lower:
                print("[warn] stopping repeats because the arm did not return to the lower pose.")
                break
            if repeat + 1 < args.repeats:
                print("Place the next random ball if needed before the next prediction.")
                time.sleep(args.between_repeat)

        if scored:
            correct = sum(1 for expected, predicted in scored if expected == predicted)
            expected_text = "".join(expected for expected, _ in scored)
            predicted_text = "".join(predicted for _, predicted in scored)
            print(
                f"sequence_expected={expected_text} "
                f"sequence_predicted={predicted_text} "
                f"accuracy={correct}/{len(scored)} ({correct / len(scored):.1%})"
            )
        elif predictions and args.same_ball_vote:
            counts: dict[str, int] = {}
            confidence_sums: dict[str, float] = {}
            for result in predictions:
                label = str(result["label"])
                counts[label] = counts.get(label, 0) + 1
                confidence_sums[label] = confidence_sums.get(label, 0.0) + float(result["confidence"])
            majority_label = sorted(
                counts,
                key=lambda label: (counts[label], confidence_sums[label] / counts[label]),
                reverse=True,
            )[0]
            mean_confidence = confidence_sums[majority_label] / counts[majority_label]
            count_text = ", ".join(f"{label}:{counts[label]}" for label in sorted(counts))
            print(
                f"majority_prediction={majority_label} "
                f"votes={count_text} "
                f"mean_confidence={mean_confidence:.2f}"
            )
        elif predictions:
            predicted_text = "".join(str(result["label"]) for result in predictions)
            print(f"sequence_predicted={predicted_text} count={len(predictions)}")
        else:
            print("no valid prediction: all samples failed the model quality filter.")
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
