#!/usr/bin/env python3
"""Shared feature extraction and model helpers for ball tactile classification."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


FINGER_NAMES = ["little", "ring", "middle", "index", "thumb_bend", "thumb_swing"]
TOUCH_FINGER_NAMES = ["little", "ring", "middle", "index", "thumb"]
FRICTION_FINGERS = ["index", "middle", "thumb"]
PRIMARY_SIZE_FINGERS = ["little", "ring", "middle", "index", "thumb_bend"]
THUMB_NAMES = ["thumb_bend", "thumb_swing"]
CORE_GRASP_FINGERS = ["index", "middle", "thumb_bend", "thumb_swing"]
FRICTION_NORMAL_FLOOR = 1.0

BALL_READY_OPEN = {
    "little": 1800.0,
    "ring": 1800.0,
    "middle": 1800.0,
    "index": 1800.0,
    "thumb_bend": 1500.0,
    "thumb_swing": 1050.0,
}

BALL_SAFE_CLOSED = {
    "little": 1200.0,
    "ring": 1220.0,
    "middle": 1350.0,
    "index": 1350.0,
    "thumb_bend": 1350.0,
    "thumb_swing": 900.0,
}

CLOSE_PHASES = [
    ("little_thumb", ("little", "thumb_swing"), 0.00),
    ("ring_thumb", ("ring", "thumb_bend"), 0.15),
    ("middle_index", ("middle", "index"), 0.30),
]

CLOSE_STEP_BY_NAME = {
    "little": 60.0,
    "ring": 55.0,
    "middle": 45.0,
    "index": 45.0,
    "thumb_bend": 30.0,
    "thumb_swing": 55.0,
}

DEFAULT_FEATURE_COLUMNS = [
    "final_angle_middle",
    "size_closure_mean",
    "final_force_delta_middle",
    "hover_thumb_force_delta_max",
]

MISS_GRASP_RULE = {
    "max_active_contacts": 3.0,
    "min_size_closure_mean": 380.0,
    "max_size_contact_angle_mean": 1360.0,
    "min_hover_sample_count": 5.0,
    "max_hover_thumb_force_delta_mean": 120.0,
    "max_hover_force_delta_sum_mean": 650.0,
    "max_final_force_delta_sum": 420.0,
}


@dataclass
class TouchFrame:
    elapsed_s: float
    angles: dict[str, float]
    forces: dict[str, float]
    tactile: dict[str, dict[str, dict[str, float]] | dict[str, float]] | None = None


@dataclass
class TouchTrial:
    label: str
    trial_id: str
    repeat_index: int
    baseline_angles: dict[str, float]
    baseline_forces: dict[str, float]
    frames: list[TouchFrame] = field(default_factory=list)
    hover_frames: list[TouchFrame] = field(default_factory=list)
    squeeze_baseline_angles: dict[str, float] | None = None
    squeeze_baseline_forces: dict[str, float] | None = None
    squeeze_frames: list[TouchFrame] = field(default_factory=list)
    squeeze_baseline_source: str = ""
    squeeze_hover_tail_count: int = 10
    squeeze_middle_ready: bool | None = None
    squeeze_middle_contact_delta: float | None = None
    squeeze_middle_seek_steps: int = 0
    squeeze_command_sample_index: int | None = None
    contact_threshold: float = 120.0
    lift_force_delta: float | None = None
    weight_g: float | None = None
    notes: str = ""


def now_trial_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def force_delta(forces: dict[str, float], baseline: dict[str, float], name: str) -> float:
    return abs(float(forces.get(name, 0.0)) - float(baseline.get(name, 0.0)))


def tactile_finger_value(frame: TouchFrame, finger: str, key: str) -> float | None:
    if not frame.tactile:
        return None
    fingers = frame.tactile.get("fingers")
    if not isinstance(fingers, dict):
        return None
    values = fingers.get(finger)
    if not isinstance(values, dict):
        return None
    number = values.get(key)
    if number is None:
        return None
    return float(number)


def tactile_friction_ratio(frame: TouchFrame, finger: str) -> float | None:
    normal = tactile_finger_value(frame, finger, "normal")
    tangential = tactile_finger_value(frame, finger, "tangential")
    if normal is None or tangential is None:
        return None
    if normal < FRICTION_NORMAL_FLOOR:
        return 0.0
    return tangential / normal


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def stddev(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.pstdev(values) if len(values) >= 2 else 0.0


def drift(values: list[float]) -> float:
    return values[-1] - values[0] if len(values) >= 2 else 0.0


def parse_series(value: str | None) -> list[float]:
    if not value:
        return []
    result = []
    for part in value.split(";"):
        number = numeric_or_none(part)
        if number is not None:
            result.append(number)
    return result


def average_series(series_list: list[list[float]]) -> list[float]:
    if not series_list:
        return []
    length = max(len(series) for series in series_list)
    result = []
    for idx in range(length):
        values = [series[idx] for series in series_list if idx < len(series)]
        if values:
            result.append(sum(values) / len(values))
    return result


def resample_series(values: list[float], length: int) -> list[float]:
    if not values:
        return [0.0] * length
    if length <= 1:
        return [values[-1]]
    if len(values) == 1:
        return [values[0]] * length
    result = []
    for idx in range(length):
        pos = idx * (len(values) - 1) / (length - 1)
        left = int(pos)
        right = min(left + 1, len(values) - 1)
        frac = pos - left
        result.append(values[left] * (1.0 - frac) + values[right] * frac)
    return result


def normalized_series_distance(left: list[float], right: list[float]) -> float:
    length = max(len(left), len(right), 2)
    left_r = resample_series(left, length)
    right_r = resample_series(right, length)
    scale = max(max(left_r, default=0.0), max(right_r, default=0.0), 1.0)
    return sum(((a - b) / scale) ** 2 for a, b in zip(left_r, right_r, strict=True)) ** 0.5


def curve_shape_metrics(series: list[float]) -> dict[str, float]:
    if not series:
        return {
            "curve_peak": 0.0,
            "curve_last": 0.0,
            "curve_rebound": 0.0,
            "curve_late_slope": 0.0,
            "curve_peak_pos": 0.0,
            "curve_rebound_ratio": 0.0,
        }
    peak = max(series)
    last = series[-1]
    peak_idx = series.index(peak)
    late = series[len(series) // 2 :]
    early_rise = peak - series[0]
    return {
        "curve_peak": peak,
        "curve_last": last,
        "curve_rebound": peak - last,
        "curve_late_slope": late[-1] - late[0] if len(late) >= 2 else 0.0,
        "curve_peak_pos": peak_idx / max(1, len(series) - 1),
        "curve_rebound_ratio": (peak - last) / max(1.0, early_rise),
    }


def ab_shape_decision(
    series: list[float],
    late_slope_threshold: float = -5.0,
    rebound_threshold: float = 15.0,
    peak_pos_threshold: float = 0.5,
    min_a_score: int = 2,
) -> tuple[str, float, dict[str, float]]:
    """Classify A/B from squeeze curve shape.

    A's middle-finger curve usually peaks earlier and rebounds; B usually keeps
    rising or stays high. The three checks vote for the A-like curve shape.
    """
    metrics = curve_shape_metrics(series)
    score_a = 0
    if metrics["curve_late_slope"] < late_slope_threshold:
        score_a += 1
    if metrics["curve_rebound"] >= rebound_threshold:
        score_a += 1
    if metrics["curve_peak_pos"] <= peak_pos_threshold:
        score_a += 1

    label = "A" if score_a >= min_a_score else "B"
    confidence = score_a / 3.0 if label == "A" else (3 - score_a) / 3.0
    metrics["ab_shape_score_a"] = float(score_a)
    metrics["ab_shape_min_a_score"] = float(min_a_score)
    metrics["ab_shape_late_slope_threshold"] = float(late_slope_threshold)
    metrics["ab_shape_rebound_threshold"] = float(rebound_threshold)
    metrics["ab_shape_peak_pos_threshold"] = float(peak_pos_threshold)
    return label, confidence, metrics


def detect_missed_grasp(row: dict[str, object]) -> tuple[bool, dict[str, float]]:
    """Heuristic gate for empty/missed grasps.

    This is intentionally separate from the A/B/C centroid model. Missed grasps
    should short-circuit before the ball classifier so the live dashboard can
    show "没抓到" instead of forcing a ball label.
    """
    active_contacts = numeric_or_none(row.get("active_contact_count")) or 0.0
    size_closure_mean = numeric_or_none(row.get("size_closure_mean")) or 0.0
    size_contact_angle_mean = numeric_or_none(row.get("size_contact_angle_mean")) or 0.0
    hover_sample_count = numeric_or_none(row.get("hover_sample_count")) or 0.0
    hover_thumb_force_delta_mean = numeric_or_none(row.get("hover_thumb_force_delta_mean")) or 0.0
    hover_force_delta_sum_mean = numeric_or_none(row.get("hover_force_delta_sum_mean")) or 0.0
    final_force_delta_sum = numeric_or_none(row.get("final_force_delta_sum")) or 0.0

    low_hover_after_lift = (
        hover_sample_count >= MISS_GRASP_RULE["min_hover_sample_count"]
        and hover_thumb_force_delta_mean <= MISS_GRASP_RULE["max_hover_thumb_force_delta_mean"]
        and hover_force_delta_sum_mean <= MISS_GRASP_RULE["max_hover_force_delta_sum_mean"]
    )
    deep_empty_close = (
        active_contacts <= MISS_GRASP_RULE["max_active_contacts"]
        and size_closure_mean >= MISS_GRASP_RULE["min_size_closure_mean"]
        and size_contact_angle_mean <= MISS_GRASP_RULE["max_size_contact_angle_mean"]
        and hover_thumb_force_delta_mean <= MISS_GRASP_RULE["max_hover_thumb_force_delta_mean"]
        and hover_force_delta_sum_mean <= MISS_GRASP_RULE["max_hover_force_delta_sum_mean"]
        and final_force_delta_sum <= MISS_GRASP_RULE["max_final_force_delta_sum"]
    )
    hit = low_hover_after_lift or deep_empty_close
    return hit, {
        "active_contact_count": float(active_contacts),
        "size_closure_mean": float(size_closure_mean),
        "size_contact_angle_mean": float(size_contact_angle_mean),
        "hover_sample_count": float(hover_sample_count),
        "hover_thumb_force_delta_mean": float(hover_thumb_force_delta_mean),
        "hover_force_delta_sum_mean": float(hover_force_delta_sum_mean),
        "final_force_delta_sum": float(final_force_delta_sum),
        "low_hover_after_lift": float(low_hover_after_lift),
        "deep_empty_close": float(deep_empty_close),
    }


def extract_features(trial: TouchTrial) -> dict[str, float | str]:
    """Convert one repeated-touch trial into one classifier-ready row."""
    contacts: dict[str, tuple[float, float, float]] = {}
    for frame in trial.frames:
        for name in FINGER_NAMES:
            if name in contacts:
                continue
            delta = force_delta(frame.forces, trial.baseline_forces, name)
            if delta >= trial.contact_threshold:
                contacts[name] = (frame.elapsed_s, float(frame.angles[name]), delta)

    final = trial.frames[-1] if trial.frames else TouchFrame(0.0, trial.baseline_angles, trial.baseline_forces)
    final_force_deltas = [
        force_delta(final.forces, trial.baseline_forces, name)
        for name in FINGER_NAMES
    ]

    contact_angles = [
        contacts[name][1]
        for name in PRIMARY_SIZE_FINGERS
        if name in contacts
    ]
    closure_values = [
        float(trial.baseline_angles.get(name, 0.0)) - contacts[name][1]
        for name in PRIMARY_SIZE_FINGERS
        if name in contacts
    ]
    contact_times = [value[0] for value in contacts.values()]

    row: dict[str, float | str] = {
        "label": trial.label,
        "trial_id": trial.trial_id,
        "repeat_index": float(trial.repeat_index),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "contact_threshold": float(trial.contact_threshold),
        "size_contact_angle_mean": mean(contact_angles),
        "size_closure_mean": mean(closure_values),
        "active_contact_count": float(len(contacts)),
        "contact_order_span_s": (max(contact_times) - min(contact_times)) if len(contact_times) >= 2 else 0.0,
        "final_force_delta_mean": mean(final_force_deltas),
        "final_force_delta_max": max(final_force_deltas) if final_force_deltas else 0.0,
        "final_force_delta_sum": sum(final_force_deltas),
        "lift_force_delta": "" if trial.lift_force_delta is None else float(trial.lift_force_delta),
        "weight_g": "" if trial.weight_g is None else float(trial.weight_g),
        "notes": trial.notes,
    }

    if trial.hover_frames:
        hover_force_delta_by_name = {
            name: [
                force_delta(frame.forces, trial.baseline_forces, name)
                for frame in trial.hover_frames
            ]
            for name in FINGER_NAMES
        }
        hover_force_sums = [
            sum(hover_force_delta_by_name[name][idx] for name in FINGER_NAMES)
            for idx in range(len(trial.hover_frames))
        ]
        hover_thumb_force = [
            sum(hover_force_delta_by_name[name][idx] for name in THUMB_NAMES)
            for idx in range(len(trial.hover_frames))
        ]
        row.update(
            {
                "hover_sample_count": float(len(trial.hover_frames)),
                "hover_duration_s": trial.hover_frames[-1].elapsed_s - trial.hover_frames[0].elapsed_s
                if len(trial.hover_frames) >= 2
                else 0.0,
                "hover_thumb_force_delta_mean": mean(hover_thumb_force),
                "hover_thumb_force_delta_max": max(hover_thumb_force),
                "hover_thumb_force_delta_std": stddev(hover_thumb_force),
                "hover_thumb_force_delta_last": hover_thumb_force[-1],
                "hover_thumb_force_delta_drift": drift(hover_thumb_force),
                "hover_force_delta_sum_mean": mean(hover_force_sums),
                "hover_force_delta_sum_max": max(hover_force_sums),
                "hover_force_delta_sum_std": stddev(hover_force_sums),
                "hover_force_delta_sum_last": hover_force_sums[-1],
                "hover_force_delta_sum_drift": drift(hover_force_sums),
            }
        )
        for name in FINGER_NAMES:
            values = hover_force_delta_by_name[name]
            row[f"hover_force_delta_{name}_mean"] = mean(values)
            row[f"hover_force_delta_{name}_max"] = max(values)
            row[f"hover_force_delta_{name}_std"] = stddev(values)
            row[f"hover_force_delta_{name}_last"] = values[-1]
            row[f"hover_force_delta_{name}_drift"] = drift(values)
        hover_proximity_by_finger = {
            finger: [
                value
                for value in (tactile_finger_value(frame, finger, "proximity") for frame in trial.hover_frames)
                if value is not None
            ]
            for finger in TOUCH_FINGER_NAMES
        }
        proximity_values = [
            value
            for values in hover_proximity_by_finger.values()
            for value in values
        ]
        row["hover_proximity_sample_count"] = float(max((len(values) for values in hover_proximity_by_finger.values()), default=0))
        row["hover_proximity_mean"] = mean(proximity_values)
        row["hover_proximity_max"] = max(proximity_values) if proximity_values else ""
        for finger in TOUCH_FINGER_NAMES:
            values = hover_proximity_by_finger[finger]
            row[f"hover_proximity_{finger}_series"] = ";".join(f"{value:.1f}" for value in values)
            row[f"hover_proximity_{finger}_mean"] = mean(values)
            row[f"hover_proximity_{finger}_max"] = max(values) if values else ""
            row[f"hover_proximity_{finger}_last"] = values[-1] if values else ""
            row[f"hover_proximity_{finger}_drift"] = drift(values) if values else ""
    else:
        row.update(
            {
                "hover_sample_count": "",
                "hover_duration_s": "",
                "hover_thumb_force_delta_mean": "",
                "hover_thumb_force_delta_max": "",
                "hover_thumb_force_delta_std": "",
                "hover_thumb_force_delta_last": "",
                "hover_thumb_force_delta_drift": "",
                "hover_force_delta_sum_mean": "",
                "hover_force_delta_sum_max": "",
                "hover_force_delta_sum_std": "",
                "hover_force_delta_sum_last": "",
                "hover_force_delta_sum_drift": "",
            }
        )
        for name in FINGER_NAMES:
            row[f"hover_force_delta_{name}_mean"] = ""
            row[f"hover_force_delta_{name}_max"] = ""
            row[f"hover_force_delta_{name}_std"] = ""
            row[f"hover_force_delta_{name}_last"] = ""
            row[f"hover_force_delta_{name}_drift"] = ""
        row["hover_proximity_sample_count"] = ""
        row["hover_proximity_mean"] = ""
        row["hover_proximity_max"] = ""
        for finger in TOUCH_FINGER_NAMES:
            row[f"hover_proximity_{finger}_series"] = ""
            row[f"hover_proximity_{finger}_mean"] = ""
            row[f"hover_proximity_{finger}_max"] = ""
            row[f"hover_proximity_{finger}_last"] = ""
            row[f"hover_proximity_{finger}_drift"] = ""

    if trial.squeeze_frames and trial.squeeze_baseline_angles is not None and trial.squeeze_baseline_forces is not None:
        hover_tail = trial.hover_frames[-max(0, trial.squeeze_hover_tail_count):]
        squeeze_window_frames = [
            *hover_tail,
            *trial.squeeze_frames,
        ]
        squeeze_force_delta_by_name = {
            name: [
                force_delta(frame.forces, trial.squeeze_baseline_forces or {}, name)
                for frame in trial.squeeze_frames
            ]
            for name in FINGER_NAMES
        }
        squeeze_force_sums = [
            sum(squeeze_force_delta_by_name[name][idx] for name in CORE_GRASP_FINGERS)
            for idx in range(len(trial.squeeze_frames))
        ]
        squeeze_thumb_force = [
            squeeze_force_delta_by_name["thumb_bend"][idx] + squeeze_force_delta_by_name["thumb_swing"][idx]
            for idx in range(len(trial.squeeze_frames))
        ]
        last_squeeze_frame = trial.squeeze_frames[-1]
        angle_deltas = [
            max(0.0, float((trial.squeeze_baseline_angles or {}).get(name, 0.0)) - float(last_squeeze_frame.angles.get(name, 0.0)))
            for name in CORE_GRASP_FINGERS
        ]
        squeeze_angle_delta_mean = mean(angle_deltas)
        squeeze_force_delta_sum_last = squeeze_force_sums[-1]
        row.update(
            {
                "squeeze_sample_count": float(len(trial.squeeze_frames)),
                "squeeze_duration_s": trial.squeeze_frames[-1].elapsed_s - trial.squeeze_frames[0].elapsed_s
                if len(trial.squeeze_frames) >= 2
                else 0.0,
                "squeeze_baseline_source": trial.squeeze_baseline_source,
                "squeeze_middle_ready": "" if trial.squeeze_middle_ready is None else float(trial.squeeze_middle_ready),
                "squeeze_middle_contact_delta": ""
                if trial.squeeze_middle_contact_delta is None
                else float(trial.squeeze_middle_contact_delta),
                "squeeze_middle_seek_steps": float(trial.squeeze_middle_seek_steps),
                "squeeze_command_sample_index": ""
                if trial.squeeze_command_sample_index is None
                else float(trial.squeeze_command_sample_index),
                "tactile_window_sample_count": float(len(squeeze_window_frames)),
                "tactile_friction_command_sample_index": ""
                if trial.squeeze_command_sample_index is None
                else float(len(hover_tail) + trial.squeeze_command_sample_index),
                "squeeze_force_delta_sum_mean": mean(squeeze_force_sums),
                "squeeze_force_delta_sum_max": max(squeeze_force_sums),
                "squeeze_force_delta_sum_last": squeeze_force_delta_sum_last,
                "squeeze_thumb_force_delta_mean": mean(squeeze_thumb_force),
                "squeeze_thumb_force_delta_max": max(squeeze_thumb_force),
                "squeeze_thumb_force_delta_last": squeeze_thumb_force[-1],
                "squeeze_thumb_force_delta_series": ";".join(
                    f"{value:.1f}" for value in squeeze_thumb_force
                ),
                "squeeze_middle_force_delta_max": max(squeeze_force_delta_by_name["middle"]),
                "squeeze_middle_force_delta_series": ";".join(
                    f"{value:.1f}" for value in squeeze_force_delta_by_name["middle"]
                ),
                "squeeze_force_delta_sum_series": ";".join(
                    f"{value:.1f}" for value in squeeze_force_sums
                ),
                "squeeze_angle_delta_mean": squeeze_angle_delta_mean,
                "squeeze_stiffness_sum_per_angle": squeeze_force_delta_sum_last / max(1.0, squeeze_angle_delta_mean),
            }
        )
        for name in FINGER_NAMES:
            values = squeeze_force_delta_by_name[name]
            row[f"squeeze_force_delta_{name}_mean"] = mean(values)
            row[f"squeeze_force_delta_{name}_max"] = max(values)
            row[f"squeeze_force_delta_{name}_last"] = values[-1]
            row[f"squeeze_force_delta_{name}_series"] = ";".join(
                f"{value:.1f}" for value in values
            )
        for finger in FRICTION_FINGERS:
            normals = [
                value
                for value in (tactile_finger_value(frame, finger, "normal") for frame in squeeze_window_frames)
                if value is not None
            ]
            tangentials = [
                value
                for value in (tactile_finger_value(frame, finger, "tangential") for frame in squeeze_window_frames)
                if value is not None
            ]
            ratios = [
                value
                for value in (tactile_friction_ratio(frame, finger) for frame in squeeze_window_frames)
                if value is not None
            ]
            row[f"tactile_normal_{finger}_series"] = ";".join(f"{value:.1f}" for value in normals)
            row[f"tactile_tangential_{finger}_series"] = ";".join(f"{value:.1f}" for value in tangentials)
            row[f"tactile_friction_{finger}_series"] = ";".join(f"{value:.4f}" for value in ratios)
            row[f"tactile_friction_{finger}_mean"] = mean(ratios)
            row[f"tactile_friction_{finger}_max"] = max(ratios) if ratios else ""
            row[f"tactile_normal_{finger}_mean"] = mean(normals)
            row[f"tactile_tangential_{finger}_mean"] = mean(tangentials)
    else:
        row.update(
            {
                "squeeze_sample_count": "",
                "squeeze_duration_s": "",
                "squeeze_baseline_source": trial.squeeze_baseline_source,
                "squeeze_middle_ready": "" if trial.squeeze_middle_ready is None else float(trial.squeeze_middle_ready),
                "squeeze_middle_contact_delta": ""
                if trial.squeeze_middle_contact_delta is None
                else float(trial.squeeze_middle_contact_delta),
                "squeeze_middle_seek_steps": float(trial.squeeze_middle_seek_steps),
                "squeeze_command_sample_index": "",
                "tactile_window_sample_count": "",
                "tactile_friction_command_sample_index": "",
                "squeeze_force_delta_sum_mean": "",
                "squeeze_force_delta_sum_max": "",
                "squeeze_force_delta_sum_last": "",
                "squeeze_thumb_force_delta_mean": "",
                "squeeze_thumb_force_delta_max": "",
                "squeeze_thumb_force_delta_last": "",
                "squeeze_thumb_force_delta_series": "",
                "squeeze_middle_force_delta_max": "",
                "squeeze_middle_force_delta_series": "",
                "squeeze_force_delta_sum_series": "",
                "squeeze_angle_delta_mean": "",
                "squeeze_stiffness_sum_per_angle": "",
            }
        )
        for name in FINGER_NAMES:
            row[f"squeeze_force_delta_{name}_mean"] = ""
            row[f"squeeze_force_delta_{name}_max"] = ""
            row[f"squeeze_force_delta_{name}_last"] = ""
            row[f"squeeze_force_delta_{name}_series"] = ""
        for finger in FRICTION_FINGERS:
            row[f"tactile_normal_{finger}_series"] = ""
            row[f"tactile_tangential_{finger}_series"] = ""
            row[f"tactile_friction_{finger}_series"] = ""
            row[f"tactile_friction_{finger}_mean"] = ""
            row[f"tactile_friction_{finger}_max"] = ""
            row[f"tactile_normal_{finger}_mean"] = ""
            row[f"tactile_tangential_{finger}_mean"] = ""

    for name in FINGER_NAMES:
        row[f"baseline_angle_{name}"] = float(trial.baseline_angles.get(name, 0.0))
        row[f"baseline_force_{name}"] = float(trial.baseline_forces.get(name, 0.0))
        row[f"final_angle_{name}"] = float(final.angles.get(name, 0.0))
        row[f"final_force_{name}"] = float(final.forces.get(name, 0.0))
        row[f"final_force_delta_{name}"] = force_delta(final.forces, trial.baseline_forces, name)
        if name in contacts:
            row[f"contacted_{name}"] = 1.0
            row[f"contact_time_{name}"] = contacts[name][0]
            row[f"contact_angle_{name}"] = contacts[name][1]
            row[f"contact_force_delta_{name}"] = contacts[name][2]
        else:
            row[f"contacted_{name}"] = 0.0
            row[f"contact_time_{name}"] = ""
            row[f"contact_angle_{name}"] = ""
            row[f"contact_force_delta_{name}"] = ""

    return row


def numeric_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def read_feature_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_rows_by_quality(
    rows: list[dict[str, str]],
    min_active_contacts: float = 0.0,
    min_hover_samples: float = 0.0,
    min_squeeze_samples: float = 0.0,
    reject_note_tokens: tuple[str, ...] = ("bad_grasp", "void"),
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    kept = []
    skipped = []
    for row in rows:
        active_contacts = numeric_or_none(row.get("active_contact_count")) or 0.0
        hover_samples = numeric_or_none(row.get("hover_sample_count")) or 0.0
        squeeze_samples = numeric_or_none(row.get("squeeze_sample_count")) or 0.0
        notes = row.get("notes", "").lower()
        rejected_by_note = any(token in notes for token in reject_note_tokens)
        if (
            active_contacts < min_active_contacts
            or hover_samples < min_hover_samples
            or squeeze_samples < min_squeeze_samples
            or rejected_by_note
        ):
            skipped.append(row)
        else:
            kept.append(row)
    return kept, skipped


def append_feature_row(path: Path, row: dict[str, float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            existing_fields = csv.DictReader(f).fieldnames or []
        fieldnames = list(dict.fromkeys([*existing_fields, *fieldnames]))

    if exists and fieldnames != existing_fields:
        rows = read_feature_rows(path)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def feature_columns_from_rows(
    rows: list[dict[str, str]],
    requested: list[str] | None = None,
) -> list[str]:
    if requested:
        return requested
    columns: list[str] = []
    for name in DEFAULT_FEATURE_COLUMNS:
        if any(numeric_or_none(row.get(name)) is not None for row in rows):
            columns.append(name)
    return columns


def train_nearest_centroid(
    rows: list[dict[str, str]],
    feature_columns: list[str],
    label_column: str = "label",
) -> dict[str, object]:
    clean_rows = [row for row in rows if row.get(label_column, "").strip()]
    if not clean_rows:
        raise ValueError("No labelled rows found.")
    if not feature_columns:
        raise ValueError("No usable numeric feature columns found.")

    feature_means: dict[str, float] = {}
    for feature in feature_columns:
        values = [
            value
            for row in clean_rows
            if (value := numeric_or_none(row.get(feature))) is not None
        ]
        if not values:
            raise ValueError(f"Feature has no numeric values: {feature}")
        feature_means[feature] = mean(values)

    vectors: list[tuple[str, list[float]]] = []
    for row in clean_rows:
        label = row[label_column].strip()
        vector = []
        for feature in feature_columns:
            value = numeric_or_none(row.get(feature))
            if value is None:
                value = feature_means[feature]
            vector.append(value)
        vectors.append((label, [float(value) for value in vector]))

    normalizer_mean = [
        mean(vector[i] for _, vector in vectors)
        for i in range(len(feature_columns))
    ]
    normalizer_std = []
    for i in range(len(feature_columns)):
        values = [vector[i] for _, vector in vectors]
        std = statistics.pstdev(values)
        normalizer_std.append(std if std > 1e-9 else 1.0)

    by_label: dict[str, list[list[float]]] = {}
    for label, vector in vectors:
        normalized = [
            (value - normalizer_mean[i]) / normalizer_std[i]
            for i, value in enumerate(vector)
        ]
        by_label.setdefault(label, []).append(normalized)
    if len(by_label) < 2:
        raise ValueError("Need samples from at least two labels; collect all three ball classes before use.")

    centroids = {
        label: [
            mean(vector[i] for vector in label_vectors)
            for i in range(len(feature_columns))
        ]
        for label, label_vectors in sorted(by_label.items())
    }
    counts = {label: len(label_vectors) for label, label_vectors in sorted(by_label.items())}

    return {
        "model_type": "nearest_centroid",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label_column": label_column,
        "feature_columns": feature_columns,
        "feature_fill_values": feature_means,
        "normalizer_mean": normalizer_mean,
        "normalizer_std": normalizer_std,
        "centroids": centroids,
        "counts": counts,
    }


def predict_row(row: dict[str, object], model: dict[str, object]) -> dict[str, object]:
    feature_columns = list(model["feature_columns"])
    fill_values = dict(model["feature_fill_values"])
    normalizer_mean = [float(value) for value in model["normalizer_mean"]]
    normalizer_std = [float(value) for value in model["normalizer_std"]]
    centroids = {
        str(label): [float(value) for value in vector]
        for label, vector in dict(model["centroids"]).items()
    }
    vector = []
    missing = []
    for feature in feature_columns:
        value = numeric_or_none(row.get(feature))
        if value is None:
            missing.append(feature)
            value = float(fill_values[feature])
        vector.append(float(value))
    normalized = [
        (value - normalizer_mean[i]) / normalizer_std[i]
        for i, value in enumerate(vector)
    ]

    distances = {}
    for label, centroid in centroids.items():
        distances[label] = math.sqrt(
            sum((value - centroid[i]) ** 2 for i, value in enumerate(normalized))
        )
    ranked = sorted(distances.items(), key=lambda item: item[1])
    best_label, best_distance = ranked[0]
    second_distance = ranked[1][1] if len(ranked) > 1 else best_distance
    confidence = 1.0 if len(ranked) == 1 else max(
        0.0,
        min(1.0, (second_distance - best_distance) / (second_distance + 1e-9)),
    )
    return {
        "label": best_label,
        "confidence": confidence,
        "distance": best_distance,
        "distances": distances,
        "missing_features": missing,
    }


def save_model(path: Path, model: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_model(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
