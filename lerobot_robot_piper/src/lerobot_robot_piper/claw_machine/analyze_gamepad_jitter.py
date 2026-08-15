#!/usr/bin/env python3
"""Analyze Piper gamepad telemetry recorded by lerobot_claw."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


REACH_JOINTS = (2, 3, 5)


@dataclass(frozen=True)
class Segment:
    direction: str
    rows: list[dict[str, float]]

    @property
    def duration_s(self) -> float:
        return self.rows[-1]["elapsed_s"] - self.rows[0]["elapsed_s"]


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"elapsed_s", "reach_axis"}
        for joint in range(1, 7):
            required.update({f"target_j{joint}", f"actual_j{joint}"})
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
        return [
            {name: float(value) for name, value in row.items() if name != "phase"}
            for row in reader
        ]


def split_reach_segments(
    rows: list[dict[str, float]],
    axis_threshold: float,
    min_duration_s: float,
) -> list[Segment]:
    segments: list[Segment] = []
    current: list[dict[str, float]] = []
    current_sign = 0

    def finish() -> None:
        nonlocal current, current_sign
        if current and current[-1]["elapsed_s"] - current[0]["elapsed_s"] >= min_duration_s:
            direction = "forward" if current_sign > 0 else "back"
            segments.append(Segment(direction, current))
        current = []
        current_sign = 0

    for row in rows:
        axis = row["reach_axis"]
        sign = 1 if axis >= axis_threshold else -1 if axis <= -axis_threshold else 0
        if sign == 0:
            finish()
        elif current_sign not in (0, sign):
            finish()
            current_sign = sign
            current = [row]
        else:
            current_sign = sign
            current.append(row)
    finish()
    return segments


def linear_fit(times: list[float], values: list[float]) -> tuple[float, float]:
    mean_time = sum(times) / len(times)
    mean_value = sum(values) / len(values)
    variance = sum((time_value - mean_time) ** 2 for time_value in times)
    if variance == 0.0:
        return 0.0, mean_value
    slope = sum(
        (time_value - mean_time) * (value - mean_value)
        for time_value, value in zip(times, values, strict=True)
    ) / variance
    return slope, mean_value - slope * mean_time


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def joint_metrics(
    segment: Segment,
    joint: int,
    reversal_velocity_threshold: float,
) -> dict[str, float | int]:
    times = [row["elapsed_s"] for row in segment.rows]
    actual = [row[f"actual_j{joint}"] for row in segment.rows]
    target = [row[f"target_j{joint}"] for row in segment.rows]
    tracking_errors = [
        target_value - actual_value
        for target_value, actual_value in zip(target, actual, strict=True)
    ]
    slope, intercept = linear_fit(times, actual)
    position_residuals = [
        value - (slope * time_value + intercept)
        for time_value, value in zip(times, actual, strict=True)
    ]

    velocities: list[float] = []
    for index in range(1, len(actual)):
        dt_s = times[index] - times[index - 1]
        if dt_s > 0.0:
            velocities.append((actual[index] - actual[index - 1]) / dt_s)
    velocity_residuals = [velocity - slope for velocity in velocities]
    reversals = sum(
        1
        for velocity in velocities
        if abs(velocity) >= reversal_velocity_threshold and velocity * slope < 0.0
    )

    return {
        "tracking_rms_deg": rms(tracking_errors),
        "position_residual_rms_deg": rms(position_residuals),
        "position_residual_peak_to_peak_deg": max(position_residuals) - min(position_residuals),
        "velocity_residual_rms_dps": rms(velocity_residuals) if velocity_residuals else 0.0,
        "reversals": reversals,
        "trend_dps": slope,
    }


def analyze_segment(
    segment: Segment,
    reversal_velocity_threshold: float,
) -> dict[int, dict[str, float | int]]:
    return {
        joint: joint_metrics(segment, joint, reversal_velocity_threshold)
        for joint in REACH_JOINTS
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Telemetry CSV produced by --gamepad-log-csv.")
    parser.add_argument("--axis-threshold", type=float, default=0.8)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--reversal-velocity-threshold", type=float, default=0.1)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = load_rows(args.csv)
    segments = split_reach_segments(rows, args.axis_threshold, args.min_duration)
    if not segments:
        print(
            "No sustained reach segment found. Hold forward/back beyond "
            f"{args.axis_threshold:.2f} for at least {args.min_duration:.1f} s."
        )
        return 1

    print("Lower residual RMS/peak-to-peak/reversals means smoother motion.")
    for index, segment in enumerate(segments, start=1):
        mean_axis = sum(abs(row["reach_axis"]) for row in segment.rows) / len(segment.rows)
        actual_rate_hz = (len(segment.rows) - 1) / segment.duration_s
        print(
            f"\nSegment {index}: {segment.direction}, duration={segment.duration_s:.2f}s, "
            f"samples={len(segment.rows)}, actual_rate={actual_rate_hz:.1f}Hz, "
            f"mean_axis={mean_axis:.3f}"
        )
        metrics = analyze_segment(segment, args.reversal_velocity_threshold)
        print("joint  track_RMS  pos_RMS  pos_P-P  vel_RMS  reversals  trend")
        for joint in REACH_JOINTS:
            values = metrics[joint]
            print(
                f"J{joint:<4} "
                f"{values['tracking_rms_deg']:9.4f} "
                f"{values['position_residual_rms_deg']:8.4f} "
                f"{values['position_residual_peak_to_peak_deg']:8.4f} "
                f"{values['velocity_residual_rms_dps']:8.3f} "
                f"{values['reversals']:9d} "
                f"{values['trend_dps']:7.3f} deg/s"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
