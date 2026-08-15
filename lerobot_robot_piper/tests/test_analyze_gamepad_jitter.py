import math

from lerobot_robot_piper.claw_machine.analyze_gamepad_jitter import (
    Segment,
    joint_metrics,
    split_reach_segments,
)


def make_row(elapsed_s: float, reach_axis: float, actual_j2: float) -> dict[str, float]:
    row = {"elapsed_s": elapsed_s, "reach_axis": reach_axis}
    for joint in range(1, 7):
        actual = actual_j2 if joint == 2 else 0.0
        row[f"actual_j{joint}"] = actual
        row[f"target_j{joint}"] = actual
    return row


def test_split_reach_segments_separates_forward_and_back_motion():
    rows = [
        make_row(0.0, 0.9, 0.0),
        make_row(0.5, 0.9, 0.5),
        make_row(1.0, 0.0, 1.0),
        make_row(1.5, -0.9, 0.5),
        make_row(2.0, -0.9, 0.0),
    ]

    segments = split_reach_segments(rows, axis_threshold=0.8, min_duration_s=0.5)

    assert [segment.direction for segment in segments] == ["forward", "back"]


def test_joint_metrics_reports_zero_residual_for_linear_motion():
    rows = [make_row(index * 0.1, 1.0, index * 0.2) for index in range(21)]

    metrics = joint_metrics(Segment("forward", rows), 2, 0.1)

    assert metrics["position_residual_rms_deg"] < 1e-10
    assert metrics["position_residual_peak_to_peak_deg"] < 1e-10
    assert metrics["velocity_residual_rms_dps"] < 1e-10
    assert metrics["reversals"] == 0


def test_joint_metrics_detects_oscillation_around_linear_motion():
    rows = [
        make_row(
            index * 0.05,
            1.0,
            index * 0.05 + 0.2 * math.sin(2.0 * math.pi * 4.0 * index * 0.05),
        )
        for index in range(81)
    ]

    metrics = joint_metrics(Segment("forward", rows), 2, 0.1)

    assert metrics["position_residual_rms_deg"] > 0.1
    assert metrics["position_residual_peak_to_peak_deg"] > 0.35
    assert metrics["velocity_residual_rms_dps"] > 2.0
    assert metrics["reversals"] > 0
