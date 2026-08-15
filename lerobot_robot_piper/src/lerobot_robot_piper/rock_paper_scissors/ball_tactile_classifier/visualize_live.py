#!/usr/bin/env python3
"""Render a browser dashboard for live RH56F2 ball predictions."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import sys
from pathlib import Path

try:
    from .common import FINGER_NAMES, FRICTION_FINGERS, TOUCH_FINGER_NAMES, average_series, curve_shape_metrics, detect_missed_grasp, numeric_or_none, parse_series, read_feature_rows
except ImportError:  # Allow: python3 visualize_live.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import FINGER_NAMES, FRICTION_FINGERS, TOUCH_FINGER_NAMES, average_series, curve_shape_metrics, detect_missed_grasp, numeric_or_none, parse_series, read_feature_rows  # type: ignore


CORE_FEATURES = [
    ("final_angle_middle", "middle angle"),
    ("size_closure_mean", "closure mean"),
    ("final_force_delta_middle", "middle force"),
    ("hover_thumb_force_delta_max", "thumb hover max"),
    ("squeeze_force_delta_sum_max", "squeeze force max"),
    ("squeeze_stiffness_sum_per_angle", "squeeze stiffness"),
]


HAND_POINTS = {
    "little": (60, 78),
    "ring": (105, 54),
    "middle": (150, 42),
    "index": (195, 58),
    "thumb_bend": (230, 140),
    "thumb_swing": (72, 170),
}

SQUEEZE_SERIES = [
    ("middle", "middle pressure delta"),
    ("thumb", "thumb pressure delta"),
    ("core_sum", "core sum pressure delta"),
]

FRICTION_SERIES = [
    ("index", "index friction ratio"),
    ("middle", "middle friction ratio"),
    ("thumb", "thumb friction ratio"),
]

REFERENCE_COLORS = {
    "A": "#d65f00",
    "B": "#0072b2",
    "C": "#8a3ffc",
}

DASHBOARD_REFRESH_SECONDS = 0.5

BALL_ART = {
    "A": "ball_assets/ball_A_original.jpg",
    "B": "ball_assets/ball_B_original.jpg",
    "C": "ball_assets/ball_C_original.jpg",
}

SQUEEZE_RATE_HZ = 20.0


def value(row: dict[str, str], key: str) -> float:
    return numeric_or_none(row.get(key)) or 0.0


def fmt(number: float | None, digits: int = 1) -> str:
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def color_for(value_: float, max_value: float) -> str:
    ratio = 0.0 if max_value <= 0 else max(0.0, min(1.0, value_ / max_value))
    red = int(45 + ratio * 210)
    green = int(120 - ratio * 80)
    blue = int(210 - ratio * 170)
    return f"rgb({red},{green},{blue})"


def bar(width_ratio: float, label: str, number: str) -> str:
    ratio = max(0.0, min(1.0, width_ratio))
    return (
        '<div class="bar-row">'
        f'<span class="bar-label">{html.escape(label)}</span>'
        '<div class="bar-track">'
        f'<div class="bar-fill" style="width:{ratio * 100:.1f}%"></div>'
        "</div>"
        f'<span class="bar-value">{html.escape(number)}</span>'
        "</div>"
    )


def row_display_label(row: dict[str, str]) -> str:
    label = row.get("label", "")
    if label and label != "unknown":
        return label
    return row.get("predicted_label", "") or ""


def add_series(left: list[float], right: list[float]) -> list[float]:
    length = max(len(left), len(right))
    return [
        (left[idx] if idx < len(left) else 0.0) + (right[idx] if idx < len(right) else 0.0)
        for idx in range(length)
    ]


def squeeze_series(row: dict[str, str], series_key: str) -> list[float]:
    if series_key == "middle":
        return parse_series(row.get("squeeze_middle_force_delta_series"))
    if series_key == "thumb":
        direct = parse_series(row.get("squeeze_thumb_force_delta_series"))
        if direct:
            return direct
        return add_series(
            parse_series(row.get("squeeze_force_delta_thumb_bend_series")),
            parse_series(row.get("squeeze_force_delta_thumb_swing_series")),
        )
    if series_key == "core_sum":
        return parse_series(row.get("squeeze_force_delta_sum_series"))
    return []


def squeeze_command_index(row: dict[str, str], values: list[float]) -> int:
    number = numeric_or_none(row.get("squeeze_command_sample_index"))
    if number is None:
        return 0
    return max(0, min(int(number), max(0, len(values) - 1)))


def friction_command_index(row: dict[str, str], values: list[float]) -> int:
    number = numeric_or_none(row.get("tactile_friction_command_sample_index"))
    if number is None:
        number = numeric_or_none(row.get("squeeze_command_sample_index"))
    if number is None:
        return 0
    return max(0, min(int(number), max(0, len(values) - 1)))


def squeeze_time_at(index: int, command_index: int) -> float:
    return (index - command_index) / SQUEEZE_RATE_HZ


def latest_reference_curve_rows(
    rows: list[dict[str, str]],
    latest: dict[str, str],
    series_key: str,
) -> dict[str, list[tuple[dict[str, str], list[float]]]]:
    labels = ["A", "B"]
    grouped: dict[str, list[tuple[dict[str, str], list[float]]]] = {label: [] for label in labels}
    latest_identity = (latest.get("trial_id"), latest.get("repeat_index"), latest.get("timestamp"))
    for row in rows:
        identity = (row.get("trial_id"), row.get("repeat_index"), row.get("timestamp"))
        if identity == latest_identity:
            continue
        label = row_display_label(row)
        if label not in grouped:
            continue
        notes = (row.get("notes") or "").lower()
        if "void" in notes or "bad_grasp" in notes:
            continue
        series = squeeze_series(row, series_key)
        if series:
            grouped[label].append((row, series))

    latest_trials = {
        label: max((row.get("trial_id") or "" for row, _ in items), default="")
        for label, items in grouped.items()
    }
    return {
        label: [
            (row, series)
            for row, series in items
            if latest_trials[label] and row.get("trial_id") == latest_trials[label]
        ]
        for label, items in grouped.items()
    }


def tactile_friction_series(row: dict[str, str], finger: str) -> list[float]:
    if finger not in FRICTION_FINGERS:
        return []
    return parse_series(row.get(f"tactile_friction_{finger}_series"))


def hover_proximity_series(row: dict[str, str], finger: str) -> list[float]:
    if finger not in TOUCH_FINGER_NAMES:
        return []
    return parse_series(row.get(f"hover_proximity_{finger}_series"))


def latest_reference_friction_rows(
    rows: list[dict[str, str]],
    latest: dict[str, str],
    finger: str,
) -> dict[str, list[tuple[dict[str, str], list[float]]]]:
    labels = ["A", "B"]
    grouped: dict[str, list[tuple[dict[str, str], list[float]]]] = {label: [] for label in labels}
    latest_identity = (latest.get("trial_id"), latest.get("repeat_index"), latest.get("timestamp"))
    for row in rows:
        identity = (row.get("trial_id"), row.get("repeat_index"), row.get("timestamp"))
        if identity == latest_identity:
            continue
        label = row_display_label(row)
        if label not in grouped:
            continue
        notes = (row.get("notes") or "").lower()
        if "void" in notes or "bad_grasp" in notes:
            continue
        series = tactile_friction_series(row, finger)
        if series:
            grouped[label].append((row, series))

    latest_trials = {
        label: max((row.get("trial_id") or "" for row, _ in items), default="")
        for label, items in grouped.items()
    }
    return {
        label: [
            (row, series)
            for row, series in items
            if latest_trials[label] and row.get("trial_id") == latest_trials[label]
        ]
        for label, items in grouped.items()
    }


def latest_reference_proximity_rows(
    rows: list[dict[str, str]],
    latest: dict[str, str],
    finger: str,
) -> dict[str, list[tuple[dict[str, str], list[float]]]]:
    labels = ["A", "B", "C"]
    grouped: dict[str, list[tuple[dict[str, str], list[float]]]] = {label: [] for label in labels}
    latest_identity = (latest.get("trial_id"), latest.get("repeat_index"), latest.get("timestamp"))
    for row in rows:
        identity = (row.get("trial_id"), row.get("repeat_index"), row.get("timestamp"))
        if identity == latest_identity:
            continue
        label = row_display_label(row)
        if label not in grouped:
            continue
        notes = (row.get("notes") or "").lower()
        if "void" in notes or "bad_grasp" in notes:
            continue
        series = hover_proximity_series(row, finger)
        if series:
            grouped[label].append((row, series))

    latest_trials = {
        label: max((row.get("trial_id") or "" for row, _ in items), default="")
        for label, items in grouped.items()
    }
    return {
        label: [
            (row, series)
            for row, series in items
            if latest_trials[label] and row.get("trial_id") == latest_trials[label]
        ]
        for label, items in grouped.items()
    }


def aligned_chart(
    values: list[float],
    command_index: int,
    title: str,
    reference: dict[str, list[tuple[dict[str, str], list[float]]]],
    reference_command_index,
    latest_digits: int = 1,
    height: int = 250,
    summary: str = "",
    axis_label: str = "seconds relative to squeeze command",
    command_label: str = "squeeze command",
) -> str:
    width = 760
    left = 54
    right = 18
    top = 24
    bottom = 42
    chart_w = width - left - right
    chart_h = height - top - bottom
    reference_values = [
        value_
        for items in reference.values()
        for _, series in items
        for value_ in series
    ]
    y_max = max([*values, *reference_values, 1.0]) * 1.18
    y_min = -0.06 * y_max

    all_times = [squeeze_time_at(idx, command_index) for idx in range(len(values))]
    for items in reference.values():
        for ref_row, series in items:
            ref_command = reference_command_index(ref_row, series)
            all_times.extend(squeeze_time_at(idx, ref_command) for idx in range(len(series)))
    x_min = min(all_times + [-0.8])
    x_max = max(all_times + [1.8])

    def x_at_time(seconds: float) -> float:
        if x_max <= x_min:
            return left
        return left + chart_w * (seconds - x_min) / (x_max - x_min)

    def y_at(v: float) -> float:
        span = max(1.0, y_max - y_min)
        ratio = max(0.0, min(1.0, (v - y_min) / span))
        return top + chart_h * (1.0 - ratio)

    def polyline(series: list[float], command: int, color: str, label: str, strong: bool = False) -> str:
        if not series:
            return ""
        line_points = " ".join(
            f"{x_at_time(squeeze_time_at(i, command)):.1f},{y_at(v):.1f}"
            for i, v in enumerate(series)
        )
        lx = x_at_time(squeeze_time_at(len(series) - 1, command))
        ly = y_at(series[-1])
        stroke_width = 3 if strong else 1.3
        opacity = 0.95 if strong else 0.22
        dash = ' stroke-dasharray="7 5"' if strong else ""
        label_text = (
            f'<text x="{lx - 6:.1f}" y="{ly - 8:.1f}" text-anchor="end" fill="{color}">'
            f'{html.escape(label)}</text>'
            if strong
            else ""
        )
        return (
            f'<polyline points="{line_points}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke_width}"{dash} opacity="{opacity}"/>'
            + label_text
        )

    current_points = " ".join(
        f"{x_at_time(squeeze_time_at(i, command_index)):.1f},{y_at(v):.1f}"
        for i, v in enumerate(values)
    )
    command_x = x_at_time(0.0)
    zero_y = y_at(0.0)
    command_line = (
        f'<line x1="{command_x:.1f}" y1="{top}" x2="{command_x:.1f}" y2="{height - bottom}" '
        'stroke="#8a63d2" stroke-width="2" stroke-dasharray="5 5"/>'
        f'<text x="{command_x + 5:.1f}" y="{top + 12}" fill="#8a63d2">{html.escape(command_label)}</text>'
    )
    zero_line = (
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width - right}" y2="{zero_y:.1f}" '
        'stroke="#9aa6b2" stroke-width="1" stroke-dasharray="3 4" opacity="0.7"/>'
    )

    reference_lines = []
    reference_counts = []
    label_order = [label for label in ("A", "B", "C") if label in reference]
    label_order.extend(sorted(label for label in reference if label not in label_order))
    for label in label_order:
        items = reference.get(label, [])
        color = REFERENCE_COLORS.get(label, "#526070")
        reference_counts.append(f"{label}:{len(items)}")
        aligned = []
        max_pre = max((reference_command_index(ref_row, series) for ref_row, series in items), default=0)
        max_post = max((len(series) - reference_command_index(ref_row, series) for ref_row, series in items), default=0)
        for ref_row, series in items:
            command = reference_command_index(ref_row, series)
            reference_lines.append(polyline(series, command, color, label))
            padded = [0.0] * (max_pre - command) + series + [series[-1]] * max(0, max_post - (len(series) - command))
            aligned.append(padded)
        if aligned:
            reference_lines.append(polyline(average_series(aligned), max_pre, color, f"{label} mean", strong=True))

    return (
        f'<svg class="squeeze-chart" viewBox="0 0 {width} {height}" role="img">'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#b9c2cf"/>'
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#b9c2cf"/>'
        + zero_line
        + "".join(reference_lines)
        + command_line
        + f'<polyline points="{current_points}" fill="none" stroke="#111827" stroke-width="4"/>'
        + "".join(
            f'<circle cx="{x_at_time(squeeze_time_at(i, command_index)):.1f}" cy="{y_at(v):.1f}" r="3.5" fill="#111827"><title>{i}: {v:.4f}</title></circle>'
            for i, v in enumerate(values)
        )
        + f'<text x="{left}" y="{height - 12}" fill="#526070">{html.escape(axis_label)}</text>'
        + f'<text x="{left}" y="16" fill="#526070">{html.escape(title)}</text>'
        + f'<text x="{width - right}" y="{height - 12}" text-anchor="end" fill="#17202c">'
        f'latest {values[-1]:.{latest_digits}f}</text>'
        + "</svg>"
        + '<div class="shape-summary">'
        + summary
        + f'<span>refs <strong>{html.escape(" ".join(reference_counts))}</strong></span>'
        + "</div>"
    )


def squeeze_chart(
    row: dict[str, str],
    series_key: str,
    series_label: str,
    reference: dict[str, list[tuple[dict[str, str], list[float]]]] | None = None,
) -> str:
    values = squeeze_series(row, series_key)
    reference = reference or {}
    if not values:
        return '<div class="empty-chart">No squeeze data for this grasp.</div>'
    metrics = curve_shape_metrics(values)
    stored_score = numeric_or_none(row.get("ab_shape_score_a"))
    if stored_score is None:
        shape_score = float(
            (metrics["curve_late_slope"] < -5.0)
            + (metrics["curve_rebound"] >= 15.0)
            + (metrics["curve_peak_pos"] <= 0.5)
        )
    else:
        shape_score = stored_score
    shape_label = "A-like" if shape_score >= 2 else "B-like"
    summary = (
        f'<span>late slope <strong>{metrics["curve_late_slope"]:.1f}</strong></span>'
        f'<span>rebound <strong>{metrics["curve_rebound"]:.1f}</strong></span>'
        f'<span>peak pos <strong>{metrics["curve_peak_pos"]:.2f}</strong></span>'
        f'<span>shape <strong>{shape_label}</strong> score_A <strong>{shape_score:.0f}/3</strong></span>'
    )
    return aligned_chart(
        values,
        squeeze_command_index(row, values),
        series_label,
        reference,
        squeeze_command_index,
        latest_digits=1,
        height=260,
        summary=summary,
    )


def friction_chart(
    row: dict[str, str],
    finger: str,
    series_label: str,
    reference: dict[str, list[tuple[dict[str, str], list[float]]]] | None = None,
) -> str:
    values = tactile_friction_series(row, finger)
    reference = reference or {}
    if not values:
        return '<div class="empty-chart">No touchData friction data for this grasp.</div>'
    summary = (
        f'<span>mean <strong>{sum(values) / len(values):.3f}</strong></span>'
        f'<span>max <strong>{max(values):.3f}</strong></span>'
    )
    return aligned_chart(
        values,
        friction_command_index(row, values),
        f"{series_label} = tangential / normal",
        reference,
        friction_command_index,
        latest_digits=3,
        height=230,
        summary=summary,
    )


def friction_charts(row: dict[str, str], reference_rows: list[dict[str, str]]) -> str:
    charts = []
    for finger, label in FRICTION_SERIES:
        reference = latest_reference_friction_rows(reference_rows, row, finger)
        charts.append(
            '<div class="squeeze-card">'
            f"<h3>{html.escape(label)}</h3>"
            + friction_chart(row, finger, label, reference)
            + "</div>"
        )
    return '<div class="squeeze-grid">' + "".join(charts) + "</div>"


def proximity_chart(
    row: dict[str, str],
    finger: str,
    series_label: str,
    reference: dict[str, list[tuple[dict[str, str], list[float]]]] | None = None,
) -> str:
    values = hover_proximity_series(row, finger)
    reference = reference or {}
    if not values:
        return '<div class="empty-chart">No hover proximity data for this grasp.</div>'
    summary = (
        f'<span>mean <strong>{sum(values) / len(values):.0f}</strong></span>'
        f'<span>max <strong>{max(values):.0f}</strong></span>'
        f'<span>drift <strong>{values[-1] - values[0] if len(values) >= 2 else 0.0:.0f}</strong></span>'
    )
    return aligned_chart(
        values,
        0,
        series_label,
        reference,
        lambda _row, _series: 0,
        latest_digits=0,
        height=220,
        summary=summary,
        axis_label="seconds from hover start",
        command_label="hover start",
    )


def proximity_charts(row: dict[str, str], reference_rows: list[dict[str, str]]) -> str:
    charts = []
    for finger in TOUCH_FINGER_NAMES:
        reference = latest_reference_proximity_rows(reference_rows, row, finger)
        charts.append(
            '<div class="squeeze-card">'
            f"<h3>{html.escape(finger)} proximity</h3>"
            + proximity_chart(row, finger, f"{finger} hover proximity", reference)
            + "</div>"
        )
    return '<div class="proximity-grid">' + "".join(charts) + "</div>"


def squeeze_charts(row: dict[str, str], reference_rows: list[dict[str, str]]) -> str:
    charts = []
    for series_key, series_label in SQUEEZE_SERIES:
        reference = latest_reference_curve_rows(reference_rows, row, series_key)
        charts.append(
            '<div class="squeeze-card">'
            f"<h3>{html.escape(series_label)}</h3>"
            + squeeze_chart(row, series_key, series_label, reference)
            + "</div>"
        )
    return '<div class="squeeze-grid">' + "".join(charts) + "</div>"


def latest_svg(row: dict[str, str]) -> str:
    hover_values = {
        name: value(row, f"hover_force_delta_{name}_mean")
        for name in FINGER_NAMES
    }
    final_values = {
        name: value(row, f"final_force_delta_{name}")
        for name in FINGER_NAMES
    }
    max_hover = max(hover_values.values(), default=1.0)
    max_final = max(final_values.values(), default=1.0)

    circles = []
    for name in FINGER_NAMES:
        x, y = HAND_POINTS[name]
        hover = hover_values[name]
        final = final_values[name]
        radius = 14 + 22 * (0.0 if max_hover <= 0 else min(1.0, hover / max_hover))
        fill = color_for(hover, max_hover)
        stroke = color_for(final, max_final)
        circles.append(
            f'<circle cx="{x}" cy="{y}" r="{radius:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="4"><title>{name}: hover={hover:.1f}, final={final:.1f}</title></circle>'
        )
        circles.append(
            f'<text x="{x}" y="{y + radius + 18:.1f}" text-anchor="middle">{html.escape(name)}</text>'
        )
    return (
        '<svg class="hand-map" viewBox="0 0 290 230" role="img">'
        '<path d="M60 198 C92 212, 172 214, 218 184 C244 166, 252 130, 232 106 '
        'C205 72, 179 96, 150 102 C121 108, 91 93, 68 112 C42 134, 35 176, 60 198 Z" '
        'fill="#f6f8fb" stroke="#d5dbe5" stroke-width="2"/>'
        + "".join(circles)
        + "</svg>"
    )


def proximity_snapshot(row: dict[str, str]) -> str:
    values = {
        finger: value(row, f"hover_proximity_{finger}_mean")
        for finger in TOUCH_FINGER_NAMES
    }
    max_value = max(values.values(), default=0.0)
    if max_value <= 0:
        return '<div class="proximity-snapshot muted">No hover proximity yet.</div>'
    bars = []
    for finger in TOUCH_FINGER_NAMES:
        current = values[finger]
        ratio = max(0.0, min(1.0, current / max_value))
        bars.append(
            '<div class="proximity-bar">'
            f'<span>{html.escape(finger)}</span>'
            '<div class="proximity-track">'
            f'<div class="proximity-fill" style="width:{ratio * 100:.1f}%"></div>'
            '</div>'
            f'<strong>{current:.0f}</strong>'
            '</div>'
        )
    return (
        '<div class="proximity-snapshot">'
        '<div class="proximity-title">Hover proximity</div>'
        + "".join(bars)
        + '</div>'
    )


def force_heatmap(row: dict[str, str]) -> str:
    heatmap_points = {
        "little": (62, 78),
        "ring": (108, 54),
        "middle": (151, 42),
        "index": (194, 58),
        "thumb": (224, 144),
    }
    source_names = {
        "little": ("little",),
        "ring": ("ring",),
        "middle": ("middle",),
        "index": ("index",),
        "thumb": ("thumb_bend", "thumb_swing"),
    }
    raw_values = {}
    for label, names in source_names.items():
        raw_values[label] = max(
            [
                max(
                    value(row, f"hover_force_delta_{name}_mean"),
                    value(row, f"final_force_delta_{name}"),
                )
                for name in names
            ]
            + [0.0]
        )
    display_values = {label: raw / 100.0 for label, raw in raw_values.items()}
    max_value = max(display_values.values(), default=1.0)
    total_force = sum(display_values.values())
    hottest = max(display_values, key=display_values.get) if display_values else "-"
    ab_proximity_assist = value(row, "ab_proximity_assist_triggered") >= 0.5
    bc_proximity_assist = value(row, "bc_proximity_assist_triggered") >= 0.5
    proximity_assist = ab_proximity_assist or bc_proximity_assist
    thumb_proximity = value(row, "hover_proximity_thumb_mean")
    thumb_proximity_threshold = value(row, "ab_proximity_thumb_threshold") or 169619.0
    proximity_ratio = max(0.0, min(1.0, thumb_proximity / max(1.0, thumb_proximity_threshold * 2.0)))

    blobs = []
    for name, (x, y) in heatmap_points.items():
        current = display_values[name]
        ratio = 0.0 if max_value <= 0 else max(0.0, min(1.0, current / max_value))
        radius = 13 + 30 * (ratio ** 0.5)
        color = color_for(current, max_value)
        blobs.append(
            f'<circle cx="{x}" cy="{y}" r="{radius:.1f}" fill="{color}" '
            f'fill-opacity="{0.30 + 0.62 * ratio:.2f}" stroke="{color}" stroke-width="3">'
            f'<title>{html.escape(name)}: {current:.2f} N</title></circle>'
        )
        if name == "thumb" and thumb_proximity > 0:
            proximity_radius = 18 + 34 * (proximity_ratio ** 0.5)
            proximity_opacity = 0.28 + (0.42 * proximity_ratio if proximity_assist else 0.18 * proximity_ratio)
            proximity_width = 5 if proximity_assist else 2.5
            blobs.append(
                f'<circle cx="{x}" cy="{y}" r="{proximity_radius:.1f}" fill="none" '
                f'stroke="#16a34a" stroke-opacity="{proximity_opacity:.2f}" '
                f'stroke-width="{proximity_width:.1f}">'
                f'<title>thumb proximity: {thumb_proximity:.0f}</title></circle>'
            )
        blobs.append(
            f'<text x="{x}" y="{y + 4}" text-anchor="middle" class="heatmap-value">{current:.1f}N</text>'
        )
        blobs.append(
            f'<text x="{x}" y="{y + radius + 15:.1f}" text-anchor="middle" class="heatmap-label">{html.escape(name)}</text>'
        )

    legend_items = []
    for ratio, label in ((0.18, "low"), (0.55, "mid"), (1.0, "high")):
        sample = max_value * ratio
        legend_items.append(
            '<span>'
            f'<i style="background:{color_for(sample, max_value)}"></i>'
            f'{html.escape(label)}'
            '</span>'
        )

    return (
        '<div class="heatmap-card">'
        '<div class="heatmap-head">'
        '<div><h3>Hand force delta (N)</h3>'
        '<div class="muted">forceAct delta / 100; green ring = thumb proximity</div></div>'
        f'<div class="heatmap-total">{fmt(total_force, 1)}<span>total N</span></div>'
        '</div>'
        '<svg class="force-heatmap" viewBox="0 0 290 230" role="img">'
        '<path d="M60 198 C92 212, 172 214, 218 184 C244 166, 252 130, 232 106 '
        'C205 72, 179 96, 150 102 C121 108, 91 93, 68 112 C42 134, 35 176, 60 198 Z" '
        'fill="#f8fafc" stroke="#dbe1ea" stroke-width="2"/>'
        + "".join(blobs)
        + "</svg>"
        '<div class="heatmap-foot">'
        f'<span>max <strong>{html.escape(hottest)}</strong></span>'
        '<span class="heatmap-legend">' + "".join(legend_items) + '</span>'
        '</div>'
        + (
            '<div class="proximity-assist">'
            f'<span>{("BC prox assist ON" if bc_proximity_assist else "AB prox assist ON") if proximity_assist else "thumb proximity"}</span>'
            f'<strong>{thumb_proximity:.0f}</strong>'
            '</div>'
            if thumb_proximity > 0
            else ""
        )
        + proximity_snapshot(row)
        + '</div>'
    )


def latest_ball_label(row: dict[str, str]) -> str:
    status = (row.get("prediction_status") or "").strip()
    if status == "miss_grasp" or detect_missed_grasp(row)[0]:
        return "NONE"
    for key in ("predicted_label", "stage1_predicted_label", "label"):
        label = (row.get(key) or "").strip()
        if label in BALL_ART:
            return label
    return ""


def displayed_prediction_label(row: dict[str, str]) -> str:
    if (row.get("prediction_status") or "").strip() == "classifying":
        return "?"
    if detect_missed_grasp(row)[0] or (row.get("prediction_status") or "").strip() == "miss_grasp":
        return "NONE"
    return row.get("predicted_label", "") or ""


def displayed_prediction_status(row: dict[str, str]) -> str:
    if detect_missed_grasp(row)[0] or (row.get("prediction_status") or "").strip() == "miss_grasp":
        return "miss_grasp"
    return row.get("prediction_status", "ok") or "ok"


def ball_art_panel(row: dict[str, str]) -> str:
    if displayed_prediction_label(row) == "?":
        return (
            '<div class="pending-art-shell">'
            '<div class="pending-art-frame">?</div>'
            '<div class="ball-art-meta">'
            '<div class="ball-art-title">classifying</div>'
            '<div class="ball-art-line"><strong>正在抓取/判断</strong></div>'
            '<div class="ball-art-line">结果出来后会立刻刷新到这里</div>'
            "</div>"
            "</div>"
        )
    label = latest_ball_label(row)
    if label == "NONE":
        active = fmt(numeric_or_none(row.get("miss_grasp_active_contact_count")), 0)
        closure = fmt(numeric_or_none(row.get("miss_grasp_size_closure_mean")), 1)
        angle = fmt(numeric_or_none(row.get("miss_grasp_size_contact_angle_mean")), 1)
        hover = fmt(numeric_or_none(row.get("miss_grasp_hover_sample_count")), 0)
        force_sum = fmt(numeric_or_none(row.get("miss_grasp_final_force_delta_sum")), 1)
        return (
            '<div class="miss-art-shell">'
            '<div class="miss-art-frame">'
            '<div class="miss-art-badge">没抓到</div>'
            '<div class="miss-art-text">grasp failed</div>'
            "</div>"
            '<div class="ball-art-meta">'
            '<div class="ball-art-title">latest decision</div>'
            '<div class="ball-art-line"><strong>没抓到</strong></div>'
            f'<div class="ball-art-line">active <strong>{active}</strong> closure <strong>{closure}</strong> angle <strong>{angle}</strong></div>'
            f'<div class="ball-art-line">hover <strong>{hover}</strong> force_sum <strong>{force_sum}</strong></div>'
            "</div>"
            "</div>"
        )
    if not label:
        return '<div class="empty-ball-art">No ball decision yet.</div>'
    image = BALL_ART[label]
    confidence_value = numeric_or_none(row.get("prediction_confidence"))
    if confidence_value is None:
        confidence_value = numeric_or_none(row.get("stage1_prediction_confidence"))
    confidence = fmt(confidence_value, 2)
    stage1 = (row.get("stage1_predicted_label") or "").strip()
    status = (row.get("prediction_status") or "").strip() or "ok"
    title = f"ball {label}"
    return (
        f'<div class="ball-art-shell">'
        f'<div class="ball-art-frame">'
        f'<img src="{html.escape(image)}" alt="{html.escape(title)}" class="ball-art-image"/>'
        "</div>"
        '<div class="ball-art-meta">'
        f'<div class="ball-art-badge">{html.escape(label)}</div>'
        f'<div class="ball-art-title">latest decision</div>'
        f'<div class="ball-art-line">final <strong>{html.escape(label)}</strong> '
        f'confidence <strong>{html.escape(confidence)}</strong></div>'
        f'<div class="ball-art-line">stage1 <strong>{html.escape(stage1 or "-")}</strong> '
        f'status <strong>{html.escape(status)}</strong></div>'
        "</div>"
        "</div>"
    )


def render_dashboard(
    rows: list[dict[str, str]],
    output: Path,
    last: int = 20,
    title: str = "RH56F2 tactile live dashboard",
    reference_rows: list[dict[str, str]] | None = None,
) -> None:
    rows = rows[-last:] if last > 0 else rows
    reference_rows = reference_rows if reference_rows is not None else rows
    output.parent.mkdir(parents=True, exist_ok=True)
    latest = rows[-1] if rows else {}
    feature_max = {
        key: max([value(row, key) for row in rows] + [1.0])
        for key, _ in CORE_FEATURES
    }
    force_max = max(
        [
            value(row, f"final_force_delta_{name}")
            for row in rows
            for name in FINGER_NAMES
        ]
        + [1.0]
    )
    hover_max = max(
        [
            value(row, f"hover_force_delta_{name}_mean")
            for row in rows
            for name in FINGER_NAMES
        ]
        + [1.0]
    )

    feature_cards = []
    for key, label in CORE_FEATURES:
        current = value(latest, key)
        feature_cards.append(
            '<section class="metric">'
            f"<h3>{html.escape(label)}</h3>"
            f"<strong>{fmt(current)}</strong>"
            + bar(current / feature_max[key], "latest", fmt(current))
            + "</section>"
        )

    final_bars = [
        bar(value(latest, f"final_force_delta_{name}") / force_max, name, fmt(value(latest, f"final_force_delta_{name}")))
        for name in FINGER_NAMES
    ]
    hover_bars = [
        bar(
            value(latest, f"hover_force_delta_{name}_mean") / hover_max,
            name,
            fmt(value(latest, f"hover_force_delta_{name}_mean")),
        )
        for name in FINGER_NAMES
    ]

    table_rows = []
    for idx, row in enumerate(rows, start=max(1, len(rows) - len(rows) + 1)):
        expected = row.get("label", "")
        if expected == "unknown":
            expected = ""
        table_rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{html.escape(row.get('timestamp', '-'))}</td>"
            f"<td>{html.escape(expected or '-')}</td>"
            f"<td>{html.escape(displayed_prediction_label(row) or '-')}</td>"
            f"<td>{fmt(numeric_or_none(row.get('prediction_confidence')), 2)}</td>"
            f"<td>{fmt(numeric_or_none(row.get('final_angle_middle')))}</td>"
            f"<td>{fmt(numeric_or_none(row.get('hover_thumb_force_delta_max')))}</td>"
            "</tr>"
        )

    latest_prediction = displayed_prediction_label(latest) or "-"
    latest_confidence = fmt(numeric_or_none(latest.get("prediction_confidence")), 2)
    latest_status = displayed_prediction_status(latest)
    stage1_prediction = latest.get("stage1_predicted_label", "-") or "-"
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<script>setTimeout(() => location.reload(), {int(DASHBOARD_REFRESH_SECONDS * 1000)});</script>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #f3f5f8; color: #17202c; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
header {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-end; margin-bottom: 18px; }}
h1 {{ font-size: 24px; margin: 0; }}
.prediction {{ font-size: 18px; padding: 10px 14px; border: 1px solid #cfd6e1; background: white; border-radius: 8px; }}
.grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; align-items: start; }}
.panel, .metric {{ background: white; border: 1px solid #dbe1ea; border-radius: 8px; padding: 16px; }}
.ball-panel {{ overflow: hidden; }}
.current-layout {{ display: grid; grid-template-columns: minmax(560px, 1.2fr) minmax(320px, .8fr); gap: 18px; align-items: stretch; }}
.ball-art-shell {{ display: grid; grid-template-columns: minmax(260px, 340px) 1fr; gap: 18px; align-items: center; }}
.ball-art-frame {{ position: relative; aspect-ratio: 1; overflow: hidden; background: transparent; border: 0; }}
.pending-art-shell {{ display: grid; grid-template-columns: minmax(260px, 340px) 1fr; gap: 18px; align-items: center; }}
.pending-art-frame {{ aspect-ratio: 1; display: flex; align-items: center; justify-content: center; border: 1px dashed #aab6c6; border-radius: 8px; background: #f8fafc; color: #526070; font-size: 120px; font-weight: 700; }}
.miss-art-shell {{ display: grid; grid-template-columns: minmax(260px, 340px) 1fr; gap: 18px; align-items: center; }}
.miss-art-frame {{ position: relative; aspect-ratio: 1; border-radius: 22px; overflow: hidden; background: linear-gradient(160deg, #ffe5e5, #fff7f7 55%, #fde9e9); box-shadow: inset 0 1px 0 rgba(255,255,255,0.75), 0 18px 34px rgba(153,27,27,0.10); display: flex; align-items: center; justify-content: center; flex-direction: column; gap: 10px; }}
.miss-art-frame::before {{ content: ""; position: absolute; inset: 12px; border-radius: 18px; background: radial-gradient(circle at 30% 24%, rgba(255,255,255,0.94), rgba(255,255,255,0.10) 30%, transparent 58%); pointer-events: none; }}
.miss-art-badge {{ position: relative; z-index: 1; font-size: 28px; font-weight: 800; color: #b42318; letter-spacing: 0; }}
.miss-art-text {{ position: relative; z-index: 1; color: #8a2b20; font-size: 15px; }}
.ball-art-image {{ width: 100%; height: 100%; object-fit: contain; display: block; filter: none; }}
.ball-art-meta {{ display: flex; flex-direction: column; gap: 10px; min-width: 0; }}
.ball-art-badge {{ display: inline-flex; align-items: center; justify-content: center; width: 3rem; height: 3rem; border-radius: 999px; background: #17202c; color: white; font-size: 1.2rem; font-weight: 700; }}
.ball-art-title {{ font-size: 20px; font-weight: 700; }}
.ball-art-line {{ color: #526070; font-size: 14px; }}
.empty-ball-art {{ padding: 28px; border: 1px dashed #c8d1dd; border-radius: 8px; color: #667487; background: #f6f8fb; }}
.metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }}
h2, h3 {{ margin: 0 0 12px; font-size: 16px; }}
.metric strong {{ display: block; font-size: 28px; margin-bottom: 10px; }}
.hand-map {{ width: 100%; height: auto; display: block; }}
.heatmap-card {{ height: 100%; min-height: 330px; border-left: 1px solid #e1e6ee; padding-left: 18px; display: flex; flex-direction: column; justify-content: space-between; }}
.heatmap-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
.heatmap-total {{ min-width: 76px; text-align: right; font-size: 26px; line-height: 1; font-weight: 800; font-variant-numeric: tabular-nums; }}
.heatmap-total span {{ display: block; margin-top: 5px; color: #667487; font-size: 11px; font-weight: 600; text-transform: uppercase; }}
.force-heatmap {{ width: 100%; max-height: 285px; height: auto; display: block; }}
.heatmap-value {{ font-size: 10px; font-weight: 800; fill: #17202c; paint-order: stroke; stroke: rgba(255,255,255,0.80); stroke-width: 3px; }}
.heatmap-label {{ font-size: 9px; fill: #526070; }}
.heatmap-foot {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; color: #526070; font-size: 12px; }}
.heatmap-legend {{ display: inline-flex; gap: 8px; align-items: center; }}
.heatmap-legend span {{ display: inline-flex; gap: 4px; align-items: center; }}
.heatmap-legend i {{ width: 10px; height: 10px; display: inline-block; border-radius: 999px; }}
.proximity-assist {{ margin-top: 8px; border: 1px solid #bbf7d0; background: #f0fdf4; color: #166534; border-radius: 8px; padding: 7px 9px; display: flex; justify-content: space-between; gap: 12px; align-items: center; font-size: 12px; }}
.proximity-assist strong {{ font-variant-numeric: tabular-nums; }}
.proximity-snapshot {{ border-top: 1px solid #e1e6ee; padding-top: 10px; margin-top: 10px; display: grid; gap: 6px; }}
.proximity-title {{ font-size: 12px; font-weight: 800; color: #293545; }}
.proximity-bar {{ display: grid; grid-template-columns: 58px 1fr 46px; gap: 8px; align-items: center; font-size: 11px; color: #526070; }}
.proximity-track {{ height: 8px; border-radius: 999px; overflow: hidden; background: #e8edf4; }}
.proximity-fill {{ height: 100%; border-radius: inherit; background: linear-gradient(90deg, #2c79d6, #16a34a); }}
.proximity-bar strong {{ color: #17202c; text-align: right; font-variant-numeric: tabular-nums; }}
.squeeze-chart {{ width: 100%; height: auto; display: block; }}
.empty-chart {{ padding: 28px; background: #f6f8fb; border: 1px dashed #c8d1dd; border-radius: 8px; color: #667487; }}
.shape-summary {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; font-size: 13px; color: #526070; }}
.shape-summary span {{ border: 1px solid #dbe1ea; border-radius: 6px; padding: 6px 8px; background: #f8fafc; }}
.squeeze-grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
.proximity-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
.squeeze-card {{ border: 1px solid #e1e6ee; border-radius: 8px; padding: 12px; background: #fbfcfe; }}
svg text {{ font-size: 10px; fill: #293545; }}
.bar-row {{ display: grid; grid-template-columns: 120px 1fr 72px; gap: 10px; align-items: center; margin: 9px 0; font-size: 13px; }}
.bar-track {{ height: 12px; background: #e8edf4; border-radius: 999px; overflow: hidden; }}
.bar-fill {{ height: 100%; background: linear-gradient(90deg, #2c79d6, #e5484d); }}
.bar-value {{ text-align: right; font-variant-numeric: tabular-nums; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e1e6ee; padding: 8px; text-align: left; }}
th {{ color: #526070; font-weight: 600; }}
.muted {{ color: #667487; font-size: 13px; }}
@media (max-width: 900px) {{ .grid, .metrics, .current-layout, .proximity-grid {{ grid-template-columns: 1fr; }} header {{ display: block; }} .heatmap-card {{ border-left: 0; border-top: 1px solid #e1e6ee; padding-left: 0; padding-top: 16px; }} }}
</style>
</head>
<body>
<main>
<header>
<div>
<h1>{html.escape(title)}</h1>
<div class="muted">Auto refreshes every {DASHBOARD_REFRESH_SECONDS:.1f}s. Heatmap color and area show finger force in N.</div>
</div>
<div class="prediction">stage1 <strong>{html.escape(stage1_prediction)}</strong> final <strong>{html.escape(latest_prediction)}</strong> confidence <strong>{latest_confidence}</strong> status <strong>{html.escape(latest_status)}</strong></div>
</header>
<section class="panel ball-panel" style="margin-bottom:16px">
<h2>Current ball</h2>
<div class="current-layout">
<div>{ball_art_panel(latest) if latest else '<div class="empty-ball-art">No rows yet.</div>'}</div>
<div>{force_heatmap(latest) if latest else '<div class="empty-ball-art">No force data yet.</div>'}</div>
</div>
</section>
<section class="metrics">
{''.join(feature_cards)}
</section>
<section class="grid">
<div class="panel">
<h2>Latest force bars</h2>
<h3>Final close force</h3>
{''.join(final_bars)}
<h3>Hover mean force</h3>
{''.join(hover_bars)}
</div>
</section>
<section class="panel" style="margin-top:16px">
<h2>Hover proximity curves</h2>
<div class="muted">SDK touchData proximity sampled during hover. Reference curves show the latest labelled A/B/C batches when available.</div>
{proximity_charts(latest, reference_rows) if latest else '<p>No rows yet.</p>'}
</section>
<section class="panel" style="margin-top:16px">
<h2>Squeeze pressure curves</h2>
{squeeze_charts(latest, reference_rows) if latest else '<p>No rows yet.</p>'}
</section>
<section class="panel" style="margin-top:16px">
<h2>TouchData friction ratio curves</h2>
<div class="muted">Friction ratio uses SDK touchData tangential / normal. The curve starts from the last hover samples and continues through squeeze.</div>
{friction_charts(latest, reference_rows) if latest else '<p>No rows yet.</p>'}
</section>
<section class="panel" style="margin-top:16px">
<h2>Recent predictions</h2>
<table>
<thead><tr><th>#</th><th>time</th><th>expected</th><th>predicted</th><th>conf</th><th>middle angle</th><th>thumb hover max</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
</section>
</main>
</body>
</html>
"""
    output.write_text(html_text, encoding="utf-8")


def render_dashboard_from_csv(
    samples: Path,
    output: Path,
    last: int = 20,
    reference_samples: Path | None = None,
) -> None:
    rows = read_feature_rows(samples) if samples.exists() else []
    reference_rows = read_feature_rows(reference_samples) if reference_samples is not None and reference_samples.exists() else None
    render_dashboard(rows, output, last=last, reference_rows=reference_rows)


def _string_row(row: dict[str, object]) -> dict[str, str]:
    return {str(key): "" if value is None else str(value) for key, value in row.items()}


def render_dashboard_preview_from_csv(
    samples: Path,
    output: Path,
    preview_row: dict[str, object],
    last: int = 20,
    reference_samples: Path | None = None,
) -> None:
    rows = read_feature_rows(samples) if samples.exists() else []
    reference_rows = read_feature_rows(reference_samples) if reference_samples is not None and reference_samples.exists() else None
    render_dashboard([*rows, _string_row(preview_row)], output, last=last, reference_rows=reference_rows)


def render_pending_dashboard_from_csv(
    samples: Path,
    output: Path,
    last: int = 20,
    reference_samples: Path | None = None,
    notes: str = "live classifying",
) -> None:
    render_dashboard_preview_from_csv(
        samples,
        output,
        {
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "label": "unknown",
            "predicted_label": "?",
            "stage1_predicted_label": "?",
            "prediction_confidence": "",
            "prediction_distance": "",
            "prediction_status": "classifying",
            "notes": notes,
        },
        last=last,
        reference_samples=reference_samples,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path(__file__).with_name("live_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("live_dashboard.html"))
    parser.add_argument("--reference-samples", type=Path, default=Path(__file__).with_name("samples.csv"))
    parser.add_argument("--last", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    render_dashboard_from_csv(args.samples, args.output, args.last, args.reference_samples)
    print(f"wrote dashboard: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
