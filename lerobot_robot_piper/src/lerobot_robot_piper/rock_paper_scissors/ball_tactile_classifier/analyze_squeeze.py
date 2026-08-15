#!/usr/bin/env python3
"""Compare squeeze-test features by ball label before training a classifier."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import statistics
import sys
from pathlib import Path

try:
    from .common import (
        ab_shape_decision,
        average_series,
        curve_shape_metrics,
        filter_rows_by_quality,
        normalized_series_distance,
        numeric_or_none,
        parse_series,
        read_feature_rows,
    )
except ImportError:  # Allow: python3 analyze_squeeze.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (  # type: ignore
        ab_shape_decision,
        average_series,
        curve_shape_metrics,
        filter_rows_by_quality,
        normalized_series_distance,
        numeric_or_none,
        parse_series,
        read_feature_rows,
    )


SQUEEZE_FEATURES = [
    "squeeze_force_delta_sum_max",
    "squeeze_force_delta_sum_last",
    "squeeze_thumb_force_delta_max",
    "squeeze_middle_force_delta_max",
    "squeeze_angle_delta_mean",
    "squeeze_stiffness_sum_per_angle",
]

SQUEEZE_SERIES = [
    ("middle", "squeeze_middle_force_delta_series"),
    ("thumb", "squeeze_thumb_force_delta_series"),
    ("core_sum", "squeeze_force_delta_sum_series"),
]

SHAPE_FEATURE_SETS = [
    ("rebound+late_slope", ("curve_rebound", "curve_late_slope")),
    ("peak+rebound", ("curve_peak", "curve_rebound")),
    ("peak+rebound+late_slope", ("curve_peak", "curve_rebound", "curve_late_slope")),
]


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def mad(values: list[float]) -> float:
    if not values:
        return 0.0
    center = median(values)
    return median([abs(value - center) for value in values])


def add_series(left: list[float], right: list[float]) -> list[float]:
    length = max(len(left), len(right))
    return [
        (left[idx] if idx < len(left) else 0.0) + (right[idx] if idx < len(right) else 0.0)
        for idx in range(length)
    ]


def row_series(row: dict[str, str], series_key: str) -> list[float]:
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


def has_bad_note(row: dict[str, str]) -> bool:
    notes = (row.get("notes") or "").lower()
    return "void" in notes or "bad_grasp" in notes


def filter_by_metadata(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    result = rows
    if args.trial_id:
        wanted = set(args.trial_id)
        result = [row for row in result if row.get("trial_id") in wanted]
    if args.trial_prefix:
        prefixes = tuple(args.trial_prefix)
        result = [row for row in result if (row.get("trial_id") or "").startswith(prefixes)]
    if args.since_date:
        result = [row for row in result if (row.get("timestamp") or "")[:10] >= args.since_date]
    if args.exclude_bad_notes:
        result = [row for row in result if not has_bad_note(row)]
    return result


def best_threshold_rule(
    values: list[tuple[str, float]],
    left_label: str,
    right_label: str,
) -> tuple[int, int, str, float] | None:
    if not values:
        return None
    unique = sorted(set(value for _, value in values))
    if len(unique) == 1:
        candidates = [unique[0]]
    else:
        candidates = [(unique[idx] + unique[idx + 1]) / 2.0 for idx in range(len(unique) - 1)]
        candidates.insert(0, unique[0] - 1e-6)
        candidates.append(unique[-1] + 1e-6)

    best: tuple[int, int, str, float] | None = None
    for threshold in candidates:
        for direction in (">=", "<="):
            correct = 0
            for label, value in values:
                if direction == ">=":
                    predicted = left_label if value >= threshold else right_label
                else:
                    predicted = left_label if value <= threshold else right_label
                correct += int(predicted == label)
            candidate = (correct, len(values), direction, threshold)
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best


def centroid_predict(
    train_rows: list[tuple[str, dict[str, float]]],
    metrics: dict[str, float],
    feature_names: tuple[str, ...],
    labels: list[str],
) -> str | None:
    scales = {}
    for feature in feature_names:
        values = [row_metrics[feature] for _, row_metrics in train_rows]
        if not values:
            return None
        span = max(values) - min(values)
        scales[feature] = max(1.0, span)

    distances: dict[str, float] = {}
    for label in labels:
        candidates = [row_metrics for row_label, row_metrics in train_rows if row_label == label]
        if not candidates:
            return None
        centroid = {
            feature: statistics.fmean(row_metrics[feature] for row_metrics in candidates)
            for feature in feature_names
        }
        distances[label] = sum(
            ((metrics[feature] - centroid[feature]) / scales[feature]) ** 2
            for feature in feature_names
        ) ** 0.5
    return min(distances, key=distances.get)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path(__file__).with_name("samples.csv"))
    parser.add_argument("--labels", default="AB", help="Labels to compare, e.g. AB or ABC.")
    parser.add_argument("--min-hover-samples", type=float, default=5.0)
    parser.add_argument("--min-squeeze-samples", type=float, default=5.0)
    parser.add_argument(
        "--trial-id",
        action="append",
        default=[],
        help="Only include this trial_id. Can be repeated.",
    )
    parser.add_argument(
        "--trial-prefix",
        action="append",
        default=[],
        help="Only include trial_id values starting with this prefix, e.g. 20260730_.",
    )
    parser.add_argument(
        "--since-date",
        help="Only include rows whose timestamp date is >= YYYY-MM-DD.",
    )
    parser.add_argument(
        "--exclude-bad-notes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore rows whose notes contain void or bad_grasp.",
    )
    parser.add_argument("--show-rows", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_feature_rows(args.samples)
    raw_count = len(rows)
    rows = filter_by_metadata(rows, args)
    metadata_count = len(rows)
    rows, skipped = filter_rows_by_quality(
        rows,
        min_active_contacts=1,
        min_hover_samples=args.min_hover_samples,
        min_squeeze_samples=args.min_squeeze_samples,
    )
    labels = [label for label in args.labels.replace(",", "") if label.strip()]
    grouped: dict[str, list[dict[str, str]]] = {
        label: [row for row in rows if row.get("label") == label]
        for label in labels
    }

    print(
        "quality filter: "
        f"min_hover_samples={args.min_hover_samples:g}, "
        f"min_squeeze_samples={args.min_squeeze_samples:g}, "
        f"used={sum(len(value) for value in grouped.values())}, skipped={skipped and len(skipped)}"
    )
    if raw_count != metadata_count:
        print(f"metadata filter: raw={raw_count}, after_metadata={metadata_count}")
    if args.trial_id:
        print("trial_id filter: " + ", ".join(args.trial_id))
    if args.trial_prefix:
        print("trial_prefix filter: " + ", ".join(args.trial_prefix))
    if args.since_date:
        print(f"since_date filter: {args.since_date}")
    for label in labels:
        print(f"{label}: {len(grouped[label])} squeeze samples")
    if any(len(grouped[label]) == 0 for label in labels):
        print("not enough squeeze-test data yet.")
        return 0

    if args.show_rows:
        print()
        print("rows:")
        for label in labels:
            print(label)
            for row in grouped[label]:
                middle = numeric_or_none(row.get("squeeze_middle_force_delta_max"))
                total = numeric_or_none(row.get("squeeze_force_delta_sum_max"))
                stiffness = numeric_or_none(row.get("squeeze_stiffness_sum_per_angle"))
                series = row.get("squeeze_middle_force_delta_series", "")
                print(
                    f"  {row.get('timestamp', '-')} "
                    f"trial={row.get('trial_id', '-')} repeat={row.get('repeat_index', '-')} "
                    f"middle_max={middle or 0:.1f} total_max={total or 0:.1f} "
                    f"stiffness={stiffness or 0:.1f} series={series or '-'}"
                )

    print()
    print("feature summary: min / median / max")
    values_by_feature: dict[str, dict[str, list[float]]] = {}
    for feature in SQUEEZE_FEATURES:
        values_by_feature[feature] = {}
        print(feature)
        for label in labels:
            values = [
                value
                for row in grouped[label]
                if (value := numeric_or_none(row.get(feature))) is not None
            ]
            values_by_feature[feature][label] = values
            if values:
                print(
                    f"  {label}: "
                    f"{min(values):.1f} / {median(values):.1f} / {max(values):.1f}"
                )
            else:
                print(f"  {label}: -")

    if len(labels) == 2:
        left, right = labels
        print()
        print(f"separation estimate: {left} vs {right}")
        for feature in SQUEEZE_FEATURES:
            left_values = values_by_feature[feature][left]
            right_values = values_by_feature[feature][right]
            if not left_values or not right_values:
                continue
            gap = abs(median(left_values) - median(right_values))
            spread = mad(left_values) + mad(right_values)
            score = gap / max(1.0, spread)
            print(
                f"{feature}: median_gap={gap:.1f}, "
                f"spread={spread:.1f}, score={score:.2f}"
            )

    print()
    print("curve shape summary from stored time series")
    metrics_by_series: dict[str, list[tuple[str, dict[str, str], dict[str, float]]]] = {}
    for series_name, series_field in SQUEEZE_SERIES:
        print(f"{series_name} ({series_field})")
        metrics_by_series[series_name] = []
        shape_values: dict[str, dict[str, list[float]]] = {
            metric: {label: [] for label in labels}
            for metric in curve_shape_metrics([0.0]).keys()
        }
        for label in labels:
            for row in grouped[label]:
                series = row_series(row, series_name)
                if not series:
                    continue
                metrics = curve_shape_metrics(series)
                metrics_by_series[series_name].append((label, row, metrics))
                for metric, value in metrics.items():
                    shape_values[metric][label].append(value)
        for metric, by_label in shape_values.items():
            if not all(by_label[label] for label in labels):
                continue
            values_text = []
            for label in labels:
                values_text.append(f"{label}_median={median(by_label[label]):.2f}")
            line = f"  {metric}: " + " ".join(values_text)
            if len(labels) == 2:
                left, right = labels
                left_values = by_label[left]
                right_values = by_label[right]
                gap = abs(median(left_values) - median(right_values))
                spread = mad(left_values) + mad(right_values)
                line += f" gap={gap:.2f} score={gap / max(1.0, spread):.2f}"
            print(line)

    if len(labels) == 2:
        left, right = labels
        print()
        print(f"best single-threshold curve rules: {left} vs {right}")
        for series_name, _ in SQUEEZE_SERIES:
            metric_rows = metrics_by_series.get(series_name, [])
            if not metric_rows:
                continue
            print(series_name)
            for metric in curve_shape_metrics([0.0]).keys():
                values = [(label, metrics[metric]) for label, _, metrics in metric_rows]
                best = best_threshold_rule(values, left, right)
                if best is None:
                    continue
                correct, total, direction, threshold = best
                print(
                    f"  {metric}: {correct}/{total} ({correct / total:.1%}) "
                    f"rule: {left} if {metric} {direction} {threshold:.2f} else {right}"
                )

        print()
        print("leave-one-out shape feature sets")
        for series_name, _ in SQUEEZE_SERIES:
            metric_rows = metrics_by_series.get(series_name, [])
            if not metric_rows:
                continue
            print(series_name)
            for set_name, feature_names in SHAPE_FEATURE_SETS:
                correct = 0
                total = 0
                for idx, (label, _, metrics) in enumerate(metric_rows):
                    train_rows = [
                        (other_label, other_metrics)
                        for other_idx, (other_label, _, other_metrics) in enumerate(metric_rows)
                        if other_idx != idx
                    ]
                    predicted = centroid_predict(train_rows, metrics, feature_names, labels)
                    if predicted is None:
                        continue
                    total += 1
                    correct += int(predicted == label)
                if total:
                    print(f"  {set_name}: {correct}/{total} ({correct / total:.1%})")

    print()
    print("curve leave-one-out nearest reference:")
    for series_name, _ in SQUEEZE_SERIES:
        curve_rows = []
        for label in labels:
            for row in grouped[label]:
                series = row_series(row, series_name)
                if series:
                    curve_rows.append((label, row, series_name, series))
        if not curve_rows:
            continue
        print()
        print(series_name)
        correct = 0
        total = 0
        for idx, (label, row, series_name, series) in enumerate(curve_rows):
            references = {}
            for candidate in labels:
                references[candidate] = average_series([
                    other_series
                    for other_idx, (other_label, _, other_series_name, other_series) in enumerate(curve_rows)
                    if other_idx != idx and other_label == candidate and other_series_name == series_name
                ])
            if any(not references[candidate] for candidate in labels):
                continue
            distances = {
                candidate: normalized_series_distance(series, references[candidate])
                for candidate in labels
            }
            predicted = min(distances, key=distances.get)
            total += 1
            correct += int(predicted == label)
            distance_text = " ".join(f"dist_{candidate}={distances[candidate]:.2f}" for candidate in labels)
            print(
                f"  {row.get('timestamp', '-')} expected={label} predicted={predicted} "
                f"series={series_name} {distance_text}"
            )
        if total:
            print(f"{series_name}_curve_accuracy={correct}/{total} ({correct / total:.1%})")

    if set(labels) == {"A", "B"}:
        print()
        print("A/B shape rule by squeeze series")
        for series_name, _ in SQUEEZE_SERIES:
            shape_rows = [
                (label, row, row_series(row, series_name))
                for label in labels
                for row in grouped[label]
                if row_series(row, series_name)
            ]
            if not shape_rows:
                continue
            print(series_name)
            correct = 0
            for label, row, series in shape_rows:
                predicted, confidence, metrics = ab_shape_decision(series)
                correct += int(predicted == label)
                print(
                    f"  {row.get('timestamp', '-')} expected={label} predicted={predicted} "
                    f"conf={confidence:.2f} late_slope={metrics['curve_late_slope']:.1f} "
                    f"rebound={metrics['curve_rebound']:.1f} peak_pos={metrics['curve_peak_pos']:.2f} "
                    f"score_A={metrics['ab_shape_score_a']:.0f}"
                )
            total = len(shape_rows)
            print(f"{series_name}_shape_accuracy={correct}/{total} ({correct / total:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
