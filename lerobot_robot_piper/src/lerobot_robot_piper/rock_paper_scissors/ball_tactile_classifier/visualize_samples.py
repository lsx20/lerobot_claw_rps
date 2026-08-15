#!/usr/bin/env python3
"""Visualize collected tactile feature rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .common import FINGER_NAMES, numeric_or_none, read_feature_rows
except ImportError:  # Allow: python3 visualize_samples.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import FINGER_NAMES, numeric_or_none, read_feature_rows  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path(__file__).with_name("samples.csv"))
    parser.add_argument("--last", type=int, default=30)
    return parser.parse_args()


def grouped_values(rows: list[dict[str, str]], column: str) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for row in rows:
        value = numeric_or_none(row.get(column))
        if value is None:
            continue
        result.setdefault(row.get("label", "unknown"), []).append(value)
    return result


def main() -> int:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required: python3 -m pip install matplotlib") from exc

    rows = read_feature_rows(args.samples)
    if args.last > 0:
        rows = rows[-args.last :]
    if not rows:
        raise SystemExit(f"No rows found in {args.samples}")

    labels = [row.get("label", "") for row in rows]
    x = list(range(len(rows)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0][0].set_title("Size: closure mean at first contact")
    axes[0][0].plot(x, [numeric_or_none(row.get("size_closure_mean")) or 0.0 for row in rows], marker="o")
    axes[0][0].set_xticks(x, labels, rotation=45, ha="right")

    axes[0][1].set_title("Final force delta sum")
    axes[0][1].plot(x, [numeric_or_none(row.get("final_force_delta_sum")) or 0.0 for row in rows], marker="o")
    axes[0][1].set_xticks(x, labels, rotation=45, ha="right")

    latest = rows[-1]
    force_values = [
        numeric_or_none(latest.get(f"final_force_delta_{name}")) or 0.0
        for name in FINGER_NAMES
    ]
    axes[1][0].set_title(f"Latest pseudo tactile heatmap: {latest.get('label', 'unknown')}")
    bars = axes[1][0].bar(FINGER_NAMES, force_values)
    max_force = max(force_values) if force_values else 1.0
    for bar, value in zip(bars, force_values, strict=True):
        ratio = 0.0 if max_force <= 0 else min(1.0, value / max_force)
        bar.set_color((ratio, 0.25, 1.0 - ratio))
    axes[1][0].tick_params(axis="x", rotation=30)

    columns = [
        "size_closure_mean",
        "final_force_delta_sum",
        "hover_thumb_force_delta_mean",
        "hover_force_delta_sum_mean",
    ]
    axes[1][1].set_title("Class feature spread")
    for column in columns:
        values_by_label = grouped_values(rows, column)
        if not values_by_label:
            continue
        xs = []
        ys = []
        for idx, (_, values) in enumerate(sorted(values_by_label.items())):
            xs.extend([idx] * len(values))
            ys.extend(values)
        axes[1][1].scatter(xs, ys, label=column, alpha=0.7)
    axes[1][1].legend(loc="best")

    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
