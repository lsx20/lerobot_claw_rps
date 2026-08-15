#!/usr/bin/env python3
"""Plot stored squeeze pressure curves for selected trials."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from .common import average_series, parse_series, resample_series
except ImportError:
    from common import average_series, parse_series, resample_series  # type: ignore


SERIES = [
    ("thumb", "squeeze_thumb_force_delta_series", "thumb force delta"),
    ("core_sum", "squeeze_force_delta_sum_series", "core four-finger force delta"),
    ("middle", "squeeze_middle_force_delta_series", "middle force delta"),
]

COLORS = {
    "A": "#d65f00",
    "B": "#0072b2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path(__file__).with_name("samples.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("squeeze_curves_current.png"))
    parser.add_argument("--labels", default="AB")
    parser.add_argument("--trial-id", action="append", default=[])
    parser.add_argument("--trial-prefix", action="append", default=[])
    parser.add_argument("--since-date")
    parser.add_argument("--resample", type=int, default=80)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    return parser.parse_args()


def read_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    with args.samples.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = set(args.labels.replace(",", ""))
    rows = [row for row in rows if row.get("label") in labels and row.get("squeeze_sample_count")]
    if args.trial_id:
        wanted = set(args.trial_id)
        rows = [row for row in rows if row.get("trial_id") in wanted]
    if args.trial_prefix:
        prefixes = tuple(args.trial_prefix)
        rows = [row for row in rows if (row.get("trial_id") or "").startswith(prefixes)]
    if args.since_date:
        rows = [row for row in rows if (row.get("timestamp") or "")[:10] >= args.since_date]
    return rows


def command_index(row: dict[str, str]) -> int:
    try:
        return int(float(row.get("squeeze_command_sample_index") or 0.0))
    except ValueError:
        return 0


def aligned_seconds(length: int, command_sample: int, rate_hz: float) -> list[float]:
    return [(idx - command_sample) / rate_hz for idx in range(length)]


def median_early_slope(rows: list[dict[str, str]], field: str, rate_hz: float) -> float:
    slopes = []
    for row in rows:
        values = parse_series(row.get(field))
        start = command_index(row)
        end = min(len(values) - 1, start + max(2, int(rate_hz * 0.5)))
        if len(values) >= 2 and end > start:
            slopes.append((values[end] - values[start]) / ((end - start) / rate_hz))
    if not slopes:
        return 0.0
    slopes.sort()
    return slopes[len(slopes) // 2]


def main() -> int:
    args = parse_args()
    rows = read_rows(args)
    grouped = {label: [row for row in rows if row.get("label") == label] for label in sorted(set(args.labels.replace(",", "")))}
    if not any(grouped.values()):
        raise SystemExit("no squeeze rows matched the filters")

    fig, axes = plt.subplots(len(SERIES), 1, figsize=(11, 10), sharex=True)
    fig.suptitle("Squeeze pressure curves, aligned at squeeze command", fontsize=15)

    for axis, (series_name, field, title) in zip(axes, SERIES, strict=True):
        max_len = max((len(parse_series(row.get(field))) for row in rows), default=0)
        x_mean = aligned_seconds(args.resample, int(args.resample * 0.15), args.rate_hz)
        for label, label_rows in grouped.items():
            color = COLORS.get(label, "black")
            resampled = []
            for row in label_rows:
                values = parse_series(row.get(field))
                if not values:
                    continue
                x = aligned_seconds(len(values), command_index(row), args.rate_hz)
                axis.plot(x, values, color=color, alpha=0.22, linewidth=1.2)
                resampled.append(resample_series(values, args.resample))
            if resampled:
                mean_values = average_series(resampled)
                axis.plot(x_mean, mean_values, color=color, linewidth=3.0, label=f"{label} mean")
        axis.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
        axis.set_title(title, loc="left")
        axis.set_ylabel("force delta")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left")
        axis.text(
            0.99,
            0.92,
            " | ".join(
                f"{label} early slope={median_early_slope(label_rows, field, args.rate_hz):.0f}/s"
                for label, label_rows in grouped.items()
                if label_rows
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#333333",
        )
        if max_len:
            axis.set_xlim(left=-0.4)

    axes[-1].set_xlabel("seconds relative to squeeze command")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"saved plot: {args.output}")
    for label, label_rows in grouped.items():
        trial_ids = sorted(set(row.get("trial_id", "-") for row in label_rows))
        print(f"{label}: {len(label_rows)} rows, trials={','.join(trial_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
