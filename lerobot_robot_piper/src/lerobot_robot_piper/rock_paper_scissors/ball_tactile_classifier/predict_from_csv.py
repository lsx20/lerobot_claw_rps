#!/usr/bin/env python3
"""Predict ball labels for collected tactile sample rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .common import filter_rows_by_quality, load_model, predict_row, read_feature_rows
except ImportError:  # Allow: python3 predict_from_csv.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import filter_rows_by_quality, load_model, predict_row, read_feature_rows  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path(__file__).with_name("samples.csv"))
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("model.json"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--min-active-contacts",
        type=float,
        default=None,
        help="Skip rows with fewer active contacts. Default uses the value stored in the model.",
    )
    parser.add_argument(
        "--min-hover-samples",
        type=float,
        default=None,
        help="Skip rows with fewer hover samples. Default uses the value stored in the model.",
    )
    parser.add_argument(
        "--min-squeeze-samples",
        type=float,
        default=None,
        help="Skip rows with fewer squeeze samples. Default uses the value stored in the model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = load_model(args.model)
    rows = read_feature_rows(args.samples)
    if args.limit > 0:
        rows = rows[-args.limit :]
    min_active_contacts = (
        float(model.get("min_active_contacts", 0.0))
        if args.min_active_contacts is None
        else args.min_active_contacts
    )
    min_hover_samples = (
        float(model.get("min_hover_samples", 0.0))
        if args.min_hover_samples is None
        else args.min_hover_samples
    )
    min_squeeze_samples = (
        float(model.get("min_squeeze_samples", 0.0))
        if args.min_squeeze_samples is None
        else args.min_squeeze_samples
    )
    rows, skipped = filter_rows_by_quality(
        rows,
        min_active_contacts=min_active_contacts,
        min_hover_samples=min_hover_samples,
        min_squeeze_samples=min_squeeze_samples,
    )
    print("offline prediction only: this command reads CSV rows and does not move the RH56F2 hand.")
    if min_active_contacts > 0 or min_hover_samples > 0 or min_squeeze_samples > 0:
        print(
            f"quality filter: min_active_contacts={min_active_contacts:g}, "
            f"min_hover_samples={min_hover_samples:g}, "
            f"min_squeeze_samples={min_squeeze_samples:g}, "
            f"used={len(rows)}, skipped={len(skipped)}"
        )

    total = 0
    correct = 0
    confusion: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows, start=1):
        result = predict_row(row, model)
        expected = row.get(str(model.get("label_column", "label")), "")
        predicted = str(result["label"])
        if expected:
            total += 1
            if predicted == expected:
                correct += 1
            confusion[(expected, predicted)] = confusion.get((expected, predicted), 0) + 1
        print(
            f"{index:03d} expected={expected or '-'} "
            f"predicted={predicted} "
            f"confidence={float(result['confidence']):.2f} "
            f"distance={float(result['distance']):.2f}"
        )
    if total:
        print(f"accuracy={correct}/{total} ({correct / total:.1%})")
        mistakes = [
            (expected, predicted, count)
            for (expected, predicted), count in sorted(confusion.items())
            if expected != predicted
        ]
        if mistakes:
            print("mistakes: " + ", ".join(f"{e}->{p}:{c}" for e, p, c in mistakes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
