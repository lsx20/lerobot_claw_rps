#!/usr/bin/env python3
"""Train a small nearest-centroid classifier from collected tactile samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .common import (
        feature_columns_from_rows,
        filter_rows_by_quality,
        read_feature_rows,
        save_model,
        train_nearest_centroid,
    )
except ImportError:  # Allow: python3 train_classifier.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (  # type: ignore
        feature_columns_from_rows,
        filter_rows_by_quality,
        read_feature_rows,
        save_model,
        train_nearest_centroid,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path(__file__).with_name("samples.csv"))
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("model.json"))
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--min-active-contacts",
        type=float,
        default=1.0,
        help="Skip rows with fewer active contacts. Default skips zero-contact rows.",
    )
    parser.add_argument(
        "--min-hover-samples",
        type=float,
        default=0.0,
        help="Skip rows with fewer hover samples. Use this after collecting lift/hover data.",
    )
    parser.add_argument(
        "--min-squeeze-samples",
        type=float,
        default=0.0,
        help="Skip rows with fewer squeeze samples. Use this after collecting --squeeze-test data.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Optional explicit feature list. Default uses stable size/contact/force features.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_feature_rows(args.samples)
    rows, skipped = filter_rows_by_quality(
        rows,
        min_active_contacts=args.min_active_contacts,
        min_hover_samples=args.min_hover_samples,
        min_squeeze_samples=args.min_squeeze_samples,
    )
    features = feature_columns_from_rows(rows, args.features)
    model = train_nearest_centroid(rows, features, args.label_column)
    model["source_samples"] = str(args.samples)
    model["min_active_contacts"] = args.min_active_contacts
    model["min_hover_samples"] = args.min_hover_samples
    model["min_squeeze_samples"] = args.min_squeeze_samples
    save_model(args.model, model)

    print(f"saved model: {args.model}")
    print(
        f"quality filter: min_active_contacts={args.min_active_contacts:g}, "
        f"min_hover_samples={args.min_hover_samples:g}, "
        f"min_squeeze_samples={args.min_squeeze_samples:g}, "
        f"used={len(rows)}, skipped={len(skipped)}"
    )
    print("features: " + ", ".join(features))
    for label, count in dict(model["counts"]).items():
        print(f"  {label}: {count} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
