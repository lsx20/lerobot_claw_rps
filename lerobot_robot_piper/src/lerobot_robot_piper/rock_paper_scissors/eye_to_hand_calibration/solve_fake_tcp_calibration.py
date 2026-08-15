#!/usr/bin/env python3
"""Solve fixed-D405 camera point to Piper fake-TCP target calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import apply_transform, load_point_pairs, solve_rigid_transform, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("fake_tcp_samples.csv"))
    parser.add_argument("--output", type=Path, default=Path("fake_tcp_calibration.json"))
    parser.add_argument(
        "--exclude",
        default="",
        help="comma-separated zero-based sample indices to exclude from fitting, e.g. 4,10",
    )
    parser.add_argument(
        "--holdout-every",
        type=int,
        default=0,
        help="optional: use every Nth sample as holdout validation instead of fitting it",
    )
    return parser.parse_args()


def parse_exclude(value: str, sample_count: int) -> np.ndarray:
    if not value.strip():
        return np.array([], dtype=int)
    indices = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    bad = [index for index in indices if index < 0 or index >= sample_count]
    if bad:
        raise ValueError(f"excluded sample indices out of range: {bad}")
    return np.asarray(indices, dtype=int)


def split_indices(sample_count: int, holdout_every: int, excluded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.setdiff1d(np.arange(sample_count), excluded)
    if len(all_indices) < 3:
        raise ValueError("fewer than 3 samples remain after exclusions")
    if holdout_every <= 1:
        return all_indices, np.array([], dtype=int)
    holdout = all_indices[(all_indices + 1) % holdout_every == 0]
    train = np.setdiff1d(all_indices, holdout)
    if len(train) < 3:
        raise ValueError("holdout split leaves fewer than 3 training samples")
    return train, holdout


def error_stats(predicted: np.ndarray, target: np.ndarray) -> dict[str, object]:
    vectors = predicted - target
    distances = np.linalg.norm(vectors, axis=1)
    return {
        "count": int(len(distances)),
        "rms_m": float(np.sqrt(np.mean(distances**2))) if len(distances) else None,
        "mean_m": float(np.mean(distances)) if len(distances) else None,
        "median_m": float(np.median(distances)) if len(distances) else None,
        "max_m": float(np.max(distances)) if len(distances) else None,
        "errors_m": [float(value) for value in distances],
        "error_vectors_m": vectors.tolist(),
    }


def main() -> int:
    args = parse_args()
    camera_points, fake_tcp_points, rows = load_point_pairs(args.input)
    excluded_idx = parse_exclude(args.exclude, len(camera_points))
    train_idx, holdout_idx = split_indices(len(camera_points), args.holdout_every, excluded_idx)

    transform = solve_rigid_transform(camera_points[train_idx], fake_tcp_points[train_idx])
    train_predicted = apply_transform(transform, camera_points[train_idx])
    all_predicted = apply_transform(transform, camera_points)

    result: dict[str, object] = {
        "calibration_type": "fixed_camera_to_fake_tcp_target_point",
        "meaning": "target_base_point = T_base_fake_tcp_target_from_camera * camera_point",
        "input": str(args.input),
        "sample_count": int(len(camera_points)),
        "excluded_sample_indices": [int(index) for index in excluded_idx],
        "train_sample_count": int(len(train_idx)),
        "holdout_sample_count": int(len(holdout_idx)),
        "units": {
            "translation": "meter",
            "rotation_matrix": "dimensionless",
        },
        "T_base_fake_tcp_target_from_camera": transform.tolist(),
        "train_error": error_stats(train_predicted, fake_tcp_points[train_idx]),
        "all_error": error_stats(all_predicted, fake_tcp_points),
        "sample_rows": rows,
    }
    if len(holdout_idx):
        holdout_predicted = apply_transform(transform, camera_points[holdout_idx])
        result["holdout_error"] = error_stats(holdout_predicted, fake_tcp_points[holdout_idx])

    write_json(args.output, result)
    print(f"wrote {args.output}")
    print(f"samples: {len(camera_points)} train: {len(train_idx)} holdout: {len(holdout_idx)}")
    if len(excluded_idx):
        print(f"excluded: {','.join(str(int(index)) for index in excluded_idx)}")
    print(f"train RMS: {result['train_error']['rms_m']:.6f} m")
    print(f"all RMS:   {result['all_error']['rms_m']:.6f} m")
    print(f"all max:   {result['all_error']['max_m']:.6f} m")
    if len(holdout_idx):
        holdout_error = result["holdout_error"]
        print(f"holdout RMS: {holdout_error['rms_m']:.6f} m")
        print(f"holdout max: {holdout_error['max_m']:.6f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
