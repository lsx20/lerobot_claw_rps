#!/usr/bin/env python3
"""Solve fixed-camera 2D point to Piper fake-TCP XY calibration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("fake_tcp_samples.csv"))
    parser.add_argument("--output", type=Path, default=Path("fake_tcp_planar_calibration.json"))
    parser.add_argument(
        "--source",
        choices=("camera_xy", "pixel"),
        default="camera_xy",
        help="2D camera input: camera_x_m,camera_y_m or pixel_x,pixel_y",
    )
    parser.add_argument(
        "--target",
        choices=("xy", "xy_rpy"),
        default="xy",
        help="fit fake TCP XY only, or XY plus RX,RY,RZ",
    )
    parser.add_argument("--exclude", default="", help="comma-separated zero-based sample indices to ignore")
    return parser.parse_args()


def parse_exclude(value: str, sample_count: int) -> set[int]:
    if not value.strip():
        return set()
    indices = {int(item.strip()) for item in value.split(",") if item.strip()}
    bad = sorted(index for index in indices if index < 0 or index >= sample_count)
    if bad:
        raise ValueError(f"excluded sample indices out of range: {bad}")
    return indices


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 3:
        raise ValueError("at least 3 samples are required for 2D affine calibration")
    return rows


def source_xy(rows: list[dict[str, str]], source: str) -> np.ndarray:
    if source == "camera_xy":
        return np.array([[float(row["camera_x_m"]), float(row["camera_y_m"])] for row in rows], dtype=float)
    return np.array([[float(row["pixel_x"]), float(row["pixel_y"])] for row in rows], dtype=float)


def target_values(rows: list[dict[str, str]], target: str) -> tuple[np.ndarray, list[str], list[str]]:
    if target == "xy":
        fields = ["fake_tcp_x_m", "fake_tcp_y_m"]
        units = ["m", "m"]
    else:
        fields = ["fake_tcp_x_m", "fake_tcp_y_m", "fake_tcp_rx_deg", "fake_tcp_ry_deg", "fake_tcp_rz_deg"]
        units = ["m", "m", "deg", "deg", "deg"]
    return np.array([[float(row[field]) for field in fields] for row in rows], dtype=float), fields, units


def solve_affine(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    if source_points.shape[0] < 3:
        raise ValueError("at least 3 remaining samples are required")
    design = np.column_stack([source_points, np.ones(len(source_points))])
    if np.linalg.matrix_rank(design, tol=1e-9) < 3:
        raise ValueError("source points are degenerate; spread samples across the workspace")
    matrix_t, *_ = np.linalg.lstsq(design, target_points, rcond=None)
    return matrix_t.T


def apply_affine(affine_2x3: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    design = np.column_stack([points_xy, np.ones(len(points_xy))])
    return design @ affine_2x3.T


def stats(predicted: np.ndarray, target: np.ndarray) -> dict[str, object]:
    vectors = predicted[:, :2] - target[:, :2]
    distances = np.linalg.norm(vectors, axis=1)
    result: dict[str, object] = {
        "rms_m": float(np.sqrt(np.mean(distances**2))),
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "max_m": float(np.max(distances)),
        "errors_m": [float(value) for value in distances],
        "error_vectors_m": vectors.tolist(),
    }
    if predicted.shape[1] > 2:
        rpy_errors = predicted[:, 2:] - target[:, 2:]
        result.update(
            {
                "rpy_rms_deg": float(np.sqrt(np.mean(rpy_errors**2))),
                "rpy_max_abs_deg": float(np.max(np.abs(rpy_errors))),
                "rpy_error_vectors_deg": rpy_errors.tolist(),
            }
        )
    return result


def main() -> int:
    args = parse_args()
    rows = load_rows(args.input)
    excluded = parse_exclude(args.exclude, len(rows))
    used_indices = [index for index in range(len(rows)) if index not in excluded]
    if len(used_indices) < 3:
        raise ValueError("fewer than 3 samples remain after exclusions")

    src_all = source_xy(rows, args.source)
    dst_all, target_fields, target_units = target_values(rows, args.target)
    src_used = src_all[used_indices]
    dst_used = dst_all[used_indices]
    affine = solve_affine(src_used, dst_used)
    used_pred = apply_affine(affine, src_used)
    all_pred = apply_affine(affine, src_all)

    payload = {
        "calibration_type": "fixed_camera_2d_to_fake_tcp_xy",
        "meaning": "fake_tcp_xy_base_m = affine_2x3 * [camera_u, camera_v, 1]",
        "source": args.source,
        "target": args.target,
        "target_fields": target_fields,
        "target_units": target_units,
        "input": str(args.input),
        "sample_count": len(rows),
        "used_sample_indices": used_indices,
        "excluded_sample_indices": sorted(excluded),
        "affine_nx3": affine.tolist(),
        "affine_2x3": affine[:2].tolist(),
        "used_error": stats(used_pred, dst_used),
        "all_error": stats(all_pred, dst_all),
        "sample_rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}")
    print(f"source: {args.source}")
    print(f"target: {args.target}")
    print(f"samples: {len(rows)} used: {len(used_indices)} excluded: {sorted(excluded)}")
    print(f"used RMS: {payload['used_error']['rms_m']:.6f} m")
    print(f"used max: {payload['used_error']['max_m']:.6f} m")
    print(f"all RMS:  {payload['all_error']['rms_m']:.6f} m")
    print(f"all max:  {payload['all_error']['max_m']:.6f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
