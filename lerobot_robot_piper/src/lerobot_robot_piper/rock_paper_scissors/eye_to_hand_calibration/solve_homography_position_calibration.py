#!/usr/bin/env python3
"""Solve homography from image pixel (u,v) to Piper base tabletop (X,Y)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("homography_position_samples.csv"))
    parser.add_argument("--output", type=Path, default=Path("homography_position_calibration.json"))
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


def load_samples(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 4:
        raise ValueError("at least 4 samples are required for homography")
    pixels = np.array([[float(row["camera_pixel_u"]), float(row["camera_pixel_v"])] for row in rows], dtype=float)
    base_xy = np.array([[float(row["base_x_m"]), float(row["base_y_m"])] for row in rows], dtype=float)
    return pixels, base_xy, rows


def solve_homography(pixels: np.ndarray, base_xy: np.ndarray) -> np.ndarray:
    if len(pixels) < 4:
        raise ValueError("at least 4 remaining samples are required")
    rows = []
    for (u, v), (x, y) in zip(pixels, base_xy, strict=True):
        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, u * x, v * x, x])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, u * y, v * y, y])
    matrix = np.asarray(rows, dtype=float)
    if np.linalg.matrix_rank(matrix, tol=1e-9) < 8:
        raise ValueError("sample pixels are degenerate; spread samples across the workspace")
    _, _, vh_mat = np.linalg.svd(matrix)
    homography = vh_mat[-1].reshape(3, 3)
    if abs(homography[2, 2]) > 1e-12:
        homography = homography / homography[2, 2]
    return homography


def apply_homography(homography: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    pixel_h = np.column_stack([pixels, np.ones(len(pixels), dtype=float)])
    mapped_h = (homography @ pixel_h.T).T
    return mapped_h[:, :2] / mapped_h[:, 2:3]


def error_stats(predicted: np.ndarray, target: np.ndarray) -> dict[str, object]:
    vectors = predicted - target
    distances = np.linalg.norm(vectors, axis=1)
    return {
        "rms_m": float(np.sqrt(np.mean(distances**2))),
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "max_m": float(np.max(distances)),
        "errors_m": [float(value) for value in distances],
        "error_vectors_m": vectors.tolist(),
    }


def main() -> int:
    args = parse_args()
    pixels, base_xy, rows = load_samples(args.input)
    excluded = parse_exclude(args.exclude, len(rows))
    used_indices = [index for index in range(len(rows)) if index not in excluded]
    homography = solve_homography(pixels[used_indices], base_xy[used_indices])
    used_pred = apply_homography(homography, pixels[used_indices])
    all_pred = apply_homography(homography, pixels)
    payload = {
        "calibration_type": "pixel_to_base_xy_homography",
        "meaning": "[X,Y,1]^T ~ H_pixel_to_base_xy * [u,v,1]^T",
        "input": str(args.input),
        "sample_count": len(rows),
        "used_sample_indices": used_indices,
        "excluded_sample_indices": sorted(excluded),
        "H_pixel_to_base_xy": homography.tolist(),
        "used_error": error_stats(used_pred, base_xy[used_indices]),
        "all_error": error_stats(all_pred, base_xy),
        "sample_rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}")
    print(f"samples: {len(rows)} used: {len(used_indices)} excluded: {sorted(excluded)}")
    print(f"used RMS: {payload['used_error']['rms_m']:.6f} m")
    print(f"used max: {payload['used_error']['max_m']:.6f} m")
    print(f"all RMS:  {payload['all_error']['rms_m']:.6f} m")
    print(f"all max:  {payload['all_error']['max_m']:.6f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
