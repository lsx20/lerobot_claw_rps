#!/usr/bin/env python3
"""Solve a point-target eye-in-hand transform from paired D405/Piper samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3) + skew(rotvec)
    axis = rotvec / angle
    cross = skew(axis)
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def rpy_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(roll), -np.sin(roll)], [0.0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def load_samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 6:
        raise ValueError("at least 6 samples are required")
    camera_points = np.array(
        [[float(row[name]) for name in ("camera_x_m", "camera_y_m", "camera_z_m")] for row in rows],
        dtype=float,
    )
    base_poses = np.array(
        [[float(row[name]) for name in ("base_x_m", "base_y_m", "base_z_m", "base_rx_deg", "base_ry_deg", "base_rz_deg")] for row in rows],
        dtype=float,
    )
    return camera_points, base_poses


def base_transforms(base_poses: np.ndarray) -> list[np.ndarray]:
    return [
        make_transform(rpy_to_matrix(*pose[3:]), pose[:3])
        for pose in base_poses
    ]


def predict(params: np.ndarray, camera_points: np.ndarray, base_tf: list[np.ndarray]) -> np.ndarray:
    tool_camera = make_transform(rotvec_to_matrix(params[:3]), params[3:6])
    camera_h = np.column_stack([camera_points, np.ones(len(camera_points))])
    predictions = []
    for transform, point in zip(base_tf, camera_h, strict=True):
        predictions.append((transform @ tool_camera @ point)[:3])
    return np.asarray(predictions)


def residual(params: np.ndarray, camera_points: np.ndarray, base_tf: list[np.ndarray]) -> np.ndarray:
    predictions = predict(params, camera_points, base_tf)
    target = params[6:9]
    return (predictions - target).reshape(-1)


def optimize(camera_points: np.ndarray, base_tf: list[np.ndarray], iterations: int, starts: int, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    base_positions = np.array([transform[:3, 3] for transform in base_tf])
    best_params = None
    best_cost = float("inf")
    for start in range(starts):
        params = np.zeros(9, dtype=float)
        if start:
            params[:3] = rng.uniform(-np.pi, np.pi, 3)
            params[3:6] = rng.normal(0.0, 0.08, 3)
        params[6:9] = np.mean(base_positions, axis=0)
        damping = 1e-3
        current = float(np.dot(residual(params, camera_points, base_tf), residual(params, camera_points, base_tf)))
        for _ in range(iterations):
            error = residual(params, camera_points, base_tf)
            jacobian = np.zeros((len(error), 9), dtype=float)
            steps = np.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
            for index, step in enumerate(steps):
                trial = params.copy()
                trial[index] += step
                jacobian[:, index] = (residual(trial, camera_points, base_tf) - error) / step
            normal = jacobian.T @ jacobian + damping * np.diag(np.diag(jacobian.T @ jacobian) + 1e-12)
            try:
                update = np.linalg.solve(normal, -(jacobian.T @ error))
            except np.linalg.LinAlgError:
                break
            if np.linalg.norm(update) < 1e-9:
                break
            candidate = params + update
            candidate_cost = float(np.dot(residual(candidate, camera_points, base_tf), residual(candidate, camera_points, base_tf)))
            if candidate_cost < current:
                params, current = candidate, candidate_cost
                damping = max(damping * 0.5, 1e-9)
            else:
                damping = min(damping * 10.0, 1e12)
        if current < best_cost:
            best_params, best_cost = params, current
    if best_params is None:
        raise RuntimeError("calibration optimizer failed")
    return best_params, best_cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("eye_hand_samples_for_calibration.csv"))
    parser.add_argument("--output", type=Path, default=Path("eye_hand_calibration.json"))
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    camera_points, base_poses = load_samples(args.input)
    transforms = base_transforms(base_poses)
    params, cost = optimize(camera_points, transforms, args.iterations, args.starts, args.seed)
    tool_camera = make_transform(rotvec_to_matrix(params[:3]), params[3:6])
    predictions = predict(params, camera_points, transforms)
    target = params[6:9]
    errors = np.linalg.norm(predictions - target, axis=1)
    result = {
        "input": str(args.input),
        "sample_count": int(len(camera_points)),
        "rpy_convention": "R_base_tool = Rz(RZ) @ Ry(RY) @ Rx(RX)",
        "T_tool_camera": tool_camera.tolist(),
        "fixed_target_base_m": target.tolist(),
        "residual_point_rms_m": float(np.sqrt(np.mean(errors**2))),
        "residual_coordinate_rms_m": float(np.sqrt(np.mean(residual(params, camera_points, transforms) ** 2))),
        "residual_max_m": float(np.max(errors)),
        "sample_errors_m": errors.tolist(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
