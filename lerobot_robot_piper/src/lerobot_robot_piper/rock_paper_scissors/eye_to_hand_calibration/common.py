#!/usr/bin/env python3
"""Shared helpers for fixed D405 to fake-TCP target calibration."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CSV_FIELDS = [
    "unix_time",
    "camera_x_m",
    "camera_y_m",
    "camera_z_m",
    "fake_tcp_x_m",
    "fake_tcp_y_m",
    "fake_tcp_z_m",
    "fake_tcp_rx_deg",
    "fake_tcp_ry_deg",
    "fake_tcp_rz_deg",
    "pixel_x",
    "pixel_y",
    "depth_m",
    "confidence",
]


@dataclass(frozen=True)
class Detection3D:
    confidence: float
    box: tuple[int, int, int, int]
    pixel: tuple[int, int]
    depth_m: float
    camera_xyz_m: tuple[float, float, float]


def parse_roi(value: str) -> tuple[int, int, int, int]:
    values = tuple(int(item.strip()) for item in value.split(","))
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1")
    return values


def read_piper_pose(piper: object) -> tuple[float, float, float, float, float, float]:
    """Read Piper end pose as meters and degrees.

    Piper SDK reports translation in 0.001 mm and rotation in 0.001 degree.
    In this calibration the end-pose position is treated as the fake TCP target.
    """
    end_pose = piper.GetArmEndPoseMsgs().end_pose
    position_m = [
        float(value) / 1_000_000.0
        for value in (end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis)
    ]
    rotation_deg = [
        float(value) / 1000.0
        for value in (end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis)
    ]
    return tuple(position_m + rotation_deg)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def apply_transform(transform: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim == 1:
        return (transform @ np.array([points[0], points[1], points[2], 1.0], dtype=float))[:3]
    points_h = np.column_stack([points, np.ones(len(points), dtype=float)])
    return (transform @ points_h.T).T[:, :3]


def solve_rigid_transform(source_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    """Return T_target_source that best maps source points onto target points."""
    source = np.asarray(source_xyz, dtype=float)
    target = np.asarray(target_xyz, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape Nx3")
    if len(source) < 3:
        raise ValueError("at least 3 point pairs are required")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    if np.linalg.matrix_rank(source_zero, tol=1e-6) < 2:
        raise ValueError("source points must not be collinear")
    if np.linalg.matrix_rank(target_zero, tol=1e-6) < 2:
        raise ValueError("target points must not be collinear")

    covariance = source_zero.T @ target_zero
    u_mat, _, vh_mat = np.linalg.svd(covariance)
    rotation = vh_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0:
        vh_mat[-1, :] *= -1.0
        rotation = vh_mat.T @ u_mat.T
    translation = target_center - rotation @ source_center
    return make_transform(rotation, translation)


def load_point_pairs(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 3:
        raise ValueError("at least 3 samples are required")
    camera_points = np.array(
        [[float(row[name]) for name in ("camera_x_m", "camera_y_m", "camera_z_m")] for row in rows],
        dtype=float,
    )
    fake_tcp_points = np.array(
        [[float(row[name]) for name in ("fake_tcp_x_m", "fake_tcp_y_m", "fake_tcp_z_m")] for row in rows],
        dtype=float,
    )
    return camera_points, fake_tcp_points, rows


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load_transform_json(path: Path) -> tuple[dict[str, object], np.ndarray]:
    payload = json.loads(path.read_text())
    key = "T_base_fake_tcp_target_from_camera"
    if key not in payload:
        raise KeyError(f"{path} does not contain {key}")
    return payload, np.asarray(payload[key], dtype=float)
