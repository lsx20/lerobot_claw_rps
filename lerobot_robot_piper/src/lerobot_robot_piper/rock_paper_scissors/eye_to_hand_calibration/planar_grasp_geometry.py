#!/usr/bin/env python3
"""Planar fixed-camera grasp geometry for Piper tabletop ball picking."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


JOINT_LIMITS_DEG = [
    (-150.0, 150.0),
    (0.0, 180.0),
    (-170.0, 0.0),
    (-100.0, 100.0),
    (-70.0, 70.0),
    (-120.0, 120.0),
]

PIPER_JOINT_ORIGINS = [
    ((0.0, 0.0, 0.123), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.5708, -0.1359, -3.1416)),
    ((0.28503, 0.0, 0.0), (0.0, 0.0, -1.7939)),
    ((-0.021984, -0.25075, 0.0), (1.5708, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-1.5708, 0.0, 0.0)),
    ((8.8259e-05, -0.091, 0.0), (1.5708, 0.0, 0.0)),
]


def transform_xyz_rpy(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def transform_rotz(angle_rad: float) -> np.ndarray:
    transform = np.eye(4)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    transform[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return transform


def piper_fk(joints_deg: list[float]) -> np.ndarray:
    transform = np.eye(4)
    for (xyz, rpy), joint_deg in zip(PIPER_JOINT_ORIGINS, joints_deg, strict=True):
        transform = transform @ transform_xyz_rpy(xyz, rpy) @ transform_rotz(math.radians(joint_deg))
    return transform


def wrap_deg(angle: float) -> float:
    wrapped = (angle + 180.0) % 360.0 - 180.0
    if wrapped == -180.0 and angle > 0:
        return 180.0
    return wrapped


def radial_flange_xy(
    ball_xy_m: tuple[float, float],
    radial_offset_mm: float,
) -> tuple[tuple[float, float], float]:
    radius = math.hypot(ball_xy_m[0], ball_xy_m[1])
    if radius < 1e-6:
        raise ValueError("ball XY is too close to base origin; radial direction is undefined")
    unit = (ball_xy_m[0] / radius, ball_xy_m[1] / radius)
    offset_m = radial_offset_mm / 1000.0
    flange_xy_m = (ball_xy_m[0] - offset_m * unit[0], ball_xy_m[1] - offset_m * unit[1])
    theta_deg = math.degrees(math.atan2(ball_xy_m[1], ball_xy_m[0]))
    return flange_xy_m, theta_deg


def radial_target(
    ball_xy_m: tuple[float, float],
    radial_offset_mm: float,
    rz_offset_deg: float,
    rx_deg: float,
    ry_deg: float,
) -> tuple[tuple[float, float], tuple[float, float, float], float]:
    flange_xy_m, theta_deg = radial_flange_xy(ball_xy_m, radial_offset_mm)
    rpy = (rx_deg, ry_deg, wrap_deg(rz_offset_deg + theta_deg))
    return flange_xy_m, rpy, theta_deg


def solve_planar_joint_target(
    flange_xy_m: tuple[float, float],
    fixed_z_mm: float,
    j4_deg: float = 0.0,
    j6_deg: float = 0.0,
    j5_seed_deg: float = 13.0,
) -> tuple[list[float], np.ndarray, float]:
    target_xyz_m = np.array([flange_xy_m[0], flange_xy_m[1], fixed_z_mm / 1000.0])
    theta_deg = math.degrees(math.atan2(flange_xy_m[1], flange_xy_m[0]))
    j1_deg = wrap_deg(theta_deg)
    j1_deg = max(JOINT_LIMITS_DEG[0][0], min(JOINT_LIMITS_DEG[0][1], j1_deg))
    j4_deg = max(JOINT_LIMITS_DEG[3][0], min(JOINT_LIMITS_DEG[3][1], j4_deg))
    j6_deg = max(JOINT_LIMITS_DEG[5][0], min(JOINT_LIMITS_DEG[5][1], j6_deg))

    def residual(values: np.ndarray) -> np.ndarray:
        joints = [j1_deg, values[0], values[1], j4_deg, values[2], j6_deg]
        actual_xyz = piper_fk(joints)[:3, 3]
        regularization = 0.05 * np.array([values[0] - 58.0, values[1] + 21.0, values[2] - j5_seed_deg])
        return np.concatenate([(actual_xyz - target_xyz_m) * 1000.0, regularization])

    seeds = [
        [58.0, -21.0, j5_seed_deg],
        [55.0, -17.0, j5_seed_deg],
        [70.0, -65.0, 34.0],
        [45.0, -30.0, 0.0],
        [80.0, -80.0, 30.0],
    ]
    best: tuple[float, np.ndarray] | None = None
    for seed in seeds:
        result = least_squares(
            residual,
            seed,
            bounds=(
                [JOINT_LIMITS_DEG[1][0], JOINT_LIMITS_DEG[2][0], JOINT_LIMITS_DEG[4][0]],
                [JOINT_LIMITS_DEG[1][1], JOINT_LIMITS_DEG[2][1], JOINT_LIMITS_DEG[4][1]],
            ),
            max_nfev=2000,
        )
        joints = [j1_deg, result.x[0], result.x[1], j4_deg, result.x[2], j6_deg]
        actual_xyz = piper_fk(joints)[:3, 3]
        error_mm = float(np.linalg.norm((actual_xyz - target_xyz_m) * 1000.0))
        if best is None or error_mm < best[0]:
            best = (error_mm, result.x)
    if best is None:
        raise RuntimeError("planar joint solve failed")
    joints = [j1_deg, float(best[1][0]), float(best[1][1]), j4_deg, float(best[1][2]), j6_deg]
    return joints, piper_fk(joints), best[0]
