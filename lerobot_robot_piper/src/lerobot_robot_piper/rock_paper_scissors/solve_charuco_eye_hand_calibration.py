#!/usr/bin/env python3
"""Solve eye-in-hand transform from Charuco board pose samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from charuco_eye_hand_common import make_transform, rodrigues_to_matrix
from solve_eye_hand_calibration import rpy_to_matrix


def transform_from_rotvec_translation(rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return make_transform(Rotation.from_rotvec(rotvec).as_matrix(), translation)


def transform_to_params(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            Rotation.from_matrix(transform[:3, :3]).as_rotvec(),
            transform[:3, 3],
        ]
    )


def load_samples(path: Path, min_corners: int, max_reprojection_rms: float) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    with path.open(newline="") as sample_file:
        rows = list(csv.DictReader(sample_file))

    base_tool: list[np.ndarray] = []
    camera_board: list[np.ndarray] = []
    kept_indices: list[int] = []
    for index, row in enumerate(rows, start=1):
        corners = int(float(row.get("charuco_corners", 0) or 0))
        reprojection_rms = float(row.get("reprojection_rms_px", 0.0) or 0.0)
        if corners < min_corners or reprojection_rms > max_reprojection_rms:
            continue
        pose = [float(row[name]) for name in ("base_x_m", "base_y_m", "base_z_m", "base_rx_deg", "base_ry_deg", "base_rz_deg")]
        base_tool.append(make_transform(rpy_to_matrix(*pose[3:]), np.asarray(pose[:3], dtype=float)))
        rvec = np.asarray([float(row[name]) for name in ("camera_board_rvec_x", "camera_board_rvec_y", "camera_board_rvec_z")], dtype=float)
        tvec = np.asarray([float(row[name]) for name in ("camera_board_x_m", "camera_board_y_m", "camera_board_z_m")], dtype=float)
        camera_board.append(make_transform(rodrigues_to_matrix(rvec), tvec))
        kept_indices.append(index)
    if len(base_tool) < 6:
        raise SystemExit(
            f"Need at least 6 kept samples; kept {len(base_tool)} after filters "
            f"min_corners={min_corners}, max_reprojection_rms={max_reprojection_rms}."
        )
    return base_tool, camera_board, kept_indices


def mean_transform(transforms: list[np.ndarray]) -> np.ndarray:
    rotations = Rotation.from_matrix([transform[:3, :3] for transform in transforms])
    mean_rotation = rotations.mean().as_matrix()
    mean_translation = np.mean([transform[:3, 3] for transform in transforms], axis=0)
    return make_transform(mean_rotation, mean_translation)


def residuals(
    params: np.ndarray,
    base_tool: list[np.ndarray],
    camera_board: list[np.ndarray],
    translation_scale: float,
    rotation_scale: float,
) -> np.ndarray:
    tool_camera = transform_from_rotvec_translation(params[:3], params[3:6])
    base_board_target = transform_from_rotvec_translation(params[6:9], params[9:12])
    target_rotation = Rotation.from_matrix(base_board_target[:3, :3])
    values: list[float] = []
    for base_tool_i, camera_board_i in zip(base_tool, camera_board, strict=True):
        predicted = base_tool_i @ tool_camera @ camera_board_i
        translation_error = (predicted[:3, 3] - base_board_target[:3, 3]) / translation_scale
        rotation_error = (target_rotation.inv() * Rotation.from_matrix(predicted[:3, :3])).as_rotvec() / rotation_scale
        values.extend(translation_error.tolist())
        values.extend(rotation_error.tolist())
    return np.asarray(values, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("eye_hand_samples_charuco.csv"))
    parser.add_argument("--output", type=Path, default=Path("eye_hand_calibration_charuco.json"))
    parser.add_argument("--initial", type=Path, default=Path("eye_hand_calibration_yolo11x_clean2.json"))
    parser.add_argument("--min-corners", type=int, default=0)
    parser.add_argument("--max-reprojection-rms", type=float, default=float("inf"))
    parser.add_argument("--translation-scale", type=float, default=0.01)
    parser.add_argument("--rotation-scale", type=float, default=0.05)
    args = parser.parse_args()

    base_tool, camera_board, kept_indices = load_samples(args.input, args.min_corners, args.max_reprojection_rms)
    if args.initial.exists():
        initial_tool_camera = np.asarray(json.loads(args.initial.read_text())["T_tool_camera"], dtype=float)
    else:
        initial_tool_camera = np.eye(4, dtype=float)
    predicted = [base_tool_i @ initial_tool_camera @ camera_board_i for base_tool_i, camera_board_i in zip(base_tool, camera_board, strict=True)]
    initial_base_board = mean_transform(predicted)
    initial_params = np.concatenate([transform_to_params(initial_tool_camera), transform_to_params(initial_base_board)])

    result = least_squares(
        residuals,
        initial_params,
        args=(base_tool, camera_board, args.translation_scale, args.rotation_scale),
        max_nfev=3000,
        x_scale="jac",
    )
    if not result.success:
        raise SystemExit(f"optimization failed: {result.message}")

    tool_camera = transform_from_rotvec_translation(result.x[:3], result.x[3:6])
    base_board = transform_from_rotvec_translation(result.x[6:9], result.x[9:12])
    predicted_final = [base_tool_i @ tool_camera @ camera_board_i for base_tool_i, camera_board_i in zip(base_tool, camera_board, strict=True)]
    translation_errors = np.asarray([np.linalg.norm(transform[:3, 3] - base_board[:3, 3]) for transform in predicted_final])
    target_rotation = Rotation.from_matrix(base_board[:3, :3])
    rotation_errors_deg = np.asarray(
        [
            np.degrees(np.linalg.norm((target_rotation.inv() * Rotation.from_matrix(transform[:3, :3])).as_rotvec()))
            for transform in predicted_final
        ]
    )
    payload = {
        "input": str(args.input),
        "sample_count": len(base_tool),
        "kept_sample_indices": kept_indices,
        "min_corners_filter": args.min_corners,
        "max_reprojection_rms_filter": args.max_reprojection_rms,
        "method": "Charuco board full-pose eye-in-hand optimization",
        "rpy_convention": "R_base_tool = Rz(RZ) @ Ry(RY) @ Rx(RX)",
        "T_tool_camera": tool_camera.tolist(),
        "T_base_board": base_board.tolist(),
        "translation_rms_m": float(np.sqrt(np.mean(translation_errors**2))),
        "translation_max_m": float(np.max(translation_errors)),
        "rotation_rms_deg": float(np.sqrt(np.mean(rotation_errors_deg**2))),
        "rotation_max_deg": float(np.max(rotation_errors_deg)),
        "sample_translation_errors_m": translation_errors.tolist(),
        "sample_rotation_errors_deg": rotation_errors_deg.tolist(),
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
