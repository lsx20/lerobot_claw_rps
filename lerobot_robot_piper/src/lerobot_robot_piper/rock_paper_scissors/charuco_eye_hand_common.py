#!/usr/bin/env python3
"""Shared Charuco helpers for D405/Piper eye-in-hand calibration."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CharucoPose:
    rvec: tuple[float, float, float]
    tvec: tuple[float, float, float]
    corner_count: int
    reprojection_rms_px: float = 0.0


def make_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    dictionary_name: str = "DICT_4X4_50",
) -> tuple[object, object]:
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        float(square_length_m),
        float(marker_length_m),
        dictionary,
    )
    return board, dictionary


def camera_matrix_from_intrinsics(intrinsics: object) -> np.ndarray:
    return np.array(
        [
            [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
            [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distortion_from_intrinsics(intrinsics: object) -> np.ndarray:
    coeffs = list(getattr(intrinsics, "coeffs", []) or [])
    if len(coeffs) < 5:
        coeffs += [0.0] * (5 - len(coeffs))
    return np.asarray(coeffs[:5], dtype=np.float64).reshape(1, 5)


def estimate_charuco_pose(
    image_bgr: np.ndarray,
    board: object,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    min_corners: int,
    pnp_method: str = "ippe",
) -> tuple[CharucoPose | None, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    display = image_bgr.copy()

    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < min_corners:
        return None, display

    draw_corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
    draw_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)
    cv2.aruco.drawDetectedCornersCharuco(display, draw_corners, draw_ids)
    object_points, image_points = board.matchImagePoints(draw_corners, draw_ids)
    if object_points is None or image_points is None or len(object_points) < min_corners:
        return None, display

    if pnp_method == "ippe":
        ok, rvecs, tvecs, reprojection_errors = cv2.solvePnPGeneric(
            object_points.astype(np.float64),
            image_points.astype(np.float64),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
    elif pnp_method == "iterative":
        ok, rvec, tvec = cv2.solvePnP(
            object_points.astype(np.float64),
            image_points.astype(np.float64),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        rvecs = (rvec,) if ok else ()
        tvecs = (tvec,) if ok else ()
        reprojection_errors = np.array([[np.inf]], dtype=np.float64)
    else:
        raise ValueError("pnp_method must be ippe or iterative")

    if not ok or not rvecs:
        ok_iterative, rvec, tvec = cv2.solvePnP(
            object_points.astype(np.float64),
            image_points.astype(np.float64),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok_iterative:
            return None, display
        rvecs = (rvec,)
        tvecs = (tvec,)
        reprojection_errors = np.array([[np.inf]], dtype=np.float64)

    best_index = 0
    if reprojection_errors is not None and len(reprojection_errors) == len(rvecs):
        best_index = int(np.argmin(np.asarray(reprojection_errors).reshape(-1)))
    rvec = np.asarray(rvecs[best_index], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvecs[best_index], dtype=np.float64).reshape(3, 1)
    projected, _ = cv2.projectPoints(object_points.astype(np.float64), rvec, tvec, camera_matrix, dist_coeffs)
    reprojection_rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points.reshape(-1, 2)) ** 2, axis=1))))

    cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, 0.04)
    return (
        CharucoPose(
            rvec=tuple(float(value) for value in rvec.reshape(3)),
            tvec=tuple(float(value) for value in tvec.reshape(3)),
            corner_count=int(len(charuco_ids)),
            reprojection_rms_px=reprojection_rms,
        ),
        display,
    )


def rodrigues_to_matrix(rvec: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rotation


def make_transform(rotation: np.ndarray, translation: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform
