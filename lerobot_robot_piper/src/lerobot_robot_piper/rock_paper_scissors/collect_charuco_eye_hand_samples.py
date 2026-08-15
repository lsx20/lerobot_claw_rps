#!/usr/bin/env python3
"""Collect read-only Charuco board poses and Piper end poses for hand-eye calibration."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from charuco_eye_hand_common import (
    camera_matrix_from_intrinsics,
    distortion_from_intrinsics,
    estimate_charuco_pose,
    make_charuco_board,
)
from collect_eye_hand_samples import read_piper_pose
from rps_prize_controller import rotate_d405_ccw_90


def pose_jump(previous: object, current: object) -> tuple[float, float]:
    r_jump = float(np.linalg.norm(np.asarray(current.rvec) - np.asarray(previous.rvec)))
    t_jump = float(np.linalg.norm(np.asarray(current.tvec) - np.asarray(previous.tvec)))
    return r_jump, t_jump


def median_pose(poses: list[object]) -> object:
    first = poses[0]
    rvec = np.median([pose.rvec for pose in poses], axis=0)
    tvec = np.median([pose.tvec for pose in poses], axis=0)
    return type(first)(
        rvec=tuple(float(value) for value in rvec),
        tvec=tuple(float(value) for value in tvec),
        corner_count=int(round(float(np.median([pose.corner_count for pose in poses])))),
        reprojection_rms_px=float(np.median([pose.reprojection_rms_px for pose in poses])),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="260322279862")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-length", type=float, default=0.024)
    parser.add_argument("--marker-length", type=float, default=0.018)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--stable-frames", type=int, default=8)
    parser.add_argument("--max-translation-jump", type=float, default=0.006)
    parser.add_argument("--max-rotation-jump", type=float, default=0.035)
    parser.add_argument("--max-reprojection-rms", type=float, default=1.5)
    parser.add_argument("--pnp-method", choices=("ippe", "iterative"), default="iterative")
    parser.add_argument("--zero-distortion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=Path("eye_hand_samples_charuco.csv"))
    args = parser.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit("pyrealsense2 is not installed in this Python environment") from exc
    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError as exc:
        raise SystemExit("piper_sdk is not installed in this Python environment") from exc

    print("SAFETY: Charuco collection is read-only.")
    print("This program does not enable, disable, or move Piper.")
    print("Fix the printed Charuco board rigidly in the workspace.")
    print("Move Piper safely through diverse poses; press s only when stable.")
    if input("Type READONLY to connect and collect read-only samples: ").strip() != "READONLY":
        print("Cancelled.")
        return 0

    board, _ = make_charuco_board(
        args.squares_x,
        args.squares_y,
        args.square_length,
        args.marker_length,
        args.dictionary,
    )

    piper = C_PiperInterface_V2(
        args.can,
        judge_flag=False,
        can_auto_init=False,
        dh_is_offset=1,
        start_sdk_fk_cal=True,
    )
    piper.ConnectPort()
    time.sleep(1.0)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    camera_matrix = camera_matrix_from_intrinsics(intrinsics)
    dist_coeffs = np.zeros((1, 5), dtype=np.float64) if args.zero_distortion else distortion_from_intrinsics(intrinsics)
    stable_poses: list[object] = []
    output_exists = args.output.exists() and args.output.stat().st_size > 0
    sample_count = 0
    image_dir = args.output.with_suffix("")
    if args.save_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", newline="") as sample_file:
        writer = csv.writer(sample_file)
        if not output_exists:
            writer.writerow(
                [
                    "unix_time",
                    "camera_board_rvec_x",
                    "camera_board_rvec_y",
                    "camera_board_rvec_z",
                    "camera_board_x_m",
                    "camera_board_y_m",
                    "camera_board_z_m",
                    "base_x_m",
                    "base_y_m",
                    "base_z_m",
                    "base_rx_deg",
                    "base_ry_deg",
                    "base_rz_deg",
                    "charuco_corners",
                    "reprojection_rms_px",
                    "image_path",
                ]
            )

        try:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color = np.asanyarray(color_frame.get_data())
                board_pose, display = estimate_charuco_pose(
                    color,
                    board,
                    camera_matrix,
                    dist_coeffs,
                    args.min_corners,
                    args.pnp_method,
                )
                if board_pose is None or board_pose.reprojection_rms_px > args.max_reprojection_rms:
                    stable_poses.clear()
                else:
                    if stable_poses:
                        r_jump, t_jump = pose_jump(stable_poses[-1], board_pose)
                        if r_jump > args.max_rotation_jump or t_jump > args.max_translation_jump:
                            stable_poses.clear()
                    stable_poses = (stable_poses + [board_pose])[-args.stable_frames :]

                stable = median_pose(stable_poses) if len(stable_poses) >= args.stable_frames else None
                view = rotate_d405_ccw_90(display)
                status = (
                    f"stable {len(stable_poses)}/{args.stable_frames} err={board_pose.reprojection_rms_px:.2f}px"
                    if board_pose is not None
                    else "Charuco NOT FOUND"
                )
                color_text = (0, 255, 0) if stable is not None else (0, 0, 255)
                cv2.putText(view, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color_text, 2)
                cv2.putText(view, f"samples={sample_count}  s:save q:quit", (20, view.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
                cv2.imshow("Charuco eye-hand collection", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    if stable is None:
                        print(f"Not stable: {len(stable_poses)}/{args.stable_frames}")
                        continue
                    piper_pose = read_piper_pose(piper)
                    image_path = ""
                    if args.save_images:
                        image_path = str(image_dir / f"sample_{sample_count + 1:03d}.png")
                        cv2.imwrite(image_path, color)
                    writer.writerow(
                        [
                            f"{time.time():.6f}",
                            *[f"{value:.9f}" for value in stable.rvec],
                            *[f"{value:.9f}" for value in stable.tvec],
                            *[f"{value:.6f}" for value in piper_pose],
                            stable.corner_count,
                            f"{stable.reprojection_rms_px:.4f}",
                            image_path,
                        ]
                    )
                    sample_file.flush()
                    sample_count += 1
                    print(
                        f"saved #{sample_count}: "
                        f"camera_board_t={stable.tvec}, rvec={stable.rvec}, "
                        f"corners={stable.corner_count}, reproj={stable.reprojection_rms_px:.3f}px, "
                        f"piper_pose={piper_pose}"
                    )
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            try:
                piper.DisconnectPort()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
