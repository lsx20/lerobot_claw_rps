#!/usr/bin/env python3
"""Validate D405-to-Piper target conversion without moving the robot."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from collect_eye_hand_samples import read_piper_pose
from rps_prize_controller import D405BallCamera, rotate_d405_ccw_90, rotate_d405_observation_ccw_90
from solve_eye_hand_calibration import make_transform, rpy_to_matrix


def parse_roi(value: str) -> tuple[int, int, int, int]:
    values = tuple(int(item.strip()) for item in value.split(","))
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="260322279862")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--calibration", type=Path, default=Path("eye_hand_calibration_latest.json"))
    parser.add_argument("--roi", type=parse_roi, default=(0, 0, 640, 480))
    parser.add_argument("--min-radius", type=float, default=45.0)
    parser.add_argument("--ball-color", choices=("auto", "orange", "green", "white"), default="auto")
    parser.add_argument("--shape-profile", choices=("original", "legacy", "strict"), default="original")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError as exc:
        raise SystemExit("piper_sdk is not installed in this Python environment") from exc

    calibration = json.loads(args.calibration.read_text())
    tool_camera = np.asarray(calibration["T_tool_camera"], dtype=float)
    print(f"calibration: {args.calibration}")
    print(f"point RMS: {calibration['residual_point_rms_m']:.4f} m")
    print(f"point max: {calibration['residual_max_m']:.4f} m")
    print("SAFETY: read-only validation; no Piper motion or enable/disable commands.")

    piper = C_PiperInterface_V2(
        args.can,
        judge_flag=False,
        can_auto_init=False,
        dh_is_offset=1,
        start_sdk_fk_cal=True,
    )
    piper.ConnectPort()
    time.sleep(1.0)
    camera = D405BallCamera(args.serial, args.width, args.height, args.fps, args.roi, args.min_radius, args.ball_color, shape_profile=args.shape_profile)
    camera.start()
    try:
        while True:
            packet = camera.read()
            if packet is None:
                continue
            color, depth, depth_frame = packet
            observation = camera.detect_ball(color, depth, depth_frame)
            display = rotate_d405_ccw_90(color)
            if observation is not None:
                piper_pose = read_piper_pose(piper)
                base_tool = make_transform(rpy_to_matrix(*piper_pose[3:]), np.asarray(piper_pose[:3]))
                camera_point = np.array([*observation.camera_xyz_m, 1.0])
                base_point = base_tool @ tool_camera @ camera_point
                rotated_observation = rotate_d405_observation_ccw_90(observation, color.shape[1])
                cv2.circle(display, rotated_observation.pixel, int(rotated_observation.radius_px), (0, 255, 0), 2)
                cv2.drawMarker(display, rotated_observation.pixel, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(display, f"base target=({base_point[0]:.3f},{base_point[1]:.3f},{base_point[2]:.3f})m", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(display, "p: print target   q: quit", (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("p"):
                    print(f"camera_xyz_m={observation.camera_xyz_m}")
                    print(f"piper_pose={piper_pose}")
                    print(f"target_base_m={tuple(float(value) for value in base_point[:3])}")
                elif key == ord("q"):
                    break
            else:
                cv2.putText(display, "ball: NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(display, "q: quit", (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.imshow("Piper target validation", display)
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        try:
            piper.DisconnectPort()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
