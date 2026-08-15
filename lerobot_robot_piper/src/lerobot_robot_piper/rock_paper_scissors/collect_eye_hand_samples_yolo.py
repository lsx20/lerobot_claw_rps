#!/usr/bin/env python3
"""Collect read-only eye-hand samples using YOLO26 ball detections."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from rps_prize_controller import BallObservation, rotate_d405_ccw_90, rotate_d405_observation_ccw_90
from test_yolo_d405_ball import depth_at_box
from collect_eye_hand_samples import parse_roi, read_piper_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="260322279862")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--roi", type=parse_roi, default=(0, 0, 640, 480))
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu", help="ultralytics device, e.g. cpu or 0")
    parser.add_argument("--stable-frames", type=int, default=10)
    parser.add_argument("--max-pixel-jump", type=float, default=40.0)
    parser.add_argument("--max-depth-jump", type=float, default=0.04)
    parser.add_argument("--output", type=Path, default=Path("eye_hand_samples_yolo.csv"))
    return parser.parse_args()


def median_observation(observations: list[BallObservation]) -> BallObservation | None:
    if not observations:
        return None
    pixels = np.median([item.pixel for item in observations], axis=0)
    xyz = np.median([item.camera_xyz_m for item in observations], axis=0)
    return BallObservation(
        pixel=(int(round(pixels[0])), int(round(pixels[1]))),
        depth_m=float(np.median([item.depth_m for item in observations])),
        camera_xyz_m=tuple(float(value) for value in xyz),
        radius_px=float(np.median([item.radius_px for item in observations])),
    )


def main() -> int:
    args = parse_args()
    try:
        import pyrealsense2 as rs
        from piper_sdk import C_PiperInterface_V2
    except ImportError as exc:
        raise SystemExit("pyrealsense2 and piper_sdk are required") from exc

    model = YOLO(args.model)
    sports_ball_ids = {class_id for class_id, name in model.names.items() if name == "sports ball"}
    if not sports_ball_ids:
        raise SystemExit("The selected YOLO model has no sports ball class")

    print("SAFETY: YOLO26 read-only calibration collection.")
    print("This program does not enable, disable, or move Piper.")
    print("Keep the arm mechanically supported and use only your approved manual method.")
    if input("Type READONLY to continue: ").strip() != "READONLY":
        print("Cancelled.")
        return 0

    piper = C_PiperInterface_V2(args.can, judge_flag=False, can_auto_init=False, dh_is_offset=1, start_sdk_fk_cal=True)
    piper.ConnectPort()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    observations: list[BallObservation] = []
    output_exists = args.output.exists() and args.output.stat().st_size > 0

    try:
        with args.output.open("a", newline="") as sample_file:
            writer = csv.writer(sample_file)
            if not output_exists:
                writer.writerow([
                    "unix_time",
                    "camera_x_m", "camera_y_m", "camera_z_m",
                    "base_x_m", "base_y_m", "base_z_m",
                    "base_rx_deg", "base_ry_deg", "base_rz_deg",
                    "pixel_x", "pixel_y", "depth_m", "radius_px", "confidence",
                ])
            print(f"D405={args.serial}, Piper CAN={args.can}, YOLO={args.model}")
            print("Fix one ball. Change Piper pose safely, wait for stable detection, press s.")
            while True:
                frames = align.process(pipeline.wait_for_frames())
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                result = model.predict(color, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
                candidate: BallObservation | None = None
                confidence = 0.0
                if result.boxes is not None:
                    for box_tensor, confidence_tensor, class_tensor in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                        if int(class_tensor.item()) not in sports_ball_ids:
                            continue
                        box = [int(value) for value in box_tensor.tolist()]
                        x0, y0, x1, y1 = box
                        x0 = max(0, min(args.width - 1, x0))
                        y0 = max(0, min(args.height - 1, y0))
                        x1 = max(x0 + 1, min(args.width, x1))
                        y1 = max(y0 + 1, min(args.height, y1))
                        if not (args.roi[0] <= (x0 + x1) // 2 <= args.roi[2] and args.roi[1] <= (y0 + y1) // 2 <= args.roi[3]):
                            continue
                        depth_result = depth_at_box(depth_frame, depth_image, (x0, y0, x1, y1), depth_scale)
                        if depth_result is None:
                            continue
                        depth_m, pixel = depth_result
                        point = rs.rs2_deproject_pixel_to_point(depth_frame.profile.as_video_stream_profile().intrinsics, list(pixel), depth_m)
                        candidate = BallObservation(pixel, depth_m, tuple(float(value) for value in point), max(x1 - x0, y1 - y0) / 2.0)
                        confidence = float(confidence_tensor.item())
                        break

                if candidate is None:
                    observations.clear()
                else:
                    if observations:
                        previous = observations[-1]
                        pixel_jump = float(np.hypot(candidate.pixel[0] - previous.pixel[0], candidate.pixel[1] - previous.pixel[1]))
                        depth_jump = abs(candidate.depth_m - previous.depth_m)
                        if pixel_jump > args.max_pixel_jump or depth_jump > args.max_depth_jump:
                            observations.clear()
                    observations = (observations + [candidate])[-args.stable_frames:]
                stable = median_observation(observations) if len(observations) >= args.stable_frames else None
                display = rotate_d405_ccw_90(color)
                if candidate is None:
                    cv2.putText(display, "BALL NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    rotated_candidate = rotate_d405_observation_ccw_90(candidate, color.shape[1])
                    cv2.drawMarker(display, rotated_candidate.pixel, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(display, f"BALL conf={confidence:.2f} xyz=({candidate.camera_xyz_m[0]:.3f},{candidate.camera_xyz_m[1]:.3f},{candidate.camera_xyz_m[2]:.3f})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(display, f"stable={len(observations)}/{args.stable_frames}  s:save q:quit", (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imshow("YOLO26 eye-hand collection", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    if stable is None:
                        print(f"Not stable: {len(observations)}/{args.stable_frames}")
                        continue
                    pose = read_piper_pose(piper)
                    writer.writerow([time.time(), *stable.camera_xyz_m, *pose, stable.pixel[0], stable.pixel[1], stable.depth_m, stable.radius_px, confidence])
                    sample_file.flush()
                    print(f"saved: camera={stable.camera_xyz_m}, piper_pose={pose}")
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
