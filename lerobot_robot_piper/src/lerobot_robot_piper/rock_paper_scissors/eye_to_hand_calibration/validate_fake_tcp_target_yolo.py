#!/usr/bin/env python3
"""Validate fixed-D405 camera point to fake-TCP target conversion with YOLO."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

THIS_DIR = Path(__file__).resolve().parent
RPS_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(RPS_DIR) not in sys.path:
    sys.path.insert(0, str(RPS_DIR))

from common import Detection3D, apply_transform, load_transform_json, parse_roi
from rps_prize_controller import (
    rotate_d405_box_ccw_90,
    rotate_d405_ccw_90,
    rotate_d405_point_ccw_90,
)
from test_yolo_d405_ball import depth_at_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="315122271151")
    parser.add_argument("--calibration", type=Path, default=Path("fake_tcp_calibration.json"))
    parser.add_argument("--model", default=str(RPS_DIR / "yolo26n.pt"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--roi", type=parse_roi, default=(0, 0, 640, 480))
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu", help="ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--z-offset-m", type=float, default=0.0, help="optional target Z offset after conversion")
    return parser.parse_args()


def predict_kwargs(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        kwargs["device"] = args.device
    return kwargs


def best_yolo_ball(
    result: object,
    depth_frame: object,
    depth_image: np.ndarray,
    depth_scale: float,
    intrinsics: object,
    ball_class_ids: set[int],
    roi: tuple[int, int, int, int],
    width: int,
    height: int,
) -> Detection3D | None:
    best: Detection3D | None = None
    if result.boxes is None:
        return None
    for box_tensor, confidence_tensor, class_tensor in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        if int(class_tensor.item()) not in ball_class_ids:
            continue
        confidence = float(confidence_tensor.item())
        x0, y0, x1, y1 = [int(value) for value in box_tensor.tolist()]
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(x0 + 1, min(width, x1))
        y1 = max(y0 + 1, min(height, y1))
        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2
        if not (roi[0] <= center_x <= roi[2] and roi[1] <= center_y <= roi[3]):
            continue
        depth_result = depth_at_box(depth_frame, depth_image, (x0, y0, x1, y1), depth_scale)
        if depth_result is None:
            continue
        depth_m, pixel = depth_result
        point = rs2_deproject_pixel_to_point(intrinsics, list(pixel), depth_m)
        detection = Detection3D(
            confidence=confidence,
            box=(x0, y0, x1, y1),
            pixel=pixel,
            depth_m=depth_m,
            camera_xyz_m=tuple(float(value) for value in point),
        )
        if best is None or confidence > best.confidence:
            best = detection
    return best


def format_realsense_devices(rs_module: object) -> str:
    devices = []
    for device in rs_module.context().query_devices():
        name = device.get_info(rs_module.camera_info.name)
        serial = device.get_info(rs_module.camera_info.serial_number)
        devices.append(f"{name} serial={serial}")
    return "\n".join(f"  {item}" for item in devices) if devices else "  none"


def main() -> int:
    args = parse_args()
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit("pyrealsense2 and ultralytics are required") from exc

    global rs2_deproject_pixel_to_point
    rs2_deproject_pixel_to_point = rs.rs2_deproject_pixel_to_point

    calibration, transform = load_transform_json(args.calibration)
    print(f"calibration: {args.calibration}")
    print(f"meaning: {calibration.get('meaning')}")
    all_error = calibration.get("all_error", {})
    if isinstance(all_error, dict):
        print(f"fit RMS: {all_error.get('rms_m')} m")
        print(f"fit max: {all_error.get('max_m')} m")
    print("SAFETY: read-only validation; no Piper connection or motion commands.")

    model = YOLO(args.model)
    ball_class_ids = {class_id for class_id, name in model.names.items() if str(name).lower() == "sports ball"}
    if not ball_class_ids:
        raise SystemExit(f"The selected model has no sports ball class: {model.names}")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)
    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        raise SystemExit(
            f"Could not start RealSense D405 with serial={args.serial!r}.\n"
            f"Connected RealSense devices:\n{format_realsense_devices(rs)}\n"
            f"Use the shown serial, or pass --serial \"\" to use any connected RealSense."
        ) from exc
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            result = model.predict(color, **predict_kwargs(args))[0]
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            detection = best_yolo_ball(
                result,
                depth_frame,
                depth_image,
                depth_scale,
                intrinsics,
                ball_class_ids,
                args.roi,
                args.width,
                args.height,
            )
            display = rotate_d405_ccw_90(color)
            if detection is None:
                cv2.putText(display, "YOLO BALL NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2)
            else:
                target = apply_transform(transform, np.asarray(detection.camera_xyz_m))
                target[2] += args.z_offset_m
                rotated_box = rotate_d405_box_ccw_90(detection.box, color.shape[1])
                rotated_center = rotate_d405_point_ccw_90(detection.pixel, color.shape[1])
                cv2.rectangle(display, (rotated_box[0], rotated_box[1]), (rotated_box[2], rotated_box[3]), (0, 255, 0), 2)
                cv2.drawMarker(display, rotated_center, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(display, f"camera=({detection.camera_xyz_m[0]:.3f},{detection.camera_xyz_m[1]:.3f},{detection.camera_xyz_m[2]:.3f})m", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
                cv2.putText(display, f"fake TCP target=({target[0]:.3f},{target[1]:.3f},{target[2]:.3f})m", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
                cv2.putText(display, "p:print target q:quit", (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
                cv2.imshow("Fixed D405 fake-TCP target validation", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("p"):
                    print(f"confidence={detection.confidence:.3f}")
                    print(f"camera_xyz_m={detection.camera_xyz_m}")
                    print(f"fake_tcp_target_m={tuple(float(value) for value in target)}")
                elif key == ord("q"):
                    break
                continue

            cv2.putText(display, "q:quit", (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            cv2.imshow("Fixed D405 fake-TCP target validation", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
