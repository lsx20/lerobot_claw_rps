#!/usr/bin/env python3
"""Run YOLO-World with one generic ball class on the hand-mounted D405."""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from ultralytics import YOLOWorld

from test_yolo_d405_ball import depth_at_box
from rps_prize_controller import rotate_d405_box_ccw_90, rotate_d405_ccw_90, rotate_d405_point_ccw_90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="260322279862")
    parser.add_argument("--model", default="yolov8s-worldv2.pt")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="ultralytics device, e.g. cpu or 0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit("pyrealsense2 is not installed in this Python environment") from exc

    model = YOLOWorld(args.model)
    model.set_classes(["ball"])
    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    print("SAFETY: YOLO-World and D405 read-only test.")
    print("No Piper connection, enable, disable, or motion commands are used.")
    print("Prompt: ball. Press q to quit; press p to print the best target.")
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            result = model.predict(color, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            display = rotate_d405_ccw_90(color)
            detections: list[tuple[float, tuple[int, int, int, int], float, tuple[float, float, float]]] = []
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            if result.boxes is not None:
                for box_tensor, confidence_tensor in zip(result.boxes.xyxy, result.boxes.conf):
                    confidence = float(confidence_tensor.item())
                    x0, y0, x1, y1 = [int(value) for value in box_tensor.tolist()]
                    x0 = max(0, min(args.width - 1, x0))
                    y0 = max(0, min(args.height - 1, y0))
                    x1 = max(x0 + 1, min(args.width, x1))
                    y1 = max(y0 + 1, min(args.height, y1))
                    depth_result = depth_at_box(depth_frame, depth_image, (x0, y0, x1, y1), depth_scale)
                    if depth_result is None:
                        continue
                    depth_m, center = depth_result
                    point = rs.rs2_deproject_pixel_to_point(intrinsics, list(center), depth_m)
                    point_xyz = tuple(float(value) for value in point)
                    rotated_box = rotate_d405_box_ccw_90((x0, y0, x1, y1), color.shape[1])
                    rotated_center = rotate_d405_point_ccw_90(center, color.shape[1])
                    detections.append((confidence, rotated_box, depth_m, point_xyz))
                    cv2.rectangle(display, (rotated_box[0], rotated_box[1]), (rotated_box[2], rotated_box[3]), (0, 255, 0), 2)
                    cv2.drawMarker(display, rotated_center, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(display, f"BALL {confidence:.2f} depth={depth_m:.3f}m", (rotated_box[0], max(24, rotated_box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
            detections.sort(key=lambda item: item[0], reverse=True)
            cv2.putText(display, f"balls={len(detections)}  q:quit p:print", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow("YOLO-World D405 ball test", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p") and detections:
                best = detections[0]
                print(f"confidence={best[0]:.3f} depth_m={best[2]:.4f} camera_xyz_m={best[3]}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
