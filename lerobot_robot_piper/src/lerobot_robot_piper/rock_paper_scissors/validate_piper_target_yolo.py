#!/usr/bin/env python3
"""Validate YOLO D405-to-Piper target conversion without moving the robot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from collect_eye_hand_samples import parse_roi, read_piper_pose
from rps_prize_controller import (
    rotate_d405_box_ccw_90,
    rotate_d405_ccw_90,
    rotate_d405_point_ccw_90,
)
from solve_eye_hand_calibration import make_transform, rpy_to_matrix
from test_yolo_d405_ball import depth_at_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="260322279862")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--calibration", type=Path, default=Path("eye_hand_calibration_yolo.json"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--roi", type=parse_roi, default=(0, 0, 640, 480))
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cpu", help="ultralytics device, e.g. cpu or 0")
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


def main() -> int:
    args = parse_args()
    try:
        import pyrealsense2 as rs
        from piper_sdk import C_PiperInterface_V2
    except ImportError as exc:
        raise SystemExit("pyrealsense2 and piper_sdk are required") from exc

    calibration = json.loads(args.calibration.read_text())
    tool_camera = np.asarray(calibration["T_tool_camera"], dtype=float)
    print(f"calibration: {args.calibration}")
    print(f"point RMS: {calibration.get('residual_point_rms_m', float('nan')):.4f} m")
    print(f"point max: {calibration.get('residual_max_m', float('nan')):.4f} m")
    print("SAFETY: YOLO read-only validation; no Piper motion or enable/disable commands.")

    model = YOLO(args.model)
    sports_ball_ids = {
        class_id
        for class_id, name in model.names.items()
        if str(name).lower() == "sports ball"
    }
    if not sports_ball_ids:
        raise SystemExit(f"The selected YOLO model has no sports ball class: {model.names}")

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
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
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
            display = rotate_d405_ccw_90(color)
            best: tuple[float, tuple[int, int, int, int], float, tuple[int, int], tuple[float, float, float]] | None = None
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            if result.boxes is not None:
                for box_tensor, confidence_tensor, class_tensor in zip(
                    result.boxes.xyxy,
                    result.boxes.conf,
                    result.boxes.cls,
                ):
                    if int(class_tensor.item()) not in sports_ball_ids:
                        continue
                    confidence = float(confidence_tensor.item())
                    x0, y0, x1, y1 = [int(value) for value in box_tensor.tolist()]
                    x0 = max(0, min(args.width - 1, x0))
                    y0 = max(0, min(args.height - 1, y0))
                    x1 = max(x0 + 1, min(args.width, x1))
                    y1 = max(y0 + 1, min(args.height, y1))
                    center_x = (x0 + x1) // 2
                    center_y = (y0 + y1) // 2
                    if not (
                        args.roi[0] <= center_x <= args.roi[2]
                        and args.roi[1] <= center_y <= args.roi[3]
                    ):
                        continue
                    depth_result = depth_at_box(depth_frame, depth_image, (x0, y0, x1, y1), depth_scale)
                    if depth_result is None:
                        continue
                    depth_m, center = depth_result
                    point = rs.rs2_deproject_pixel_to_point(intrinsics, list(center), depth_m)
                    point_xyz = tuple(float(value) for value in point)
                    if best is None or confidence > best[0]:
                        best = (confidence, (x0, y0, x1, y1), depth_m, center, point_xyz)

            if best is None:
                cv2.putText(display, "YOLO BALL NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            else:
                confidence, box, depth_m, center, camera_xyz = best
                piper_pose = read_piper_pose(piper)
                base_tool = make_transform(rpy_to_matrix(*piper_pose[3:]), np.asarray(piper_pose[:3]))
                camera_point = np.array([*camera_xyz, 1.0])
                base_point = base_tool @ tool_camera @ camera_point
                rotated_box = rotate_d405_box_ccw_90(box, color.shape[1])
                rotated_center = rotate_d405_point_ccw_90(center, color.shape[1])
                cv2.rectangle(display, (rotated_box[0], rotated_box[1]), (rotated_box[2], rotated_box[3]), (0, 255, 0), 2)
                cv2.drawMarker(display, rotated_center, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(display, f"YOLO ball conf={confidence:.2f} depth={depth_m:.3f}m", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
                cv2.putText(display, f"base target=({base_point[0]:.3f},{base_point[1]:.3f},{base_point[2]:.3f})m", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
                cv2.putText(display, "p: print target   q: quit", (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imshow("YOLO D405 Piper target validation", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("p"):
                    print(f"confidence={confidence:.3f}")
                    print(f"camera_xyz_m={camera_xyz}")
                    print(f"piper_pose={piper_pose}")
                    print(f"target_base_m={tuple(float(value) for value in base_point[:3])}")
                elif key == ord("q"):
                    break
                continue

            cv2.putText(display, "q: quit", (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow("YOLO D405 Piper target validation", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
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
