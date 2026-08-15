#!/usr/bin/env python3
"""Run pretrained YOLO sports-ball detection on the hand-mounted D405."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

from rps_prize_controller import rotate_d405_box_ccw_90, rotate_d405_ccw_90, rotate_d405_point_ccw_90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="260322279862")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--device", default="cpu", help="ultralytics device, e.g. cpu or 0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--detect-zoom", type=float, default=1.0, help="center crop zoom before YOLO detection; 1.0 disables it")
    parser.add_argument("--show-detect-view", action="store_true", help="display the zoomed image used for YOLO instead of the original camera view")
    parser.add_argument("--rs-timeout-ms", type=int, default=15000, help="RealSense frame wait timeout")
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=960)
    return parser.parse_args()


def format_realsense_devices(rs_module: object) -> str:
    devices = []
    for device in rs_module.context().query_devices():
        name = device.get_info(rs_module.camera_info.name)
        serial = device.get_info(rs_module.camera_info.serial_number)
        usb_type = device.get_info(rs_module.camera_info.usb_type_descriptor)
        devices.append(f"{name} serial={serial} usb={usb_type}")
    return "\n".join(f"  {item}" for item in devices) if devices else "  none"


def zoom_for_detection(image: np.ndarray, zoom: float) -> tuple[np.ndarray, tuple[int, int, float, float]]:
    if zoom <= 1.0:
        return image, (0, 0, 1.0, 1.0)
    height, width = image.shape[:2]
    crop_width = max(1, int(round(width / zoom)))
    crop_height = max(1, int(round(height / zoom)))
    crop_x0 = (width - crop_width) // 2
    crop_y0 = (height - crop_height) // 2
    crop = image[crop_y0:crop_y0 + crop_height, crop_x0:crop_x0 + crop_width]
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    scale_x = crop_width / width
    scale_y = crop_height / height
    return resized, (crop_x0, crop_y0, scale_x, scale_y)


def unzoom_box(box: tuple[int, int, int, int], transform: tuple[int, int, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    crop_x0, crop_y0, scale_x, scale_y = transform
    x0, y0, x1, y1 = box
    mapped_x0 = int(round(crop_x0 + x0 * scale_x))
    mapped_y0 = int(round(crop_y0 + y0 * scale_y))
    mapped_x1 = int(round(crop_x0 + x1 * scale_x))
    mapped_y1 = int(round(crop_y0 + y1 * scale_y))
    mapped_x0 = max(0, min(width - 1, mapped_x0))
    mapped_y0 = max(0, min(height - 1, mapped_y0))
    mapped_x1 = max(mapped_x0 + 1, min(width, mapped_x1))
    mapped_y1 = max(mapped_y0 + 1, min(height, mapped_y1))
    return mapped_x0, mapped_y0, mapped_x1, mapped_y1


def depth_at_box(depth_frame: object, depth_image: np.ndarray, box: tuple[int, int, int, int], depth_scale: float) -> tuple[float, tuple[int, int]] | None:
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    inner_x0 = x0 + int(width * 0.30)
    inner_x1 = x1 - int(width * 0.30)
    inner_y0 = y0 + int(height * 0.30)
    inner_y1 = y1 - int(height * 0.30)
    region = depth_image[inner_y0:inner_y1, inner_x0:inner_x1].astype(np.float32) * depth_scale
    valid = region[region > 0.05]
    if valid.size == 0:
        return None
    depth_m = float(np.median(valid))
    center = ((x0 + x1) // 2, (y0 + y1) // 2)
    return depth_m, center


def main() -> int:
    args = parse_args()
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit("pyrealsense2 is not installed in this Python environment") from exc

    model = YOLO(args.model)
    ball_class_ids = {class_id for class_id, name in model.names.items() if name == "sports ball"}
    if not ball_class_ids:
        raise SystemExit(f"Model does not contain a 'sports ball' class: {model.names}")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)
    print(f"Connected RealSense devices:\n{format_realsense_devices(rs)}")
    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        raise SystemExit(
            f"Could not start RealSense camera with serial={args.serial!r}: {exc}\n"
            f"Connected RealSense devices:\n{format_realsense_devices(rs)}\n"
            f"Check the serial, USB3 connection, permissions, and whether another program is using the camera."
        ) from exc
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    print("SAFETY: YOLO and D405 read-only test.")
    print("No Piper connection, enable, disable, or motion commands are used.")
    if args.detect_zoom > 1.0:
        print(f"YOLO detection zoom: {args.detect_zoom:.2f}x center crop. Field of view is narrower.")
    print("Press q to quit; press p to print the best detected ball.")
    window = "YOLO D405 ball test"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.window_width, args.window_height)
    last_print: tuple[float, tuple[float, float, float], float] | None = None
    try:
        while True:
            try:
                frames = align.process(pipeline.wait_for_frames(args.rs_timeout_ms))
            except RuntimeError as exc:
                raise SystemExit(
                    f"RealSense frame timeout after {args.rs_timeout_ms} ms: {exc}\n"
                    f"Connected RealSense devices:\n{format_realsense_devices(rs)}\n"
                    f"Try reconnecting the D405 on USB3, closing realsense-viewer/other camera users, "
                    f"or testing a lower stream rate such as --fps 15."
                ) from exc
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            detect_color, zoom_transform = zoom_for_detection(color, args.detect_zoom)
            result = model.predict(detect_color, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            display_source = detect_color if args.show_detect_view else color
            display = rotate_d405_ccw_90(display_source)
            detections: list[tuple[float, tuple[int, int, int, int], float, tuple[float, float, float]]] = []
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            if result.boxes is not None:
                for box_tensor, confidence_tensor, class_tensor in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    class_id = int(class_tensor.item())
                    confidence = float(confidence_tensor.item())
                    if class_id not in ball_class_ids:
                        continue
                    detect_box = tuple(int(value) for value in box_tensor.tolist())
                    x0, y0, x1, y1 = unzoom_box(detect_box, zoom_transform, args.width, args.height)
                    depth_result = depth_at_box(depth_frame, depth_image, (x0, y0, x1, y1), depth_scale)
                    if depth_result is None:
                        continue
                    depth_m, center = depth_result
                    point = rs.rs2_deproject_pixel_to_point(intrinsics, list(center), depth_m)
                    point_xyz = tuple(float(value) for value in point)
                    if args.show_detect_view:
                        draw_box = detect_box
                        draw_center = ((detect_box[0] + detect_box[2]) // 2, (detect_box[1] + detect_box[3]) // 2)
                    else:
                        draw_box = (x0, y0, x1, y1)
                        draw_center = center
                    rotated_box = rotate_d405_box_ccw_90(draw_box, display_source.shape[1])
                    rotated_center = rotate_d405_point_ccw_90(draw_center, display_source.shape[1])
                    detections.append((confidence, rotated_box, depth_m, point_xyz))
                    cv2.rectangle(display, (rotated_box[0], rotated_box[1]), (rotated_box[2], rotated_box[3]), (0, 255, 0), 2)
                    cv2.drawMarker(display, rotated_center, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(display, f"BALL {confidence:.2f} depth={depth_m:.3f}m", (rotated_box[0], max(24, rotated_box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
            detections.sort(key=lambda item: item[0], reverse=True)
            if detections:
                best = detections[0]
                last_print = (best[2], best[3], best[0])
                cv2.putText(display, f"balls={len(detections)}  q:quit p:print", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            else:
                cv2.putText(display, "BALL NOT FOUND  q:quit", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p") and last_print is not None:
                print(f"confidence={last_print[2]:.3f} depth_m={last_print[0]:.4f} camera_xyz_m={last_print[1]}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
