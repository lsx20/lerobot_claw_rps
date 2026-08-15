#!/usr/bin/env python3
"""ROS2 read-only D405 YOLO ball target publisher.

Publishes the selected ball as geometry_msgs/PointStamped in the D405 optical
camera frame. This script does not connect to Piper and never sends motion,
enable, disable, gripper, or hand commands.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from rps_prize_controller import rotate_d405_box_ccw_90, rotate_d405_ccw_90, rotate_d405_point_ccw_90
from test_yolo_d405_ball import depth_at_box


@dataclass(frozen=True)
class Detection:
    confidence: float
    box: tuple[int, int, int, int]
    pixel: tuple[int, int]
    depth_m: float
    camera_xyz_m: tuple[float, float, float]


def parse_roi(value: str) -> tuple[int, int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected x0,y0,x1,y1")
    try:
        x0, y0, x1, y1 = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI values must be integers") from exc
    if x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError("ROI must satisfy x1>x0 and y1>y0")
    return x0, y0, x1, y1


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="260322279862")
    parser.add_argument("--model", default="yolo11x.pt")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--roi", type=parse_roi, default=(0, 0, 640, 480))
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--max-pixel-jump", type=float, default=30.0)
    parser.add_argument("--max-depth-jump", type=float, default=0.03)
    parser.add_argument("--frame-id", default="d405_color_optical_frame")
    parser.add_argument("--topic", default="/d405/ball_point_camera")
    parser.add_argument("--no-window", action="store_true")
    return parser.parse_known_args()


def median_detection(detections: list[Detection]) -> Detection:
    pixels = np.median([item.pixel for item in detections], axis=0)
    xyz = np.median([item.camera_xyz_m for item in detections], axis=0)
    depth_m = float(np.median([item.depth_m for item in detections]))
    confidence = float(np.median([item.confidence for item in detections]))
    widths = [item.box[2] - item.box[0] for item in detections]
    heights = [item.box[3] - item.box[1] for item in detections]
    center_x = int(round(float(pixels[0])))
    center_y = int(round(float(pixels[1])))
    width = int(round(float(np.median(widths))))
    height = int(round(float(np.median(heights))))
    return Detection(
        confidence=confidence,
        box=(center_x - width // 2, center_y - height // 2, center_x + width // 2, center_y + height // 2),
        pixel=(center_x, center_y),
        depth_m=depth_m,
        camera_xyz_m=tuple(float(value) for value in xyz),
    )


def candidate_score(detection: Detection, width: int, height: int) -> float:
    image_center = (width / 2.0, height / 2.0)
    center_distance = float(np.hypot(detection.pixel[0] - image_center[0], detection.pixel[1] - image_center[1]))
    return detection.confidence - 0.001 * center_distance


def main() -> int:
    args, ros_args = parse_args()
    if args.stable_frames <= 0:
        raise SystemExit("--stable-frames must be positive")

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit("pyrealsense2 is not installed in this Python environment") from exc
    try:
        import rclpy
        from geometry_msgs.msg import PointStamped
        from rclpy.node import Node
    except ImportError as exc:
        raise SystemExit("ROS2 rclpy is not available. Run: source /opt/ros/humble/setup.bash") from exc

    class TargetNode(Node):
        def __init__(self) -> None:
            super().__init__("d405_yolo_target_node")
            self.publisher = self.create_publisher(PointStamped, args.topic, 10)

    model = YOLO(args.model)
    ball_class_ids = {class_id for class_id, name in model.names.items() if str(name).lower() == "sports ball"}
    if not ball_class_ids:
        raise SystemExit(f"Model does not contain a 'sports ball' class: {model.names}")

    rclpy.init(args=ros_args)
    node = TargetNode()

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    stable_buffer: deque[Detection] = deque(maxlen=args.stable_frames)
    last_log = 0.0

    print("SAFETY: ROS2 D405 YOLO target publisher is read-only.")
    print("No Piper connection, enable, disable, gripper, or motion commands are used.")
    print(f"publishing {args.topic} in frame {args.frame_id}")

    try:
        while rclpy.ok():
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                rclpy.spin_once(node, timeout_sec=0.0)
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            result = model.predict(color, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            detections: list[Detection] = []

            if result.boxes is not None:
                for box_tensor, confidence_tensor, class_tensor in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    if int(class_tensor.item()) not in ball_class_ids:
                        continue
                    x0, y0, x1, y1 = [int(value) for value in box_tensor.tolist()]
                    x0 = max(0, min(args.width - 1, x0))
                    y0 = max(0, min(args.height - 1, y0))
                    x1 = max(x0 + 1, min(args.width, x1))
                    y1 = max(y0 + 1, min(args.height, y1))
                    center = ((x0 + x1) // 2, (y0 + y1) // 2)
                    if not (args.roi[0] <= center[0] <= args.roi[2] and args.roi[1] <= center[1] <= args.roi[3]):
                        continue
                    depth_result = depth_at_box(depth_frame, depth_image, (x0, y0, x1, y1), depth_scale)
                    if depth_result is None:
                        continue
                    depth_m, pixel = depth_result
                    point = rs.rs2_deproject_pixel_to_point(intrinsics, list(pixel), depth_m)
                    detections.append(
                        Detection(
                            confidence=float(confidence_tensor.item()),
                            box=(x0, y0, x1, y1),
                            pixel=pixel,
                            depth_m=depth_m,
                            camera_xyz_m=tuple(float(value) for value in point),
                        )
                    )

            best = max(detections, key=lambda item: candidate_score(item, args.width, args.height), default=None)
            stable = None
            if best is None:
                stable_buffer.clear()
            else:
                if stable_buffer:
                    previous = stable_buffer[-1]
                    pixel_jump = float(np.hypot(best.pixel[0] - previous.pixel[0], best.pixel[1] - previous.pixel[1]))
                    depth_jump = abs(best.depth_m - previous.depth_m)
                    if pixel_jump > args.max_pixel_jump or depth_jump > args.max_depth_jump:
                        stable_buffer.clear()
                stable_buffer.append(best)
                if len(stable_buffer) >= args.stable_frames:
                    stable = median_detection(list(stable_buffer))

            display = rotate_d405_ccw_90(color)
            if best is None:
                cv2.putText(display, "YOLO BALL NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                rotated_box = rotate_d405_box_ccw_90(best.box, color.shape[1])
                rotated_center = rotate_d405_point_ccw_90(best.pixel, color.shape[1])
                cv2.rectangle(display, (rotated_box[0], rotated_box[1]), (rotated_box[2], rotated_box[3]), (0, 255, 0), 2)
                cv2.drawMarker(display, rotated_center, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(
                    display,
                    f"BALL conf={best.confidence:.2f} depth={best.depth_m:.3f}m stable={len(stable_buffer)}/{args.stable_frames}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 0),
                    2,
                )

            if stable is not None:
                message = PointStamped()
                message.header.stamp = node.get_clock().now().to_msg()
                message.header.frame_id = args.frame_id
                message.point.x = stable.camera_xyz_m[0]
                message.point.y = stable.camera_xyz_m[1]
                message.point.z = stable.camera_xyz_m[2]
                node.publisher.publish(message)
                now = time.monotonic()
                if now - last_log >= 1.0:
                    print(
                        f"published conf={stable.confidence:.3f} "
                        f"pixel={stable.pixel} depth={stable.depth_m:.4f}m "
                        f"camera_xyz_m={stable.camera_xyz_m}"
                    )
                    last_log = now

            if not args.no_window:
                cv2.imshow("ROS2 D405 YOLO target - q to quit", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        pipeline.stop()
        if not args.no_window:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
