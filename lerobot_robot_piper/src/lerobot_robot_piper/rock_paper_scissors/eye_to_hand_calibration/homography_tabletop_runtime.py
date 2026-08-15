#!/usr/bin/env python3
"""Runtime helpers for fixed-camera tabletop homography grasp targets."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

THIS_DIR = Path(__file__).resolve().parent
RPS_DIR = THIS_DIR.parent
PACKAGE_DIR = RPS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(RPS_DIR) not in sys.path:
    sys.path.insert(0, str(RPS_DIR))
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from ball_tactile_classifier.common import BALL_READY_OPEN, FINGER_NAMES
from common import parse_roi
from rh56f2_hand import RH56F2Hand, RH56F2HandConfig
from rps_prize_controller import rotate_d405_box_ccw_90, rotate_d405_ccw_90, rotate_d405_point_ccw_90
from test_yolo_d405_ball import unzoom_box, zoom_for_detection


@dataclass(frozen=True)
class PixelDetection:
    confidence: float
    box: tuple[int, int, int, int]
    pixel: tuple[int, int]


@dataclass(frozen=True)
class PoseFieldMatch:
    pose: tuple[float, float, float, float, float, float]
    distance_m: float
    index: int


CLAW_START_POSE_MM_DEG = (161.039, 0.257, 332.985, 172.196, 49.686, 169.902)
CLAW_START_JOINTS_DEG = (0.091, 46.504, -45.622, 0.000, 43.982, 6.571)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calibration", type=Path, default=Path("homography_position_calibration.json"))
    parser.add_argument("--fixed-z-mm", type=float, required=True)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--serial", default="315122271151")
    parser.add_argument("--model", default=str(RPS_DIR / "yolo26n.pt"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--roi", type=parse_roi, default=(0, 0, 640, 480))
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detect-zoom", type=float, default=1.0, help="center crop zoom before YOLO detection; output pixels use original image coordinates")
    parser.add_argument("--show-detect-view", action="store_true", help="display the zoomed image used for YOLO instead of the original camera view")
    parser.add_argument("--stable-frames", type=int, default=10)
    parser.add_argument("--max-pixel-jump", type=float, default=20.0)
    parser.add_argument("--speed", type=int, default=8)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=40.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-baudrate", type=int, default=115200)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--hand-force", type=int, default=600)
    parser.add_argument("--hand-settle", type=float, default=0.6)
    parser.add_argument("--no-hand-pose", action="store_true", help="do not command RH56F2 pre-grasp pose")


def parse_rpy(value: str) -> tuple[float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected RX,RY,RZ in degrees")
    return tuple(parts)


def predict_kwargs(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {"conf": args.conf, "imgsz": args.imgsz, "verbose": False}
    if args.device:
        kwargs["device"] = args.device
    return kwargs


def load_homography(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    if payload.get("calibration_type") != "pixel_to_base_xy_homography":
        raise SystemExit(f"{path} is not a pixel_to_base_xy_homography calibration")
    return np.asarray(payload["H_pixel_to_base_xy"], dtype=float)


def apply_homography(homography: np.ndarray, pixel: tuple[int, int]) -> tuple[float, float]:
    point = np.array([float(pixel[0]), float(pixel[1]), 1.0], dtype=float)
    mapped = homography @ point
    if abs(float(mapped[2])) < 1e-12:
        raise RuntimeError("homography mapped target with near-zero scale")
    return (float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2]))


def best_ball_pixel(
    result: object,
    ball_class_ids: set[int],
    roi: tuple[int, int, int, int],
    width: int,
    height: int,
    zoom_transform: tuple[int, int, float, float] = (0, 0, 1.0, 1.0),
) -> PixelDetection | None:
    if result.boxes is None:
        return None
    best: PixelDetection | None = None
    for box_tensor, confidence_tensor, class_tensor in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        if int(class_tensor.item()) not in ball_class_ids:
            continue
        confidence = float(confidence_tensor.item())
        x0, y0, x1, y1 = [int(value) for value in box_tensor.tolist()]
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(x0 + 1, min(width, x1))
        y1 = max(y0 + 1, min(height, y1))
        x0, y0, x1, y1 = unzoom_box((x0, y0, x1, y1), zoom_transform, width, height)
        pixel = ((x0 + x1) // 2, (y0 + y1) // 2)
        if not (roi[0] <= pixel[0] <= roi[2] and roi[1] <= pixel[1] <= roi[3]):
            continue
        detection = PixelDetection(confidence=confidence, box=(x0, y0, x1, y1), pixel=pixel)
        if best is None or detection.confidence > best.confidence:
            best = detection
    return best


def update_stable(
    detections: list[PixelDetection],
    detection: PixelDetection | None,
    stable_frames: int,
    max_pixel_jump: float,
) -> list[PixelDetection]:
    if detection is None:
        return []
    if detections:
        previous = detections[-1]
        jump = float(np.hypot(detection.pixel[0] - previous.pixel[0], detection.pixel[1] - previous.pixel[1]))
        if jump > max_pixel_jump:
            detections = []
    return (detections + [detection])[-stable_frames:]


def median_detection(detections: list[PixelDetection]) -> PixelDetection:
    pixels = np.median([item.pixel for item in detections], axis=0)
    boxes = np.median([item.box for item in detections], axis=0)
    return PixelDetection(
        confidence=float(np.median([item.confidence for item in detections])),
        box=tuple(int(round(value)) for value in boxes),
        pixel=(int(round(pixels[0])), int(round(pixels[1]))),
    )


def map_detection_to_zoom_view(detection: PixelDetection | None, transform: tuple[int, int, float, float], width: int, height: int) -> PixelDetection | None:
    if detection is None:
        return None
    crop_x0, crop_y0, scale_x, scale_y = transform
    x0, y0, x1, y1 = detection.box
    mapped_x0 = int(round((x0 - crop_x0) / scale_x))
    mapped_y0 = int(round((y0 - crop_y0) / scale_y))
    mapped_x1 = int(round((x1 - crop_x0) / scale_x))
    mapped_y1 = int(round((y1 - crop_y0) / scale_y))
    mapped_x0 = max(0, min(width - 1, mapped_x0))
    mapped_y0 = max(0, min(height - 1, mapped_y0))
    mapped_x1 = max(mapped_x0 + 1, min(width, mapped_x1))
    mapped_y1 = max(mapped_y0 + 1, min(height, mapped_y1))
    pixel = ((mapped_x0 + mapped_x1) // 2, (mapped_y0 + mapped_y1) // 2)
    return PixelDetection(detection.confidence, (mapped_x0, mapped_y0, mapped_x1, mapped_y1), pixel)


def draw_preview(
    color: np.ndarray,
    detection: PixelDetection | None,
    target_xy_m: tuple[float, float] | None,
    stable_count: int,
    stable_frames: int,
) -> np.ndarray:
    display = rotate_d405_ccw_90(color)
    if detection is None:
        cv2.putText(display, "YOLO BALL NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2)
    else:
        rotated_box = rotate_d405_box_ccw_90(detection.box, color.shape[1])
        rotated_pixel = rotate_d405_point_ccw_90(detection.pixel, color.shape[1])
        cv2.rectangle(display, (rotated_box[0], rotated_box[1]), (rotated_box[2], rotated_box[3]), (0, 255, 0), 2)
        cv2.drawMarker(display, rotated_pixel, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(
            display,
            f"pixel=({detection.pixel[0]},{detection.pixel[1]}) conf={detection.confidence:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    if target_xy_m is not None:
        cv2.putText(
            display,
            f"target XY=({target_xy_m[0]:.3f},{target_xy_m[1]:.3f})m",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    cv2.putText(
        display,
        f"stable={stable_count}/{stable_frames}  q:quit",
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )
    return display


def detect_stable_pixel(args: argparse.Namespace, homography: np.ndarray) -> PixelDetection:
    import pyrealsense2 as rs

    model = YOLO(args.model)
    ball_class_ids = {class_id for class_id, name in model.names.items() if str(name).lower() == "sports ball"}
    if not ball_class_ids:
        raise SystemExit(f"The selected model has no sports ball class: {model.names}")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    detections: list[PixelDetection] = []
    try:
        pipeline.start(config)
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            detect_color, zoom_transform = zoom_for_detection(color, args.detect_zoom)
            result = model.predict(detect_color, **predict_kwargs(args))[0]
            detection = best_ball_pixel(result, ball_class_ids, args.roi, args.width, args.height, zoom_transform)
            detections = update_stable(detections, detection, args.stable_frames, args.max_pixel_jump)
            stable = median_detection(detections) if len(detections) >= args.stable_frames else None
            preview_target = apply_homography(homography, stable.pixel) if stable is not None else None
            if args.preview:
                display_color = detect_color if args.show_detect_view else color
                display_detection = map_detection_to_zoom_view(stable or detection, zoom_transform, args.width, args.height) if args.show_detect_view else stable or detection
                cv2.imshow(
                    "D405 homography tabletop target",
                    draw_preview(display_color, display_detection, preview_target, len(detections), args.stable_frames),
                )
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise SystemExit("Cancelled from preview window.")
            if stable is not None:
                return stable
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def load_pose_field(path: Path) -> list[tuple[float, float, float, float, float, float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no pose samples")
    fields = ("base_x_m", "base_y_m", "base_z_m", "base_rx_deg", "base_ry_deg", "base_rz_deg")
    return [tuple(float(row[field]) for field in fields) for row in rows]


def nearest_pose_field(path: Path, target_xy_m: tuple[float, float]) -> PoseFieldMatch:
    poses = load_pose_field(path)
    distances = [
        float(np.hypot(pose[0] - target_xy_m[0], pose[1] - target_xy_m[1]))
        for pose in poses
    ]
    index = int(np.argmin(distances))
    return PoseFieldMatch(pose=poses[index], distance_m=distances[index], index=index)


def format_pregrasp_pose() -> str:
    return ", ".join(f"{name}={BALL_READY_OPEN[name]:.0f}" for name in FINGER_NAMES)


def command_pregrasp(args: argparse.Namespace) -> None:
    if args.no_hand_pose:
        print("RH56F2 pre-grasp skipped by --no-hand-pose")
        return
    hand = RH56F2Hand(
        RH56F2HandConfig(
            port=args.hand_port,
            baudrate=args.hand_baudrate,
            hand_id=args.hand_id,
            speed=args.hand_speed,
            force=args.hand_force,
            mode=0,
        )
    )
    print(f"Setting RH56F2 pre-grasp pose: port={args.hand_port} id={args.hand_id}")
    print(f"pose: {format_pregrasp_pose()}")
    hand.connect()
    try:
        accepted = hand.set_angles(BALL_READY_OPEN)
        print(f"pre-grasp command accepted={accepted} ack={hand.last_write_ack}")
        if args.hand_settle > 0:
            time.sleep(args.hand_settle)
    finally:
        hand.disconnect()


def build_movep_command(
    args: argparse.Namespace,
    target_xy_m: tuple[float, float],
    rpy_deg: tuple[float, float, float],
) -> tuple[list[str], str]:
    return build_movep_command_from_pose(
        args,
        (
            target_xy_m[0] * 1000.0,
            target_xy_m[1] * 1000.0,
            args.fixed_z_mm,
            rpy_deg[0],
            rpy_deg[1],
            rpy_deg[2],
        ),
    )


def build_movep_command_from_pose(
    args: argparse.Namespace,
    pose_mm_deg: tuple[float, float, float, float, float, float],
) -> tuple[list[str], str]:
    target = (
        f"{pose_mm_deg[0]:.3f},"
        f"{pose_mm_deg[1]:.3f},"
        f"{pose_mm_deg[2]:.3f},"
        f"{pose_mm_deg[3]:.3f},"
        f"{pose_mm_deg[4]:.3f},"
        f"{pose_mm_deg[5]:.3f}"
    )
    cmd = [
        sys.executable,
        str(PACKAGE_DIR / "movep_to_pose.py"),
        "--can",
        args.can,
        "--target",
        target,
        "--speed",
        str(args.speed),
        "--duration",
        str(args.duration),
        "--rate-hz",
        str(args.rate_hz),
    ]
    return cmd, target


def print_movep_command(label: str, cmd: list[str], target: str) -> None:
    print(f"{label}_movep_target_mm_deg={target}")
    print(f"{label} command:")
    print(" ".join(cmd))


def print_and_maybe_execute_sequence(
    args: argparse.Namespace,
    commands: list[tuple[str, list[str], str]],
) -> int:
    for label, cmd, target in commands:
        print_movep_command(label, cmd, target)
    if not args.execute:
        print("dry run only. Add --execute to run movep_to_pose.py.")
        return 0
    command_pregrasp(args)
    for label, cmd, _ in commands:
        print(f"running {label} MOVE_P...")
        code = subprocess.run(cmd, input="YES\n\n", text=True, check=False).returncode
        if code != 0:
            print(f"{label} MOVE_P failed with exit code {code}; stopping sequence.")
            return code
    return 0


def print_and_maybe_execute(args: argparse.Namespace, cmd: list[str], target: str) -> int:
    return print_and_maybe_execute_sequence(args, [("target", cmd, target)])
