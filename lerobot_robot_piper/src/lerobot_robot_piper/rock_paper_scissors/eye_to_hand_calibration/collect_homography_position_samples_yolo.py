#!/usr/bin/env python3
"""Collect fixed-camera homography samples: image pixel (u,v) -> Piper base (X,Y)."""

from __future__ import annotations

import argparse
import csv
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

from ball_tactile_classifier.common import BALL_READY_OPEN, BALL_SAFE_CLOSED, FINGER_NAMES
from common import parse_roi, read_piper_pose
from rh56f2_hand import RH56F2Hand, RH56F2HandConfig
from rps_prize_controller import rotate_d405_box_ccw_90, rotate_d405_ccw_90, rotate_d405_point_ccw_90
from test_yolo_d405_ball import unzoom_box, zoom_for_detection


FIELDS = [
    "unix_time",
    "camera_pixel_u",
    "camera_pixel_v",
    "base_x_m",
    "base_y_m",
    "base_z_m",
    "base_rx_deg",
    "base_ry_deg",
    "base_rz_deg",
    "confidence",
]


@dataclass(frozen=True)
class PixelDetection:
    confidence: float
    box: tuple[int, int, int, int]
    pixel: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="315122271151")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--model", default=str(RPS_DIR / "yolo26n.pt"))
    parser.add_argument("--output", type=Path, default=Path("homography_position_samples.csv"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--roi", type=parse_roi, default=(0, 0, 640, 480))
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detect-zoom", type=float, default=1.0, help="center crop zoom before YOLO detection; saved pixels use original image coordinates")
    parser.add_argument("--show-detect-view", action="store_true", help="display the zoomed image used for YOLO instead of the original camera view")
    parser.add_argument("--stable-frames", type=int, default=8)
    parser.add_argument("--max-pixel-jump", type=float, default=20.0)
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-baudrate", type=int, default=115200)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--hand-force", type=int, default=600)
    parser.add_argument("--hand-settle", type=float, default=1.0)
    parser.add_argument(
        "--hand-pose",
        choices=("ball_safe_closed", "ball_ready_open"),
        default="ball_ready_open",
        help="RH56F2 pose to command before sampling. Default matches tactile classification pre-grasp pose.",
    )
    parser.add_argument("--no-hand-pose", action="store_true", help="do not connect RH56F2 or command hand pose")
    return parser.parse_args()


def require_can_up(can_name: str) -> None:
    operstate_path = Path("/sys/class/net") / can_name / "operstate"
    if not operstate_path.exists():
        raise SystemExit(f"CAN interface {can_name!r} does not exist. Check USB-CAN connection.")
    state = operstate_path.read_text().strip()
    if state != "up":
        raise SystemExit(
            f"CAN interface {can_name!r} is {state.upper()}, so Piper pose cannot be read.\n"
            f"Bring it up first:\n"
            f"  sudo ip link set {can_name} down\n"
            f"  sudo ip link set {can_name} up type can bitrate 1000000"
        )


def predict_kwargs(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {"conf": args.conf, "imgsz": args.imgsz, "verbose": False}
    if args.device:
        kwargs["device"] = args.device
    return kwargs


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


def stable_update(
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


def median_detection(detections: list[PixelDetection]) -> PixelDetection | None:
    if not detections:
        return None
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


def draw_display(
    color: np.ndarray,
    detection: PixelDetection | None,
    locked: PixelDetection | None,
    stable_count: int,
    stable_frames: int,
    sample_count: int,
) -> np.ndarray:
    display = rotate_d405_ccw_90(color)
    cv2.putText(display, "D405 view - homography position sampling", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if detection is None:
        cv2.putText(display, "YOLO BALL NOT FOUND", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2)
    else:
        rotated_box = rotate_d405_box_ccw_90(detection.box, color.shape[1])
        rotated_pixel = rotate_d405_point_ccw_90(detection.pixel, color.shape[1])
        cv2.rectangle(display, (rotated_box[0], rotated_box[1]), (rotated_box[2], rotated_box[3]), (0, 255, 0), 2)
        cv2.drawMarker(display, rotated_pixel, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(display, f"pixel=({detection.pixel[0]},{detection.pixel[1]}) conf={detection.confidence:.2f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if locked is None:
        cv2.putText(display, "step 1: wait stable, press c to lock pixel", (20, display.shape[0] - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
    else:
        cv2.putText(display, f"LOCKED pixel=({locked.pixel[0]},{locked.pixel[1]})", (20, display.shape[0] - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2)
    cv2.putText(display, f"stable={stable_count}/{stable_frames} samples={sample_count}  c:lock s:save x:clear g:pregrasp q:quit", (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    return display


def count_existing_samples(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def format_realsense_devices(rs_module: object) -> str:
    devices = []
    for device in rs_module.context().query_devices():
        name = device.get_info(rs_module.camera_info.name)
        serial = device.get_info(rs_module.camera_info.serial_number)
        devices.append(f"{name} serial={serial}")
    return "\n".join(f"  {item}" for item in devices) if devices else "  none"


def selected_hand_pose(args: argparse.Namespace) -> dict[str, float]:
    if args.hand_pose == "ball_safe_closed":
        return BALL_SAFE_CLOSED
    return BALL_READY_OPEN


def format_hand_pose(pose: dict[str, float]) -> str:
    return ", ".join(f"{name}={pose[name]:.0f}" for name in FINGER_NAMES)


def connect_and_command_hand(args: argparse.Namespace) -> RH56F2Hand | None:
    if args.no_hand_pose:
        print("RH56F2 hand pose skipped by --no-hand-pose")
        return None
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
    pose = selected_hand_pose(args)
    print(f"Setting RH56F2 hand pose: {args.hand_pose} port={args.hand_port} id={args.hand_id}")
    print(f"pose: {format_hand_pose(pose)}")
    hand.connect()
    accepted = hand.set_angles(pose)
    print(f"hand command accepted={accepted} ack={hand.last_write_ack}")
    if args.hand_settle > 0:
        time.sleep(args.hand_settle)
    return hand


def main() -> int:
    args = parse_args()
    try:
        import pyrealsense2 as rs
        from piper_sdk import C_PiperInterface_V2
    except ImportError as exc:
        raise SystemExit("pyrealsense2, piper_sdk, and ultralytics are required") from exc

    model = YOLO(args.model)
    ball_class_ids = {class_id for class_id, name in model.names.items() if str(name).lower() == "sports ball"}
    if not ball_class_ids:
        raise SystemExit(f"The selected model has no sports ball class: {model.names}")

    print("SAFETY: homography position collection is read-only for Piper.")
    print("It does not enable, disable, or move the arm.")
    print("By default it commands RH56F2 to the tactile-classifier pre-grasp pose.")
    print("Per sample: c locks image pixel, then move fake TCP to the matching tabletop point, s saves current base XY.")
    if input("Type READONLY to continue: ").strip() != "READONLY":
        print("Cancelled.")
        return 0

    require_can_up(args.can)
    piper = C_PiperInterface_V2(args.can, judge_flag=False, can_auto_init=False, dh_is_offset=1, start_sdk_fk_cal=True)
    piper.ConnectPort()
    time.sleep(1.0)
    hand = connect_and_command_hand(args)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = None
    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        raise SystemExit(
            f"Could not start RealSense D405 with serial={args.serial!r}.\n"
            f"Connected RealSense devices:\n{format_realsense_devices(rs)}"
        ) from exc

    detections: list[PixelDetection] = []
    locked: PixelDetection | None = None
    output_exists = args.output.exists() and args.output.stat().st_size > 0
    sample_count = count_existing_samples(args.output)
    try:
        with args.output.open("a", newline="") as sample_file:
            writer = csv.writer(sample_file)
            if not output_exists:
                writer.writerow(FIELDS)
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color = np.asanyarray(color_frame.get_data())
                detect_color, zoom_transform = zoom_for_detection(color, args.detect_zoom)
                result = model.predict(detect_color, **predict_kwargs(args))[0]
                detection = best_ball_pixel(result, ball_class_ids, args.roi, args.width, args.height, zoom_transform)
                detections = stable_update(detections, detection, args.stable_frames, args.max_pixel_jump)
                stable = median_detection(detections) if len(detections) >= args.stable_frames else None
                display_color = detect_color if args.show_detect_view else color
                display_detection = map_detection_to_zoom_view(stable or detection, zoom_transform, args.width, args.height) if args.show_detect_view else stable or detection
                display_locked = map_detection_to_zoom_view(locked, zoom_transform, args.width, args.height) if args.show_detect_view else locked
                display = draw_display(display_color, display_detection, display_locked, len(detections), args.stable_frames, sample_count)
                cv2.imshow("D405 view - homography position sampling", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    if stable is None:
                        print(f"Cannot lock yet: not stable ({len(detections)}/{args.stable_frames})")
                    else:
                        locked = stable
                        print(f"locked pixel: {locked.pixel}")
                if key == ord("x"):
                    locked = None
                    print("cleared locked pixel")
                if key == ord("g"):
                    if hand is None:
                        print("Hand is not connected; run without --no-hand-pose to use g:hand pose.")
                    else:
                        accepted = hand.set_angles(selected_hand_pose(args))
                        print(f"hand command accepted={accepted} ack={hand.last_write_ack}")
                if key == ord("s"):
                    if locked is None:
                        print("No locked pixel. First wait stable and press c.")
                        continue
                    pose = read_piper_pose(piper)
                    writer.writerow([
                        f"{time.time():.6f}",
                        locked.pixel[0],
                        locked.pixel[1],
                        *[f"{value:.6f}" for value in pose],
                        f"{locked.confidence:.6f}",
                    ])
                    sample_file.flush()
                    sample_count += 1
                    print(f"saved #{sample_count}: pixel={locked.pixel}, base_xy=({pose[0]:.6f},{pose[1]:.6f})")
                    locked = None
    finally:
        if profile is not None:
            pipeline.stop()
        cv2.destroyAllWindows()
        try:
            piper.DisconnectPort()
        except Exception:
            pass
        if hand is not None:
            try:
                hand.disconnect()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
