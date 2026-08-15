#!/usr/bin/env python3
"""Collect fixed-D405 camera point to Piper fake-TCP target samples with YOLO."""

from __future__ import annotations

import argparse
import csv
import sys
import time
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

from common import CSV_FIELDS, Detection3D, parse_roi, read_piper_pose
from rh56f2_hand import DEFAULT_OPEN, RH56F2Hand, RH56F2HandConfig
from rps_prize_controller import (
    rotate_d405_box_ccw_90,
    rotate_d405_ccw_90,
    rotate_d405_point_ccw_90,
)
from test_yolo_d405_ball import depth_at_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="315122271151")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--model", default=str(RPS_DIR / "yolo26n.pt"))
    parser.add_argument("--output", type=Path, default=Path("fake_tcp_samples.csv"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--roi", type=parse_roi, default=(0, 0, 640, 480))
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu", help="ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--stable-frames", type=int, default=10)
    parser.add_argument("--max-pixel-jump", type=float, default=35.0)
    parser.add_argument("--max-depth-jump", type=float, default=0.03)
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-baudrate", type=int, default=115200)
    parser.add_argument("--hand-speed", type=int, default=2500)
    parser.add_argument("--hand-force", type=int, default=1500)
    parser.add_argument("--hand-settle", type=float, default=1.0)
    parser.add_argument("--no-open-hand", action="store_true", help="do not connect RH56F2 or command open pose")
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


def median_detection(detections: list[Detection3D]) -> Detection3D | None:
    if not detections:
        return None
    xyz = np.median([item.camera_xyz_m for item in detections], axis=0)
    pixel = np.median([item.pixel for item in detections], axis=0)
    box = np.median([item.box for item in detections], axis=0)
    return Detection3D(
        confidence=float(np.median([item.confidence for item in detections])),
        box=tuple(int(round(value)) for value in box),
        pixel=(int(round(pixel[0])), int(round(pixel[1]))),
        depth_m=float(np.median([item.depth_m for item in detections])),
        camera_xyz_m=tuple(float(value) for value in xyz),
    )


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
    for box_tensor, confidence_tensor, class_tensor in zip(
        result.boxes.xyxy,
        result.boxes.conf,
        result.boxes.cls,
    ):
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
        if best is None or detection.confidence > best.confidence:
            best = detection
    return best


def stable_update(
    detections: list[Detection3D],
    detection: Detection3D | None,
    stable_frames: int,
    max_pixel_jump: float,
    max_depth_jump: float,
) -> list[Detection3D]:
    if detection is None:
        return []
    if detections:
        previous = detections[-1]
        pixel_jump = float(np.hypot(detection.pixel[0] - previous.pixel[0], detection.pixel[1] - previous.pixel[1]))
        depth_jump = abs(detection.depth_m - previous.depth_m)
        if pixel_jump > max_pixel_jump or depth_jump > max_depth_jump:
            detections = []
    return (detections + [detection])[-stable_frames:]


def draw_display(
    color: np.ndarray,
    detection: Detection3D | None,
    locked: Detection3D | None,
    stable_count: int,
    stable_frames: int,
    sample_count: int,
) -> np.ndarray:
    display = rotate_d405_ccw_90(color)
    cv2.putText(display, "D405 view", (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if detection is None:
        cv2.putText(display, "YOLO BALL NOT FOUND", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2)
    else:
        rotated_box = rotate_d405_box_ccw_90(detection.box, color.shape[1])
        rotated_center = rotate_d405_point_ccw_90(detection.pixel, color.shape[1])
        cv2.rectangle(display, (rotated_box[0], rotated_box[1]), (rotated_box[2], rotated_box[3]), (0, 255, 0), 2)
        cv2.drawMarker(display, rotated_center, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        x, y, z = detection.camera_xyz_m
        cv2.putText(display, f"camera=({x:.3f},{y:.3f},{z:.3f})m conf={detection.confidence:.2f}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
    if locked is None:
        cv2.putText(display, "step 1: wait stable, press c to lock camera point", (20, display.shape[0] - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
    else:
        x, y, z = locked.camera_xyz_m
        cv2.putText(display, f"LOCKED camera=({x:.3f},{y:.3f},{z:.3f})m", (20, display.shape[0] - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2)
    cv2.putText(display, f"stable={stable_count}/{stable_frames} samples={sample_count}  c:lock s:save x:clear o:open q:quit", (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
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


def require_can_up(can_name: str) -> None:
    operstate_path = Path("/sys/class/net") / can_name / "operstate"
    if not operstate_path.exists():
        raise SystemExit(
            f"CAN interface {can_name!r} does not exist. Check USB-CAN connection and run: ip link show"
        )
    state = operstate_path.read_text().strip()
    if state != "up":
        raise SystemExit(
            f"CAN interface {can_name!r} is {state.upper()}, so Piper pose cannot be read.\n"
            f"Bring it up first:\n"
            f"  sudo ip link set {can_name} down\n"
            f"  sudo ip link set {can_name} up type can bitrate 1000000\n"
            f"  ip -details link show {can_name}"
        )


def connect_and_open_hand(args: argparse.Namespace) -> RH56F2Hand | None:
    if args.no_open_hand:
        print("RH56F2 open skipped by --no-open-hand")
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
    print(f"Opening RH56F2 hand: port={args.hand_port} id={args.hand_id}")
    hand.connect()
    accepted = hand.set_angles(DEFAULT_OPEN)
    print(f"open command accepted={accepted} ack={hand.last_write_ack}")
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

    global rs2_deproject_pixel_to_point
    rs2_deproject_pixel_to_point = rs.rs2_deproject_pixel_to_point

    model = YOLO(args.model)
    ball_class_ids = {class_id for class_id, name in model.names.items() if str(name).lower() == "sports ball"}
    if not ball_class_ids:
        raise SystemExit(f"The selected model has no sports ball class: {model.names}")

    print("SAFETY: read-only sample collection.")
    print("This program does not enable, disable, or move Piper.")
    print("By default it opens RH56F2 once and keeps the hand connection alive.")
    print("For each sample, manually place the fake TCP where you want the camera point to map.")
    if input("Type READONLY to continue: ").strip() != "READONLY":
        print("Cancelled.")
        return 0

    require_can_up(args.can)
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
    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        raise SystemExit(
            f"Could not start RealSense D405 with serial={args.serial!r}.\n"
            f"Connected RealSense devices:\n{format_realsense_devices(rs)}\n"
            f"Use the shown serial, or pass --serial \"\" to use any connected RealSense."
        ) from exc
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    hand = connect_and_open_hand(args)

    detections: list[Detection3D] = []
    locked_detection: Detection3D | None = None
    output_exists = args.output.exists() and args.output.stat().st_size > 0
    sample_count = count_existing_samples(args.output)

    try:
        with args.output.open("a", newline="") as sample_file:
            writer = csv.writer(sample_file)
            if not output_exists:
                writer.writerow(CSV_FIELDS)
            print(f"D405={args.serial}, Piper CAN={args.can}, YOLO={args.model}")
            print("Workflow per sample:")
            print("  1) Keep hand/arm out of D405 view, wait stable, press c to lock camera point.")
            print("  2) Move fake TCP to the desired target point, press s to save pair.")
            print("  x clears the locked camera point; o opens the hand again.")
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
                detections = stable_update(
                    detections,
                    detection,
                    args.stable_frames,
                    args.max_pixel_jump,
                    args.max_depth_jump,
                )
                stable = median_detection(detections) if len(detections) >= args.stable_frames else None
                display = draw_display(
                    color,
                    stable or detection,
                    locked_detection,
                    len(detections),
                    args.stable_frames,
                    sample_count,
                )
                cv2.imshow("D405 view - fake TCP sample collection", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    if stable is None:
                        print(f"Cannot lock yet: not stable ({len(detections)}/{args.stable_frames})")
                    else:
                        locked_detection = stable
                        print(f"locked camera point: {locked_detection.camera_xyz_m}")
                if key == ord("x"):
                    locked_detection = None
                    print("cleared locked camera point")
                if key == ord("o"):
                    if hand is None:
                        print("Hand is not connected; run without --no-open-hand to use o:open.")
                    else:
                        accepted = hand.set_angles(DEFAULT_OPEN)
                        print(f"open command accepted={accepted} ack={hand.last_write_ack}")
                if key == ord("s"):
                    if locked_detection is None:
                        print("No locked camera point. First keep the view clear, wait stable, press c.")
                        continue
                    pose = read_piper_pose(piper)
                    writer.writerow([
                        f"{time.time():.6f}",
                        *[f"{value:.6f}" for value in locked_detection.camera_xyz_m],
                        *[f"{value:.6f}" for value in pose],
                        locked_detection.pixel[0],
                        locked_detection.pixel[1],
                        f"{locked_detection.depth_m:.6f}",
                        f"{locked_detection.confidence:.6f}",
                    ])
                    sample_file.flush()
                    sample_count += 1
                    print(f"saved #{sample_count}: camera={locked_detection.camera_xyz_m}, fake_tcp_pose={pose}")
                    locked_detection = None
    finally:
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
