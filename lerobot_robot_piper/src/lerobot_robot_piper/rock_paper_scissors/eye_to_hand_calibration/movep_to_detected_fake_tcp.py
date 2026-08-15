#!/usr/bin/env python3
"""Detect a ball with fixed D405 and move Piper fake TCP with movep_to_pose.py."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from common import apply_transform, load_transform_json
from rh56f2_hand import DEFAULT_CLOSED, RH56F2Hand, RH56F2HandConfig


def parse_rpy(value: str) -> tuple[float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected RX,RY,RZ in degrees")
    return tuple(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=Path("fake_tcp_calibration.json"))
    parser.add_argument("--can", default="can0")
    parser.add_argument("--serial", default="315122271151")
    parser.add_argument("--model", default=str(THIS_DIR.parent / "yolo26n.pt"))
    parser.add_argument("--rpy", type=parse_rpy, default=(176.0, 45.0, -170.0), help="MOVE_P target RX,RY,RZ in degrees")
    parser.add_argument("--safe-z-mm", type=float, default=180.0, help="override target Z for first validation move")
    parser.add_argument("--z-offset-mm", type=float, default=0.0, help="add to converted target Z before safe-z override")
    parser.add_argument("--speed", type=int, default=3)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-baudrate", type=int, default=115200)
    parser.add_argument("--hand-speed", type=int, default=2500)
    parser.add_argument("--hand-force", type=int, default=1500)
    parser.add_argument("--hand-settle", type=float, default=1.0)
    parser.add_argument("--no-fist", action="store_true", help="do not command RH56F2 closed/fist before MOVE_P")
    parser.add_argument("--execute", action="store_true", help="actually run movep_to_pose.py; default only prints command")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=960)
    return parser.parse_args()


def detect_once(args: argparse.Namespace) -> tuple[float, float, float]:
    import cv2
    import numpy as np
    import pyrealsense2 as rs
    from ultralytics import YOLO

    sys.path.insert(0, str(THIS_DIR.parent))
    from test_yolo_d405_ball import depth_at_box

    model = YOLO(args.model)
    ball_ids = {class_id for class_id, name in model.names.items() if str(name).lower() == "sports ball"}
    if not ball_ids:
        raise SystemExit(f"The selected model has no sports ball class: {model.names}")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    try:
        best = None
        for _ in range(30):
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            result = model.predict(color, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            if result.boxes is None:
                continue
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            for box_tensor, confidence_tensor, class_tensor in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if int(class_tensor.item()) not in ball_ids:
                    continue
                confidence = float(confidence_tensor.item())
                x0, y0, x1, y1 = [int(value) for value in box_tensor.tolist()]
                x0 = max(0, min(639, x0))
                y0 = max(0, min(479, y0))
                x1 = max(x0 + 1, min(640, x1))
                y1 = max(y0 + 1, min(480, y1))
                depth_result = depth_at_box(depth_frame, depth_image, (x0, y0, x1, y1), depth_scale)
                if depth_result is None:
                    continue
                depth_m, pixel = depth_result
                point = rs.rs2_deproject_pixel_to_point(intrinsics, list(pixel), depth_m)
                if best is None or confidence > best[0]:
                    best = (confidence, tuple(float(value) for value in point))
        if best is None:
            raise SystemExit("No YOLO sports ball detected.")
        return best[1]
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def command_fist(args: argparse.Namespace) -> None:
    if args.no_fist:
        print("RH56F2 fist skipped by --no-fist")
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
    print(f"Closing RH56F2 hand to fist: port={args.hand_port} id={args.hand_id}")
    hand.connect()
    try:
        accepted = hand.set_angles(DEFAULT_CLOSED)
        print(f"fist command accepted={accepted} ack={hand.last_write_ack}")
        if args.hand_settle > 0:
            time.sleep(args.hand_settle)
    finally:
        hand.disconnect()


def main() -> int:
    args = parse_args()
    _, transform = load_transform_json(args.calibration)
    camera_xyz = detect_once(args)
    target_m = apply_transform(transform, camera_xyz)
    target_mm = [float(value) * 1000.0 for value in target_m]
    target_mm[2] += args.z_offset_mm
    if args.safe_z_mm is not None:
        target_mm[2] = args.safe_z_mm
    rx, ry, rz = args.rpy
    target = f"{target_mm[0]:.3f},{target_mm[1]:.3f},{target_mm[2]:.3f},{rx:.3f},{ry:.3f},{rz:.3f}"
    movep = PACKAGE_DIR / "movep_to_pose.py"
    cmd = [
        sys.executable,
        str(movep),
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
    print(f"camera_xyz_m={camera_xyz}")
    print(f"converted_target_m={tuple(float(value) for value in target_m)}")
    print(f"movep_target_mm_deg={target}")
    print("command:")
    print(" ".join(cmd))
    if not args.execute:
        print("dry run only. Add --execute to run movep_to_pose.py.")
        return 0
    command_fist(args)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
