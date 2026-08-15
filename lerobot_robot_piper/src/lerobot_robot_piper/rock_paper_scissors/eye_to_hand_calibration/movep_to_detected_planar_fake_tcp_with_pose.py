#!/usr/bin/env python3
"""Detect a ball and move Piper using planar XY plus fitted RPY calibration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = THIS_DIR.parents[1]
RPS_DIR = THIS_DIR.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from rh56f2_hand import DEFAULT_CLOSED, RH56F2Hand, RH56F2HandConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=Path("fake_tcp_planar_pose_calibration.json"))
    parser.add_argument("--fixed-z-mm", type=float, required=True)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--serial", default="315122271151")
    parser.add_argument("--model", default=str(RPS_DIR / "yolo26n.pt"))
    parser.add_argument("--speed", type=int, default=3)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-baudrate", type=int, default=115200)
    parser.add_argument("--hand-speed", type=int, default=2500)
    parser.add_argument("--hand-force", type=int, default=1500)
    parser.add_argument("--hand-settle", type=float, default=1.0)
    parser.add_argument("--no-fist", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=960)
    return parser.parse_args()


def load_calibration(path: Path) -> tuple[dict[str, object], np.ndarray]:
    payload = json.loads(path.read_text())
    if payload.get("calibration_type") != "fixed_camera_2d_to_fake_tcp_xy":
        raise SystemExit(f"{path} is not a planar fake TCP calibration")
    if payload.get("target") != "xy_rpy":
        raise SystemExit(f"{path} does not contain fitted pose target. Re-solve with --target xy_rpy")
    return payload, np.asarray(payload["affine_nx3"], dtype=float)


def detect_once(args: argparse.Namespace) -> tuple[tuple[float, float, float], tuple[int, int]]:
    import cv2
    import pyrealsense2 as rs
    from ultralytics import YOLO

    sys.path.insert(0, str(RPS_DIR))
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
                    best = (confidence, tuple(float(value) for value in point), pixel)
        if best is None:
            raise SystemExit("No YOLO sports ball detected.")
        return best[1], best[2]
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
    calibration, affine = load_calibration(args.calibration)
    camera_xyz, pixel = detect_once(args)
    if calibration["source"] == "camera_xy":
        source = np.array([camera_xyz[0], camera_xyz[1], 1.0], dtype=float)
    else:
        source = np.array([pixel[0], pixel[1], 1.0], dtype=float)
    target_values = affine @ source
    target = (
        f"{target_values[0] * 1000.0:.3f},"
        f"{target_values[1] * 1000.0:.3f},"
        f"{args.fixed_z_mm:.3f},"
        f"{target_values[2]:.3f},"
        f"{target_values[3]:.3f},"
        f"{target_values[4]:.3f}"
    )
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
    print(f"pixel={pixel}")
    print(
        "fake_tcp_xy_rpy="
        f"({target_values[0]:.6f}, {target_values[1]:.6f}, "
        f"{target_values[2]:.3f}, {target_values[3]:.3f}, {target_values[4]:.3f})"
    )
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
