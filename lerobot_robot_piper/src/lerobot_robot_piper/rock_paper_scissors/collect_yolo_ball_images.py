#!/usr/bin/env python3
"""Capture raw D405 color images for the three-class YOLO ball dataset."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from rps_prize_controller import rotate_d405_ccw_90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="260322279862", help="D405 serial number")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-root", type=Path, default=Path("yolo_ball_dataset"))
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--auto-save", action="store_true", help="save one training image at the selected interval")
    parser.add_argument("--interval", type=float, default=1.0, help="auto-save interval in seconds")
    return parser.parse_args()


def next_index(directory: Path) -> int:
    indices = []
    for path in directory.glob("image_*.jpg"):
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max(indices, default=0) + 1


def draw_status(frame: np.ndarray, train_count: int, val_count: int, message: str) -> np.ndarray:
    output = frame.copy()
    cv2.putText(output, "s: save train   v: save val   q: quit", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(output, f"train={train_count}  val={val_count}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    if message:
        cv2.putText(output, message, (20, output.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return output


def main() -> int:
    args = parse_args()
    if not 1 <= args.jpg_quality <= 100:
        raise SystemExit("--jpg-quality must be between 1 and 100")
    if args.interval <= 0.0:
        raise SystemExit("--interval must be greater than zero")

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit("pyrealsense2 is not installed in this Python environment") from exc

    train_dir = args.output_root / "images" / "train"
    val_dir = args.output_root / "images" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    train_index = next_index(train_dir)
    val_index = next_index(val_dir)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    print("SAFETY: this program only captures D405 color images.")
    print("It does not connect to, enable, disable, or move Piper.")
    print(f"D405 serial: {args.serial}")
    print(f"Dataset root: {args.output_root}")
    print("Keys: s=save train, v=save validation, q=quit")

    pipeline.start(config)
    message = "Place balls in the real D405 view"
    last_save_time = 0.0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            rotated = rotate_d405_ccw_90(color)
            display = draw_status(rotated, train_index - 1, val_index - 1, message)
            cv2.imshow("D405 YOLO image collector", display)
            key = cv2.waitKey(1) & 0xFF
            now = time.monotonic()
            if key == ord("q"):
                break
            if args.auto_save and now - last_save_time >= args.interval:
                destination = train_dir / f"image_{train_index:06d}.jpg"
                train_index += 1
                last_save_time = now
                ok = cv2.imwrite(str(destination), rotated, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
                if not ok:
                    message = f"FAILED: {destination}"
                    print(message)
                else:
                    message = f"Auto-saved train: {destination.name}"
                    print(f"saved {destination}")
                continue
            if key not in (ord("s"), ord("v")):
                continue
            if now - last_save_time < 0.4:
                continue
            last_save_time = now
            if key == ord("s"):
                destination = train_dir / f"image_{train_index:06d}.jpg"
                train_index += 1
                split_name = "train"
            else:
                destination = val_dir / f"image_{val_index:06d}.jpg"
                val_index += 1
                split_name = "val"
            ok = cv2.imwrite(str(destination), rotated, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
            if not ok:
                message = f"FAILED: {destination}"
                print(message)
                continue
            message = f"Saved {split_name}: {destination.name}"
            print(f"saved {destination}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    print(f"Images saved under {args.output_root / 'images'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
