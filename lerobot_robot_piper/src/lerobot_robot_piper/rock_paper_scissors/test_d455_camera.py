#!/usr/bin/env python3
"""Minimal Intel RealSense D455 color-stream smoke test.

Run this before connecting any robot motion code:

    python3 test_d455_camera.py

Press q to quit.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="", help="Optional RealSense serial number.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the preview.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 is not installed.")
        print("Install RealSense SDK first, then run: python3 -m pip install pyrealsense2")
        return 1

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    try:
        pipeline.start(config)
    except RuntimeError as exc:
        print(f"Failed to start D455 color stream: {exc}")
        print("Check USB3 cable, camera power, permissions, and realsense-viewer.")
        return 1

    window = "D455 color test - press q"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    last_t = time.time()
    frames = 0
    shown_fps = 0.0

    print("D455 color stream started. Press q in the preview window to quit.")
    try:
        while True:
            rs_frames = pipeline.wait_for_frames()
            color_frame = rs_frames.get_color_frame()
            if not color_frame:
                continue

            # The stream is requested as bgr8, so OpenCV can display it directly.
            frame = np.asanyarray(color_frame.get_data())
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            frames += 1
            now = time.time()
            if now - last_t >= 1.0:
                shown_fps = frames / (now - last_t)
                frames = 0
                last_t = now

            cv2.putText(
                frame,
                f"D455 {args.width}x{args.height}@{args.fps} FPS {shown_fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
