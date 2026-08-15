#!/usr/bin/env python3
"""Observe D405 prize-ball detections without commanding the robot."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from rps_prize_controller import (
    BallObservation,
    D405BallCamera,
    rotate_d405_ccw_90,
    rotate_d405_observation_ccw_90,
)


def parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1") from exc
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1 with x1>x0 and y1>y0")
    return values


def draw_observation(frame: np.ndarray, observation: BallObservation | None, stable_xyz: tuple[float, float, float] | None) -> np.ndarray:
    output = frame.copy()
    if observation is None:
        cv2.putText(output, "ball: NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return output
    x, y = observation.pixel
    cv2.circle(output, (x, y), max(5, int(observation.radius_px)), (0, 255, 0), 2)
    cv2.drawMarker(output, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    cx, cy, cz = observation.camera_xyz_m
    cv2.putText(output, f"BALL pixel=({x},{y}) depth={observation.depth_m:.3f} m", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(output, f"camera xyz=({cx:.3f}, {cy:.3f}, {cz:.3f}) m", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    if stable_xyz is not None:
        cv2.putText(output, f"stable xyz=({stable_xyz[0]:.3f}, {stable_xyz[1]:.3f}, {stable_xyz[2]:.3f}) m", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
    return output


def draw_candidates(frame: np.ndarray, candidates: list[BallObservation]) -> np.ndarray:
    output = frame.copy()
    colors = {"white": (255, 255, 255), "orange": (0, 140, 255), "green": (0, 255, 0)}
    for candidate in candidates:
        x, y = candidate.pixel
        draw_color = colors.get(candidate.color, (255, 0, 255))
        cv2.circle(output, (x, y), max(5, int(candidate.radius_px)), draw_color, 2)
        cv2.putText(output, candidate.color.upper(), (x - 35, y - int(candidate.radius_px) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2)
    return output


def median_observation(observations: list[BallObservation]) -> BallObservation | None:
    if not observations:
        return None
    pixels = np.median([item.pixel for item in observations], axis=0)
    xyz = np.median([item.camera_xyz_m for item in observations], axis=0)
    return BallObservation(
        pixel=(int(round(pixels[0])), int(round(pixels[1]))),
        depth_m=float(np.median([item.depth_m for item in observations])),
        camera_xyz_m=tuple(float(value) for value in xyz),
        radius_px=float(np.median([item.radius_px for item in observations])),
    )


def write_sample(writer: csv.writer, observation: BallObservation, stable_xyz: tuple[float, float, float] | None) -> None:
    xyz = stable_xyz or observation.camera_xyz_m
    writer.writerow([
        f"{time.time():.6f}",
        observation.pixel[0],
        observation.pixel[1],
        f"{observation.depth_m:.6f}",
        f"{xyz[0]:.6f}",
        f"{xyz[1]:.6f}",
        f"{xyz[2]:.6f}",
        f"{observation.radius_px:.2f}",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="260322279862", help="D405 serial number")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--roi", type=parse_roi, default=(0, 0, 640, 480))
    parser.add_argument("--stable-frames", type=int, default=10)
    parser.add_argument("--max-pixel-jump", type=float, default=35.0)
    parser.add_argument("--max-depth-jump", type=float, default=0.03)
    parser.add_argument("--max-radius-jump", type=float, default=25.0)
    parser.add_argument("--min-radius", type=float, default=20.0)
    parser.add_argument("--ball-color", choices=("auto", "orange", "green", "white"), default="auto")
    parser.add_argument("--min-circularity", type=float, default=0.40)
    parser.add_argument("--max-aspect-ratio", type=float, default=1.55)
    parser.add_argument("--min-extent", type=float, default=0.50)
    parser.add_argument("--output", type=Path, default=Path("d405_ball_samples.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    camera = D405BallCamera(
        args.serial,
        args.width,
        args.height,
        args.fps,
        args.roi,
        args.min_radius,
        args.ball_color,
        args.min_circularity,
        args.max_aspect_ratio,
        args.min_extent,
    )
    observations: list[BallObservation] = []
    with args.output.open("w", newline="") as sample_file:
        writer = csv.writer(sample_file)
        writer.writerow(["unix_time", "pixel_x", "pixel_y", "depth_m", "stable_x_m", "stable_y_m", "stable_z_m", "radius_px"])
        camera.start()
        print(f"D405 started: serial={args.serial}, roi={args.roi}")
        print("Show one prize ball. Press s to save a sample, q to quit.")
        try:
            while True:
                packet = camera.read()
                if packet is None:
                    continue
                color, depth, depth_frame = packet
                candidates = camera.detect_balls(color, depth, depth_frame)
                observation = candidates[0] if candidates else None
                if observation is not None:
                    if observations:
                        previous = observations[-1]
                        pixel_jump = float(np.hypot(
                            observation.pixel[0] - previous.pixel[0],
                            observation.pixel[1] - previous.pixel[1],
                        ))
                        depth_jump = abs(observation.depth_m - previous.depth_m)
                        radius_jump = abs(observation.radius_px - previous.radius_px)
                        if pixel_jump > args.max_pixel_jump or depth_jump > args.max_depth_jump or radius_jump > args.max_radius_jump:
                            observations.clear()
                    observations.append(observation)
                    observations = observations[-args.stable_frames :]
                else:
                    observations.clear()
                stable_xyz = None
                if len(observations) >= args.stable_frames:
                    stable_xyz = tuple(float(value) for value in np.median([item.camera_xyz_m for item in observations], axis=0))
                display = rotate_d405_ccw_90(color)
                rotated_candidates = [rotate_d405_observation_ccw_90(candidate, color.shape[1]) for candidate in candidates]
                rotated_observation = (
                    rotate_d405_observation_ccw_90(observation, color.shape[1])
                    if observation is not None
                    else None
                )
                display = draw_candidates(display, rotated_candidates)
                display = draw_observation(display, rotated_observation, stable_xyz)
                cv2.putText(display, "s: save sample   q: quit", (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imshow("D405 ball observation", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    if observation is None:
                        print("No ball detected; sample not saved.")
                    elif stable_xyz is None:
                        print(f"Detection is not stable yet ({len(observations)}/{args.stable_frames}); sample not saved.")
                    else:
                        write_sample(writer, observation, stable_xyz)
                        sample_file.flush()
                        print(f"Saved: pixel={observation.pixel}, camera_xyz={stable_xyz or observation.camera_xyz_m}")
                elif key == ord("q"):
                    break
        finally:
            camera.stop()
            cv2.destroyAllWindows()
    print(f"Samples written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
