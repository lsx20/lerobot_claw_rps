#!/usr/bin/env python3
"""Collect paired D405 ball and Piper end-pose samples without motion commands."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from rps_prize_controller import BallObservation, D405BallCamera, rotate_d405_ccw_90, rotate_d405_observation_ccw_90

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_roi(value: str) -> tuple[int, int, int, int]:
    values = tuple(int(item.strip()) for item in value.split(","))
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1")
    return values


def read_piper_pose(piper: object) -> tuple[float, float, float, float, float, float]:
    end_pose = piper.GetArmEndPoseMsgs().end_pose
    position_m = [
        float(value) / 1_000_000.0
        for value in (end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis)
    ]
    rotation_deg = [
        float(value) / 1000.0
        for value in (end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis)
    ]
    return tuple(position_m + rotation_deg)


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


def draw_frame(frame: np.ndarray, observation: BallObservation | None, sample_count: int) -> np.ndarray:
    output = frame.copy()
    if observation is None:
        text = "ball: NOT FOUND"
        color = (0, 0, 255)
    else:
        x, y = observation.pixel
        cv2.circle(output, observation.pixel, max(5, int(observation.radius_px)), (0, 255, 0), 2)
        cv2.drawMarker(output, observation.pixel, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        text = f"xyz=({observation.camera_xyz_m[0]:.3f},{observation.camera_xyz_m[1]:.3f},{observation.camera_xyz_m[2]:.3f})m"
        color = (0, 255, 0)
    cv2.putText(output, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(output, f"samples={sample_count}  s: save  q: quit", (20, output.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="260322279862")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--roi", type=parse_roi, default=(0, 0, 640, 480))
    parser.add_argument("--min-radius", type=float, default=45.0)
    parser.add_argument("--ball-color", choices=("auto", "orange", "green", "white"), default="auto")
    parser.add_argument("--stable-frames", type=int, default=10)
    parser.add_argument("--max-pixel-jump", type=float, default=40.0)
    parser.add_argument("--max-depth-jump", type=float, default=0.04)
    parser.add_argument("--max-radius-jump", type=float, default=25.0)
    parser.add_argument("--output", type=Path, default=Path("eye_hand_samples.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError as exc:
        raise SystemExit("piper_sdk is not installed in this Python environment") from exc

    print("SAFETY: this program does not enable, disable, or move Piper.")
    print("Keep the arm mechanically supported; a disabled arm may drop.")
    print("Move the arm only with your approved safe teach/manual method.")
    answer = input("Type READONLY to connect and collect read-only samples: ").strip()
    if answer != "READONLY":
        print("Cancelled.")
        return 0

    piper = C_PiperInterface_V2(
        args.can,
        judge_flag=False,
        can_auto_init=False,
        dh_is_offset=1,
        start_sdk_fk_cal=True,
    )
    piper.ConnectPort()
    time.sleep(1.0)
    camera = D405BallCamera(args.serial, args.width, args.height, args.fps, args.roi, args.min_radius, args.ball_color)
    camera.start()
    observations: list[BallObservation] = []
    output_exists = args.output.exists() and args.output.stat().st_size > 0
    with args.output.open("a", newline="") as sample_file:
        writer = csv.writer(sample_file)
        if not output_exists:
            writer.writerow([
                "unix_time",
                "camera_x_m", "camera_y_m", "camera_z_m",
                "base_x_m", "base_y_m", "base_z_m",
                "base_rx_deg", "base_ry_deg", "base_rz_deg",
                "pixel_x", "pixel_y", "depth_m", "radius_px",
            ])
        print(f"D405={args.serial}, Piper CAN={args.can}")
        print("Fix one ball in the workspace. Move to a safe pose, wait for stable xyz, press s.")
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
                        pixel_jump = float(np.hypot(observation.pixel[0] - previous.pixel[0], observation.pixel[1] - previous.pixel[1]))
                        depth_jump = abs(observation.depth_m - previous.depth_m)
                        radius_jump = abs(observation.radius_px - previous.radius_px)
                        if pixel_jump > args.max_pixel_jump or depth_jump > args.max_depth_jump or radius_jump > args.max_radius_jump:
                            observations.clear()
                    observations = (observations + [observation])[-args.stable_frames :]
                else:
                    observations.clear()
                stable = median_observation(observations) if len(observations) >= args.stable_frames else None
                display = rotate_d405_ccw_90(color)
                displayed_observation = (
                    rotate_d405_observation_ccw_90(stable or observation, color.shape[1])
                    if (stable or observation) is not None
                    else None
                )
                display = draw_frame(display, displayed_observation, 0)
                cv2.imshow("Eye-hand sample collection", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    if stable is None:
                        print(f"Not stable: {len(observations)}/{args.stable_frames} frames")
                        continue
                    pose = read_piper_pose(piper)
                    writer.writerow([
                        f"{time.time():.6f}",
                        *[f"{value:.6f}" for value in stable.camera_xyz_m],
                        *[f"{value:.6f}" for value in pose],
                        stable.pixel[0], stable.pixel[1], f"{stable.depth_m:.6f}", f"{stable.radius_px:.2f}",
                    ])
                    sample_file.flush()
                    print(f"saved #{sum(1 for _ in args.output.open()) - 1}: camera={stable.camera_xyz_m}, piper_pose={pose}")
                elif key == ord("q"):
                    break
        finally:
            camera.stop()
            cv2.destroyAllWindows()
            try:
                piper.DisconnectPort()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
