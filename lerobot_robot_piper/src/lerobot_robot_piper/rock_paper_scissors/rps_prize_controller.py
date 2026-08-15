#!/usr/bin/env python3
"""RPS game plus D405 prize-ball detection and pick orchestration."""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

GESTURES = ("Rock", "Paper", "Scissors")
BEATS = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}


def rotate_d405_ccw_90(frame: np.ndarray) -> np.ndarray:
    return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate_d405_point_ccw_90(point: tuple[int, int], width: int) -> tuple[int, int]:
    x, y = point
    return (y, width - 1 - x)


def rotate_d405_observation_ccw_90(observation: BallObservation, width: int) -> BallObservation:
    return BallObservation(
        pixel=rotate_d405_point_ccw_90(observation.pixel, width),
        depth_m=observation.depth_m,
        camera_xyz_m=observation.camera_xyz_m,
        radius_px=observation.radius_px,
        color=observation.color,
    )


def rotate_d405_box_ccw_90(box: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    points = [
        rotate_d405_point_ccw_90((x0, y0), width),
        rotate_d405_point_ccw_90((x0, y1), width),
        rotate_d405_point_ccw_90((x1, y0), width),
        rotate_d405_point_ccw_90((x1, y1), width),
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


class GameState(str, Enum):
    WAIT_GESTURE = "WAIT_GESTURE"
    SHOW_RESULT = "SHOW_RESULT"
    WAIT_DRAW_BUTTON = "WAIT_DRAW_BUTTON"
    FIND_BALL = "FIND_BALL"
    PICK_BALL = "PICK_BALL"
    DONE = "DONE"


@dataclass(frozen=True)
class BallObservation:
    pixel: tuple[int, int]
    depth_m: float
    camera_xyz_m: tuple[float, float, float]
    radius_px: float
    color: str = "unknown"


class PickBackend(Protocol):
    def pick(self, ball: BallObservation) -> bool: ...


def choose_system_gesture(
    player: str,
    win_probability: float,
    tie_probability: float,
    rng: random.Random | None = None,
) -> str:
    if player not in GESTURES:
        raise ValueError(f"unknown player gesture: {player}")
    if not 0.0 <= win_probability <= 1.0:
        raise ValueError("win_probability must be between 0 and 1")
    if not 0.0 <= tie_probability < 1.0:
        raise ValueError("tie_probability must be between 0 and 1")
    if win_probability + tie_probability > 1.0:
        raise ValueError("win_probability + tie_probability must be <= 1")
    randomizer = rng or random
    roll = randomizer.random()
    if roll < tie_probability:
        return player
    if roll < tie_probability + win_probability:
        return next(gesture for gesture in GESTURES if BEATS[gesture] == player)
    return BEATS[player]


class UnoSerial:
    """Line-based UNO protocol. Commands and events are ASCII lines."""

    def __init__(self, port: str, baudrate: int = 115200, dry_run: bool = False):
        self.port = port
        self.baudrate = baudrate
        self.dry_run = dry_run
        self.serial = None

    def connect(self) -> None:
        if self.dry_run:
            return
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("Install pyserial to use UNO: python3 -m pip install pyserial") from exc
        self.serial = serial.Serial(self.port, self.baudrate, timeout=0.05)
        time.sleep(2.0)
        self.send("HOST_READY")

    def close(self) -> None:
        if self.serial is not None:
            self.serial.close()
            self.serial = None

    def send(self, command: str) -> None:
        print(f"UNO << {command}")
        if self.serial is not None:
            self.serial.write((command.strip() + "\n").encode("ascii"))

    def poll(self) -> list[str]:
        if self.serial is None:
            return []
        events: list[str] = []
        while self.serial.in_waiting:
            line = self.serial.readline().decode("ascii", errors="ignore").strip()
            if line:
                print(f"UNO >> {line}")
                events.append(line)
        return events


class D405BallCamera:
    def __init__(
        self,
        serial_number: str,
        width: int,
        height: int,
        fps: int,
        roi: tuple[int, int, int, int],
        min_radius_px: float = 20.0,
        ball_color: str = "auto",
        min_circularity: float = 0.35,
        max_aspect_ratio: float = 1.80,
        min_extent: float = 0.0,
        shape_profile: str = "legacy",
    ):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("Install pyrealsense2 before using D405") from exc
        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial_number:
            self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.align = rs.align(rs.stream.color)
        self.roi = roi
        self.min_radius_px = min_radius_px
        if ball_color not in {"auto", "orange", "green", "white"}:
            raise ValueError("ball_color must be auto, orange, green, or white")
        self.ball_color = ball_color
        self.min_circularity = min_circularity
        self.max_aspect_ratio = max_aspect_ratio
        self.min_extent = min_extent
        if shape_profile not in {"original", "legacy", "strict"}:
            raise ValueError("shape_profile must be original, legacy, or strict")
        self.shape_profile = shape_profile
        self.depth_scale = 0.001
        self.profile = None

    def start(self) -> None:
        self.profile = self.pipeline.start(self.config)
        sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(sensor.get_depth_scale())

    def stop(self) -> None:
        self.pipeline.stop()

    def read(self) -> tuple[np.ndarray, np.ndarray, object] | None:
        frames = self.align.process(self.pipeline.wait_for_frames())
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            return None
        return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data()), depth

    def detect_balls(self, color: np.ndarray, depth_image: np.ndarray, depth_frame: object) -> list[BallObservation]:
        x0, y0, x1, y1 = self.roi
        roi = color[y0:y1, x0:x1]
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if self.shape_profile == "original":
            masks = {"ball": cv2.inRange(hsv, np.array([0, 85, 45]), np.array([90, 255, 255]))}
        else:
            masks = {
                "orange": cv2.inRange(hsv, np.array([0, 45, 30]), np.array([38, 255, 255])),
                "green": cv2.inRange(hsv, np.array([20, 45, 30]), np.array([110, 255, 255])),
                "white": cv2.inRange(hsv, np.array([0, 0, 90]), np.array([179, 110, 255])),
            }
        if self.ball_color != "auto":
            masks = {self.ball_color: masks[self.ball_color]}
        candidates: list[tuple[float, BallObservation]] = []
        max_radius = min(320.0, min(roi.shape[:2]) * 0.70)
        for color_name, raw_mask in masks.items():
            mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 250.0:
                    continue
                perimeter = cv2.arcLength(contour, True)
                if perimeter <= 0.0:
                    continue
                circularity = 4.0 * np.pi * area / (perimeter * perimeter)
                if circularity < self.min_circularity:
                    continue
                _, _, width, height = cv2.boundingRect(contour)
                if width == 0 or height == 0 or max(width, height) / min(width, height) > self.max_aspect_ratio:
                    continue
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                if radius < self.min_radius_px or radius > max_radius:
                    continue
                extent = area / float(width * height)
                if self.shape_profile == "strict":
                    hull_area = cv2.contourArea(cv2.convexHull(contour))
                    if hull_area <= 0.0 or area / hull_area < 0.60:
                        continue
                    if extent < self.min_extent:
                        continue
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        ellipse_width, ellipse_height = ellipse[1]
                        if min(ellipse_width, ellipse_height) <= 0.0:
                            continue
                        ellipse_aspect = max(ellipse_width, ellipse_height) / min(ellipse_width, ellipse_height)
                        if ellipse_aspect > self.max_aspect_ratio:
                            continue
                px, py = int(x0 + cx), int(y0 + cy)
                contour_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                contour_hsv = hsv[contour_mask > 0]
                if contour_hsv.size == 0:
                    continue
                median_hue, median_saturation, median_value = np.median(contour_hsv, axis=0)
                if color_name == "white" and median_saturation > 105:
                    continue
                if color_name in {"orange", "green"} and median_saturation < 48:
                    continue
                contour_depth = depth_image[y0:y1, x0:x1][contour_mask > 0].astype(np.float32) * self.depth_scale
                valid_depths = contour_depth[contour_depth > 0.05]
                if valid_depths.size == 0:
                    depth_values = [
                        float(depth_frame.get_distance(px + dx, py + dy))
                        for dx in range(-6, 7)
                        for dy in range(-6, 7)
                        if 0 <= px + dx < depth_image.shape[1] and 0 <= py + dy < depth_image.shape[0]
                    ]
                    valid_depths = np.asarray([value for value in depth_values if value > 0.05], dtype=np.float32)
                depth_m = float(np.median(valid_depths)) if valid_depths.size else 0.0
                if depth_m <= 0.05:
                    continue
                point = self.rs.rs2_deproject_pixel_to_point(depth_frame.profile.as_video_stream_profile().intrinsics, [px, py], depth_m)
                observation = BallObservation((px, py), depth_m, tuple(float(v) for v in point), float(radius), color_name)
                score = area * circularity * extent * min(1.0, area / 1200.0)
                candidates.append((float(score), observation))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in candidates]

    def detect_ball(self, color: np.ndarray, depth_image: np.ndarray, depth_frame: object) -> BallObservation | None:
        candidates = self.detect_balls(color, depth_image, depth_frame)
        return candidates[0] if candidates else None


class D435iGestureCamera:
    """Fixed third-person color camera used only for gesture recognition."""

    def __init__(self, serial_number: str, width: int, height: int, fps: int):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("Install pyrealsense2 before using D435i") from exc
        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial_number:
            self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    def start(self) -> None:
        self.pipeline.start(self.config)

    def read(self) -> np.ndarray | None:
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())

    def stop(self) -> None:
        self.pipeline.stop()


class DryRunPicker:
    def pick(self, ball: BallObservation) -> bool:
        print(f"DRY RUN pick ball pixel={ball.pixel} xyz={ball.camera_xyz_m}")
        return True


class PiperPickBackend:
    """Adapter for the existing direct Piper/RH56F2 claw cycle."""

    def __init__(self, piper: object, hand: object | None, args: argparse.Namespace, start_pose: list[int], drop_pose: list[int], transform: tuple[float, float, float, float, float, float]):
        self.piper = piper
        self.hand = hand
        self.args = args
        self.start_pose = list(start_pose)
        self.drop_pose = list(drop_pose)
        self.transform = transform

    def pick(self, ball: BallObservation) -> bool:
        from claw_arm_grasp import run_pick_cycle
        x_offset, y_offset, z_offset, x_scale, y_scale, z_scale = self.transform
        x, y, z = ball.camera_xyz_m
        hover = list(self.start_pose)
        hover[0] = int(round((x_offset + x * x_scale) * 1000.0))
        hover[1] = int(round((y_offset + y * y_scale) * 1000.0))
        hover[2] = int(round((z_offset + z * z_scale) * 1000.0))
        return bool(run_pick_cycle(self.piper, self.hand, self.args, self.start_pose, hover, self.drop_pose))


class RPSPrizeController:
    def __init__(self, camera: D405BallCamera, uno: UnoSerial, picker: PickBackend, args: argparse.Namespace):
        self.camera = camera
        self.uno = uno
        self.picker = picker
        self.args = args
        self.rng = random.Random(args.seed)
        self.state = GameState.WAIT_GESTURE
        self.last_player: str | None = None
        self.last_system: str | None = None

    @staticmethod
    def outcome(player: str, system: str) -> str:
        if player == system:
            return "TIE"
        return "SYSTEM_WIN" if BEATS[system] == player else "PLAYER_WIN"

    def run_round(self, player: str) -> str:
        system = choose_system_gesture(player, self.args.system_win_probability, self.args.tie_probability, self.rng)
        result = self.outcome(player, system)
        self.last_player, self.last_system = player, system
        self.uno.send(f"ROUND {player.upper()} {system.upper()} {result}")
        print(f"player={player} system={system} result={result}")
        if result != "PLAYER_WIN":
            self.uno.send("LOCK")
            return result
        self.uno.send("UNLOCK")
        self.state = GameState.WAIT_DRAW_BUTTON
        return result

    def wait_for_button(self) -> bool:
        deadline = time.monotonic() + self.args.button_timeout
        while time.monotonic() < deadline:
            if "BUTTON" in self.uno.poll():
                return True
            time.sleep(0.02)
        return False

    def find_ball(self) -> BallObservation | None:
        deadline = time.monotonic() + self.args.ball_timeout
        detections: list[BallObservation] = []
        while time.monotonic() < deadline:
            packet = self.camera.read()
            if packet is None:
                continue
            color, depth, depth_frame = packet
            ball = self.camera.detect_ball(color, depth, depth_frame)
            if ball is not None:
                detections.append(ball)
                if len(detections) >= self.args.ball_stable_frames:
                    xyz = np.median([item.camera_xyz_m for item in detections[-self.args.ball_stable_frames:]], axis=0)
                    return BallObservation(detections[-1].pixel, detections[-1].depth_m, tuple(float(v) for v in xyz), detections[-1].radius_px)
        return None

    def run_draw(self) -> bool:
        self.state = GameState.FIND_BALL
        self.uno.send("DRAW")
        ball = self.find_ball()
        if ball is None:
            self.uno.send("ERROR BALL_NOT_FOUND")
            return False
        self.state = GameState.PICK_BALL
        self.uno.send(f"BALL {ball.pixel[0]} {ball.pixel[1]} {ball.depth_m:.3f}")
        ok = self.picker.pick(ball)
        self.uno.send("PRIZE_OK" if ok else "ERROR PICK_FAILED")
        self.state = GameState.DONE
        return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gesture-serial", default="", help="D435i serial number for third-person gesture view")
    parser.add_argument("--ball-serial", default="", help="D405 serial number for hand-mounted first-person ball view")
    parser.add_argument("--uno-port", default="/dev/ttyACM0")
    parser.add_argument("--uno-baudrate", type=int, default=115200)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--roi", default="0,0,640,480", help="Ball ROI x0,y0,x1,y1")
    parser.add_argument("--system-win-probability", type=float, default=0.65)
    parser.add_argument("--tie-probability", type=float, default=0.05)
    parser.add_argument("--button-timeout", type=float, default=30.0)
    parser.add_argument("--ball-timeout", type=float, default=15.0)
    parser.add_argument("--ball-stable-frames", type=int, default=5)
    parser.add_argument("--ball-color", choices=("auto", "orange", "green", "white"), default="auto")
    parser.add_argument("--gesture-model", type=Path, default=Path("gesture_recognizer.task"))
    parser.add_argument("--stable-frames", type=int, default=5)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roi = tuple(int(value) for value in args.roi.split(","))
    if len(roi) != 4:
        raise SystemExit("--roi must be x0,y0,x1,y1")
    gesture_camera = D435iGestureCamera(args.gesture_serial, args.width, args.height, args.fps)
    ball_camera = D405BallCamera(args.ball_serial, args.width, args.height, args.fps, roi, ball_color=args.ball_color)
    uno = UnoSerial(args.uno_port, args.uno_baudrate, args.dry_run)
    picker = DryRunPicker()
    controller = RPSPrizeController(ball_camera, uno, picker, args)
    from rh56f2_rps_demo import HandGestureRecognizer

    recognizer = HandGestureRecognizer(args.gesture_model, 0.7, 0.5, 0.5, 0.5)
    uno.connect()
    gesture_camera.start()
    ball_camera.start()
    print("RPS prize controller ready. Show Rock, Paper, or Scissors to the fixed D435i.")
    stable_gesture = None
    stable_count = 0
    try:
        while True:
            gesture_frame = gesture_camera.read()
            if gesture_frame is None:
                continue
            annotated, gesture = recognizer.get_gesture(gesture_frame)
            if gesture in GESTURES:
                if gesture == stable_gesture:
                    stable_count += 1
                else:
                    stable_gesture, stable_count = gesture, 1
            else:
                stable_gesture, stable_count = None, 0
            if not args.no_window:
                cv2.putText(annotated, f"gesture: {gesture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("RPS prize controller - q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if stable_gesture in GESTURES and stable_count >= args.stable_frames:
                result = controller.run_round(stable_gesture)
                stable_gesture, stable_count = None, 0
                if result == "PLAYER_WIN":
                    print("Player won. Press the unlocked UNO button to draw a prize ball.")
                    if controller.wait_for_button():
                        controller.run_draw()
                    else:
                        print("Draw button timeout.")
                time.sleep(1.0)
    except KeyboardInterrupt:
        return 0
    finally:
        recognizer.close()
        ball_camera.stop()
        gesture_camera.stop()
        uno.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
