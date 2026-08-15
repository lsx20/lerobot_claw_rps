#!/usr/bin/env python3
"""D455 + MediaPipe rock-paper-scissors demo for RH56F2.

The human shows rock/paper/scissors to the D455. The RH56F2 answers with
the winning gesture:

    human rock     -> robot paper
    human paper    -> robot scissors
    human scissors -> robot rock

Press q to quit.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rh56f2_hand import DEFAULT_CLOSED, DEFAULT_OPEN, RH56F2Hand, RH56F2HandConfig


PAPER = dict(DEFAULT_OPEN)
ROCK = dict(DEFAULT_CLOSED)
SCISSORS = dict(DEFAULT_CLOSED)
SCISSORS.update(
    {
        "little": 900,
        "ring": 900,
        "middle": 1720,
        "index": 1720,
        "thumb_bend": 1130,
        "thumb_swing": 1700,
    }
)

POSES = {
    "Rock": ROCK,
    "Paper": PAPER,
    "Scissors": SCISSORS,
}

WINNING_REPLY = {
    "Rock": "Paper",
    "Paper": "Scissors",
    "Scissors": "Rock",
}

MEDIAPIPE_TASK_GESTURES = {
    "Closed_Fist": "Rock",
    "CLOSED_FIST": "Rock",
    "closed_fist": "Rock",
    "Open_Palm": "Paper",
    "OPEN_PALM": "Paper",
    "open_palm": "Paper",
    "Victory": "Scissors",
    "VICTORY": "Scissors",
    "victory": "Scissors",
}

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


class HandGestureRecognizer:
    def __init__(
        self,
        model_path: Path,
        min_detection: float,
        min_tracking: float,
        min_presence: float,
        min_score: float,
    ):
        if not model_path.exists():
            raise RuntimeError(
                f"Gesture model not found: {model_path}\n"
                "Download it with:\n"
                "  wget -O gesture_recognizer.task "
                "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
                "gesture_recognizer/float16/1/gesture_recognizer.task"
            )

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe Tasks API is not available. Try: python3 -m pip install --upgrade mediapipe"
            ) from exc

        options = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        )
        self.mp = mp
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.min_score = min_score
        self._last_timestamp_ms = 0
        self.last_category = "None"
        self.last_score = 0.0
        self.last_hand_count = 0
        self.last_fallback = "Unknown"

    def close(self) -> None:
        self.recognizer.close()

    @staticmethod
    def _draw_landmarks(frame_bgr: np.ndarray, landmarks: list[object]) -> None:
        h, w = frame_bgr.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame_bgr, points[start], points[end], (0, 220, 255), 2)
        for point in points:
            cv2.circle(frame_bgr, point, 4, (0, 80, 255), -1)

    @staticmethod
    def _distance_3d(p1: object, p2: object) -> float:
        return float(((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5)

    def _classify_landmarks(self, landmarks: list[object]) -> str:
        wrist = landmarks[0]
        fingers_straight: list[bool] = []
        for tip_idx, pip_idx in ((8, 6), (12, 10), (16, 14), (20, 18)):
            fingers_straight.append(
                self._distance_3d(landmarks[tip_idx], wrist)
                > self._distance_3d(landmarks[pip_idx], wrist)
            )

        up_count = fingers_straight.count(True)
        if up_count <= 1:
            return "Rock"
        if up_count == 2 and fingers_straight[0] and fingers_straight[1]:
            return "Scissors"
        if up_count >= 3:
            return "Paper"
        return "Unknown"

    def get_gesture(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, str]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        results = self.recognizer.recognize_for_video(mp_image, timestamp_ms)
        gesture = "Unknown"
        self.last_hand_count = len(results.hand_landmarks)
        self.last_category = "None"
        self.last_score = 0.0
        self.last_fallback = "Unknown"
        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            self._draw_landmarks(frame_bgr, landmarks)
            self.last_fallback = self._classify_landmarks(landmarks)
        if results.gestures and results.gestures[0]:
            category = results.gestures[0][0]
            self.last_category = category.category_name
            self.last_score = float(category.score)
            if category.score >= self.min_score:
                gesture = MEDIAPIPE_TASK_GESTURES.get(category.category_name, "Unknown")
        if gesture == "Unknown" and self.last_hand_count:
            gesture = self.last_fallback
        return frame_bgr, gesture


class D455ColorCamera:
    def __init__(self, serial: str, width: int, height: int, fps: int):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed. Install RealSense SDK, then run: "
                "python3 -m pip install pyrealsense2"
            ) from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial:
            self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.started = False

    def start(self) -> None:
        self.pipeline.start(self.config)
        self.started = True

    def read(self) -> np.ndarray | None:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def stop(self) -> None:
        if self.started:
            self.pipeline.stop()
            self.started = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="", help="Optional D455 serial number.")
    parser.add_argument(
        "--gesture-model",
        type=Path,
        default=Path("gesture_recognizer.task"),
        help="Path to the MediaPipe Gesture Recognizer .task model.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-baudrate", type=int, default=115200)
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--hand-force", type=int, default=1500)
    parser.add_argument(
        "--stage-delay",
        type=float,
        default=0.12,
        help="Seconds between staged finger/thumb commands inside one gesture.",
    )
    parser.add_argument("--stable-frames", type=int, default=5)
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.8,
        help="Minimum seconds to wait after a finished robot reply before accepting a new one.",
    )
    parser.add_argument(
        "--motion-settle",
        type=float,
        default=0.8,
        help="Extra seconds to keep the robot in Moving state after sending a hand pose.",
    )
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Do not connect the RH56F2.")
    parser.add_argument("--min-detection", type=float, default=0.7)
    parser.add_argument("--min-tracking", type=float, default=0.5)
    parser.add_argument("--min-presence", type=float, default=0.5)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument(
        "--print-vision",
        action="store_true",
        help="Print raw MediaPipe category, score, and hand count about once per second.",
    )
    return parser.parse_args()


def safe_set_pose(hand: RH56F2Hand, name: str, delay: float = 0.18) -> None:
    pose = POSES[name]
    if name == "Paper":
        hand.set_angles({"thumb_swing": PAPER["thumb_swing"], "thumb_bend": PAPER["thumb_bend"]})
        time.sleep(delay)
        hand.set_angles(PAPER)
    elif name == "Rock":
        hand.set_angles(
            {
                "little": ROCK["little"],
                "ring": ROCK["ring"],
                "middle": ROCK["middle"],
                "index": ROCK["index"],
            }
        )
        time.sleep(delay)
        hand.set_angles(ROCK)
    elif name == "Scissors":
        hand.set_angles(
            {
                "little": SCISSORS["little"],
                "ring": SCISSORS["ring"],
                "middle": SCISSORS["middle"],
                "index": SCISSORS["index"],
            }
        )
        time.sleep(delay)
        hand.set_angles(SCISSORS)
    else:
        hand.set_angles(pose)


def main() -> int:
    args = parse_args()

    hand: RH56F2Hand | None = None
    if not args.dry_run:
        hand = RH56F2Hand(
            RH56F2HandConfig(
                port=args.hand_port,
                baudrate=args.hand_baudrate,
                hand_id=args.hand_id,
                speed=args.hand_speed,
                force=args.hand_force,
            )
        )
        print(f"Connecting RH56F2 on {args.hand_port}, id={args.hand_id}...")
        hand.connect()
        safe_set_pose(hand, "Paper")
        print("RH56F2 ready.")
    else:
        print("Dry run: RH56F2 is disabled.")

    print("Starting D455 camera...")
    try:
        camera = D455ColorCamera(args.serial, args.width, args.height, args.fps)
        camera.start()
    except Exception:
        if hand is not None:
            hand.disconnect()
        raise

    try:
        recognizer = HandGestureRecognizer(
            args.gesture_model,
            args.min_detection,
            args.min_tracking,
            args.min_presence,
            args.min_score,
        )
    except Exception:
        camera.stop()
        if hand is not None:
            hand.disconnect()
        raise
    window = "RH56F2 RPS demo - press q"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    moving_lock = threading.Lock()
    moving = False
    last_raw = "Unknown"
    stable_count = 0
    last_triggered = "Unknown"
    last_trigger_t = 0.0
    last_vision_print_t = 0.0

    def execute_reply(robot_gesture: str) -> None:
        nonlocal moving, last_trigger_t
        with moving_lock:
            moving = True
        try:
            print(f"Robot: {robot_gesture}")
            if hand is not None:
                safe_set_pose(hand, robot_gesture, args.stage_delay)
            time.sleep(args.motion_settle)
        except Exception as exc:
            print(f"Motion failed: {exc}")
        finally:
            with moving_lock:
                last_trigger_t = time.time()
                moving = False

    print("Demo ready. Show rock/paper/scissors to the D455. Press q to quit.")
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            annotated, gesture = recognizer.get_gesture(frame)
            if args.print_vision and time.time() - last_vision_print_t >= 1.0:
                print(
                    "vision: "
                    f"raw={recognizer.last_category} "
                    f"score={recognizer.last_score:.2f} "
                    f"hands={recognizer.last_hand_count} "
                    f"fallback={recognizer.last_fallback} "
                    f"mapped={gesture}"
                )
                last_vision_print_t = time.time()
            if gesture == last_raw:
                stable_count += 1
            else:
                last_raw = gesture
                stable_count = 1

            with moving_lock:
                is_moving = moving

            now = time.time()
            stable = gesture in WINNING_REPLY and stable_count >= args.stable_frames
            cooled_down = now - last_trigger_t >= args.cooldown
            new_choice = gesture != last_triggered

            if stable and cooled_down and new_choice and not is_moving:
                robot_gesture = WINNING_REPLY[gesture]
                print(f"Human: {gesture} -> Robot wins with: {robot_gesture}")
                last_triggered = gesture
                last_trigger_t = now
                threading.Thread(target=execute_reply, args=(robot_gesture,), daemon=True).start()

            status = "Moving" if is_moving else "Ready"
            cv2.putText(
                annotated,
                f"Human: {gesture}  stable={stable_count}/{args.stable_frames}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                annotated,
                f"Status: {status}",
                (24, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if is_moving else (0, 180, 0),
                2,
            )
            cv2.putText(
                annotated,
                (
                    f"Raw: {recognizer.last_category} "
                    f"{recognizer.last_score:.2f} "
                    f"hands={recognizer.last_hand_count} "
                    f"fallback={recognizer.last_fallback}"
                ),
                (24, 122),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 180, 255),
                2,
            )
            cv2.imshow(window, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        camera.stop()
        recognizer.close()
        cv2.destroyAllWindows()
        if hand is not None:
            try:
                safe_set_pose(hand, "Paper")
                time.sleep(0.5)
            except Exception:
                pass
            hand.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
