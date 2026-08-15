#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import select
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from lerobot_robot_piper import PiperRH56F2Follower, PiperRH56F2FollowerConfig
from lerobot_robot_piper.rh56f2_hand import DEFAULT_CLOSED, DEFAULT_OPEN, HAND_NAMES


ARM_KEYS = [f"joint_{i}.pos" for i in range(1, 7)]
DEFAULT_WAYPOINTS = Path.home() / "piper_ball_waypoints.json"
DEFAULT_HAND_LANDMARKER_MODEL = (
    Path(__file__).resolve().parents[1] / "assets" / "hand_landmarker.task"
)

WEB_STATE = {
    "jpeg": None,
    "status": {},
}
WEB_STATE_LOCK = threading.Lock()


@dataclass
class GestureResult:
    label: str
    score: float
    open_fingers: int
    pinch_ratio: float
    wrist_x: float
    wrist_y: float


class MediaPipeHandGestureDetector:
    """MediaPipe Tasks hand landmark wrapper for open/grasp trigger detection."""

    def __init__(
        self,
        model_path: Path,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "mediapipe is not installed. Run this script in keyboard mode, or install mediapipe."
            ) from exc

        if not model_path.exists():
            raise FileNotFoundError(f"Hand landmarker model not found: {model_path}")

        self.mp = mp
        self.vision = vision
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    @staticmethod
    def _dist(a, b) -> float:
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5

    def _draw_landmarks(self, frame, landmarks) -> None:
        h, w = frame.shape[:2]
        for connection in self.connections:
            start = landmarks[connection.start]
            end = landmarks[connection.end]
            p1 = (int(start.x * w), int(start.y * h))
            p2 = (int(end.x * w), int(end.y * h))
            cv2.line(frame, p1, p2, (0, 220, 255), 2)
        for point in landmarks:
            center = (int(point.x * w), int(point.y * h))
            cv2.circle(frame, center, 3, (0, 255, 0), -1)

    def process(self, frame) -> tuple[GestureResult | None, object]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None, frame

        lm = result.hand_landmarks[0]
        self._draw_landmarks(frame, lm)

        # Finger extension test using wrist-to-tip vs wrist-to-pip distance.
        fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]
        open_fingers = 0
        wrist = lm[0]
        for tip, pip in fingers:
            if self._dist(wrist, lm[tip]) > self._dist(wrist, lm[pip]) * 1.08:
                open_fingers += 1

        palm_width = max(self._dist(lm[5], lm[17]), 1e-6)
        pinch_ratio = self._dist(lm[4], lm[8]) / palm_width

        if open_fingers >= 3 and pinch_ratio > 0.35:
            label = "OPEN"
            score = min(1.0, 0.5 + 0.12 * open_fingers + 0.2 * min(pinch_ratio, 1.0))
        elif open_fingers <= 1 or pinch_ratio < 0.18:
            label = "GRASP"
            score = min(1.0, 0.5 + 0.2 * (4 - open_fingers) + 0.2 * max(0.0, 0.18 - pinch_ratio))
        else:
            label = "NEUTRAL"
            score = 0.5

        palm_points = [lm[i] for i in (5, 9, 13, 17)]
        palm_x = sum(point.x for point in palm_points) / len(palm_points)
        palm_y = sum(point.y for point in palm_points) / len(palm_points)
        return GestureResult(label, score, open_fingers, pinch_ratio, palm_x, palm_y), frame

    def close(self) -> None:
        self.landmarker.close()


def load_waypoints(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Waypoints file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def lerp(a: float, b: float, t: float) -> float:
    t = t * t * (3.0 - 2.0 * t)
    return a + (b - a) * t


def move_arm(
    robot: PiperRH56F2Follower,
    target: dict[str, float],
    seconds: float,
    hz: float,
    locked_joint6: float | None,
    name: str,
) -> None:
    obs = robot.get_observation()
    start = {key: float(obs[key]) for key in ARM_KEYS}
    planned = {key: float(target[key]) for key in ARM_KEYS}
    if locked_joint6 is not None:
        planned["joint_6.pos"] = locked_joint6
    max_delta = max(abs(planned[key] - start[key]) for key in ARM_KEYS)
    print(f"Arm move {name}: max_delta={max_delta:.3f} deg")
    print(f"  start : {start}")
    print(f"  target: {planned}")
    steps = max(1, int(seconds * hz))
    for i in range(steps):
        t = (i + 1) / steps
        action = {key: lerp(start[key], float(target[key]), t) for key in ARM_KEYS}
        if locked_joint6 is not None:
            action["joint_6.pos"] = locked_joint6
        robot.send_action(action)
        time.sleep(1.0 / hz)
    end = {key: float(robot.get_observation()[key]) for key in ARM_KEYS}
    print(f"  end   : {end}")


def set_hand(robot: PiperRH56F2Follower, pose: dict[str, float], repeats: int, dt: float) -> None:
    action = {f"hand.{name}.pos": float(pose[name]) for name in HAND_NAMES}
    for _ in range(repeats):
        robot.send_action(action)
        time.sleep(dt)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def hand_pose_from_gesture(gesture: GestureResult) -> dict[str, float]:
    finger_open = clamp(gesture.open_fingers / 4.0, 0.0, 1.0)
    pinch_open = clamp((gesture.pinch_ratio - 0.15) / 0.35, 0.0, 1.0)
    openness = clamp((finger_open + pinch_open) / 2.0, 0.0, 1.0)
    return {
        name: float(DEFAULT_CLOSED[name] + (DEFAULT_OPEN[name] - DEFAULT_CLOSED[name]) * openness)
        for name in HAND_NAMES
    }


def pregrasp_hand_pose_from_gesture(gesture: GestureResult, min_open: float) -> dict[str, float]:
    """Map hand openness, but never allow full closure during follow."""
    raw = hand_pose_from_gesture(gesture)
    min_open = clamp(min_open, 0.0, 1.0)
    protected = {}
    for name in HAND_NAMES:
        protected_min = DEFAULT_CLOSED[name] + (DEFAULT_OPEN[name] - DEFAULT_CLOSED[name]) * min_open
        if DEFAULT_OPEN[name] >= DEFAULT_CLOSED[name]:
            protected[name] = max(raw[name], protected_min)
        else:
            protected[name] = min(raw[name], protected_min)
    return protected


def connect_robot(args) -> PiperRH56F2Follower:
    cfg = PiperRH56F2FollowerConfig(
        id="piper_rh56f2_vision_gesture",
        can_port=args.can_port,
        speed_rate=args.speed_rate,
        max_arm_delta_deg=args.max_arm_delta_deg,
        prompt_before_disable=True,
        clip_joint6_to_sdk_limits=False,
        hand_port=args.hand_port,
        hand_baudrate=args.hand_baudrate,
        hand_id=args.hand_id,
        hand_speed=args.hand_speed,
        hand_force=args.hand_force,
        max_hand_delta=args.max_hand_delta,
    )
    robot = PiperRH56F2Follower(cfg)
    robot.connect()
    return robot


def execute_pick_place(robot: PiperRH56F2Follower, waypoints: dict, args) -> None:
    required = ["home", "pre_grasp", "grasp", "pre_place", "place"]
    missing = [name for name in required if name not in waypoints]
    if missing:
        raise ValueError(f"Missing waypoints: {missing}")

    locked_joint6 = None
    if args.lock_joint6:
        locked_joint6 = float(robot.get_observation()["joint_6.pos"])
        print(f"Locking joint_6 at startup value: {locked_joint6:.3f} deg")
        for pose in waypoints.values():
            pose["joint_6.pos"] = locked_joint6

    print("LeRobot action: open hand")
    set_hand(robot, DEFAULT_OPEN, repeats=20, dt=0.02)

    print("LeRobot action: move to ball")
    move_arm(robot, waypoints["home"], args.move_seconds, args.control_hz, locked_joint6, "home")
    move_arm(robot, waypoints["pre_grasp"], args.move_seconds, args.control_hz, locked_joint6, "pre_grasp")
    move_arm(robot, waypoints["grasp"], args.approach_seconds, args.control_hz, locked_joint6, "grasp")

    print("LeRobot action: grasp")
    set_hand(robot, DEFAULT_CLOSED, repeats=60, dt=0.02)

    print("LeRobot action: lift and place")
    move_arm(robot, waypoints["pre_grasp"], args.approach_seconds, args.control_hz, locked_joint6, "pre_grasp")
    move_arm(robot, waypoints["pre_place"], args.move_seconds, args.control_hz, locked_joint6, "pre_place")
    move_arm(robot, waypoints["place"], args.approach_seconds, args.control_hz, locked_joint6, "place")

    print("LeRobot action: release")
    set_hand(robot, DEFAULT_OPEN, repeats=60, dt=0.02)

    print("LeRobot action: retreat")
    move_arm(robot, waypoints["pre_place"], args.approach_seconds, args.control_hz, locked_joint6, "pre_place")
    move_arm(robot, waypoints["home"], args.move_seconds, args.control_hz, locked_joint6, "home")


def execute_grasp_place_from_follow(
    robot: PiperRH56F2Follower,
    waypoints: dict,
    args,
    locked_joint6: float | None,
) -> None:
    required = ["grasp", "pre_grasp", "pre_place", "place", "home"]
    missing = [name for name in required if name not in waypoints]
    if missing:
        raise ValueError(f"Missing waypoints: {missing}")

    if locked_joint6 is not None:
        for pose in waypoints.values():
            pose["joint_6.pos"] = locked_joint6

    hover_pose = {key: float(robot.get_observation()[key]) for key in ARM_KEYS}
    if locked_joint6 is not None:
        hover_pose["joint_6.pos"] = locked_joint6

    descent_target = dict(hover_pose)
    if args.relative_grasp_from_follow:
        # Reuse the taught pre_grasp -> grasp descent shape, but apply it from
        # the current follow pose. Keep joint_1 fixed so the arm descends from
        # the tracked horizontal position instead of returning to the old grasp yaw.
        for key in ["joint_2.pos", "joint_3.pos", "joint_4.pos", "joint_5.pos"]:
            descent_target[key] = hover_pose[key] + (
                float(waypoints["grasp"][key]) - float(waypoints["pre_grasp"][key])
            )
        print("Follow trigger: relative descent target from current follow pose")
    else:
        descent_target = dict(waypoints["grasp"])
        if locked_joint6 is not None:
            descent_target["joint_6.pos"] = locked_joint6
        print("Follow trigger: absolute grasp waypoint")
    print(f"  hover : {hover_pose}")
    print(f"  target: {descent_target}")

    print("Follow trigger: keep hand open before descent")
    set_hand(robot, DEFAULT_OPEN, repeats=15, dt=0.02)

    print("Follow trigger: descend to grasp")
    move_arm(robot, descent_target, args.approach_seconds, args.control_hz, locked_joint6, "relative_grasp")

    print("Follow trigger: close hand")
    set_hand(robot, DEFAULT_CLOSED, repeats=60, dt=0.02)

    print("Follow trigger: lift and place")
    move_arm(robot, hover_pose, args.approach_seconds, args.control_hz, locked_joint6, "follow_hover")
    move_arm(robot, waypoints["pre_place"], args.move_seconds, args.control_hz, locked_joint6, "pre_place")
    move_arm(robot, waypoints["place"], args.approach_seconds, args.control_hz, locked_joint6, "place")

    print("Follow trigger: release")
    set_hand(robot, DEFAULT_OPEN, repeats=60, dt=0.02)

    print("Follow trigger: retreat")
    move_arm(robot, waypoints["pre_place"], args.approach_seconds, args.control_hz, locked_joint6, "pre_place")
    move_arm(robot, waypoints["home"], args.move_seconds, args.control_hz, locked_joint6, "home")


def build_follow_action(
    origin: dict[str, float],
    gesture: GestureResult,
    args,
    locked_joint6: float | None,
) -> dict[str, float]:
    dx = gesture.wrist_x - args.follow_center_x
    dy = gesture.wrist_y - args.follow_center_y
    j1_offset = clamp(dx * args.follow_j1_gain, -args.follow_max_j1_offset, args.follow_max_j1_offset)
    y_sign = -1.0 if args.invert_follow_y else 1.0
    j2_offset = clamp(y_sign * dy * args.follow_j2_gain, -args.follow_max_j2_offset, args.follow_max_j2_offset)

    action = {
        "joint_1.pos": origin["joint_1.pos"] + j1_offset,
        "joint_2.pos": origin["joint_2.pos"] + j2_offset,
        "joint_3.pos": origin["joint_3.pos"],
        "joint_4.pos": origin["joint_4.pos"],
        "joint_5.pos": origin["joint_5.pos"],
    }
    if locked_joint6 is not None:
        action["joint_6.pos"] = locked_joint6
    return action


def maybe_confirm_execution(args) -> bool:
    if args.dry_run:
        print("DRY RUN: gesture trigger detected; robot action skipped.")
        return False
    if args.auto_execute:
        return True
    confirm = input("Gesture trigger detected. Type RUN to execute pick-place, or Enter to ignore: ").strip()
    return confirm == "RUN"


def put_status(frame, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        y += 28


def poll_terminal_key() -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    return sys.stdin.readline().strip().lower()


def update_web_state(frame, status: dict) -> None:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return
    with WEB_STATE_LOCK:
        WEB_STATE["jpeg"] = encoded.tobytes()
        WEB_STATE["status"] = copy.deepcopy(status)


class VisionViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args) -> None:
        return

    def _send_text(self, text: str, content_type: str = "text/html; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_text(self._index_html())
            return
        if self.path == "/status.json":
            with WEB_STATE_LOCK:
                status = copy.deepcopy(WEB_STATE["status"])
            self._send_text(json.dumps(status, ensure_ascii=False, indent=2), "application/json; charset=utf-8")
            return
        if self.path == "/stream.mjpg":
            self._send_mjpeg()
            return
        self.send_error(404)

    def _send_mjpeg(self) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        while True:
            with WEB_STATE_LOCK:
                jpeg = WEB_STATE["jpeg"]
            if jpeg is None:
                time.sleep(0.05)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                break

    @staticmethod
    def _index_html() -> str:
        title = html.escape("LeRobot Vision Gesture Viewer")
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #111; color: #eee; }}
    main {{ display: grid; grid-template-columns: minmax(480px, 1fr) 420px; gap: 16px; padding: 16px; }}
    img {{ width: 100%; background: #222; border: 1px solid #333; }}
    pre {{ margin: 0; padding: 12px; background: #1d1d1d; border: 1px solid #333; overflow: auto; max-height: calc(100vh - 32px); }}
    h1 {{ font-size: 18px; margin: 0 0 10px; }}
    .hint {{ color: #aaa; margin-bottom: 12px; font-size: 13px; }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>{title}</h1>
      <div class="hint">Live camera with MediaPipe hand landmarks and follow targets. Refresh if the stream stalls.</div>
      <img src="/stream.mjpg" alt="camera stream">
    </section>
    <section>
      <h1>Status</h1>
      <pre id="status">{{}}</pre>
    </section>
  </main>
  <script>
    async function refreshStatus() {{
      try {{
        const res = await fetch('/status.json', {{cache: 'no-store'}});
        document.getElementById('status').textContent = JSON.stringify(await res.json(), null, 2);
      }} catch (err) {{
        document.getElementById('status').textContent = String(err);
      }}
    }}
    setInterval(refreshStatus, 300);
    refreshStatus();
  </script>
</body>
</html>"""


def start_web_viewer(host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), VisionViewerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Web viewer: http://{host}:{port}")
    return server


def run(args) -> None:
    waypoints = load_waypoints(args.waypoints)
    robot = connect_robot(args) if (args.execute or args.execute_follow) and not args.dry_run else None
    web_server = start_web_viewer(args.web_host, args.web_port) if args.web_viewer else None

    detector = None
    if not args.keyboard_only:
        try:
            detector = MediaPipeHandGestureDetector(
                model_path=args.hand_landmarker_model,
                min_detection_confidence=args.min_detection_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
            )
        except ModuleNotFoundError as exc:
            print(str(exc))
            print("Falling back to keyboard trigger: press g in the camera window.")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    print("Vision gesture loop started.")
    print("Show an OPEN hand first, then a GRASP hand to trigger.")
    print("Keyboard/window: g=trigger, q=quit")
    print("Terminal fallback: type g then Enter to trigger, q then Enter to quit")
    if args.follow:
        print("Follow mode: wrist x/y -> joint_1/joint_2, hand openness -> RH56F2 hand")
    if not args.execute and not args.execute_follow:
        print("Robot is NOT connected. Add --execute or --execute-follow when vision trigger is reliable.")

    open_seen = False
    grasp_start: float | None = None
    executed_once = False
    display_enabled = not args.no_display and not args.web_viewer
    last_status_print = 0.0
    last_follow_print = 0.0
    last_follow_action = 0.0
    start_time = time.monotonic()
    follow_origin = None
    locked_joint6 = None
    if args.follow and args.follow_origin_waypoint:
        if args.follow_origin_waypoint not in waypoints:
            raise ValueError(f"Missing follow origin waypoint: {args.follow_origin_waypoint}")
        follow_origin = {key: float(waypoints[args.follow_origin_waypoint][key]) for key in ARM_KEYS}
    if robot is not None and args.execute_follow:
        if follow_origin is None:
            follow_origin = {key: float(robot.get_observation()[key]) for key in ARM_KEYS}
        if args.lock_joint6:
            locked_joint6 = float(robot.get_observation()["joint_6.pos"])
            follow_origin["joint_6.pos"] = locked_joint6
        print(f"Follow origin: {follow_origin}")
        if locked_joint6 is not None:
            print(f"Follow locking joint_6 at {locked_joint6:.3f} deg")
        if args.follow_origin_waypoint and not args.skip_move_to_follow_origin:
            print(f"Moving to follow origin waypoint: {args.follow_origin_waypoint}")
            set_hand(robot, DEFAULT_OPEN, repeats=20, dt=0.02)
            move_arm(
                robot,
                follow_origin,
                args.move_seconds,
                args.control_hz,
                locked_joint6,
                args.follow_origin_waypoint,
            )

    try:
        while True:
            if args.max_seconds is not None and time.monotonic() - start_time >= args.max_seconds:
                print(f"Reached --max-seconds={args.max_seconds}; exiting.")
                break

            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed.")
                time.sleep(0.1)
                continue

            gesture = None
            if detector is not None:
                gesture, frame = detector.process(frame)

            now = time.monotonic()
            label = "NO_HAND" if gesture is None else gesture.label
            score = 0.0 if gesture is None else gesture.score
            follow_action = None
            hand_pose = None

            if args.follow and gesture is not None:
                if follow_origin is None and robot is not None:
                    follow_origin = {key: float(robot.get_observation()[key]) for key in ARM_KEYS}
                    if args.lock_joint6:
                        locked_joint6 = follow_origin["joint_6.pos"]

                origin = follow_origin
                if origin is None:
                    origin = {
                        "joint_1.pos": 0.0,
                        "joint_2.pos": 0.0,
                        "joint_3.pos": 0.0,
                        "joint_4.pos": 0.0,
                        "joint_5.pos": 0.0,
                        "joint_6.pos": 0.0,
                    }
                follow_action = build_follow_action(origin, gesture, args, locked_joint6)
                if args.allow_follow_full_grasp:
                    hand_pose = hand_pose_from_gesture(gesture)
                else:
                    hand_pose = pregrasp_hand_pose_from_gesture(gesture, args.follow_min_hand_open)

                if now - last_follow_print >= 1.0 / args.follow_print_hz:
                    print(
                        "follow "
                        f"gesture={label} palm=({gesture.wrist_x:.2f},{gesture.wrist_y:.2f}) "
                        f"pinch={gesture.pinch_ratio:.2f} fingers={gesture.open_fingers} "
                        f"arm={follow_action}"
                    )
                    last_follow_print = now

                if (
                    args.execute_follow
                    and not args.dry_run
                    and robot is not None
                    and now - last_follow_action >= 1.0 / args.follow_action_hz
                ):
                    robot.send_action(follow_action)
                    if args.follow_hand:
                        robot.send_action({f"hand.{name}.pos": value for name, value in hand_pose.items()})
                    last_follow_action = now

            if label == "OPEN":
                open_seen = True
                grasp_start = None
            elif label == "GRASP" and open_seen:
                if grasp_start is None:
                    grasp_start = now
                elif now - grasp_start >= args.trigger_hold_seconds:
                    if maybe_confirm_execution(args) and robot is not None:
                        if args.execute_follow:
                            execute_grasp_place_from_follow(robot, waypoints, args, locked_joint6)
                        else:
                            execute_pick_place(robot, waypoints, args)
                    open_seen = False
                    grasp_start = None
                    executed_once = True
                    if args.once:
                        break
            elif label not in {"GRASP", "OPEN"}:
                grasp_start = None

            hold = 0.0 if grasp_start is None else now - grasp_start
            status = {
                "gesture": label,
                "score": round(score, 3),
                "open_seen": open_seen,
                "grasp_hold": round(hold, 3),
                "trigger_hold_seconds": args.trigger_hold_seconds,
                "mode": "EXECUTE_FOLLOW" if args.execute_follow else ("EXECUTE" if args.execute else "VISION_ONLY"),
                "follow": args.follow,
                "execute_follow": args.execute_follow,
                "palm": None
                if gesture is None
                else {"x": round(gesture.wrist_x, 3), "y": round(gesture.wrist_y, 3)},
                "pinch_ratio": None if gesture is None else round(gesture.pinch_ratio, 3),
                "open_fingers": None if gesture is None else gesture.open_fingers,
                "follow_action": follow_action,
                "hand_pose": hand_pose,
            }

            if display_enabled:
                put_status(
                    frame,
                    [
                        f"gesture={label} score={score:.2f}",
                        f"open_seen={open_seen} grasp_hold={hold:.1f}/{args.trigger_hold_seconds:.1f}s",
                        f"mode={'EXECUTE' if args.execute else 'VISION_ONLY'}",
                    ],
                )
                try:
                    cv2.imshow("LeRobot vision gesture pick-place", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("g"):
                        if maybe_confirm_execution(args) and robot is not None:
                            execute_pick_place(robot, waypoints, args)
                        executed_once = True
                        if args.once:
                            break
                except cv2.error as exc:
                    print(f"OpenCV GUI is unavailable, switching to terminal mode: {exc}")
                    display_enabled = False

            if args.web_viewer:
                put_status(
                    frame,
                    [
                        f"gesture={label} score={score:.2f}",
                        f"palm={status['palm']} pinch={status['pinch_ratio']} fingers={status['open_fingers']}",
                        f"open_seen={open_seen} grasp_hold={hold:.1f}/{args.trigger_hold_seconds:.1f}s",
                        f"mode={status['mode']}",
                    ],
                )
                update_web_state(frame, status)

            terminal_key = poll_terminal_key() if args.terminal_trigger or not display_enabled else None
            if terminal_key == "q":
                break
            if terminal_key == "g":
                if maybe_confirm_execution(args) and robot is not None:
                    if args.execute_follow:
                        execute_grasp_place_from_follow(robot, waypoints, args, locked_joint6)
                    else:
                        execute_pick_place(robot, waypoints, args)
                executed_once = True
                if args.once:
                    break

            if not display_enabled:
                if now - last_status_print >= 1.0:
                    print(f"camera_ok gesture={label} score={score:.2f} open_seen={open_seen}")
                    last_status_print = now
                time.sleep(0.01)

            if executed_once and args.once:
                break
    finally:
        cap.release()
        if detector is not None:
            detector.close()
        if display_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        if robot is not None:
            robot.disconnect()
        if web_server is not None:
            web_server.shutdown()
            web_server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental LeRobot vision gesture trigger for Piper + RH56F2 pick-place."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--web-viewer", action="store_true")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8765)
    parser.add_argument("--keyboard-only", action="store_true")
    parser.add_argument("--terminal-trigger", action="store_true")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--follow", action="store_true", help="Track wrist/hand and print follow actions.")
    parser.add_argument("--execute-follow", action="store_true", help="Execute small-range LeRobot follow actions.")
    parser.add_argument("--follow-hand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-follow-full-grasp", action="store_true")
    parser.add_argument("--follow-min-hand-open", type=float, default=0.55)
    parser.add_argument("--follow-origin-waypoint", default="pre_grasp")
    parser.add_argument("--skip-move-to-follow-origin", action="store_true")
    parser.add_argument("--relative-grasp-from-follow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--follow-center-x", type=float, default=0.5)
    parser.add_argument("--follow-center-y", type=float, default=0.6)
    parser.add_argument("--invert-follow-y", action="store_true")
    parser.add_argument("--follow-j1-gain", type=float, default=40.0)
    parser.add_argument("--follow-j2-gain", type=float, default=35.0)
    parser.add_argument("--follow-max-j1-offset", type=float, default=20.0)
    parser.add_argument("--follow-max-j2-offset", type=float, default=15.0)
    parser.add_argument("--follow-action-hz", type=float, default=8.0)
    parser.add_argument("--follow-print-hz", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true", help="Connect robot and allow pick-place execution.")
    parser.add_argument("--dry-run", action="store_true", help="Detect triggers but skip robot actions.")
    parser.add_argument("--auto-execute", action="store_true", help="Run on gesture without typing RUN.")
    parser.add_argument("--once", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trigger-hold-seconds", type=float, default=0.8)
    parser.add_argument("--min-detection-confidence", type=float, default=0.65)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.65)
    parser.add_argument("--hand-landmarker-model", type=Path, default=DEFAULT_HAND_LANDMARKER_MODEL)

    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-baudrate", type=int, default=115200)
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--hand-force", type=int, default=1500)
    parser.add_argument("--speed-rate", type=int, default=25)
    parser.add_argument("--max-arm-delta-deg", type=float, default=5.0)
    parser.add_argument("--max-hand-delta", type=float, default=120.0)
    parser.add_argument("--waypoints", type=Path, default=DEFAULT_WAYPOINTS)
    parser.add_argument("--lock-joint6", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--move-seconds", type=float, default=4.0)
    parser.add_argument("--approach-seconds", type=float, default=2.0)
    parser.add_argument("--control-hz", type=float, default=50.0)
    args = parser.parse_args()

    if args.execute_follow:
        args.follow = True
    if args.auto_execute and not (args.execute or args.execute_follow):
        print("--auto-execute has no effect without --execute/--execute-follow.")
    if args.dry_run and (args.execute or args.execute_follow):
        print("--dry-run is set; robot will not connect and motion actions will be skipped.")

    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
