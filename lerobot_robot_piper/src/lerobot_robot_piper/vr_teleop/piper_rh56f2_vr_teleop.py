#!/usr/bin/env python3
"""VR teleop bridge for Piper arm + RH56F2 hand.

This is the hardware-side bridge:

  VR frame -> ee.* target pose + hand.* target angles -> PiperRH56F2Follower

The first version accepts JSON lines so it can be connected to different VR
frontends without tying the robot driver to one specific headset SDK. The
official vr_teleop input code can later call ``VRFrameTeleop.step(...)``
directly instead of going through stdin.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lerobot_robot_piper.config_piper_rh56f2_follower import PiperRH56F2FollowerConfig
from lerobot_robot_piper.piper_rh56f2_follower import EE_POSE_NAMES, PiperRH56F2Follower
from lerobot_robot_piper.rh56f2_hand import (
    DEFAULT_CLOSED,
    DEFAULT_OPEN,
    HAND_NAMES,
    RH56F2Hand,
    RH56F2HandConfig,
)


def _clip_value(goal: float, current: float, max_delta: float | None) -> float:
    if max_delta is None:
        return goal
    return current + min(max(goal - current, -max_delta), max_delta)


@dataclass
class VRFrame:
    """One normalized VR tracking frame.

    ``wrist_xyz_m`` is the operator hand/wrist position in meters.
    ``wrist_rpy_deg`` is optional roll/pitch/yaw in degrees.
    ``finger_curls`` maps RH56F2 finger names to 0.0=open and 1.0=closed.
    ``deadman`` must be true before motion commands are sent.
    """

    wrist_xyz_m: tuple[float, float, float]
    wrist_rpy_deg: tuple[float, float, float] | None
    finger_curls: dict[str, float]
    deadman: bool
    landmarks: list[float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "VRFrame":
        wrist = data.get("wrist_xyz_m")
        if not isinstance(wrist, list | tuple) or len(wrist) != 3:
            raise ValueError("frame needs wrist_xyz_m: [x, y, z]")

        raw_rpy = data.get("wrist_rpy_deg")
        rpy = None
        if raw_rpy is not None:
            if not isinstance(raw_rpy, list | tuple) or len(raw_rpy) != 3:
                raise ValueError("wrist_rpy_deg must be [rx, ry, rz]")
            rpy = tuple(float(v) for v in raw_rpy)

        curls_raw = data.get("finger_curls", {})
        if not isinstance(curls_raw, dict):
            raise ValueError("finger_curls must be an object")

        return cls(
            wrist_xyz_m=tuple(float(v) for v in wrist),
            wrist_rpy_deg=rpy,
            finger_curls={str(k): float(v) for k, v in curls_raw.items()},
            deadman=bool(data.get("deadman", False)),
            landmarks=(
                [float(v) for v in data["landmarks"]]
                if isinstance(data.get("landmarks"), list)
                else None
            ),
        )


def _parse_csv_line(message: str, prefix: str, count: int) -> list[float] | None:
    prefix = prefix.lower()
    for line in message.splitlines():
        if not line.strip().lower().startswith(prefix):
            continue
        _, _, rest = line.partition(":")
        values: list[float] = []
        for item in rest.split(","):
            try:
                values.append(float(item.strip()))
            except ValueError:
                break
        if len(values) == count:
            return values
    return None


def _parse_right_wrist(message: str) -> list[float] | None:
    return _parse_csv_line(message, "right wrist", 7)


def _parse_right_landmarks(message: str) -> list[float] | None:
    return _parse_csv_line(message, "right landmarks", 63)


def _distance(points: list[float], first: int, second: int) -> float:
    first_offset = first * 3
    second_offset = second * 3
    return math.sqrt(
        sum(
            (points[first_offset + axis] - points[second_offset + axis]) ** 2
            for axis in range(3)
        )
    )


def _point(points: list[float], index: int) -> np.ndarray:
    offset = index * 3
    return np.asarray(points[offset : offset + 3], dtype=float)


def _joint_angle(points: list[float], first: int, vertex: int, second: int) -> float:
    """Return the 3-D angle at a hand landmark in radians."""
    a = _point(points, first) - _point(points, vertex)
    b = _point(points, second) - _point(points, vertex)
    denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
    cosine = float(np.dot(a, b) / denominator)
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def landmarks_to_simple_curls(points: list[float]) -> dict[str, float]:
    """Convert 21 VR landmarks into safe 0=open, 1=closed curl values."""
    if len(points) != 63:
        raise ValueError("expected 63 hand landmark values")

    finger_indices = {
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "little": (17, 18, 19, 20),
    }
    curls: dict[str, float] = {}
    for name, (mcp, pip, dip, tip) in finger_indices.items():
        pip_bend = (math.pi - _joint_angle(points, mcp, pip, dip)) / (math.pi - 0.55)
        dip_bend = (math.pi - _joint_angle(points, pip, dip, tip)) / (math.pi - 0.55)
        angle_curl = 0.5 * (pip_bend + dip_bend)

        # Use the finger shape relative to the wrist and palm size. This is
        # invariant to image translation and hand scale, and gives a larger
        # signal when the fingertip folds toward the palm in a fist.
        mcp_from_wrist = _distance(points, mcp, 0) / max(_distance(points, 0, 9), 1e-6)
        tip_from_wrist = _distance(points, tip, 0) / max(_distance(points, 0, 9), 1e-6)
        fold_curl = (mcp_from_wrist - tip_from_wrist + 0.10) / 0.45

        # Blend relative folding and joint angles, then add gain so a real
        # fist reaches the RH56F2 closed range instead of stopping halfway.
        curls[name] = float(np.clip(1.35 * (0.65 * fold_curl + 0.35 * angle_curl), 0.0, 1.0))

    # Normalize thumb measurements by palm size.  These are distances between
    # landmarks, so moving or rotating the whole hand does not change them.
    palm_scale = max(_distance(points, 0, 9), 1e-6)
    thumb_to_wrist = _distance(points, 4, 0) / palm_scale
    thumb_to_index_mcp = _distance(points, 4, 5) / palm_scale
    thumb_to_middle_mcp = _distance(points, 4, 9) / palm_scale
    thumb_bend_angle = (math.pi - _joint_angle(points, 2, 3, 4)) / (math.pi - 0.65)

    curls["thumb_bend"] = float(np.clip(
        0.65 * ((1.05 - thumb_to_wrist) / 0.55) + 0.35 * thumb_bend_angle,
        0.0,
        1.0,
    ))
    # Use both index- and middle-MCP distances.  A thumb across the palm is
    # close to both; an abducted thumb is far from both.  This is much less
    # sensitive to wrist rotation than a screen-space left/right test.
    thumb_open_fraction = np.clip(
        0.55 * ((thumb_to_index_mcp - 0.28) / 0.62)
        + 0.45 * ((thumb_to_middle_mcp - 0.36) / 0.62),
        0.0,
        1.0,
    )
    curls["thumb_swing"] = float(1.0 - thumb_open_fraction)
    return curls


class Quest3UDPInput:
    """Read the upstream Quest 3 Hand Tracking Streamer UDP format."""

    def __init__(self, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("0.0.0.0", port))
        self.socket.setblocking(False)
        self.port = port
        self.received_packets = 0
        self.valid_frames = 0
        self.invalid_packets = 0

    def close(self) -> None:
        self.socket.close()

    def poll(self) -> VRFrame | None:
        latest: bytes | None = None
        while True:
            try:
                latest, _ = self.socket.recvfrom(65536)
            except BlockingIOError:
                break
        if latest is None:
            return None

        self.received_packets += 1

        message = latest.decode("utf-8", errors="ignore")
        wrist = _parse_right_wrist(message)
        landmarks = _parse_right_landmarks(message)
        if wrist is None or landmarks is None:
            self.invalid_packets += 1
            return None

        # Quest/Unity frame -> Piper frame. Keep orientation disabled until
        # the real wrist-axis calibration is confirmed on the hardware.
        wrist_xyz_m = (-wrist[0], -wrist[2], wrist[1])
        self.valid_frames += 1
        return VRFrame(
            wrist_xyz_m=wrist_xyz_m,
            wrist_rpy_deg=None,
            finger_curls=landmarks_to_simple_curls(landmarks),
            deadman=True,
            landmarks=landmarks,
        )


class VisionProInput:
    """Read Apple Vision Pro Tracking Streamer frames through avp_stream."""

    # Vision Pro exposes a 25-joint hand skeleton. These are the 21 joints
    # used by the MediaPipe/AnyDex hand convention; metacarpals are skipped.
    VP_TO_MEDIAPIPE = (
        0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24
    )
    AVP_TO_ROBOT = np.asarray(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    def __init__(self, ip: str):
        try:
            from avp_stream import VisionProStreamer
        except ImportError as exc:
            raise RuntimeError(
                "Vision Pro input needs avp-stream. Install it with: "
                "python3 -m pip install avp-stream"
            ) from exc

        self.ip = ip
        self.streamer = VisionProStreamer(ip=ip, record=False)
        self.received_frames = 0
        self.valid_frames = 0

    def close(self) -> None:
        cleanup = getattr(self.streamer, "cleanup", None)
        if callable(cleanup):
            cleanup()

    def poll(self) -> VRFrame | None:
        data = self.streamer.get_latest()
        if data is None or data.right is None:
            return None

        hand = data.right
        if hand.shape[0] <= max(self.VP_TO_MEDIAPIPE):
            return None

        points = np.asarray(
            [hand[index][:3, 3] for index in self.VP_TO_MEDIAPIPE],
            dtype=np.float64,
        )
        if np.allclose(points, 0.0):
            return None

        self.received_frames += 1
        self.valid_frames += 1
        wrist_avp = np.asarray(hand[0][:3, 3], dtype=np.float64)
        wrist_robot = self.AVP_TO_ROBOT @ wrist_avp
        landmarks = points.reshape(-1).tolist()
        return VRFrame(
            wrist_xyz_m=tuple(float(value) for value in wrist_robot),
            wrist_rpy_deg=None,
            finger_curls=landmarks_to_simple_curls(landmarks),
            deadman=True,
            landmarks=[float(value) for value in landmarks],
        )


class ArmPoseMapper:
    """Map relative VR wrist motion to Piper end-effector pose commands."""

    def __init__(self, xyz_scale: float, rpy_scale: float):
        self.xyz_scale = xyz_scale
        self.rpy_scale = rpy_scale
        self._vr_anchor: tuple[float, float, float] | None = None
        self._rpy_anchor: tuple[float, float, float] | None = None
        self._ee_anchor: dict[str, float] | None = None

    def reset(self, frame: VRFrame, current_ee: dict[str, float]) -> None:
        self._vr_anchor = frame.wrist_xyz_m
        self._rpy_anchor = frame.wrist_rpy_deg
        self._ee_anchor = dict(current_ee)

    def map(self, frame: VRFrame, current_ee: dict[str, float]) -> dict[str, float]:
        if self._vr_anchor is None or self._ee_anchor is None:
            self.reset(frame, current_ee)

        assert self._vr_anchor is not None
        assert self._ee_anchor is not None

        dx = (frame.wrist_xyz_m[0] - self._vr_anchor[0]) * 1000.0 * self.xyz_scale
        dy = (frame.wrist_xyz_m[1] - self._vr_anchor[1]) * 1000.0 * self.xyz_scale
        dz = (frame.wrist_xyz_m[2] - self._vr_anchor[2]) * 1000.0 * self.xyz_scale

        target = {
            "ee.x": self._ee_anchor["ee.x"] + dx,
            "ee.y": self._ee_anchor["ee.y"] + dy,
            "ee.z": self._ee_anchor["ee.z"] + dz,
            "ee.rx": self._ee_anchor["ee.rx"],
            "ee.ry": self._ee_anchor["ee.ry"],
            "ee.rz": self._ee_anchor["ee.rz"],
        }

        if frame.wrist_rpy_deg is not None and self._rpy_anchor is not None:
            target["ee.rx"] += (frame.wrist_rpy_deg[0] - self._rpy_anchor[0]) * self.rpy_scale
            target["ee.ry"] += (frame.wrist_rpy_deg[1] - self._rpy_anchor[1]) * self.rpy_scale
            target["ee.rz"] += (frame.wrist_rpy_deg[2] - self._rpy_anchor[2]) * self.rpy_scale

        return target


class RH56F2SimpleRetargeter:
    """Simple curl-to-register mapping before full AnyDexRetarget integration."""

    def __init__(self, thumb_swing_closed: float = 500.0):
        # RH56F2 SDK right-hand calibration from the supplied controller:
        # fingers 1740->900, thumb bend 1450->1100, thumb swing 1750->500.
        self.open_pose = {
            "little": 1740.0,
            "ring": 1740.0,
            "middle": 1740.0,
            "index": 1740.0,
            "thumb_bend": 1450.0,
            "thumb_swing": 1750.0,
        }
        self.closed_pose = {
            "little": 900.0,
            "ring": 900.0,
            "middle": 900.0,
            "index": 900.0,
            "thumb_bend": 1100.0,
            "thumb_swing": 500.0,
        }
        self.thumb_swing_closed = float(thumb_swing_closed)

    def map(self, curls: dict[str, float]) -> dict[str, float]:
        action: dict[str, float] = {}
        for name in HAND_NAMES:
            curl = float(curls.get(name, curls.get("all", 0.0)))
            curl = min(max(curl, 0.0), 1.0)
            opened = self.open_pose[name]
            closed = self.thumb_swing_closed if name == "thumb_swing" else self.closed_pose[name]
            action[f"hand.{name}.pos"] = opened + curl * (closed - opened)
        return action


def parse_qpos_groups(value: str) -> list[list[int]]:
    """Parse five RH56F2 finger groups, e.g. ``0,1;2,3;4,5;6,7;8,9``."""
    groups: list[list[int]] = []
    for group in value.split(";"):
        indices = [int(item.strip()) for item in group.split(",") if item.strip()]
        if not indices:
            raise ValueError("each AnyDex qpos group must contain an index")
        groups.append(indices)
    if len(groups) != 5:
        raise ValueError("AnyDex qpos groups must be thumb,index,middle,ring,little")
    return groups


class AnyDexRH56F2Retargeter:
    """Use AnyDexRetarget as a hand-pose front end for RH56F2.

    AnyDexRetarget targets a modeled hand, not RH56F2. We therefore convert
    the modeled joint angles into normalized curls before generating RH56F2
    register targets. Joint names are used instead of hard-coded qpos
    positions because Inspire's qpos order is not thumb,index,middle,ring,
    little and includes mimic joints.
    """

    def __init__(self, config_path: Path, anydex_root: Path, qpos_groups: str):
        if str(anydex_root) not in sys.path:
            sys.path.insert(0, str(anydex_root))
        try:
            from anydexretarget import Retargeter
        except ImportError as exc:
            raise RuntimeError(
                "AnyDexRetarget is unavailable. Install its dependencies and "
                "pass --anydex-root to its checkout."
            ) from exc

        self.retargeter = Retargeter.from_yaml(str(config_path), "right")
        lower = np.asarray(self.retargeter.optimizer.opt_lower_bounds, dtype=float)
        upper = np.asarray(self.retargeter.optimizer.opt_upper_bounds, dtype=float)
        self.lower = lower
        self.span = np.maximum(upper - lower, 1e-6)

        joint_names = list(self.retargeter.optimizer.robot.dof_joint_names)
        self.joint_index = {name: index for index, name in enumerate(joint_names)}
        required = {
            "index_proximal_joint",
            "middle_proximal_joint",
            "ring_proximal_joint",
            "pinky_proximal_joint",
            "thumb_proximal_yaw_joint",
            "thumb_proximal_pitch_joint",
        }
        missing = sorted(required - self.joint_index.keys())
        if missing:
            raise ValueError(
                "AnyDex hand model is missing the joints needed for RH56F2: "
                + ", ".join(missing)
            )

        # This is the physical RH56F2 order, not the AnyDex qpos order.
        self.channel_joint = {
            "index": "index_proximal_joint",
            "middle": "middle_proximal_joint",
            "ring": "ring_proximal_joint",
            "little": "pinky_proximal_joint",
            "thumb_swing": "thumb_proximal_yaw_joint",
            "thumb_bend": "thumb_proximal_pitch_joint",
        }
        self.register_mapper = RH56F2SimpleRetargeter()
        self._last_debug_time = 0.0

    def map_landmarks(self, landmarks: list[float]) -> dict[str, float]:
        if len(landmarks) != 63:
            raise ValueError("AnyDex input needs 63 hand landmark values")
        qpos = np.asarray(
            self.retargeter.retarget(np.asarray(landmarks, dtype=float).reshape(21, 3)),
            dtype=float,
        )
        curls: dict[str, float] = {}
        for channel, joint_name in self.channel_joint.items():
            index = self.joint_index[joint_name]
            curls[channel] = float(
                np.clip((qpos[index] - self.lower[index]) / self.span[index], 0.0, 1.0)
            )

        # The old adapter forced this channel to zero, which made thumb
        # abduction/adduction invisible to the real hand.
        if time.monotonic() - self._last_debug_time >= 0.5:
            print(
                "AnyDex curls: "
                + json.dumps({name: round(value, 3) for name, value in curls.items()}, sort_keys=True),
                flush=True,
            )
            self._last_debug_time = time.monotonic()
        return self.register_mapper.map(curls)


class DryRunRobot:
    """Small stand-in used to verify mapping before connecting real hardware."""

    def __init__(self, start_ee: dict[str, float]):
        self.obs = dict(start_ee)
        self.obs.update({f"hand.{name}.pos": DEFAULT_OPEN[name] for name in HAND_NAMES})

    def connect(self) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return True

    def get_observation(self) -> dict[str, float]:
        return dict(self.obs)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.obs.update(action)
        print(json.dumps(action, sort_keys=True))
        return action

    def disconnect(self) -> None:
        pass


class HandOnlyRobot:
    """Real RH56F2 hand with a fake arm pose for low-risk teleop checks."""

    def __init__(
        self,
        start_ee: dict[str, float],
        hand_port: str,
        hand_id: int,
        hand_speed: int,
        hand_force: int,
        max_hand_delta: float | None,
    ):
        self.obs = dict(start_ee)
        self.max_hand_delta = max_hand_delta
        self.hand = RH56F2Hand(
            RH56F2HandConfig(
                port=hand_port,
                hand_id=hand_id,
                speed=hand_speed,
                force=hand_force,
            )
        )

    def connect(self) -> None:
        self.hand.connect()

    @property
    def is_connected(self) -> bool:
        return self.hand.is_connected

    def get_observation(self) -> dict[str, float]:
        obs = dict(self.obs)
        hand_pos = self.hand.read_positions("angleAct")
        obs.update({f"hand.{name}.pos": value for name, value in hand_pos.items()})
        hand_force = self.hand.read_positions("forceAct")
        obs.update({f"hand.{name}.force": value for name, value in hand_force.items()})
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.obs.update({name: float(action[name]) for name in EE_POSE_NAMES if name in action})
        hand_action = {
            name: float(action[f"hand.{name}.pos"])
            for name in HAND_NAMES
            if f"hand.{name}.pos" in action
        }
        if not hand_action:
            return {}

        current = self.hand.read_positions("angleAct")
        clipped = {
            name: _clip_value(value, current[name], self.max_hand_delta)
            for name, value in hand_action.items()
        }
        self.hand.set_angles(clipped)
        return {f"hand.{name}.pos": value for name, value in clipped.items()}

    def disconnect(self) -> None:
        self.hand.disconnect()


class VRFrameTeleop:
    def __init__(self, robot: object, xyz_scale: float, rpy_scale: float, hand_mapper: object):
        self.robot = robot
        self.arm_mapper = ArmPoseMapper(xyz_scale=xyz_scale, rpy_scale=rpy_scale)
        self.hand_mapper = hand_mapper
        self._last_deadman = False

    def _hand_action(self, frame: VRFrame) -> dict[str, float]:
        if isinstance(self.hand_mapper, AnyDexRH56F2Retargeter):
            if frame.landmarks is None:
                return {}
            return self.hand_mapper.map_landmarks(frame.landmarks)
        return self.hand_mapper.map(frame.finger_curls)

    def step(self, frame: VRFrame) -> dict[str, float]:
        obs = self.robot.get_observation()
        current_ee = {name: float(obs[name]) for name in EE_POSE_NAMES}

        if not frame.deadman:
            self._last_deadman = False
            self.arm_mapper.reset(frame, current_ee)
            return {}

        if not self._last_deadman:
            self.arm_mapper.reset(frame, current_ee)
        self._last_deadman = True

        action = self.arm_mapper.map(frame, current_ee)
        action.update(self._hand_action(frame))
        return self.robot.send_action(action)


def iter_json_frames(stream: Iterable[str]) -> Iterable[VRFrame]:
    for line in stream:
        line = line.strip()
        if not line:
            continue
        yield VRFrame.from_dict(json.loads(line))


def build_robot(args: argparse.Namespace) -> object:
    start_ee = {
        "ee.x": args.start_x,
        "ee.y": args.start_y,
        "ee.z": args.start_z,
        "ee.rx": args.start_rx,
        "ee.ry": args.start_ry,
        "ee.rz": args.start_rz,
    }

    if not args.connect:
        return DryRunRobot(start_ee)

    if args.hand_only:
        return HandOnlyRobot(
            start_ee=start_ee,
            hand_port=args.hand_port,
            hand_id=args.hand_id,
            hand_speed=args.hand_speed,
            hand_force=args.hand_force,
            max_hand_delta=args.max_hand_delta,
        )

    return PiperRH56F2Follower(
        PiperRH56F2FollowerConfig(
            can_port=args.can,
            speed_rate=args.speed,
            max_ee_delta_mm=args.max_ee_delta_mm,
            max_ee_delta_deg=args.max_ee_delta_deg,
            hand_port=args.hand_port,
            hand_id=args.hand_id,
            hand_speed=args.hand_speed,
            hand_force=args.hand_force,
            max_hand_delta=args.max_hand_delta,
        )
    )


def build_input(args: argparse.Namespace) -> Quest3UDPInput | VisionProInput | None:
    if args.input_source == "quest3":
        return Quest3UDPInput(args.port)
    if args.input_source == "avp":
        return VisionProInput(args.avp_ip)
    return None


def build_hand_mapper(args: argparse.Namespace) -> object:
    if args.hand_mode == "simple":
        return RH56F2SimpleRetargeter(thumb_swing_closed=args.thumb_swing_closed)
    if args.hand_config is None:
        raise ValueError("--hand-config is required with --hand-mode anydex")
    return AnyDexRH56F2Retargeter(
        config_path=args.hand_config,
        anydex_root=args.anydex_root,
        qpos_groups=args.anydex_qpos_groups,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true", help="Connect real Piper + RH56F2 hardware.")
    parser.add_argument("--input-source", choices=["stdin", "quest3", "avp"], default="quest3")
    parser.add_argument("--port", type=int, default=9000, help="Quest 3 UDP input port.")
    parser.add_argument("--avp-ip", default="192.168.1.100", help="Apple Vision Pro IP address.")
    parser.add_argument("--hand-mode", choices=["simple", "anydex"], default="simple")
    parser.add_argument("--hand-config", type=Path)
    parser.add_argument("--anydex-root", type=Path, default=Path("third_party/AnyDexRetarget"))
    parser.add_argument(
        "--anydex-qpos-groups",
        default="0,1;2,3;4,5;6,7;8,9",
        help="Five qpos groups: thumb,index,middle,ring,little.",
    )
    parser.add_argument("--hand-only", action="store_true", help="Only connect RH56F2; keep Piper disabled.")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--hand-force", type=int, default=1500)
    parser.add_argument(
        "--thumb-swing-closed",
        type=float,
        default=500.0,
        help="RH56F2 thumb side-swing target when fully curled; calibrate on hardware.",
    )
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--xyz-scale", type=float, default=0.6)
    parser.add_argument("--rpy-scale", type=float, default=0.5)
    parser.add_argument("--max-ee-delta-mm", type=float, default=10.0)
    parser.add_argument("--max-ee-delta-deg", type=float, default=5.0)
    parser.add_argument("--max-hand-delta", type=float, default=80.0)
    parser.add_argument("--start-x", type=float, default=300.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-z", type=float, default=250.0)
    parser.add_argument("--start-rx", type=float, default=0.0)
    parser.add_argument("--start-ry", type=float, default=0.0)
    parser.add_argument("--start-rz", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robot = build_robot(args)
    hand_mapper = build_hand_mapper(args)
    teleop = VRFrameTeleop(
        robot,
        xyz_scale=args.xyz_scale,
        rpy_scale=args.rpy_scale,
        hand_mapper=hand_mapper,
    )
    interval_s = 1.0 / args.rate_hz
    print(f"Initializing input source: {args.input_source}", flush=True)
    vr_input: Quest3UDPInput | VisionProInput | None = None
    last_frame_time = time.monotonic()
    last_status_time = time.monotonic()

    try:
        vr_input = build_input(args)
        print("Input source initialized.", flush=True)
        robot.connect()
        if vr_input is None:
            print("VR teleop ready. Reading normalized JSON frames from stdin.")
            for frame in iter_json_frames(sys.stdin):
                started = time.time()
                teleop.step(frame)
                elapsed = time.time() - started
                if elapsed < interval_s:
                    time.sleep(interval_s - elapsed)
        else:
            if isinstance(vr_input, Quest3UDPInput):
                print(
                    f"VR teleop ready. Listening for Quest 3 UDP on 0.0.0.0:{vr_input.port}. "
                    "Waiting for valid hand frames...",
                    flush=True,
                )
            else:
                print(
                    f"Vision Pro teleop ready. Connecting to {vr_input.ip}. "
                    "Waiting for right-hand frames...",
                    flush=True,
                )
            while True:
                started = time.time()
                frame = vr_input.poll()
                if frame is not None:
                    last_frame_time = time.monotonic()
                    action = teleop.step(frame)
                elif time.monotonic() - last_frame_time > 0.25:
                    teleop.step(
                        VRFrame(
                            wrist_xyz_m=(0.0, 0.0, 0.0),
                            wrist_rpy_deg=None,
                            finger_curls={},
                            deadman=False,
                        )
                    )
                if time.monotonic() - last_status_time >= 2.0:
                    if isinstance(vr_input, Quest3UDPInput):
                        print(
                            "Quest 3 status: "
                            f"packets={vr_input.received_packets} "
                            f"valid={vr_input.valid_frames} "
                            f"invalid={vr_input.invalid_packets}",
                            flush=True,
                        )
                    else:
                        print(
                            "Vision Pro status: "
                            f"frames={vr_input.received_frames} "
                            f"valid_hand={vr_input.valid_frames}",
                            flush=True,
                        )
                    last_status_time = time.monotonic()
                elapsed = time.time() - started
                if elapsed < interval_s:
                    time.sleep(interval_s - elapsed)
    except KeyboardInterrupt:
        print("\nStopping VR teleoperation.")
    finally:
        if vr_input is not None:
            vr_input.close()
        if getattr(robot, "is_connected", False):
            robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
