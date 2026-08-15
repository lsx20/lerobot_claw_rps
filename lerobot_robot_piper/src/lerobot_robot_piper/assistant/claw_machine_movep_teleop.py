#!/usr/bin/env python3
"""Keyboard teleop for a claw-machine style pick cycle.

Manual phase:
  w/s: MOVE_J reach forward/back
  a/d: MOVE_J base left/right
  space: open while descending, close at grab height, lift/drop, open, close while returning

Manual keyboard control uses MOVE_J for smoother operator feel.
The automatic vertical pick cycle uses the official Piper MOVE_P demo call:
  MotionCtrl_2(0x01, 0x00, speed, 0x00)
  EndPoseCtrl(X, Y, Z, RX, RY, RZ)

It does not use MOVE_J and does not disable motors automatically.
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

from piper_sdk import C_PiperInterface_V2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from rh56f2_hand import (  # noqa: E402
        DEFAULT_CLOSED,
        DEFAULT_OPEN,
        RH56F2Hand,
        RH56F2HandConfig,
    )
except Exception:  # pragma: no cover - lets arm-only mode still run.
    DEFAULT_CLOSED = {}
    DEFAULT_OPEN = {}
    RH56F2Hand = None
    RH56F2HandConfig = None


HELP = """
Controls

  w / s       reach forward / back with J2,J3,J5
  a / d       J1 left / right
  space       open while descending, close at grab height, lift/drop, open, close while returning
  p           print current pose
  r           reset the teleop reference to current pose
  q           quit

Use Ctrl-C only if the key loop is unresponsive.
"""

DEFAULT_START_POSE = [325809, 23336, 268832, 173269, 27298, 172930]


class RawTerminal:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def read_key(timeout_s: float = 0.1) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return None
    key = sys.stdin.read(1)
    while select.select([sys.stdin], [], [], 0.0)[0]:
        key = sys.stdin.read(1)
    return key


def parse_pose_mm_deg(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("expected X,Y,Z,RX,RY,RZ")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pose values must be numbers") from exc
    return [int(round(item * 1000.0)) for item in values]


def parse_xy_axis(value: str) -> tuple[float, float]:
    name = value.strip().lower()
    aliases = {
        "x+": (1.0, 0.0),
        "+x": (1.0, 0.0),
        "x-": (-1.0, 0.0),
        "-x": (-1.0, 0.0),
        "y+": (0.0, 1.0),
        "+y": (0.0, 1.0),
        "y-": (0.0, -1.0),
        "-y": (0.0, -1.0),
    }
    if name in aliases:
        return aliases[name]
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected x+, x-, y+, y-, or DX,DY")
    try:
        dx, dy = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("axis values must be numbers") from exc
    norm = (dx * dx + dy * dy) ** 0.5
    if norm <= 0:
        raise argparse.ArgumentTypeError("axis vector cannot be zero")
    return dx / norm, dy / norm


def end_pose_raw(piper: C_PiperInterface_V2) -> list[int]:
    ep = piper.GetArmEndPoseMsgs().end_pose
    return [ep.X_axis, ep.Y_axis, ep.Z_axis, ep.RX_axis, ep.RY_axis, ep.RZ_axis]


def pose_mm_deg(pose: list[int]) -> str:
    return (
        f"X={pose[0] / 1000.0:8.3f} Y={pose[1] / 1000.0:8.3f} "
        f"Z={pose[2] / 1000.0:8.3f} RX={pose[3] / 1000.0:8.3f} "
        f"RY={pose[4] / 1000.0:8.3f} RZ={pose[5] / 1000.0:8.3f}"
    )


def joints_deg(piper: C_PiperInterface_V2) -> list[float]:
    js = piper.GetArmJointMsgs().joint_state
    return [
        js.joint_1 / 1000.0,
        js.joint_2 / 1000.0,
        js.joint_3 / 1000.0,
        js.joint_4 / 1000.0,
        js.joint_5 / 1000.0,
        js.joint_6 / 1000.0,
    ]


def joint_limits_ok(joints: list[float]) -> bool:
    limits = [
        (-150.0, 150.0),
        (0.0, 180.0),
        (-170.0, 0.0),
        (-100.0, 100.0),
        (-70.0, 70.0),
        (-120.0, 120.0),
    ]
    return all(lo <= value <= hi for value, (lo, hi) in zip(joints, limits, strict=True))


def fmt_joints(values: list[float]) -> str:
    return " ".join(f"{value:8.3f}" for value in values)


def arm_status_code(piper: C_PiperInterface_V2) -> int:
    return int(piper.GetArmStatus().arm_status.arm_status)


def ctrl_mode_code(piper: C_PiperInterface_V2) -> int:
    return int(piper.GetArmStatus().arm_status.ctrl_mode)


def mode_feed_code(piper: C_PiperInterface_V2) -> int:
    return int(piper.GetArmStatus().arm_status.mode_feed)


def has_real_feedback(piper: C_PiperInterface_V2) -> bool:
    status = piper.GetArmStatus()
    pose = piper.GetArmEndPoseMsgs()
    joints = piper.GetArmJointMsgs()
    ep = pose.end_pose
    js = joints.joint_state
    return (
        status.Hz > 0
        or pose.Hz > 0
        or joints.Hz > 0
        or any(
            value != 0
            for value in (
                ep.X_axis,
                ep.Y_axis,
                ep.Z_axis,
                ep.RX_axis,
                ep.RY_axis,
                ep.RZ_axis,
                js.joint_1,
                js.joint_2,
                js.joint_3,
                js.joint_4,
                js.joint_5,
                js.joint_6,
            )
        )
    )


def wait_for_real_feedback(piper: C_PiperInterface_V2, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if has_real_feedback(piper):
            return
        time.sleep(0.05)
    raise RuntimeError("No real Piper feedback received; refusing to command motion.")


def enable_all(piper: C_PiperInterface_V2, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    count = 0
    while time.time() < deadline:
        piper.EnableArm(7, 0x02)
        time.sleep(0.02)
        enable_status = list(piper.GetArmEnableStatus())
        if count % 10 == 0:
            print(f"enable status: {enable_status}")
        if enable_status and all(enable_status):
            return True
        count += 1
    return False


def wait_for_movep_ready(
    piper: C_PiperInterface_V2,
    speed: int,
    timeout_s: float,
) -> bool:
    deadline = time.time() + timeout_s
    count = 0
    while time.time() < deadline:
        piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
        piper.EnableArm(7, 0x02)
        time.sleep(0.05)
        status = piper.GetArmStatus()
        enable_status = list(piper.GetArmEnableStatus())
        ctrl_mode = ctrl_mode_code(piper)
        mode_feed = mode_feed_code(piper)
        arm_status = arm_status_code(piper)
        if count % 10 == 0:
            print(
                "mode ready check: "
                f"ctrl=0x{ctrl_mode:x} mode=0x{mode_feed:x} "
                f"arm=0x{arm_status:x} hz={status.Hz:.1f} "
                f"enable={enable_status}"
            )
        if (
            ctrl_mode == 0x01
            and mode_feed == 0x00
            and arm_status == 0x00
            and all(enable_status)
        ):
            return True
        count += 1
    return False


def wait_for_movej_ready(
    piper: C_PiperInterface_V2,
    speed: int,
    timeout_s: float,
) -> bool:
    deadline = time.time() + timeout_s
    count = 0
    while time.time() < deadline:
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        piper.EnableArm(7, 0x02)
        time.sleep(0.05)
        status = piper.GetArmStatus()
        enable_status = list(piper.GetArmEnableStatus())
        ctrl_mode = ctrl_mode_code(piper)
        mode_feed = mode_feed_code(piper)
        arm_status = arm_status_code(piper)
        if count % 10 == 0:
            print(
                "movej ready check: "
                f"ctrl=0x{ctrl_mode:x} mode=0x{mode_feed:x} "
                f"arm=0x{arm_status:x} hz={status.Hz:.1f} "
                f"enable={enable_status}"
            )
        if (
            ctrl_mode == 0x01
            and mode_feed == 0x01
            and arm_status == 0x00
            and all(enable_status)
        ):
            return True
        count += 1
    return False


def pose_error_mm_deg(actual: list[int], target: list[int]) -> tuple[float, float]:
    xyz_error = max(abs(actual[idx] - target[idx]) for idx in range(3)) / 1000.0
    rpy_error = max(abs(actual[idx] - target[idx]) for idx in range(3, 6)) / 1000.0
    return xyz_error, rpy_error


def send_movep_for(
    piper: C_PiperInterface_V2,
    target: list[int],
    speed: int,
    duration_s: float,
    rate_hz: float,
    label: str,
    position_tolerance_mm: float | None = None,
    rpy_tolerance_deg: float | None = None,
    require_reached: bool = False,
) -> bool:
    interval_s = 1.0 / rate_hz
    deadline = time.time() + duration_s
    count = 0
    while time.time() < deadline:
        piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
        piper.EndPoseCtrl(*target)
        time.sleep(interval_s)

        if count % max(1, int(rate_hz / 2)) == 0:
            actual = end_pose_raw(piper)
            xyz_error, rpy_error = pose_error_mm_deg(actual, target)
            enable_status = list(piper.GetArmEnableStatus())
            arm_status = arm_status_code(piper)
            motion_status = int(piper.GetArmStatus().arm_status.motion_status)
            print(
                f"\r{label}: arm=0x{arm_status:x} motion=0x{motion_status:x} "
                f"xyz_err={xyz_error:.3f}mm rpy_err={rpy_error:.3f}deg "
                f"enable={enable_status} pose=[{pose_mm_deg(actual)}]",
                end="",
                flush=True,
            )
            if not all(enable_status) or arm_status != 0x00:
                print()
                return False
            if (
                position_tolerance_mm is not None
                and rpy_tolerance_deg is not None
                and motion_status == 0x00
                and xyz_error <= position_tolerance_mm
                and rpy_error <= rpy_tolerance_deg
            ):
                print()
                return True
        count += 1
    actual = end_pose_raw(piper)
    xyz_error, rpy_error = pose_error_mm_deg(actual, target)
    reached = (
        position_tolerance_mm is not None
        and rpy_tolerance_deg is not None
        and xyz_error <= position_tolerance_mm
        and rpy_error <= rpy_tolerance_deg
    )
    state = "done" if reached or not require_reached else "timeout"
    print(
        f"{label} {state}: xyz_err={xyz_error:.3f}mm rpy_err={rpy_error:.3f}deg "
        f"pose=[{pose_mm_deg(actual)}]"
    )
    if require_reached and not reached:
        return False
    return True


def send_movej_for(
    piper: C_PiperInterface_V2,
    target: list[float],
    speed: int,
    duration_s: float,
    rate_hz: float,
    label: str,
) -> bool:
    if not joint_limits_ok(target):
        print(f"\n[warn] MOVE_J target outside limits: {fmt_joints(target)}")
        return False
    raw = [int(round(joint * 1000.0)) for joint in target]
    interval_s = 1.0 / rate_hz
    deadline = time.time() + duration_s
    while time.time() < deadline:
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        piper.JointCtrl(*raw)
        time.sleep(interval_s)
        actual = joints_deg(piper)
        enable_status = list(piper.GetArmEnableStatus())
        arm_status = arm_status_code(piper)
        error = max(abs(actual[idx] - target[idx]) for idx in range(6))
        print(
            f"\r{label}: arm=0x{arm_status:x} err={error:.3f}deg "
            f"enable={enable_status} joints={fmt_joints(actual)}",
            end="",
            flush=True,
        )
        if not all(enable_status) or arm_status != 0x00:
            print()
            return False
        if error <= 0.8:
            print()
            return True
    print()
    return True


def send_movej_once(
    piper: C_PiperInterface_V2,
    target: list[float],
    speed: int,
) -> bool:
    if not joint_limits_ok(target):
        print(f"\n[warn] MOVE_J target outside limits: {fmt_joints(target)}")
        return False
    raw = [int(round(joint * 1000.0)) for joint in target]
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    piper.JointCtrl(*raw)
    enable_status = list(piper.GetArmEnableStatus())
    arm_status = arm_status_code(piper)
    if not all(enable_status) or arm_status != 0x00:
        print(f"\n[warn] MOVE_J send failed: arm=0x{arm_status:x} enable={enable_status}")
        return False
    return True


def key_to_joint_move(key: str) -> tuple[str, int] | None:
    key = key.lower()
    if key == "w":
        return "forward", 0
    if key == "s":
        return "back", 1
    if key == "d":
        return "right", 2
    if key == "a":
        return "left", 3
    return None


def apply_joint_step(
    target: list[float],
    direction: int,
    j1_step: float,
    reach_step: float,
    reach_transition_j2: float,
    reach_pre_gains: tuple[float, float, float],
    reach_post_gains: tuple[float, float, float],
    locked_j4: float,
    locked_j5: float,
    locked_j6: float,
) -> tuple[list[float], str]:
    def apply_reach_gains(
        joints: list[float],
        signed_step: float,
        gains: tuple[float, float, float],
    ) -> None:
        joints[1] += signed_step * gains[0]
        joints[2] += signed_step * gains[1]
        joints[4] += signed_step * gains[2]

    def apply_reach(joints: list[float], sign: float) -> str:
        j2 = joints[1]
        label = "pre"
        if sign > 0:
            if j2 < reach_transition_j2:
                pre_j2_delta = min(
                    reach_step * reach_pre_gains[0],
                    reach_transition_j2 - j2,
                )
                if pre_j2_delta > 0:
                    pre_step = pre_j2_delta / reach_pre_gains[0]
                    apply_reach_gains(joints, pre_step, reach_pre_gains)
                remaining = reach_step - max(0.0, pre_j2_delta / reach_pre_gains[0])
                if remaining > 1e-6:
                    apply_reach_gains(joints, remaining, reach_post_gains)
                    label = "pre->post"
            else:
                apply_reach_gains(joints, reach_step, reach_post_gains)
                label = "post"
        else:
            if j2 > reach_transition_j2:
                post_j2_delta = min(
                    reach_step * reach_post_gains[0],
                    j2 - reach_transition_j2,
                )
                if post_j2_delta > 0:
                    post_step = post_j2_delta / reach_post_gains[0]
                    apply_reach_gains(joints, -post_step, reach_post_gains)
                remaining = reach_step - max(0.0, post_j2_delta / reach_post_gains[0])
                if remaining > 1e-6:
                    apply_reach_gains(joints, -remaining, reach_pre_gains)
                    label = "post->pre"
                else:
                    label = "post"
            else:
                apply_reach_gains(joints, -reach_step, reach_pre_gains)
                label = "pre"
        return label

    next_target = list(target)
    if direction == 0:
        phase = apply_reach(next_target, 1.0)
    elif direction == 1:
        phase = apply_reach(next_target, -1.0)
        next_target[4] = locked_j5
    elif direction == 2:
        next_target[0] += j1_step
        phase = "base"
    elif direction == 3:
        next_target[0] -= j1_step
        phase = "base"
    else:
        raise ValueError(f"bad direction: {direction}")
    next_target[3] = locked_j4
    next_target[5] = locked_j6
    return next_target, phase


def set_hand(hand: object | None, pose: dict[str, float], label: str) -> None:
    if hand is None:
        print(f"{label}: hand disabled; skipped")
        return
    hand.set_angles(pose)
    print(f"{label}: hand command sent")


def set_hand_async(hand: object | None, pose: dict[str, float], label: str) -> None:
    if hand is None:
        print(f"{label}: hand disabled; skipped")
        return

    def worker() -> None:
        try:
            hand.set_angles(pose)
            print(f"\n{label}: hand command sent")
        except Exception as exc:
            print(f"\n[warn] {label}: hand command failed: {exc}")

    threading.Thread(target=worker, daemon=True).start()


def print_state(piper: C_PiperInterface_V2) -> None:
    print()
    print(f"pose:   {pose_mm_deg(end_pose_raw(piper))}")
    print(f"joints: {fmt_joints(joints_deg(piper))}")
    print(piper.GetArmStatus())


def prompt_before_disable(piper: C_PiperInterface_V2) -> None:
    print()
    print("WARNING: disabling Piper motors may make the arm drop.")
    print("Hold/support the arm before disabling.")
    answer = input(
        "Type D then Enter to disable arm motors, or press Enter to keep motors enabled: "
    ).strip()
    if answer == "D":
        piper.DisableArm(7)
        print("Piper arm motors disabled.")
    else:
        print("Piper arm motors left enabled.")


def run_pick_cycle(
    piper: C_PiperInterface_V2,
    hand: object | None,
    args: argparse.Namespace,
    start_pose: list[int],
    hover_pose: list[int],
    drop_pose: list[int],
) -> bool:
    grab_pose = list(hover_pose)
    grab_pose[2] = int(round(args.grab_z * 1000.0))
    lift_pose = list(hover_pose)
    if args.lift_z is not None:
        lift_pose[2] = int(round(args.lift_z * 1000.0))

    print()
    print("Running pick cycle")
    print(f"  hover: {pose_mm_deg(hover_pose)}")
    print(f"  grab:  {pose_mm_deg(grab_pose)}")
    print(f"  lift:  {pose_mm_deg(lift_pose)}")
    print(f"  drop:  {pose_mm_deg(drop_pose)}")
    print(f"  start: {pose_mm_deg(start_pose)}")

    set_hand_async(hand, DEFAULT_OPEN, "open while descending")
    if not send_movep_for(
        piper,
        grab_pose,
        args.speed,
        args.vertical_duration,
        args.rate_hz,
        "descend",
        args.auto_position_tolerance_mm,
        args.auto_rpy_tolerance_deg,
        True,
    ):
        print("[warn] descend failed")
        return False
    set_hand(hand, DEFAULT_CLOSED, "close")
    time.sleep(args.hand_settle)

    if not send_movep_for(
        piper,
        lift_pose,
        args.speed,
        args.vertical_duration,
        args.rate_hz,
        "lift",
        args.auto_position_tolerance_mm,
        args.auto_rpy_tolerance_deg,
        True,
    ):
        print("[warn] lift failed")
        return False

    if not send_movep_for(
        piper,
        drop_pose,
        args.speed,
        args.transfer_duration,
        args.rate_hz,
        "drop move",
        args.auto_position_tolerance_mm,
        args.auto_rpy_tolerance_deg,
        True,
    ):
        print("[warn] drop move failed")
        return False
    set_hand(hand, DEFAULT_OPEN, "open")
    time.sleep(args.hand_settle)

    set_hand_async(hand, DEFAULT_CLOSED, "close while returning")
    if not send_movep_for(
        piper,
        start_pose,
        args.speed,
        args.return_duration,
        args.rate_hz,
        "return",
        args.auto_position_tolerance_mm,
        args.auto_rpy_tolerance_deg,
        True,
    ):
        print("[warn] return failed")
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0")
    parser.add_argument("--speed", type=int, default=3)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--feedback-timeout", type=float, default=8.0)
    parser.add_argument("--j1-step-deg", type=float, default=2.0)
    parser.add_argument("--reach-step-deg", type=float, default=2.0)
    parser.add_argument("--reach-transition-j2-deg", type=float, default=35.0)
    parser.add_argument("--reach-pre-j2-gain", type=float, default=1.0)
    parser.add_argument("--reach-pre-j3-gain", type=float, default=-1.2)
    parser.add_argument("--reach-pre-j5-gain", type=float, default=-0.15)
    parser.add_argument("--reach-post-j2-gain", type=float, default=1.0)
    parser.add_argument("--reach-post-j3-gain", type=float, default=-1.2)
    parser.add_argument("--reach-post-j5-gain", type=float, default=-0.15)
    parser.add_argument("--reach-j2-gain", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--reach-j3-gain", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--reach-j5-gain", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--nudge-duration", type=float, default=0.5)
    parser.add_argument("--start", type=parse_pose_mm_deg, default=list(DEFAULT_START_POSE))
    parser.add_argument("--start-duration", type=float, default=8.0)
    parser.add_argument("--hover-z", type=float)
    parser.add_argument("--hover-duration", type=float, default=6.0)
    parser.add_argument("--grab-z", type=float, required=True)
    parser.add_argument("--lift-z", type=float)
    parser.add_argument("--vertical-duration", type=float, default=4.0)
    parser.add_argument("--transfer-duration", type=float, default=8.0)
    parser.add_argument("--return-duration", type=float, default=8.0)
    parser.add_argument("--auto-position-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--auto-rpy-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--drop", type=parse_pose_mm_deg)
    parser.add_argument("--hand-settle", type=float, default=1.0)
    parser.add_argument("--no-hand", action="store_true")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--hand-force", type=int, default=1500)
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.speed < 0 or args.speed > 100:
        raise ValueError("--speed must be in [0, 100]")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")
    if args.reach_step_deg <= 0:
        raise ValueError("--reach-step-deg must be positive")
    legacy_reach_gains = (
        args.reach_j2_gain,
        args.reach_j3_gain,
        args.reach_j5_gain,
    )
    if any(value is not None for value in legacy_reach_gains):
        if not all(value is not None for value in legacy_reach_gains):
            raise ValueError(
                "legacy --reach-j2-gain/--reach-j3-gain/--reach-j5-gain "
                "must be passed together"
            )
        args.reach_pre_j2_gain = args.reach_j2_gain
        args.reach_pre_j3_gain = args.reach_j3_gain
        args.reach_pre_j5_gain = args.reach_j5_gain
        args.reach_post_j2_gain = args.reach_j2_gain
        args.reach_post_j3_gain = args.reach_j3_gain
        args.reach_post_j5_gain = args.reach_j5_gain
    if args.reach_pre_j2_gain <= 0 or args.reach_post_j2_gain <= 0:
        raise ValueError("reach J2 gains must be positive")

    print("Claw-machine MOVE_P teleop")
    print(HELP)
    print(f"grab Z: {args.grab_z:.3f} mm")
    print(
        "MOVE_J keyboard: "
        f"j1_step={args.j1_step_deg:.3f} reach_step={args.reach_step_deg:.3f} "
        f"transition_j2={args.reach_transition_j2_deg:.3f}"
    )
    print(
        "Reach gains: "
        f"pre=({args.reach_pre_j2_gain:.2f},{args.reach_pre_j3_gain:.2f},"
        f"{args.reach_pre_j5_gain:.2f}) "
        f"post=({args.reach_post_j2_gain:.2f},{args.reach_post_j3_gain:.2f},"
        f"{args.reach_post_j5_gain:.2f})"
    )
    print("Keyboard uses MOVE_J; pick/drop vertical cycle uses official MOVE_P.")
    if not args.yes:
        answer = input("Type YES to continue: ").strip()
        if answer != "YES":
            print("Aborted.")
            return 1

    hand = None
    if not args.no_hand:
        if RH56F2Hand is None or RH56F2HandConfig is None:
            raise RuntimeError("RH56F2 hand module could not be imported")
        hand = RH56F2Hand(
            RH56F2HandConfig(
                port=args.hand_port,
                hand_id=args.hand_id,
                speed=args.hand_speed,
                force=args.hand_force,
            )
        )
        hand.connect()
        print("RH56F2 hand connected.")
        set_hand(hand, DEFAULT_CLOSED, "initial close")
        time.sleep(args.hand_settle)

    piper = C_PiperInterface_V2(
        args.can,
        judge_flag=False,
        can_auto_init=False,
        dh_is_offset=1,
        start_sdk_fk_cal=True,
    )
    piper.ConnectPort()
    time.sleep(1.0)

    try:
        wait_for_real_feedback(piper, args.feedback_timeout)
        print_state(piper)
        print("Selecting CAN_CTRL + MOVE_P for setup moves...")
        if not enable_all(piper, args.feedback_timeout):
            print("[warn] Arm did not enable.")
            print_state(piper)
            return 1
        if not wait_for_movep_ready(piper, args.speed, args.feedback_timeout):
            print("[warn] MOVE_P did not become ready. Reset the arm if Arm Status is 0x4.")
            print_state(piper)
            return 1

        start_pose = list(args.start) if args.start is not None else end_pose_raw(piper)
        if args.start is not None:
            print(f"Moving to configured start pose: {pose_mm_deg(start_pose)}")
            if not send_movep_for(
                piper,
                start_pose,
                args.speed,
                args.start_duration,
                args.rate_hz,
                "start",
            ):
                print("[warn] start move failed")
                print_state(piper)
                return 1

        keyboard_pose = end_pose_raw(piper)
        if args.hover_z is not None:
            keyboard_pose = list(keyboard_pose)
            keyboard_pose[2] = int(round(args.hover_z * 1000.0))
            print(f"Moving to keyboard hover Z: {pose_mm_deg(keyboard_pose)}")
            if not send_movep_for(
                piper,
                keyboard_pose,
                args.speed,
                args.hover_duration,
                args.rate_hz,
                "hover",
            ):
                print("[warn] hover move failed")
                print_state(piper)
                return 1

        print("Switching to MOVE_J for keyboard control...")
        if not wait_for_movej_ready(piper, args.speed, args.feedback_timeout):
            print("[warn] MOVE_J did not become ready.")
            print_state(piper)
            return 1
        joint_target = joints_deg(piper)
        locked_j4 = joint_target[3]
        locked_j5 = joint_target[4]
        locked_j6 = joint_target[5]

        drop_pose = list(args.drop) if args.drop is not None else list(start_pose)
        print(f"Captured start pose: {pose_mm_deg(start_pose)}")
        print(f"Keyboard pose:       {pose_mm_deg(keyboard_pose)}")
        print(f"Keyboard joints:     {fmt_joints(joint_target)}")
        print(
            "Locked joints:       "
            f"J4={locked_j4:.3f} J5-back={locked_j5:.3f} J6={locked_j6:.3f}"
        )
        print(f"Drop pose:           {pose_mm_deg(drop_pose)}")
        set_hand(hand, DEFAULT_CLOSED, "keyboard close")
        print("Keyboard control is active.")
        print("Raw keyboard mode is active: typed keys are not echoed by the terminal.")
        print("Use WASD; each accepted key will print a MOVE_J target.")

        with RawTerminal():
            while True:
                key = read_key(0.1)
                if key is None:
                    continue
                key_lower = key.lower()
                if key_lower == "q":
                    print("\nquit")
                    break
                if key_lower == "p":
                    print_state(piper)
                    continue
                if key_lower == "r":
                    joint_target = joints_deg(piper)
                    locked_j4 = joint_target[3]
                    locked_j5 = joint_target[4]
                    locked_j6 = joint_target[5]
                    print(f"\nreference reset joints: {fmt_joints(joint_target)}")
                    continue
                if key == " ":
                    if not wait_for_movep_ready(piper, args.speed, args.feedback_timeout):
                        print("[warn] MOVE_P did not become ready for pick cycle.")
                        print_state(piper)
                        break
                    hover_pose = end_pose_raw(piper)
                    ok = run_pick_cycle(piper, hand, args, start_pose, hover_pose, drop_pose)
                    if not ok and arm_status_code(piper) != 0x00:
                        print("[warn] arm is in error status; reset the arm before retrying.")
                        break
                    if not wait_for_movej_ready(piper, args.speed, args.feedback_timeout):
                        print("[warn] MOVE_J did not become ready after pick cycle.")
                        break
                    joint_target = joints_deg(piper)
                    locked_j4 = joint_target[3]
                    locked_j5 = joint_target[4]
                    locked_j6 = joint_target[5]
                    print(f"cycle {'complete' if ok else 'stopped'}; current joints={fmt_joints(joint_target)}")
                    continue

                move = key_to_joint_move(key)
                if move is None:
                    print(f"\nignored key: {key!r}")
                    continue
                label, direction = move
                next_target, phase = apply_joint_step(
                    joint_target,
                    direction,
                    args.j1_step_deg,
                    args.reach_step_deg,
                    args.reach_transition_j2_deg,
                    (
                        args.reach_pre_j2_gain,
                        args.reach_pre_j3_gain,
                        args.reach_pre_j5_gain,
                    ),
                    (
                        args.reach_post_j2_gain,
                        args.reach_post_j3_gain,
                        args.reach_post_j5_gain,
                    ),
                    locked_j4,
                    locked_j5,
                    locked_j6,
                )

                joint_target = next_target
                print(f"\rkey {label} [{phase}]: target joints {fmt_joints(joint_target)}", end="", flush=True)
                if not send_movej_once(piper, joint_target, args.speed):
                    print("[warn] nudge failed; stop sending commands and reset if needed.")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted. Motors were not disabled by this script.")
    finally:
        if hand is not None:
            try:
                hand.disconnect()
            except Exception as exc:
                print(f"[warn] hand disconnect failed: {exc}")
        try:
            prompt_before_disable(piper)
        except Exception as exc:
            print(f"[warn] final disable prompt failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
