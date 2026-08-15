#!/usr/bin/env python3
"""Shared initialization and Piper helpers for the claw-machine workflow."""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty

from piper_sdk import C_PiperInterface_V2


DEFAULT_START_POSE = [115297, 1540, 286435, -178229, 62125, -177491]
JOINT_LIMITS_DEG = [
    (-150.0, 150.0),
    (0.0, 180.0),
    (-170.0, 0.0),
    (-100.0, 100.0),
    (-70.0, 70.0),
    (-120.0, 120.0),
]

HELP = """
Controls

  w / s       reach forward / back with J2,J3,J5
  a / d       J1 left / right
  space       open while descending, close at grab height, lift/drop, open, close while returning
  p           print current pose
  r           reset the teleop reference to current joints
  q           quit

Use Ctrl-C only if the key loop is unresponsive.
"""


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


def parse_name_list(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("expected comma-separated names")
    return names


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


def fmt_joints(values: list[float]) -> str:
    return " ".join(f"{value:8.3f}" for value in values)


def joint_limits_ok(joints: list[float]) -> bool:
    return all(
        lo <= value <= hi
        for value, (lo, hi) in zip(joints, JOINT_LIMITS_DEG, strict=True)
    )


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


def wait_for_mode_ready(
    piper: C_PiperInterface_V2,
    move_mode: int,
    speed: int,
    timeout_s: float,
    label: str,
) -> bool:
    deadline = time.time() + timeout_s
    count = 0
    while time.time() < deadline:
        piper.MotionCtrl_2(0x01, move_mode, speed, 0x00)
        piper.EnableArm(7, 0x02)
        time.sleep(0.05)
        status = piper.GetArmStatus()
        enable_status = list(piper.GetArmEnableStatus())
        ctrl_mode = ctrl_mode_code(piper)
        mode_feed = mode_feed_code(piper)
        arm_status = arm_status_code(piper)
        if count % 10 == 0:
            print(
                f"{label} ready check: "
                f"ctrl=0x{ctrl_mode:x} mode=0x{mode_feed:x} "
                f"arm=0x{arm_status:x} hz={status.Hz:.1f} enable={enable_status}"
            )
        if (
            ctrl_mode == 0x01
            and mode_feed == move_mode
            and arm_status == 0x00
            and all(enable_status)
        ):
            return True
        count += 1
    return False


def wait_for_movep_ready(
    piper: C_PiperInterface_V2,
    speed: int,
    timeout_s: float,
) -> bool:
    return wait_for_mode_ready(piper, 0x00, speed, timeout_s, "movep")


def wait_for_movej_ready(
    piper: C_PiperInterface_V2,
    speed: int,
    timeout_s: float,
) -> bool:
    return wait_for_mode_ready(piper, 0x01, speed, timeout_s, "movej")


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


def connect_piper(args: argparse.Namespace) -> C_PiperInterface_V2:
    piper = C_PiperInterface_V2(
        args.can,
        judge_flag=False,
        can_auto_init=False,
        dh_is_offset=1,
        start_sdk_fk_cal=True,
    )
    piper.ConnectPort()
    time.sleep(1.0)
    return piper


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0")
    parser.add_argument("--speed", type=int, default=3)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--feedback-timeout", type=float, default=8.0)
    parser.add_argument("--j1-step-deg", type=float, default=2.0)
    parser.add_argument("--reach-step-deg", type=float, default=2.0)
    parser.add_argument("--reach-transition-j2-deg", type=float, default=90.0)
    parser.add_argument("--reach-pre-j2-gain", type=float, default=1.0)
    parser.add_argument("--reach-pre-j3-gain", type=float, default=-0.85)
    parser.add_argument("--reach-pre-j5-gain", type=float, default=-0.05)
    parser.add_argument("--reach-post-j2-gain", type=float, default=1.0)
    parser.add_argument("--reach-post-j3-gain", type=float, default=-1.2)
    parser.add_argument("--reach-post-j5-gain", type=float, default=-0.15)
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
    parser.add_argument("--pre-grab-open-settle", type=float, default=1.0)
    parser.add_argument("--drop-open-settle", type=float, default=4.0)
    parser.add_argument("--no-hand", action="store_true")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--pre-grab-open-speed", type=int, default=1800)
    parser.add_argument("--hand-force", type=int, default=1500)
    parser.add_argument("--held-force-threshold", type=float, default=130.0)
    parser.add_argument(
        "--held-force-fingers",
        type=parse_name_list,
        default=["thumb_bend", "thumb_swing", "index", "middle"],
    )
    parser.add_argument(
        "--held-force-alt-fingers",
        type=parse_name_list,
        default=["thumb_bend", "thumb_swing", "index", "ring"],
    )
    parser.add_argument("--held-check-duration", type=float, default=1.0)
    parser.add_argument("--held-check-rate-hz", type=float, default=5.0)
    parser.add_argument("--held-required-samples", type=int, default=3)
    parser.add_argument("--result-gesture", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--result-gesture-speed", type=int, default=20)
    parser.add_argument("--result-gesture-j2-back-deg", type=float, default=30.0)
    parser.add_argument("--result-gesture-j6-deg", type=float, default=90.0)
    parser.add_argument("--result-thumb-speed", type=int, default=2500)
    parser.add_argument("--result-thumb-settle", type=float, default=0.8)
    parser.add_argument("--result-gesture-duration", type=float, default=6.0)
    parser.add_argument("--result-gesture-hold-after", type=float, default=2.0)
    parser.add_argument("--result-gesture-return-duration", type=float, default=2.5)
    parser.add_argument("--control", choices=("keyboard", "gamepad"), default="keyboard")
    parser.add_argument("--gamepad-device", default="/dev/input/js0")
    parser.add_argument("--gamepad-deadzone", type=float, default=0.18)
    parser.add_argument("--gamepad-axis-x", type=int, default=0)
    parser.add_argument("--gamepad-axis-y", type=int, default=1)
    parser.add_argument("--gamepad-invert-y", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gamepad-j1-speed-dps", type=float, default=8.0)
    parser.add_argument("--gamepad-reach-speed-dps", type=float, default=6.0)
    parser.add_argument("--gamepad-axis-curve", type=float, default=1.8)
    parser.add_argument("--gamepad-print-interval", type=float, default=0.2)
    parser.add_argument("--gamepad-lead-limit-deg", type=float, default=1.5)
    parser.add_argument("--gamepad-stop-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--yes", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.speed < 0 or args.speed > 100:
        raise ValueError("--speed must be in [0, 100]")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")
    if args.reach_step_deg <= 0:
        raise ValueError("--reach-step-deg must be positive")
    if args.reach_pre_j2_gain <= 0 or args.reach_post_j2_gain <= 0:
        raise ValueError("reach J2 gains must be positive")
    if args.hand_settle < 0 or args.pre_grab_open_settle < 0 or args.drop_open_settle < 0:
        raise ValueError("hand settle times must be non-negative")
    if args.hand_speed <= 0 or args.pre_grab_open_speed <= 0:
        raise ValueError("hand speeds must be positive")
    if args.held_force_threshold < 0:
        raise ValueError("--held-force-threshold must be non-negative")
    if args.held_check_duration <= 0 or args.held_check_rate_hz <= 0:
        raise ValueError("held check duration/rate must be positive")
    if args.held_required_samples <= 0:
        raise ValueError("--held-required-samples must be positive")
    valid_hand_names = {"little", "ring", "middle", "index", "thumb_bend", "thumb_swing"}
    invalid_force_names = set(args.held_force_fingers) - valid_hand_names
    if invalid_force_names:
        raise ValueError(f"bad --held-force-fingers names: {sorted(invalid_force_names)}")
    invalid_alt_force_names = set(args.held_force_alt_fingers) - valid_hand_names
    if invalid_alt_force_names:
        raise ValueError(f"bad --held-force-alt-fingers names: {sorted(invalid_alt_force_names)}")
    if args.result_gesture_speed < 0 or args.result_gesture_speed > 100:
        raise ValueError("--result-gesture-speed must be in [0, 100]")
    if args.result_thumb_speed <= 0:
        raise ValueError("--result-thumb-speed must be positive")
    if args.result_gesture_j2_back_deg < 0 or args.result_gesture_j6_deg < 0:
        raise ValueError("result gesture angles must be non-negative")
    if (
        args.result_thumb_settle < 0
        or args.result_gesture_duration < 0
        or args.result_gesture_hold_after < 0
        or args.result_gesture_return_duration < 0
    ):
        raise ValueError("result gesture durations must be non-negative")
    if not 0.0 <= args.gamepad_deadzone < 1.0:
        raise ValueError("--gamepad-deadzone must be in [0, 1)")
    if args.gamepad_j1_speed_dps <= 0 or args.gamepad_reach_speed_dps <= 0:
        raise ValueError("gamepad speed values must be positive")
    if args.gamepad_axis_curve < 1.0:
        raise ValueError("--gamepad-axis-curve must be >= 1")
    if args.gamepad_print_interval < 0:
        raise ValueError("--gamepad-print-interval must be non-negative")
    if args.gamepad_lead_limit_deg < 0:
        raise ValueError("--gamepad-lead-limit-deg must be non-negative")
