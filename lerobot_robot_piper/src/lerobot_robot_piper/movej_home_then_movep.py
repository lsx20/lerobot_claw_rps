"""Shared Piper MOVE_J/MOVE_P utility functions.

Several standalone hardware test scripts import these helpers with a bare
``from movej_home_then_movep import ...`` because they are executed directly
from this directory. Keep this module dependency-light and hardware-focused.
"""

from __future__ import annotations

import time
from typing import Any


def arm_status_code(piper: Any) -> int:
    return int(piper.GetArmStatus().arm_status.arm_status)


def ctrl_mode_code(piper: Any) -> int:
    return int(piper.GetArmStatus().arm_status.ctrl_mode)


def mode_feed_code(piper: Any) -> int:
    return int(piper.GetArmStatus().arm_status.mode_feed)


def end_pose_raw(piper: Any) -> list[int]:
    ep = piper.GetArmEndPoseMsgs().end_pose
    return [ep.X_axis, ep.Y_axis, ep.Z_axis, ep.RX_axis, ep.RY_axis, ep.RZ_axis]


def joints_deg(piper: Any) -> list[float]:
    js = piper.GetArmJointMsgs().joint_state
    return [
        js.joint_1 / 1000.0,
        js.joint_2 / 1000.0,
        js.joint_3 / 1000.0,
        js.joint_4 / 1000.0,
        js.joint_5 / 1000.0,
        js.joint_6 / 1000.0,
    ]


def fmt(values: list[float]) -> str:
    return " ".join(f"{value:8.3f}" for value in values)


def status_feedback_is_fresh(status: Any, first_status_time: float) -> bool:
    stamp = getattr(status, "time_stamp", None)
    if stamp is None:
        return True
    return float(stamp) != float(first_status_time)


def has_real_feedback(piper: Any) -> bool:
    status = piper.GetArmStatus()
    pose = piper.GetArmEndPoseMsgs()
    joints = piper.GetArmJointMsgs()
    ep = pose.end_pose
    js = joints.joint_state
    return (
        getattr(status, "Hz", 0) > 0
        or getattr(pose, "Hz", 0) > 0
        or getattr(joints, "Hz", 0) > 0
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


def wait_for_real_feedback(piper: Any, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if has_real_feedback(piper):
            return
        time.sleep(0.05)
    raise RuntimeError("No real Piper feedback received; refusing to command motion.")


def enable_all(piper: Any, attempts: int = 120, interval_s: float = 0.02) -> bool:
    last_status: list[bool] = []
    for count in range(attempts):
        piper.EnableArm(7, 0x02)
        time.sleep(interval_s)
        last_status = list(piper.GetArmEnableStatus())
        if count % 10 == 0:
            print(f"enable status: {last_status}")
        if last_status and all(last_status):
            return True
    print(f"[warn] enable failed; final enable status={last_status}")
    return False


def select_mode(
    piper: Any,
    move_mode: int,
    speed: int,
    installation_pos: int,
    count: int = 50,
    interval_s: float = 0.02,
) -> None:
    for _ in range(count):
        piper.MotionCtrl_2(0x01, move_mode, speed, 0x00, 0, installation_pos)
        time.sleep(interval_s)


def wait_for_mode_ready(
    piper: Any,
    move_mode: int,
    speed: int,
    installation_pos: int,
    timeout_s: float,
) -> bool:
    deadline = time.time() + timeout_s
    first_status_time = float(getattr(piper.GetArmStatus(), "time_stamp", 0.0))
    count = 0
    while time.time() < deadline:
        piper.MotionCtrl_2(0x01, move_mode, speed, 0x00, 0, installation_pos)
        piper.EnableArm(7, 0x02)
        time.sleep(0.05)
        status = piper.GetArmStatus()
        enable_status = list(piper.GetArmEnableStatus())
        ctrl_mode = ctrl_mode_code(piper)
        mode_feed = mode_feed_code(piper)
        arm_status = arm_status_code(piper)
        fresh = status_feedback_is_fresh(status, first_status_time)
        if count % 10 == 0:
            print(
                "mode ready check: "
                f"ctrl=0x{ctrl_mode:x} mode=0x{mode_feed:x} "
                f"arm=0x{arm_status:x} hz={status.Hz:.1f} "
                f"fresh={fresh} enable={enable_status}"
            )
        if (
            fresh
            and ctrl_mode == 0x01
            and mode_feed == move_mode
            and arm_status == 0x00
            and enable_status
            and all(enable_status)
        ):
            return True
        count += 1
    return False


def _pose_mm_deg(pose: list[int]) -> str:
    return (
        f"X={pose[0] / 1000.0:8.3f} Y={pose[1] / 1000.0:8.3f} "
        f"Z={pose[2] / 1000.0:8.3f} RX={pose[3] / 1000.0:8.3f} "
        f"RY={pose[4] / 1000.0:8.3f} RZ={pose[5] / 1000.0:8.3f}"
    )


def print_status(piper: Any, label: str) -> None:
    status = piper.GetArmStatus()
    print(
        f"{label}: ctrl=0x{ctrl_mode_code(piper):x} "
        f"mode=0x{mode_feed_code(piper):x} "
        f"arm=0x{arm_status_code(piper):x} "
        f"hz={getattr(status, 'Hz', 0.0):.1f} "
        f"enable={list(piper.GetArmEnableStatus())} "
        f"joints={fmt(joints_deg(piper))} "
        f"pose=[{_pose_mm_deg(end_pose_raw(piper))}]"
    )


def print_driver_summary(piper: Any, label: str) -> None:
    print_status(piper, label)


def prompt_before_disable(piper: Any) -> None:
    print()
    print("WARNING: disabling Piper motors may make the arm drop.")
    print("Hold/support the arm and hand before disabling.")
    confirm = input(
        "Type D then Enter to disable arm motors, or press Enter to keep motors enabled: "
    ).strip()
    if confirm == "D":
        piper.DisableArm(7)
        print("Piper arm motors disabled.")
    else:
        print("Piper arm motors left enabled.")
