"""Prepare Piper CAN control at Web UI boot, and send DisableArm on exit."""

from __future__ import annotations

import time

from lerobot_robot_piper.piper_follower import load_piper_interface_v2

load_piper_interface_v2()

from lerobot_robot_piper.assistant.recover_piper_control_state import (
    enable_until_feedback_true,
    recover_control_mode,
    wait_for_any_feedback,
    wait_for_can_ctrl,
)
from lerobot_robot_piper.assistant.restore_piper_normal_mode import apply_restore

DEFAULT_DISABLE_EXIT_JOINTS = [0.090, 0.000, 0.000, 1.678, 1.957, 0.380]
DEFAULT_DISABLE_EXIT_SPEED = 8
DEFAULT_DISABLE_EXIT_DURATION_S = 8.0
DEFAULT_DISABLE_EXIT_SETTLE_S = 0.5
TEACH_CTRL_MODES = {0x02, 0x06, 0x07}
IDLE_TEACH_STATES = {0x00, 0x02, 0x06}


def _enum_int(value: object) -> int:
    if hasattr(value, "value"):
        try:
            return int(value.value)
        except (TypeError, ValueError):
            pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def connect_piper(can_port: str):
    C_PiperInterface_V2 = load_piper_interface_v2()
    piper = C_PiperInterface_V2(
        can_port,
        judge_flag=False,
        can_auto_init=False,
        dh_is_offset=1,
        start_sdk_fk_cal=True,
    )
    piper.ConnectPort()
    time.sleep(0.4)
    return piper


def is_teach_mode(piper) -> bool:
    status = piper.GetArmStatus().arm_status
    ctrl_mode = _enum_int(status.ctrl_mode)
    teach_status = _enum_int(status.teach_status)
    print(f"Piper boot status: ctrl_mode={status.ctrl_mode} teach_status={status.teach_status}")
    return ctrl_mode in TEACH_CTRL_MODES or teach_status not in IDLE_TEACH_STATES


def prepare_piper_can_control(can_port: str) -> None:
    piper = connect_piper(can_port)
    try:
        if not wait_for_any_feedback(piper, 3.0):
            print("[warn] no Piper feedback after setup_can0; still checking teach/CAN mode.")
        if not is_teach_mode(piper):
            print("Piper is not in teach mode; skip restore/recover.")
            return
        print("Teach mode detected; restoring normal CAN control.")
        apply_restore(piper, repeats=50, interval=0.02, wait=3.0)
        recover_control_mode(piper, speed=5, installation_pos=0x01)
        if not wait_for_can_ctrl(piper, speed=5, installation_pos=0x01, timeout_s=8.0):
            raise RuntimeError("Control Mode did not become CAN_CTRL after teach restore")
        enable_status = enable_until_feedback_true(piper, timeout_s=8.0)
        print(f"after teach restore enable: {enable_status}")
        for _ in range(20):
            piper.MotionCtrl_1(0x02, 0x00, 0x00)
            piper.MotionCtrl_2(0x01, 0x01, 5, 0x00, 0, 0x01)
            time.sleep(0.02)
        print("Piper is in CAN control mode.")
    finally:
        piper.DisconnectPort()


def run_d_disable_flow(
    can_port: str,
    joints: list[float] | None = None,
    speed: int = DEFAULT_DISABLE_EXIT_SPEED,
    duration_s: float = DEFAULT_DISABLE_EXIT_DURATION_S,
    settle_s: float = DEFAULT_DISABLE_EXIT_SETTLE_S,
    rate_hz: float = 25.0,
) -> None:
    target = list(DEFAULT_DISABLE_EXIT_JOINTS if joints is None else joints)
    piper = connect_piper(can_port)
    try:
        print("D-flow disable: moving to configured safe disable joints, then DisableArm.")
        print("disable-exit joints: " + " ".join(f"{value:8.3f}" for value in target))
        raw = [int(round(value * 1000.0)) for value in target]
        deadline = time.monotonic() + max(duration_s, 0.1)
        interval_s = 1.0 / max(rate_hz, 1.0)
        while time.monotonic() < deadline:
            piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
            piper.JointCtrl(*raw)
            time.sleep(interval_s)
        print("Disable-exit move complete; sending DisableArm(7).")
        piper.DisableArm(7)
        time.sleep(max(settle_s, 0.0))
        try:
            print(f"Piper enable status after D-flow DisableArm: {list(piper.GetArmEnableStatus())}")
        except Exception as exc:
            print(f"[warn] failed reading enable status after D-flow DisableArm: {exc}")
    finally:
        piper.DisconnectPort()
