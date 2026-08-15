#!/usr/bin/env python3
"""Arm-side keyboard MOVE_J controls for the claw-machine workflow."""

from __future__ import annotations

from argparse import Namespace

from piper_sdk import C_PiperInterface_V2

from claw_init import (
    arm_status_code,
    fmt_joints,
    joint_limits_ok,
    joints_deg,
)


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
    args: Namespace,
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
        reach_step = args.reach_step_deg
        transition = args.reach_transition_j2_deg
        pre_gains = (
            args.reach_pre_j2_gain,
            args.reach_pre_j3_gain,
            args.reach_pre_j5_gain,
        )
        post_gains = (
            args.reach_post_j2_gain,
            args.reach_post_j3_gain,
            args.reach_post_j5_gain,
        )

        label = "pre"
        if sign > 0:
            if j2 < transition:
                pre_j2_delta = min(reach_step * pre_gains[0], transition - j2)
                if pre_j2_delta > 0:
                    pre_step = pre_j2_delta / pre_gains[0]
                    apply_reach_gains(joints, pre_step, pre_gains)
                remaining = reach_step - max(0.0, pre_j2_delta / pre_gains[0])
                if remaining > 1e-6:
                    apply_reach_gains(joints, remaining, post_gains)
                    label = "pre->post"
            else:
                apply_reach_gains(joints, reach_step, post_gains)
                label = "post"
        else:
            if j2 > transition:
                post_j2_delta = min(reach_step * post_gains[0], j2 - transition)
                if post_j2_delta > 0:
                    post_step = post_j2_delta / post_gains[0]
                    apply_reach_gains(joints, -post_step, post_gains)
                remaining = reach_step - max(0.0, post_j2_delta / post_gains[0])
                if remaining > 1e-6:
                    apply_reach_gains(joints, -remaining, pre_gains)
                    label = "post->pre"
                else:
                    label = "post"
            else:
                apply_reach_gains(joints, -reach_step, pre_gains)
                label = "pre"
        return label

    next_target = list(target)
    if direction == 0:
        phase = apply_reach(next_target, 1.0)
    elif direction == 1:
        phase = apply_reach(next_target, -1.0)
        next_target[4] = locked_j5
    elif direction == 2:
        next_target[0] += args.j1_step_deg
        phase = "base"
    elif direction == 3:
        next_target[0] -= args.j1_step_deg
        phase = "base"
    else:
        raise ValueError(f"bad direction: {direction}")

    next_target[3] = locked_j4
    next_target[5] = locked_j6
    return next_target, phase


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


def capture_keyboard_reference(
    piper: C_PiperInterface_V2,
) -> tuple[list[float], float, float, float]:
    joint_target = joints_deg(piper)
    return joint_target, joint_target[3], joint_target[4], joint_target[5]
