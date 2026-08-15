#!/usr/bin/env python3
"""Arm-side MOVE_P pick/drop cycle for the claw-machine workflow."""

from __future__ import annotations

import time
from argparse import Namespace

from piper_sdk import C_PiperInterface_V2

from claw_hand_grasp import (
    close_at_grab,
    close_while_returning,
    object_held_by_force,
    open_at_drop,
    open_while_descending,
    show_thumb_gesture,
)
from claw_arm_keyboard import send_movej_once
from claw_init import fmt_joints, joints_deg, pose_mm_deg, send_movep_for, wait_for_movej_ready


def send_movej_for(
    piper: C_PiperInterface_V2,
    target: list[float],
    speed: int,
    duration_s: float,
    rate_hz: float,
    label: str,
) -> bool:
    deadline = time.time() + duration_s
    interval_s = 1.0 / rate_hz
    while time.time() < deadline:
        if not send_movej_once(piper, target, speed):
            return False
        time.sleep(interval_s)
    print(f"{label}: joints {fmt_joints(target)}")
    return True


def run_result_gesture(
    piper: C_PiperInterface_V2,
    hand: object | None,
    args: Namespace,
    held: bool,
) -> bool:
    if not args.result_gesture:
        return True
    if not wait_for_movej_ready(piper, args.speed, args.feedback_timeout):
        print("[warn] MOVE_J did not become ready for result gesture.")
        return False

    original = joints_deg(piper)
    gesture = list(original)
    gesture[1] -= args.result_gesture_j2_back_deg
    gesture[5] += args.result_gesture_j6_deg if held else -args.result_gesture_j6_deg
    label = "success thumbs-up" if held else "empty thumbs-down"

    print(f"Result gesture: {label}")
    print(f"  original joints: {fmt_joints(original)}")
    print(f"  gesture joints:  {fmt_joints(gesture)}")

    show_thumb_gesture(hand, args.result_thumb_speed)
    if args.result_thumb_settle > 0:
        time.sleep(args.result_thumb_settle)

    if not send_movej_for(
        piper,
        gesture,
        args.result_gesture_speed,
        args.result_gesture_duration,
        args.rate_hz,
        label,
    ):
        return False
    if args.result_gesture_hold_after > 0:
        if not send_movej_for(
            piper,
            gesture,
            args.result_gesture_speed,
            args.result_gesture_hold_after,
            args.rate_hz,
            "result gesture hold",
        ):
            return False
    if not send_movej_for(
        piper,
        original,
        args.result_gesture_speed,
        args.result_gesture_return_duration,
        args.rate_hz,
        "result gesture return",
    ):
        return False
    close_while_returning(hand)
    return True


def run_pick_cycle(
    piper: C_PiperInterface_V2,
    hand: object | None,
    args: Namespace,
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

    open_while_descending(hand, args.pre_grab_open_speed)
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

    if args.pre_grab_open_settle > 0:
        time.sleep(args.pre_grab_open_settle)
    close_at_grab(hand, args.hand_speed)
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

    held_at_drop = object_held_by_force(
        hand,
        args.held_force_threshold,
        args.held_force_fingers,
        args.held_force_alt_fingers,
        args.held_check_duration,
        args.held_check_rate_hz,
        args.held_required_samples,
    )
    open_at_drop(hand)
    time.sleep(args.drop_open_settle)

    close_while_returning(hand)
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
    if not run_result_gesture(piper, hand, args, held_at_drop):
        print("[warn] result gesture failed")
        return False
    return True
