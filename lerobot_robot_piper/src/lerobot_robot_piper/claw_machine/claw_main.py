#!/usr/bin/env python3
"""Modular claw-machine controller entrypoint."""

from __future__ import annotations

import sys
import time

from claw_arm_grasp import run_pick_cycle
from claw_arm_keyboard import (
    apply_joint_step,
    capture_keyboard_reference,
    key_to_joint_move,
    send_movej_once,
)
from claw_hand_keyboard import close_for_keyboard, connect_hand, disconnect_hand
from claw_gamepad import run_gamepad_loop
from claw_init import (
    HELP,
    RawTerminal,
    arm_status_code,
    build_arg_parser,
    connect_piper,
    end_pose_raw,
    fmt_joints,
    pose_mm_deg,
    print_state,
    prompt_before_disable,
    read_key,
    send_movep_for,
    validate_args,
    wait_for_movej_ready,
    wait_for_movep_ready,
    wait_for_real_feedback,
    enable_all,
)


def move_to_start_and_hover(piper, args) -> tuple[list[int], list[int]]:
    print("Selecting CAN_CTRL + MOVE_P for setup moves...")
    if not enable_all(piper, args.feedback_timeout):
        raise RuntimeError("Arm did not enable.")
    if not wait_for_movep_ready(piper, args.speed, args.feedback_timeout):
        raise RuntimeError("MOVE_P did not become ready. Reset if Arm Status is 0x4.")

    start_pose = list(args.start)
    print(f"Moving to configured start pose: {pose_mm_deg(start_pose)}")
    if not send_movep_for(
        piper,
        start_pose,
        args.speed,
        args.start_duration,
        args.rate_hz,
        "start",
    ):
        raise RuntimeError("start move failed")

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
            raise RuntimeError("hover move failed")
    return start_pose, keyboard_pose


def run_keyboard_loop(piper, hand, args, start_pose: list[int], keyboard_pose: list[int]) -> None:
    print("Switching to MOVE_J for keyboard control...")
    if not wait_for_movej_ready(piper, args.speed, args.feedback_timeout):
        raise RuntimeError("MOVE_J did not become ready.")

    joint_target, locked_j4, locked_j5, locked_j6 = capture_keyboard_reference(piper)
    drop_pose = list(args.drop) if args.drop is not None else list(start_pose)

    print(f"Captured start pose: {pose_mm_deg(start_pose)}")
    print(f"Keyboard pose:       {pose_mm_deg(keyboard_pose)}")
    print(f"Keyboard joints:     {fmt_joints(joint_target)}")
    print(
        "Locked joints:       "
        f"J4={locked_j4:.3f} J5-back={locked_j5:.3f} J6={locked_j6:.3f}"
    )
    print(f"Drop pose:           {pose_mm_deg(drop_pose)}")
    close_for_keyboard(hand)
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
                joint_target, locked_j4, locked_j5, locked_j6 = capture_keyboard_reference(piper)
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
                    print("[warn] arm is in error status; reset before retrying.")
                    break
                if not wait_for_movej_ready(piper, args.speed, args.feedback_timeout):
                    print("[warn] MOVE_J did not become ready after pick cycle.")
                    break
                joint_target, locked_j4, locked_j5, locked_j6 = capture_keyboard_reference(piper)
                print(f"cycle {'complete' if ok else 'stopped'}; current joints={fmt_joints(joint_target)}")
                continue

            move = key_to_joint_move(key)
            if move is None:
                print(f"\nignored key: {key!r}")
                continue

            label, direction = move
            joint_target, phase = apply_joint_step(
                joint_target,
                direction,
                args,
                locked_j4,
                locked_j5,
                locked_j6,
            )
            print(
                f"\rkey {label} [{phase}]: target joints {fmt_joints(joint_target)}",
                end="",
                flush=True,
            )
            if not send_movej_once(piper, joint_target, args.speed):
                print("[warn] nudge failed; stop sending commands and reset if needed.")
                break


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    print("Modular claw-machine teleop")
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
    print(f"{args.control.capitalize()} uses MOVE_J; pick/drop vertical cycle uses official MOVE_P.")
    if not args.yes:
        answer = input("Type YES to continue: ").strip()
        if answer != "YES":
            print("Aborted.")
            return 1

    hand = connect_hand(args)
    time.sleep(args.hand_settle)
    piper = connect_piper(args)

    try:
        wait_for_real_feedback(piper, args.feedback_timeout)
        print_state(piper)
        start_pose, keyboard_pose = move_to_start_and_hover(piper, args)
        if args.control == "gamepad":
            run_gamepad_loop(piper, hand, args, start_pose, keyboard_pose)
        else:
            run_keyboard_loop(piper, hand, args, start_pose, keyboard_pose)
    except KeyboardInterrupt:
        print("\nInterrupted. Motors were not disabled by this script.")
    except Exception as exc:
        print(f"\n[warn] {exc}")
        try:
            print_state(piper)
        except Exception:
            pass
        return 1
    finally:
        disconnect_hand(hand)
        try:
            prompt_before_disable(piper)
        except Exception as exc:
            print(f"[warn] final disable prompt failed: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
