#!/usr/bin/env python3
"""Run D405 YOLO ball tests in a fixed model order."""

from __future__ import annotations

import argparse
import subprocess
import sys


MODEL_ORDER = [
    ("yolo11s.pt", 0.25, 960),
    ("yolo11m.pt", 0.25, 960),
    ("yolo26s.pt", 0.25, 960),
    ("yolo26m.pt", 0.25, 960),
    ("yolo11x.pt", 0.25, 1280),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="260322279862")
    parser.add_argument("--start-index", type=int, default=1, help="1-based model index")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.start_index <= len(MODEL_ORDER):
        raise SystemExit(f"--start-index must be between 1 and {len(MODEL_ORDER)}")

    for index, (model, conf, imgsz) in enumerate(MODEL_ORDER[args.start_index - 1 :], start=args.start_index):
        command = [
            args.python,
            "test_yolo_d405_ball.py",
            "--serial",
            args.serial,
            "--model",
            model,
            "--conf",
            str(conf),
            "--imgsz",
            str(imgsz),
        ]
        print()
        print(f"[{index}/{len(MODEL_ORDER)}] {' '.join(command)}")
        print("Close with q in the OpenCV window to return here.")
        completed = subprocess.run(command, check=False)
        print(f"{model} exited with code {completed.returncode}")
        if index == len(MODEL_ORDER):
            break
        answer = input("Try next model? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
