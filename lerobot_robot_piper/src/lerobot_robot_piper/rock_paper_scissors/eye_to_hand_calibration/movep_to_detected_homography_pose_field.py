#!/usr/bin/env python3
"""Detect a tabletop ball, map pixel to base XY, and use nearest pose-field RPY."""

from __future__ import annotations

import argparse
from pathlib import Path

from homography_tabletop_runtime import (
    add_common_args,
    apply_homography,
    build_movep_command,
    detect_stable_pixel,
    load_homography,
    nearest_pose_field,
    print_and_maybe_execute,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--pose-field", type=Path, default=Path("pose_field_samples.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    homography = load_homography(args.calibration)
    detection = detect_stable_pixel(args, homography)
    target_xy_m = apply_homography(homography, detection.pixel)
    match = nearest_pose_field(args.pose_field, target_xy_m)
    rpy = (match.pose[3], match.pose[4], match.pose[5])
    print(f"pixel={detection.pixel} conf={detection.confidence:.3f}")
    print(f"target_xy_m=({target_xy_m[0]:.6f}, {target_xy_m[1]:.6f})")
    print(
        "nearest_pose_field="
        f"index={match.index} distance={match.distance_m * 1000.0:.1f}mm "
        f"sample_xy=({match.pose[0]:.6f}, {match.pose[1]:.6f}) "
        f"rpy=({rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f})"
    )
    cmd, target = build_movep_command(args, target_xy_m, rpy)
    return print_and_maybe_execute(args, cmd, target)


if __name__ == "__main__":
    raise SystemExit(main())
