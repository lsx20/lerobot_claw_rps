#!/usr/bin/env python3
"""Generate a printable Charuco board image for D405 eye-in-hand calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from charuco_eye_hand_common import make_charuco_board


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("charuco_board_7x5.png"))
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--pixels-per-square", type=int, default=220)
    parser.add_argument("--margin-squares", type=float, default=0.75)
    parser.add_argument("--square-length", type=float, default=0.024, help="metres, record/print metadata only")
    parser.add_argument("--marker-length", type=float, default=0.018, help="metres, record/print metadata only")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    args = parser.parse_args()

    board, _ = make_charuco_board(
        args.squares_x,
        args.squares_y,
        args.square_length,
        args.marker_length,
        args.dictionary,
    )
    board_width = args.squares_x * args.pixels_per_square
    board_height = args.squares_y * args.pixels_per_square
    margin = int(round(args.margin_squares * args.pixels_per_square))
    board_image = board.generateImage((board_width, board_height))
    canvas = np.full((board_height + 2 * margin, board_width + 2 * margin), 255, dtype=np.uint8)
    canvas[margin : margin + board_height, margin : margin + board_width] = board_image
    cv2.imwrite(str(args.output), canvas)

    print(f"wrote {args.output}")
    print(f"board: {args.squares_x}x{args.squares_y}, dictionary={args.dictionary}")
    print(f"metadata square_length_m={args.square_length}, marker_length_m={args.marker_length}")
    print("Print at 100% scale. Measure one printed black/white square and use that actual length for collection/solve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
