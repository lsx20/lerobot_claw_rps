#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/lerobot_robot_piper/src/lerobot_robot_piper/rock_paper_scissors"
./run_rps_homography_grasp_touch_ui.sh "$@"
