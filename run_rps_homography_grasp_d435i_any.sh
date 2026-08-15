#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$BUNDLE_ROOT/lerobot_robot_piper/src:$BUNDLE_ROOT/vendor/piper_sdk:${PYTHONPATH:-}"
cd "$BUNDLE_ROOT/lerobot_robot_piper/src/lerobot_robot_piper/rock_paper_scissors"
./run_rps_homography_grasp_d435i_any.sh "$@"
