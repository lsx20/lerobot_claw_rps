#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
BUNDLE_ROOT="$(cd ../../../.. && pwd)"
export PYTHONPATH="$BUNDLE_ROOT/lerobot_robot_piper/src:$BUNDLE_ROOT/vendor/piper_sdk:${PYTHONPATH:-}"

./run_rps_homography_grasp_d455.sh --gesture-serial "" "$@"
