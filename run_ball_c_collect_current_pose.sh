#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/lerobot_robot_piper/src/lerobot_robot_piper/rock_paper_scissors/ball_tactile_classifier"

python3 collect_lift_samples.py \
  --label C \
  --grab-pose=-48.510,193.750,210.000,180.000,64.180,-75.943 \
  --drop-pose=-48.510,193.750,210.000,180.000,64.180,-75.943 \
  "$@"
