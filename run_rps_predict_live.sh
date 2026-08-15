#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$BUNDLE_ROOT/lerobot_robot_piper/src:$BUNDLE_ROOT/vendor/piper_sdk:${PYTHONPATH:-}"
cd "$BUNDLE_ROOT/lerobot_robot_piper/src/lerobot_robot_piper/rock_paper_scissors"
python3 ball_tactile_classifier/predict_live.py \
  --model ball_tactile_classifier/model_with_newB.json \
  --visual-reference-samples ball_tactile_classifier/samples_with_newB.csv \
  --grab-pose 281.005,96.727,200.058,-178.850,58.706,-159.536 \
  --repeats 10 \
  --between-repeat 3 \
  --no-ab-squeeze-test \
  --hover-duration 1.5 \
  --lift-settle 0.0 \
  --lower-duration 2 \
  --yes
