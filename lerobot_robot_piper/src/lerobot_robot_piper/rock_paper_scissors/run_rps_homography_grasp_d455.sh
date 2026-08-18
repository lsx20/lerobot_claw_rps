#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 rps_yolo_pick.py \
  --pick-source homography \
  --can can0 \
  --hand-port /dev/ttyUSB0 \
  --hand-id 1 \
  --gesture-serial 261722071542 \
  --ball-serial 315122271151 \
  --homography-calibration eye_to_hand_calibration/homography_position_calibration.json \
  --homography-ball-model yolo26n.pt \
  --ball-device cpu \
  --fixed-grab-z 0.215 \
  --lift-z 0.285 \
  --radial-offset-mm 45 \
  --drop-pose="-185.449,434.149,135.053,68.125,72.173,172.599" \
  --speed 8 \
  --rate-hz 40 \
  --start-duration 8 \
  --planar-duration 15 \
  --vertical-duration 4 \
  --transfer-duration 8 \
  --return-duration 8 \
  --player-win-probability 1.0 \
  --tie-probability 0.0 \
  --force-player-win \
  --classify-ball \
  --ball-tactile-model ball_tactile_classifier/model.json \
  --ball-tactile-output ball_tactile_classifier/live_predictions.csv \
  --ball-tactile-visual-reference-samples ball_tactile_classifier/samples.csv \
  "$@"
