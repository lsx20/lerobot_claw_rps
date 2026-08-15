#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 rps_yolo_pick.py \
  --gesture-camera-index /dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0 \
  --gesture-camera-release-settle 0.7 \
  --pick-source homography \
  --can can0 \
  --hand-port /dev/ttyUSB0 \
  --hand-id 1 \
  --ball-serial 260322279862 \
  --homography-calibration eye_to_hand_calibration/homography_position_calibration.json \
  --homography-ball-model yolo26s.pt \
  --ball-device 0 \
  --start-ball-camera-after-win \
  --ball-start-retries 3 \
  --ball-start-retry-delay 1.0 \
  --homography-imgsz 640 \
  --homography-detect-zoom 2.0 \
  --ball-warmup 0 \
  --force-player-win \
  --fixed-grab-z 0.208 \
  --lift-z 0.285 \
  --radial-offset-mm 95 \
  --drop-pose="197.205,-281.251,245.243,175.536,55.498,119.972" \
  --grasp-start-speed 15 \
  --planar-speed 15 \
  --rps-return-speed 15 \
  --drop-approach-lift-mm 30 \
  --safe-drop-circle-shrink-mm 60 \
  --disable-exit-joints="0.090,0.000,0.000,1.678,1.957,0.380" \
  --disable-exit-speed 8 \
  --disable-exit-duration 8 \
  --yes \
  --no-one-shot \
  --speed 8 \
  --rate-hz 40 \
  --start-duration 8 \
  --planar-duration 5 \
  --vertical-duration 4 \
  --transfer-duration 8 \
  --return-duration 8 \
  --drop-open-settle 5 \
  --classify-ball \
  --no-ball-ab-squeeze-test \
  --ball-tactile-model ball_tactile_classifier/model.json \
  --ball-tactile-output ball_tactile_classifier/live_predictions.csv \
  --ball-tactile-visual-reference-samples ball_tactile_classifier/samples.csv \
  "$@"
