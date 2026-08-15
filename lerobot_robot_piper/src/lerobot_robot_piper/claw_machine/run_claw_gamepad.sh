#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python claw_main.py \
  --control gamepad \
  --gamepad-device /dev/input/js0 \
  --can can0 \
  --speed 8 \
  --rate-hz 40 \
  --gamepad-j1-speed-dps 18 \
  --gamepad-reach-speed-dps 14 \
  --gamepad-axis-curve 2.4 \
  --gamepad-deadzone 0.22 \
  --gamepad-lead-limit-deg 2.5 \
  --gamepad-stop-reset \
  --grab-z 205 \
  --lift-z 287.496 \
  --transfer-duration 20 \
  --return-duration 20 \
  --hand-speed 800 \
  --pre-grab-open-speed 1800 \
  --hand-settle 2.5 \
  --pre-grab-open-settle 0.5 \
  --drop-open-settle 5.0 \
  --held-force-threshold 130 \
  --held-force-fingers thumb_bend,thumb_swing,index,middle \
  --held-force-alt-fingers thumb_bend,thumb_swing,index,ring \
  --held-check-duration 1.0 \
  --held-check-rate-hz 5 \
  --held-required-samples 3 \
  --result-gesture \
  --result-gesture-speed 20 \
  --result-gesture-j2-back-deg 30 \
  --result-gesture-j6-deg 90 \
  --result-thumb-speed 2500 \
  --result-thumb-settle 0.8 \
  --result-gesture-duration 6.0 \
  --result-gesture-hold-after 2.0 \
  --result-gesture-return-duration 2.5 \
  --auto-position-tolerance-mm 3 \
  --auto-rpy-tolerance-deg 2 \
  --drop="197.205,-281.251,245.243,175.536,55.498,119.972"
