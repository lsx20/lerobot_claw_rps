#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/lerobot_robot_piper/src"
python3 -m lerobot_robot_piper.claw_machine.lerobot_claw \
  --control gamepad \
  --gamepad-device /dev/input/js0 \
  --gamepad-axis-x 0 \
  --gamepad-axis-y 1 \
  --gamepad-pick-button 0 \
  --gamepad-stop-button 1 \
  --can can0 \
  --hand-port /dev/ttyUSB0 \
  --yes \
  --no-classify-ball \
  --grab-z 215 \
  --lift-z 287.496 \
  --drop-pose="222.660,-305.970,257.233,-177.047,56.852,130.068" \
  --rate-hz 25 \
  --gamepad-reach-speed-dps 6 \
  --result-gesture-speed 20
