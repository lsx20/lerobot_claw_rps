#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/lerobot_robot_piper/src"
LOG="../claw_gamepad_$(date +%Y%m%d_%H%M%S).csv"
python3 -m lerobot_robot_piper.claw_machine.lerobot_claw \
  --control gamepad \
  --gamepad-device /dev/input/js0 \
  --gamepad-axis-x 0 \
  --gamepad-axis-y 1 \
  --gamepad-pick-button 0 \
  --gamepad-stop-button 1 \
  --can can0 \
  --hand-port /dev/ttyUSB0 \
  --grab-z 215 \
  --lift-z 287.496 \
  --drop="197.205,-281.251,245.243,175.536,55.498,119.972" \
  --rate-hz 25 \
  --gamepad-reach-speed-dps 6 \
  --ball-hover-duration 4.5 \
  --ball-hover-rate-hz 10.0 \
  --gamepad-log-csv "$LOG"
echo "log saved: $LOG"
python3 lerobot_robot_piper/claw_machine/analyze_gamepad_jitter.py "$LOG" || true
