#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pip install -e vendor/piper_sdk
python3 -m pip install -e lerobot_robot_piper
