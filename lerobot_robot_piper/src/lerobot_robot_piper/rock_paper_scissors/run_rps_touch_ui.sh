#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
BUNDLE_ROOT="$(cd ../../../.. && pwd)"
export PYTHONPATH="$BUNDLE_ROOT/lerobot_robot_piper/src:$BUNDLE_ROOT/vendor/piper_sdk:${PYTHONPATH:-}"
python3 rps_touch_ui_server.py "$@"
