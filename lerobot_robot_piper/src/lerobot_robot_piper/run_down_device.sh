#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python3 -m lerobot_robot_piper.down_device "$@"
