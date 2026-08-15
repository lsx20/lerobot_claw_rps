#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

"$REPO_ROOT/setup_can0.sh"

cd "$SCRIPT_DIR/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python3 -m lerobot_robot_piper.web_ui.server --manage-backends "$@"
