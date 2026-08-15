#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
../run_web_ui.sh "$@"
