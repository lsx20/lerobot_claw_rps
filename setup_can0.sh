#!/usr/bin/env bash
set -euo pipefail

sudo modprobe gs_usb 2>/dev/null || true

if ! ip link show can0 >/dev/null 2>&1; then
  echo "error: can0 not found. Check the USB-CAN adapter, driver, and cable, then run: ip link" >&2
  exit 1
fi

echo "Configuring can0..."
sudo ip link set can0 down 2>/dev/null || true

current_bitrate="$(ip -details link show can0 | grep -oP 'bitrate \K\d+' || true)"
if [[ "$current_bitrate" != "1000000" ]]; then
  echo "Setting can0 bitrate to 1000000..."
  sudo ip link set can0 type can bitrate 1000000
else
  echo "can0 bitrate is already 1000000."
fi

echo "Bringing can0 up..."
sudo ip link set can0 up
ip -details link show can0

if [[ -e /dev/ttyUSB0 ]]; then
  sudo chmod a+rw /dev/ttyUSB0
  ls -l /dev/ttyUSB0
else
  echo "warning: /dev/ttyUSB0 not found"
fi
