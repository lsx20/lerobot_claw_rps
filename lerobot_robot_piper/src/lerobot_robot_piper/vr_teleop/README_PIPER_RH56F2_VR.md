# Piper + RH56F2 VR Teleop

This folder is the integration point for VR teleoperation.

## Goal

Use VR hand tracking to control both:

- Piper arm end-effector pose: `ee.x`, `ee.y`, `ee.z`, `ee.rx`, `ee.ry`, `ee.rz`
- RH56F2 dexterous hand angles: `hand.<finger>.pos`

The hardware side reuses `PiperRH56F2Follower`, which already contains the
lessons from the claw-machine debugging work:

- Piper MOVE_P end-effector control through `ee.*`
- RH56F2 RS485 hand control through `hand.*.pos`
- hand force feedback through `hand.*.force`
- per-step motion limits for arm and hand
- no automatic motor disable on exit unless the user confirms

## Current Bridge

`piper_rh56f2_vr_teleop.py` accepts normalized JSON frames from stdin.

Example dry run:

```bash
printf '%s\n' \
  '{"deadman": false, "wrist_xyz_m": [0.0, 0.0, 0.0], "finger_curls": {"all": 0.0}}' \
  '{"deadman": true, "wrist_xyz_m": [0.02, 0.00, 0.00], "finger_curls": {"all": 0.5}}' \
  | python3 -m lerobot_robot_piper.vr_teleop.piper_rh56f2_vr_teleop
```

Real hardware mode must be explicit:

```bash
python3 -m lerobot_robot_piper.vr_teleop.piper_rh56f2_vr_teleop \
  --connect \
  --can can0 \
  --hand-port /dev/ttyUSB0
```

## Expected VR Frame

```json
{
  "deadman": true,
  "wrist_xyz_m": [0.02, 0.00, 0.01],
  "wrist_rpy_deg": [0.0, 0.0, 5.0],
  "finger_curls": {
    "thumb_bend": 0.2,
    "thumb_swing": 0.1,
    "index": 0.7,
    "middle": 0.7,
    "ring": 0.4,
    "little": 0.4
  }
}
```

## Quest 3 VR Input

This follows the first `vr_teleop` project's input path. The Quest 3 Hand
Tracking Streamer app sends the right wrist pose and 21 right-hand landmarks
to the computer as a UDP text packet. The computer listens on UDP port 9000.

The data path is:

```text
Quest 3 Hand Tracking Streamer -> UDP:9000 -> VRFrame
    -> RH56F2SimpleRetargeter -> hand.<finger>.pos -> /dev/ttyUSB0
```

Configure the Quest 3 app with the computer's LAN IP address, not
`127.0.0.1`, and port `9000`. The Quest 3 and computer must be on the same
LAN. Start the listener without hardware first:

```bash
PYTHONPATH=/home/zhiyu/robot_ws/lerobot_robot_piper/src \
python3 -u -m lerobot_robot_piper.vr_teleop.piper_rh56f2_vr_teleop \
  --input-source quest3 --port 9000
```

Check that the computer is listening:

```bash
ss -lunp | rg ':9000'
```

The status line should show packets increasing after the app starts streaming.
For the first real test, connect only RH56F2 and keep Piper disabled:

```bash
PYTHONPATH=/home/zhiyu/robot_ws/lerobot_robot_piper/src \
python3 -u -m lerobot_robot_piper.vr_teleop.piper_rh56f2_vr_teleop \
  --input-source quest3 \
  --port 9000 \
  --connect --hand-only \
  --hand-port /dev/ttyUSB0 \
  --hand-speed 1000 \
  --max-hand-delta 80 \
  --thumb-swing-closed 500
```

The command above uses the Quest 3 wrist position for the future Piper arm
mapping and sends only the hand channels when `--hand-only` is enabled.
Removing the hand from the Quest 3 stream makes the deadman state false and
stops new motion commands. Press `Ctrl-C` to stop.

## Apple Vision Pro Input

Vision Pro uses the Tracking Streamer app shown in the headset. Its screen
must show `gRPC Server Ready`. The computer connects to the Vision Pro; the
Vision Pro IP in the screenshot is `192.168.3.62`.

Install the upstream gRPC client once if needed:

```bash
python3 -m pip install avp-stream
```

Start the computer program with the Vision Pro IP:

```bash
PYTHONPATH=/home/zhiyu/robot_ws/lerobot_robot_piper/src \
python3 -u -m lerobot_robot_piper.vr_teleop.piper_rh56f2_vr_teleop \
  --input-source avp \
  --avp-ip 192.168.3.62
```

After the computer prints `Vision Pro teleop ready`, press `START` in the
Vision Pro Tracking Streamer app. The computer should then print increasing
`Vision Pro status` frame counts.

For a hand-only hardware test:

```bash
PYTHONPATH=/home/zhiyu/robot_ws/lerobot_robot_piper/src \
python3 -u -m lerobot_robot_piper.vr_teleop.piper_rh56f2_vr_teleop \
  --input-source avp \
  --avp-ip 192.168.3.62 \
  --connect --hand-only \
  --hand-port /dev/ttyUSB0 \
  --hand-speed 1000 \
  --max-hand-delta 80 \
  --thumb-swing-closed 500
```

Vision Pro and the computer must be connected to the same LAN. Do not use
`127.0.0.1`; that address means the local computer itself.

## Next Step

The current simple mapper is kept only as the hardware interface baseline.
The next upgrade is to pass the same 63 Quest 3 landmark values through
AnyDexRetarget, then adapt its modeled-hand output to RH56F2's six serial
channels.

## AnyDex Mode

AnyDexRetarget has models for several hands, but not RH56F2. Therefore its
output must be reduced to five finger-curl values before RH56F2 register
angles are sent. The qpos groups are ordered as:

```text
thumb,index,middle,ring,little
```

Example command using an Inspire model as the retargeting front end:

```bash
python3 -m lerobot_robot_piper.vr_teleop.piper_rh56f2_vr_teleop \
  --input-source quest3 \
  --hand-mode anydex \
  --anydex-root /path/to/AnyDexRetarget \
  --hand-config /path/to/AnyDexRetarget/example/config/adaptive/quest3/quest3_inspire_hand.yaml \
  --anydex-qpos-groups '0,1;2,3;4,5;6,7;8,9'
```

The example qpos groups are placeholders until RH56F2 is calibrated against
the selected AnyDex model. Do not connect real hardware until each finger
group and its open/closed direction has been checked in dry-run output.
