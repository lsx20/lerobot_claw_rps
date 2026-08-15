# Fixed Camera Tabletop Sampling

This is the new fixed-camera tabletop workflow:

```text
image pixel (u,v) -> base tabletop (X,Y)
Z is manually configured
pose is selected from a nearest-neighbor pose field
```

## Reset Samples

When the camera position changes, remove old samples/calibrations:

```bash
cd /home/zhiyu/robot_ws/lerobot_robot_piper/src/lerobot_robot_piper/rock_paper_scissors/eye_to_hand_calibration
rm -f homography_position_samples.csv homography_position_calibration.json pose_field_samples.csv
```

## 1. Position Samples

Collect image pixel to base XY pairs:

```bash
python3 collect_homography_position_samples_yolo.py \
  --can can0 \
  --serial 315122271151 \
  --output homography_position_samples.csv
```

By default the script commands RH56F2 to the same pre-grasp pose used by the
tactile classification program right before closing:

```text
little=1800 ring=1800 middle=1800 index=1800 thumb_bend=1500 thumb_swing=1050
```

Use `--no-hand-pose` only if you want to skip RH56F2 control.

Per sample:

```text
1. Put the ball/reference point in D405 view.
2. Wait for stable detection.
3. Press c to lock the image pixel.
4. Move fake TCP to the matching tabletop point.
5. Press s to save pixel + current base XY.
```

Keys:

```text
c  lock stable pixel
s  save locked pixel plus current Piper pose
x  clear locked pixel
g  send the pre-grasp hand pose again
q  quit
```

Recommended sample count:

```text
minimum: 8-12
better: 15-25
```

Cover left, right, front, back, center, and corners of the real grasp area.

## 2. Solve Homography

```bash
python3 solve_homography_position_calibration.py \
  --input homography_position_samples.csv \
  --output homography_position_calibration.json
```

Check:

```text
all RMS
all max
```

For first tests, target roughly:

```text
RMS < 10 mm
max < 20 mm
```

## 3. Pose Field Samples

This does not use the camera. Move the gripper to a good pose at different
tabletop XY positions, then save the current Piper pose.

```bash
python3 collect_pose_field_samples.py \
  --can can0 \
  --output pose_field_samples.csv
```

This script also sends the same RH56F2 pre-grasp pose by default.

Per sample:

```text
1. Manually/teach move to a useful XY location.
2. Adjust RX, RY, RZ to a pose that can grab cleanly.
3. Press Enter to save current X,Y,Z,RX,RY,RZ.
```

Recommended sample count:

```text
minimum: 3x3 = 9
better: 4x4 = 16
best: 5x5 = 25
```

Runtime should use nearest-neighbor pose selection first:

```text
target XY -> nearest saved pose sample -> RX,RY,RZ
```

## 4. Dry-Run Detection

Fixed RPY version:

```bash
python3 movep_to_detected_homography_fixed_pose.py \
  --can can0 \
  --serial 315122271151 \
  --calibration homography_position_calibration.json \
  --fixed-z-mm 230 \
  --rpy 172,55,180
```

This fixed-pose script defaults to:

```text
target mode: radial
flange XY: ball XY minus 45 mm along the base-origin radial ray
RZ: wrap(180 deg + atan2(ball_Y, ball_X))
RX,RY: from --rpy
start-first: move to claw-machine DEFAULT_START_POSE before the detected target
start duration: 10 s
```

Override with `--radial-offset-mm`, `--rz-offset-deg`, `--start-duration`, or
`--no-start-first`. Use `--target-mode xy_offset` to return to the old fixed
`--x-offset-mm/--y-offset-mm` behavior. In `--execute` mode this wrapper feeds
`YES` and then an empty line to `movep_to_pose.py`, so each segment starts and
keeps motors enabled at the end before the next segment.

For the constrained planar wrist solution, use:

```bash
python3 movep_to_detected_homography_fixed_pose.py \
  --can can0 \
  --serial 315122271151 \
  --calibration homography_position_calibration.json \
  --fixed-z-mm 230 \
  --target-mode planar_joint \
  --execute
```

In `planar_joint` mode, the start move uses MOVE_P, then the final target uses
MOVE_J with J4/J6 constrained in the same SDK session. The final disable prompt
is interactive by default, so type `D` to disable or press Enter to keep motors
enabled.

Pose-field version:

```bash
python3 movep_to_detected_homography_pose_field.py \
  --can can0 \
  --serial 315122271151 \
  --calibration homography_position_calibration.json \
  --pose-field pose_field_samples.csv \
  --fixed-z-mm 230
```

Both commands are dry-run by default. They print:

```text
YOLO pixel
mapped target base XY
movep_to_pose.py command
```

Add `--execute` only after the printed target looks reasonable. Start with a
safe high `--fixed-z-mm`, then lower Z after XY is verified.
