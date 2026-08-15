# Fixed D405 to Fake TCP Calibration

This folder is for the shortcut workflow where the fixed D405 point is mapped
directly to the Piper fake TCP target point.

The solved matrix is not a pure camera-to-robot-base external calibration. Its
meaning is:

```text
fake_tcp_target_base = T_base_fake_tcp_target_from_camera * camera_point
```

Use it only when the fake TCP target convention is the same during collection
and use.

## 1. Collect Samples

Put one detectable ball/target in the fixed D405 view. Each sample is collected
in two steps so the open hand can block the camera after the ball coordinate is
recorded:

```text
1. Keep the arm/hand out of the D405 view, wait for stable detection, press c.
2. Move Piper's fake TCP to the desired target point, press s.
```

```bash
cd /home/zhiyu/robot_ws/lerobot_robot_piper/src/lerobot_robot_piper/rock_paper_scissors/eye_to_hand_calibration
python3 collect_fake_tcp_samples_yolo.py --can can0 --serial 260322279862 --output fake_tcp_samples.csv
```

The collector is read-only for Piper. It connects to read the current end pose
but does not enable, disable, or move the arm. It opens the RH56F2 hand once
after the D405 starts and keeps the hand connection alive. Press `o` during
collection to send the open-hand command again.

Keys:

```text
c  lock the current stable camera point
s  save locked camera point plus current fake TCP pose
x  clear the locked camera point
o  open RH56F2 hand again
q  quit
```

If you do not want the collector to control the hand:

```bash
python3 collect_fake_tcp_samples_yolo.py --can can0 --serial 260322279862 --output fake_tcp_samples.csv --no-open-hand
```

Recommended sample count:

```text
minimum: 6
better: 10-20
```

Spread samples across the whole work area. Do not collect all points in one
small patch.

## 2. Solve

```bash
python3 solve_fake_tcp_calibration.py --input fake_tcp_samples.csv --output fake_tcp_calibration.json
```

Optional holdout check:

```bash
python3 solve_fake_tcp_calibration.py --input fake_tcp_samples.csv --output fake_tcp_calibration.json --holdout-every 5
```

Check `all RMS` and `all max`. For a quick tabletop workflow, a few millimeters
to around one centimeter may be usable depending on the task.

## 3. Validate Live

```bash
python3 validate_fake_tcp_target_yolo.py --calibration fake_tcp_calibration.json --serial 260322279862
```

Press `p` to print the current camera point and converted fake TCP target.

Optional Z offset after conversion:

```bash
python3 validate_fake_tcp_target_yolo.py --calibration fake_tcp_calibration.json --z-offset-m 0.02
```

## Important Constraint

If the arm orientation, fake TCP definition, or the physical offset convention
changes between collection and use, this shortcut can drift. For a general
camera-to-base calibration, collect real target points in base coordinates
instead of fake TCP target points.
