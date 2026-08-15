# RH56F2 + D435i + Hand-Mounted D405 Rock Paper Scissors

This directory contains a RH56F2 version of the OmniHand/MediaPipe demo from
the reference article.

The reference article uses OmniHand 2025 with a CAN-FD SDK:

- `OmniHand2025`
- `create_hand_by_hcan`
- ten active joint angles

This project uses your existing RH56F2 RS485 driver instead:

- `RH56F2Hand`
- `/dev/ttyUSB0`
- six register angles: little, ring, middle, index, thumb_bend, thumb_swing

## 1. Install Camera And Vision Dependencies

First make sure both the fixed D435i and the hand-mounted D405 work in Intel
RealSense Viewer.

Then install Python dependencies.

Check the Python version inside the active environment first:

```bash
python3 --version
```

For Python 3.13, use the newer MediaPipe wheel. Do not pin numpy to 1.26.4:

```bash
python3 -m pip install pyrealsense2 mediapipe==0.10.35
```

Download the MediaPipe Gesture Recognizer model once:

```bash
wget -O gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

For Python 3.10 to 3.12, the article's old version can work, but the simpler
command below is usually enough:

```bash
python3 -m pip install pyrealsense2 mediapipe
```

The demo script uses the newer MediaPipe Tasks API, so it also needs
`gesture_recognizer.task` in this directory, or a path passed with
`--gesture-model`.

If your OpenCV build has issues after changing numpy, reinstall OpenCV in the
same Python environment.

## 2. Find The RealSense Cameras

Use one of these:

```bash
realsense-viewer
lerobot-find-cameras realsense
```

Write down both serial numbers. The fixed D435i is used for third-person
gesture recognition, and the hand-mounted D405 is used for first-person
prize-ball detection. Use `--gesture-serial` and `--ball-serial` in the new
prize workflow when both cameras are connected.

## 3. Test A RealSense Color Camera Only (Legacy Smoke Test)

```bash
python3 test_d455_camera.py
```

With an explicit serial number:

```bash
python3 test_d455_camera.py --serial 1234567890
```

Press `q` to quit.

## 4. Test RH56F2 Only

Keep the arm still and make sure the hand has empty space around it.

```bash
python3 test_rh56f2_rps.py --port /dev/ttyUSB0 --cycle
```

If the hand has not reached each pose before the next command, increase the
cycle delay:

```bash
python3 test_rh56f2_rps.py --port /dev/ttyUSB0 --cycle --delay 4
```

To test faster gesture motion, increase hand speed and shorten the staged
delay:

```bash
python3 test_rh56f2_rps.py --port /dev/ttyUSB0 --cycle --speed 1200 --stage-delay 0.05 --delay 1
```

If the USB-RS485 adapter appears as another port:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

Then pass the correct port with `--port`.

## 5. Run The Full Demo

First run dry mode. This uses the camera and recognition, but does not connect
the RH56F2:

```bash
python3 rh56f2_rps_demo.py --dry-run
```

For vision debugging, print the raw MediaPipe category and confidence:

```bash
python3 rh56f2_rps_demo.py --dry-run --print-vision
```

If the raw category stays `None` but `hands=1`, the script falls back to a
landmark geometry classifier. In that case, check the `fallback=` field. The
demo can trigger as long as `fallback` becomes `Rock`, `Paper`, or `Scissors`.

If a hand is visible but the score is low, lower the thresholds:

```bash
python3 rh56f2_rps_demo.py --dry-run --print-vision --min-score 0.2 --min-detection 0.4 --min-presence 0.4
```

If the model is somewhere else:

```bash
python3 rh56f2_rps_demo.py --dry-run --gesture-model /path/to/gesture_recognizer.task
```

Then connect the RH56F2:

```bash
python3 rh56f2_rps_demo.py --hand-port /dev/ttyUSB0
```

For a faster game, increase hand speed and shorten stage, settle, and cooldown
times:

```bash
python3 rh56f2_rps_demo.py --hand-port /dev/ttyUSB0 --hand-speed 1200 --stage-delay 0.05 --motion-settle 0.5 --cooldown 0.5
```

If the hand changes too quickly or does not finish moving, increase them:

```bash
python3 rh56f2_rps_demo.py --hand-port /dev/ttyUSB0 --motion-settle 3 --cooldown 3
```

With an explicit D455 serial number:

```bash
python3 rh56f2_rps_demo.py --serial 1234567890 --hand-port /dev/ttyUSB0
```

Press `q` to quit.

## Gesture Mapping

Human gesture from MediaPipe:

- Rock: zero or one extended finger
- Scissors: index and middle extended
- Paper: three or four extended fingers

Robot reply:

- Human Rock -> RH56F2 Paper
- Human Paper -> RH56F2 Scissors
- Human Scissors -> RH56F2 Rock

## RH56F2 Pose Tuning

The current pose values are conservative starting points:

- Rock: `DEFAULT_CLOSED`
- Paper: `DEFAULT_OPEN`
- Scissors: index and middle open, ring and little closed

If your RH56F2 finger directions differ, tune the values in
`test_rh56f2_rps.py` first. After the hand looks correct, copy the same values
into `rh56f2_rps_demo.py`.

Move in small increments, about 50 to 100 register units at a time.

## 7. RPS Prize-Ball Workflow With D435i + Hand-Mounted D405 + Arduino UNO

`rps_prize_controller.py` adds the complete game orchestration:

1. The fixed third-person D435i recognizes the audience gesture.
2. The host randomly chooses the system gesture according to
   `--system-win-probability` and `--tie-probability`.
3. If the audience wins, the host sends `UNLOCK` to the UNO.
4. The audience presses the UNO button; UNO reports `BUTTON`.
5. The hand-mounted first-person D405 finds a colored ball in `--roi`, obtains
   its depth, and converts the pixel to camera coordinates.
6. The pick backend grasps and drops the ball.

Upload `uno_rps.ino` to the Arduino UNO. Connect a push button between D2 and
GND. The built-in LED indicates that the draw button is unlocked.

The two cameras are selected independently. `--gesture-serial` is the D435i
serial number and `--ball-serial` is the D405 serial number. The physical
connection is also intentionally documented this way: D435i is connected by
USB, while the hand-mounted D405 uses the computer Type-C port. The software
uses the RealSense serial number rather than assuming `/dev/video0` or USB
device order, because Linux device numbering can change after reconnecting.

Find both serial numbers with:

```bash
rs-enumerate-devices
```

Start a two-camera/UNO dry-run:

```bash
python3 rps_prize_controller.py --dry-run \
  --gesture-serial YOUR_D435I_SERIAL \
  --ball-serial YOUR_D405_SERIAL \
  --uno-port /dev/ttyACM0 \
  --gesture-model gesture_recognizer.task
```

The default system win probability is 65%; change it with:

```bash
python3 rps_prize_controller.py --system-win-probability 0.7 --tie-probability 0.05
```

`--dry-run` disables the UNO serial connection and uses a print-only pick
backend; both RealSense cameras still run. The fixed D435i is used only by the
gesture loop. The D405 is read only during the prize-ball search, after the UNO
button event, so its first-person view is not mixed with the gesture view.

The current entrypoint validates the two-camera vision and game logic, but it
does not move Piper. For real picking, construct `PiperPickBackend` with
the already-connected Piper/RH56F2 objects from `claw_machine`, and provide a
workspace transform from D405 camera coordinates to Piper coordinates:

```text
robot_x_m = x_offset + camera_x_m * x_scale
robot_y_m = y_offset + camera_y_m * y_scale
robot_z_m = z_offset + camera_z_m * z_scale
```

Because the D405 is mounted on the hand, its camera coordinates change with
the wrist pose. A fixed XYZ offset/scale is therefore only valid if the hand
approaches the prize tray with a repeatable orientation. For accurate picking,
calibrate a hand-mounted camera-to-tool transform, or use the Piper wrist pose
and a full rigid transform before commanding motion. Do not run Piper motion
before checking the generated target in dry-run mode. The default ball
detector accepts saturated colored objects; tune the ROI and detector
thresholds for the actual prize-ball color and tray background.
