# Field Debug Checklist 2026-08-11

## Current Known Changes

- CLAW `--grab-z` changed from `200` to `215` mm.
- RPS `--fixed-grab-z` changed from `0.200` to `0.215` m.
- CLAW and RPS default start pose/joints changed to:
  - joints: `0.091, 46.504, -45.622, 0.000, 43.982, 6.571`
  - pose: `161.039, 0.257, 332.985, 172.196, 49.686, 169.902`
- Drop pose changed to:
  - `197.205,-281.251,245.243,175.536,55.498,119.972`
- RH56F2 hand port unified to `/dev/ttyUSB0`.
- YOLO defaults changed to CPU on this PC.

## Still Not Fully Closed

1. RH56F2 may still report `No RH56F2 response while reading angleAct`.
2. Serial permission may still fail until `dialout` group membership takes effect after re-login.
3. D405 serial must match the real device on this PC.
4. RPS fixed-camera homography must be recalibrated for the new physical setup.
5. YOLO detection in the new venue still needs to be verified under real lighting/background.
6. CLAW tactile A/B/C classification is still likely affected by changed grasp/contact conditions.

## Before Running Anything

1. Re-login after:

```bash
sudo usermod -aG dialout lsx
```

2. Run:

```bash
cd /home/lsx/lerobot_claw_rps_bundle_20260808
./setup_can0.sh
```

3. Check current serial devices:

```bash
ls -l /dev/serial/by-id
```

4. Check current user groups:

```bash
id
```

## Recommended On-Site Order

1. Test RH56F2 hand only.

```bash
cd /home/lsx/lerobot_claw_rps_bundle_20260808/lerobot_robot_piper/src/lerobot_robot_piper/rock_paper_scissors
python3 test_rh56f2_rps.py --port /dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG0327LK-if00-port0 --cycle
```

If this fails, do not continue to CLAW or RPS. Fix hand power, cable, RS485 adapter, or USB hub first.

2. Test D405 YOLO only.

```bash
cd /home/lsx/lerobot_claw_rps_bundle_20260808/lerobot_robot_piper/src/lerobot_robot_piper/rock_paper_scissors
python3 test_yolo_d405_ball.py --serial 260322279862 --model yolo26n.pt
```

Check whether the ball is detected stably in the new environment.

3. If YOLO can already detect the ball, recalibrate homography first.

```bash
cd /home/lsx/lerobot_claw_rps_bundle_20260808/lerobot_robot_piper/src/lerobot_robot_piper/rock_paper_scissors/eye_to_hand_calibration
python3 collect_homography_position_samples_yolo.py \
  --can can0 \
  --serial 260322279862 \
  --device cpu \
  --output homography_position_samples.csv
```

Then solve:

```bash
python3 solve_homography_position_calibration.py \
  --input homography_position_samples.csv \
  --output homography_position_calibration.json
```

4. Run RPS grasp test only after hand and YOLO are both stable.

5. Run CLAW only after RH56F2 feedback is stable.

## If YOLO Is Not Stable In The New Venue

Use the lowest-manual-work path:

1. Auto-capture many D405 images:

```bash
cd /home/lsx/lerobot_claw_rps_bundle_20260808/lerobot_robot_piper/src/lerobot_robot_piper/rock_paper_scissors
python3 collect_yolo_ball_images.py \
  --serial 260322279862 \
  --output-root yolo_ball_dataset_newsite \
  --auto-save \
  --interval 1.0
```

2. Only annotate a small subset with pre-labeling assistance.

Recommended tools:

- Roboflow
- CVAT
- Label Studio

3. Retrain YOLO only after the current model is confirmed insufficient.

## If CLAW Ball Classification Is Still Wrong

Check the latest prediction row:

```bash
cd /home/lsx/lerobot_claw_rps_bundle_20260808/lerobot_robot_piper/src/lerobot_robot_piper/rock_paper_scissors/ball_tactile_classifier
tail -n 2 claw_predictions.csv
```

Focus on:

- `hover_sample_count`
- `hover_duration_s`
- `final_angle_middle`
- `size_closure_mean`
- `final_force_delta_middle`
- `hover_thumb_force_delta_max`

If hover samples are still low, increase hover duration further.
If hover samples are adequate but labels are still wrong, collect new samples and retrain for the new setup.
