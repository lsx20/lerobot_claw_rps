# ROS2 Autonomous Grasp Reference Decision

## Recommendation

Use `nero-mobile-manipulation-ros2` as the primary reference.

Reason:

- It is built around ROS 2 Humble, TF2, RGB-D perception, YOLO/VLM, and MoveIt 2, which matches Ubuntu 22.04 better than ROS1 Noetic examples.
- Its perception chain is closer to our target: RGB-D detection -> stable 3D target -> camera frame target -> TF2 transform to robot base -> planning/execution.
- It separates perception, coordinate transform, and execution into independent ROS2 nodes, which is easier to debug than one large script.
- It explicitly treats hand-eye parameters, real robot poses, and safety configuration as local machine data that must be re-measured.

Do not directly run its robot motion scripts on our Piper yet. The safe first step is to port its perception/TF architecture while keeping our current LeRobot/Piper SDK execution path as the known working baseline.

## Repository Comparison

### `wzi690421-droid/nero-mobile-manipulation-ros2`

Best fit.

Core idea:

1. RGB-D camera publishes image and organized point cloud.
2. YOLO/VLM finds the target in 2D.
3. Depth/point cloud is sampled around the detected pixel.
4. The node waits for several stable frames before locking a 3D target.
5. A `PointStamped` target in camera frame is transformed through TF2 into robot base frame.
6. MoveIt 2 checks reachability and plans approach/grasp motion.
7. Execution is separated from planning and can be disabled for dry-run validation.

Files to study:

- `external_refs/nero-mobile-manipulation-ros2/src/nero_vision_grasp_ws_cjr/src/yolo_detector/yolo_detector/detector_node.py`
- `external_refs/nero-mobile-manipulation-ros2/src/nero_vision_grasp_ws_cjr/src/yolo_detector/yolo_detector/target_pose_node.py`
- `external_refs/nero-mobile-manipulation-ros2/src/nero_vision_grasp_ws_cjr/src/yolo_detector/yolo_detector/grasp_executor_auto_node.py`
- `external_refs/nero-mobile-manipulation-ros2/src/nero_vision_grasp_ws_cjr/config/vision_grasp.yaml`

### `mu9enn/eyes_piper`

Useful but not the primary base.

Core idea:

1. Orbbec RGB-D camera provides color and depth.
2. YOLOv5 detects fruit/object.
3. ROS static TF connects camera and Piper frames.
4. MoveIt controls Piper to grasp.
5. Dataset tooling helps train custom YOLO models.

Why not primary:

- It targets ROS Noetic / catkin, not ROS2 Humble.
- It uses Orbbec Astra rather than our RealSense D405/D435i setup.
- It is Piper-specific, so keep it as a Piper/MoveIt reference only.

### `himnshu-debug/Language-based-Robot-Manipulation`

Not suitable for our current accuracy problem.

Core idea:

1. Voice command chooses object name.
2. YOLO-World detects object in webcam image.
3. A simple 2D pixel-to-robot linear mapping gives robot X/Y.
4. Dobot moves directly to the target.

Why not primary:

- No depth-based 3D localization.
- No ROS2, no Piper, no TF2.
- No collision-aware planning.
- Calibration is simpler than our current method, not more rigorous.

## What We Should Change

Current pipeline:

```text
D405 YOLO pixel/depth
  -> camera_xyz
  -> JSON T_tool_camera
  -> current Piper pose
  -> base_xyz
  -> LeRobot/Piper SDK moves
```

Recommended NERO-style pipeline:

```text
D405 ROS2 color + aligned depth/pointcloud
  -> YOLO ball detector
  -> stable PointStamped in d405_color_optical_frame
  -> TF2: d405_color_optical_frame -> piper_base
  -> target PoseStamped in piper_base
  -> current LeRobot/Piper SDK execution first
  -> MoveIt2 plan-only later
```

The key upgrade is not just changing the detector. The key upgrade is making calibration a real TF chain and validating it continuously:

```text
piper_base -> tool/end_effector -> d405_color_optical_frame -> ball_point
```

## Calibration Upgrade

The current fixed-ball method can produce a small numerical residual but still be physically wrong if the ball center/depth/pose samples are biased.

Better calibration route:

1. Mount a Charuco/AprilTag board rigidly in the workspace.
2. Use D405 to estimate full board pose, not just one ball center point.
3. Record 20-30 diverse Piper poses with large changes in X/Y/Z and wrist orientation.
4. Solve eye-in-hand calibration using `T_base_tool_i` and `T_camera_board_i`.
5. Publish the result as a ROS2 static transform from tool frame to D405 optical frame.
6. Validate on independent target points before any grasp motion.

Acceptance target:

- Independent validation target scatter: ideally below 10 mm.
- Maximum validation error: preferably below 20 mm.
- D405 detection confidence stable and depth jump below 10-15 mm.
- Same physical ball should transform to nearly the same `base_xyz` after changing arm pose.

## Immediate Safe Next Steps

1. Keep using `rps_yolo_pick.py` for the known working demo.
2. Create a ROS2-only D405 target publisher that does not move Piper.
3. Publish D405 target as `PointStamped`.
4. Add a TF broadcaster from the current JSON calibration.
5. Compare ROS2 TF output against `validate_piper_target_yolo.py`.
6. Only after the TF result matches the current Python result, replace the Python matrix calculation.
7. MoveIt2 should be introduced first in plan-only mode, not execution mode.

## Current Baseline Commands

Validate current non-ROS transform:

```bash
python3 validate_piper_target_yolo.py \
  --serial 260322279862 \
  --can can0 \
  --calibration eye_hand_calibration_yolo11x_clean2.json \
  --model yolo11x.pt \
  --roi 0,0,640,480 \
  --conf 0.35 \
  --imgsz 1280
```

Run current RPS plus one-shot ball grasp:

```bash
python3 rps_yolo_pick.py \
  --can can0 \
  --hand-port /dev/ttyUSB0 \
  --gesture-serial 261722071542 \
  --ball-serial 260322279862 \
  --calibration eye_hand_calibration_yolo11x_clean2.json \
  --ball-model yolo11x.pt \
  --ball-conf 0.35 \
  --ball-imgsz 1280 \
  --ball-stable-frames 3 \
  --ball-timeout 30 \
  --ball-warmup 0.2 \
  --rps-result-hold 2.0 \
  --fixed-grab-z 0.20 \
  --no-refine-at-hover \
  --force-player-win \
  --yes
```
