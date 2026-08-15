#!/usr/bin/env python3
"""ROS2 read-only D405 camera point to Piper base point transformer.

Subscribes to a D405 camera-frame PointStamped, reads Piper's current end pose,
applies the saved eye-in-hand matrix, and publishes a base-frame PointStamped.

This script does not enable, disable, or move Piper.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from collect_eye_hand_samples import read_piper_pose
from solve_eye_hand_calibration import make_transform, rpy_to_matrix


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--calibration", type=Path, default=Path("eye_hand_calibration_yolo11x_clean2.json"))
    parser.add_argument("--input-topic", default="/d405/ball_point_camera")
    parser.add_argument("--output-topic", default="/piper/ball_point_base")
    parser.add_argument("--base-frame", default="piper_base")
    parser.add_argument("--x-offset", type=float, default=0.0, help="metres added to transformed base X")
    parser.add_argument("--y-offset", type=float, default=0.0, help="metres added to transformed base Y")
    parser.add_argument("--z-offset", type=float, default=0.0, help="metres added to transformed base Z")
    parser.add_argument("--yes", action="store_true", help="skip READONLY confirmation")
    return parser.parse_known_args()


def main() -> int:
    args, ros_args = parse_args()
    if not args.calibration.exists():
        raise SystemExit(f"--calibration does not exist: {args.calibration}")

    try:
        import rclpy
        from geometry_msgs.msg import PointStamped
        from rclpy.node import Node
    except ImportError as exc:
        raise SystemExit("ROS2 rclpy is not available. Run: source /opt/ros/humble/setup.bash") from exc

    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError as exc:
        raise SystemExit("piper_sdk is not installed in this Python environment") from exc

    tool_camera = np.asarray(json.loads(args.calibration.read_text())["T_tool_camera"], dtype=float)
    if tool_camera.shape != (4, 4):
        raise SystemExit(f"invalid T_tool_camera shape in {args.calibration}: {tool_camera.shape}")

    print("SAFETY: ROS2 Piper target transformer is read-only.")
    print("No Piper enable, disable, gripper, hand, or motion commands are used.")
    print(f"calibration: {args.calibration}")
    print(f"subscribe: {args.input_topic}")
    print(f"publish:   {args.output_topic} in frame {args.base_frame}")
    print(f"base offset(m): ({args.x_offset}, {args.y_offset}, {args.z_offset})")
    if not args.yes and input("Type READONLY to connect to Piper CAN and transform targets: ").strip() != "READONLY":
        print("Cancelled.")
        return 0

    piper = C_PiperInterface_V2(
        args.can,
        judge_flag=False,
        can_auto_init=False,
        dh_is_offset=1,
        start_sdk_fk_cal=True,
    )
    piper.ConnectPort()
    time.sleep(1.0)

    rclpy.init(args=ros_args)

    class TransformNode(Node):
        def __init__(self) -> None:
            super().__init__("piper_target_transform_node")
            self.publisher = self.create_publisher(PointStamped, args.output_topic, 10)
            self.subscription = self.create_subscription(PointStamped, args.input_topic, self.callback, 10)
            self.last_log = 0.0

        def callback(self, message: PointStamped) -> None:
            pose = read_piper_pose(piper)
            base_tool = make_transform(rpy_to_matrix(*pose[3:]), np.asarray(pose[:3], dtype=float))
            camera_point = np.array([message.point.x, message.point.y, message.point.z, 1.0], dtype=float)
            base_point = base_tool @ tool_camera @ camera_point

            output = PointStamped()
            output.header.stamp = self.get_clock().now().to_msg()
            output.header.frame_id = args.base_frame
            output.point.x = float(base_point[0]) + args.x_offset
            output.point.y = float(base_point[1]) + args.y_offset
            output.point.z = float(base_point[2]) + args.z_offset
            self.publisher.publish(output)

            now = time.monotonic()
            if now - self.last_log >= 1.0:
                print(
                    "target_base_m="
                    f"({output.point.x:.6f}, {output.point.y:.6f}, {output.point.z:.6f}) "
                    "piper_pose="
                    f"({pose[0]:.6f}, {pose[1]:.6f}, {pose[2]:.6f}, "
                    f"{pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f})"
                )
                self.last_log = now

    node = TransformNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        try:
            piper.DisconnectPort()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
