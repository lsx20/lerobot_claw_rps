from lerobot_robot_piper.claw_machine.claw_hand_grasp import BALL_CLOSED, BALL_READY_OPEN
from lerobot_robot_piper.claw_machine.lerobot_claw import (
    ClawMachineController,
    ClawMachineTaskConfig,
    parse_pose_mm_deg,
    pose_from_raw,
)
from lerobot_robot_piper.piper_rh56f2_follower import EE_POSE_NAMES


class FakeRobot:
    def __init__(self):
        self.actions = []
        self.obs = {
            "ee.x": 100.0,
            "ee.y": 200.0,
            "ee.z": 300.0,
            "ee.rx": 1.0,
            "ee.ry": 2.0,
            "ee.rz": 3.0,
        }
        for name in ["little", "ring", "middle", "index", "thumb_bend", "thumb_swing"]:
            self.obs[f"hand.{name}.force"] = 0.0

    def get_observation(self):
        return dict(self.obs)

    def send_action(self, action):
        self.actions.append(dict(action))
        self.obs.update(action)
        return dict(action)


def test_pose_from_raw_converts_sdk_units_to_lerobot_units():
    pose = pose_from_raw([1000, 2000, 3000, 4000, 5000, 6000])
    assert pose == {
        "ee.x": 1.0,
        "ee.y": 2.0,
        "ee.z": 3.0,
        "ee.rx": 4.0,
        "ee.ry": 5.0,
        "ee.rz": 6.0,
    }


def test_parse_pose_mm_deg_uses_ee_action_keys():
    pose = parse_pose_mm_deg("1,2,3,4,5,6")
    assert list(pose) == EE_POSE_NAMES
    assert pose["ee.z"] == 3.0


def test_controller_sends_hand_pose_through_lerobot_action_keys():
    robot = FakeRobot()
    controller = ClawMachineController(
        robot,
        ClawMachineTaskConfig(
            grab_z=120.0,
            drop_pose=parse_pose_mm_deg("1,2,3,4,5,6"),
        ),
    )

    controller.set_hand_pose(BALL_CLOSED)

    assert robot.actions[-1]["hand.index.pos"] == BALL_CLOSED["index"]
    assert robot.actions[-1]["hand.thumb_swing.pos"] == BALL_CLOSED["thumb_swing"]


def test_pick_cycle_uses_ee_and_hand_actions_only():
    robot = FakeRobot()
    task = ClawMachineTaskConfig(
        grab_z=120.0,
        drop_pose=parse_pose_mm_deg("500,600,700,8,9,10"),
        rate_hz=1000.0,
        vertical_duration_s=0.001,
        transfer_duration_s=0.001,
        return_duration_s=0.001,
        hand_settle_s=0.0,
        pre_grab_open_settle_s=0.0,
        drop_open_settle_s=0.0,
        held_check_duration_s=0.001,
        held_check_rate_hz=1000.0,
    )
    controller = ClawMachineController(robot, task)

    held = controller.run_pick_cycle()

    assert held is False
    assert any(
        action.get("hand.index.pos") == BALL_READY_OPEN["index"]
        for action in robot.actions
    )
    assert any(
        action.get("hand.index.pos") == BALL_CLOSED["index"]
        for action in robot.actions
    )
    assert any(action.get("ee.z") == 120.0 for action in robot.actions)
    assert any(action.get("ee.x") == 500.0 for action in robot.actions)
