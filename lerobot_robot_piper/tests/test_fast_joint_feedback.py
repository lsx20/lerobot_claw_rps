from lerobot_robot_piper.claw_machine.lerobot_claw import (
    ClawMachineController,
    ClawMachineTaskConfig,
)


class FastJointRobot:
    def __init__(self):
        self.full_observation_calls = 0
        self.fast_joint_calls = 0

    def get_observation(self):
        self.full_observation_calls += 1
        raise AssertionError("full observation should not be used for arm-only feedback")

    def get_arm_joint_positions(self):
        self.fast_joint_calls += 1
        return {f"joint_{index}.pos": float(index) for index in range(1, 7)}

    def send_action(self, action):
        return action


def test_current_joints_uses_fast_arm_only_feedback_when_available():
    robot = FastJointRobot()
    controller = ClawMachineController(robot, ClawMachineTaskConfig(grab_z=120.0))

    joints = controller.current_joints()

    assert joints == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert robot.fast_joint_calls == 1
    assert robot.full_observation_calls == 0
