# LeRobot 抓娃娃接入说明

这个目录现在有两层代码：一层是原来的“直接硬件控制脚本”，另一层是新的
“LeRobot 风格控制层”。

## 1. 直接 SDK 控制层

这些旧文件直接控制 Piper SDK 和 RH56F2：

- `claw_main.py`
- `claw_arm_grasp.py`
- `claw_init.py`
- `claw_hand_grasp.py`

它们会到处传一个原始的 `piper_sdk.C_PiperInterface_V2` 对象，然后直接调用：

- `MotionCtrl_2(...)`
- `JointCtrl(...)`
- `EndPoseCtrl(...)`

这种写法适合调硬件，因为很直接；但它还不是 LeRobot 的任务/策略结构。

## 2. LeRobot 控制层

`lerobot_claw.py` 是新的 LeRobot 风格版本。

它只通过 LeRobot Robot 接口控制机器人：

```python
robot.get_observation()
robot.send_action(action)
```

这是你学习 LeRobot 框架时最需要抓住的点：

- `get_observation()`：读取当前机器人状态。
- `send_action()`：发送下一步动作。
- 任务流程、手写规则、未来的 policy 都应该放在这两个接口上层。

## 3. 新增的末端位姿动作

`PiperRH56F2Follower` 现在支持末端位姿动作：

```python
{
    "ee.x": 100.0,   # mm
    "ee.y": 200.0,   # mm
    "ee.z": 300.0,   # mm
    "ee.rx": 0.0,    # degree
    "ee.ry": 0.0,    # degree
    "ee.rz": 0.0,    # degree
}
```

它原本也支持关节和 RH56F2 手指动作：

```python
{
    "joint_1.pos": 0.0,
    "hand.index.pos": 1720,
}
```

所以抓娃娃 controller 现在使用：

- `ee.*`：控制 Piper 末端 MOVE_P 位姿。
- `hand.*.pos`：控制 RH56F2 手指角度。
- `hand.*.force`：读取 RH56F2 力反馈，用来判断有没有抓到东西。

## 4. 运行一次 LeRobot 抓娃娃流程

从仓库根目录运行：

```bash
python3 -m lerobot_robot_piper.claw_machine.lerobot_claw \
  --grab-z 120 \
  --drop 100,200,300,0,0,0 \
  --hand-port /dev/ttyUSB0 \
  --yes
```

注意：上面的 `--grab-z` 和 `--drop` 只是示例值，你要换成自己现场标定好的
抓取高度和投放点。

`--drop` 的格式是：

```text
X,Y,Z,RX,RY,RZ
```

单位是：

```text
X/Y/Z: mm
RX/RY/RZ: degree
```

## 5. 推荐阅读顺序

如果你想理解 LeRobot 是怎么接入硬件的，按这个顺序看：

1. `PiperRH56F2Follower.observation_features`
2. `PiperRH56F2Follower.action_features`
3. `PiperRH56F2Follower.get_observation`
4. `PiperRH56F2Follower.send_action`
5. `ClawMachineController.run_pick_cycle`

最短理解方式：

```text
硬件 SDK
-> LeRobot Robot 统一接口
-> 手写任务 controller
-> 未来可以替换成 policy / dataset / imitation learning
```

## 6. 这一步做到了什么，还没做到什么

已经做到：

- 把 Piper + RH56F2 的抓娃娃动作放到 LeRobot Robot 接口上。
- 把抓娃娃流程整理成 `ClawMachineController`。
- 让抓娃娃逻辑可以用假 robot 做单元测试，不必每次都连真硬件。

还没做到：

- 还没有采集 LeRobot 数据集。
- 还没有训练 policy。
- 还没有用 `lerobot-record` 录制抓娃娃示教数据。

现在这一步可以理解成：

```text
先把硬件控制包装成统一接口，再往数据采集和策略学习走。
```
