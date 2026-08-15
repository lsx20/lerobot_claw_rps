# 灵巧手抓娃娃机用户前端

这是面向用户的总控前端，负责在两个独立功能之间切换：

- 猜拳自主抓取
- 遥控抓取

前端位于 `lerobot_robot_piper/web_ui`，不再属于 `rock_paper_scissors` 子模块。

## 后端边界

- 总控入口：`lerobot_robot_piper/run_web_ui.sh`
- 显式掉使能入口：`lerobot_robot_piper/run_down_device.sh`
- 统一硬件后端：`rock_paper_scissors/run_rps_homography_grasp_touch_ui.sh`

Web UI 启动时会 setup `can0`，并在整个 UI 生命周期持续发送 Piper
`EnableArm(7, 0x02)`。Web UI 不自动发送 `DisableArm`，也不把遥控命令塞进
独立的 `run_claw_machine.sh` 子进程。

## 启动

启动总控前端：

```bash
cd /home/z0200/桌面/lerobot_claw_rps/lerobot_robot_piper/src/lerobot_robot_piper
./run_web_ui.sh
```

浏览器打开 `http://127.0.0.1:8765/`。

## 模式行为

选择猜拳自主抓取时，Web UI 启动统一硬件后端，并通过
`rps_touch_ui_command.json` / `rps_touch_ui_state.json` 协议发送
`start_game` 和 `draw_prize`。

选择遥控抓取并点击开始时，Web UI 向同一个硬件后端发送 `remote_run`。
后端复用 `ClawMachineController.move_to_start_and_hover()` 和
`run_keyboard_loop()`，会使用启动 Web UI 的终端读取键盘输入。

需要显式掉使能时：

```bash
./run_down_device.sh
```

只看前端效果时：

```bash
python3 web_ui/server.py --demo
```
