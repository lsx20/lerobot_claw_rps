# Piper 抓娃娃 + 猜拳迁移包

## 内容

- `lerobot_robot_piper/`：当前项目源码，包含抓娃娃和猜拳代码、球分类模型、标定/样本 CSV、运行脚本。
- `vendor/piper_sdk/`：本机使用的 Piper SDK 源码，避免另一台电脑缺 SDK。
- `run_claw_machine.sh`：抓娃娃主程序。
- `run_claw_machine_with_log.sh`：抓娃娃 + 手柄/关节频率日志分析。
- `run_rps_predict_live.sh`：猜拳/球分类触觉测试程序。
- `setup_can0.sh`：启用 CAN0。

## 新电脑首次安装

```bash
cd ~/Desktop/lerobot_claw_rps_bundle_20260808
./install_on_new_pc.sh
```

如果缺系统依赖，至少需要：Python 3.10+、CAN 驱动、`python3-pip`、`pyserial`、`lerobot` 相关依赖、相机/ROS 依赖按实际功能安装。

如果灵巧手串口打开时报权限错误，先把当前用户加入 `dialout`：

```bash
sudo usermod -aG dialout lsx
```

然后重新登录，再运行下面的脚本。

## 启动 CAN0

```bash
./setup_can0.sh
```

## 运行抓娃娃

```bash
./run_claw_machine.sh
```

A/button 0 抓取，B/button 1 急停保持。

## 运行抓娃娃并记录真实频率

```bash
./run_claw_machine_with_log.sh
```

## 运行猜拳/球触觉分类测试

```bash
./run_rps_predict_live.sh
```

## 运行猜拳抽奖触屏界面

先启动本地网页界面：

```bash
./run_rps_touch_ui.sh
```

浏览器打开 `http://127.0.0.1:8765/`。再启动配套猜拳抽奖程序：

```bash
./run_rps_homography_grasp_touch_ui.sh --yes
```

界面流程：点击“开始游戏” -> 提示用户伸手对准摄像头猜拳 -> 显示“恭喜获胜/再试一次” -> 获胜后点击“启动灵巧手自主抽奖” -> 显示抓取触觉界面 -> 抽奖完成后显示“已完成抽奖画面”。

## 硬件默认端口

- Piper CAN：`can0`
- RH56F2 灵巧手：`/dev/ttyUSB0`
- 手柄：`/dev/input/js0`

如果另一台电脑端口不同，改对应脚本里的参数。
