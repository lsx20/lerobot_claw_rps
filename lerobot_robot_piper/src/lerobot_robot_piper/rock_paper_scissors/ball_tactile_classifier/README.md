# RH56F2 三类球触觉分类

这个子目录只放“通过 RH56F2 手指角度/力反馈区分三类球”的实验程序，不改现有石头剪刀布流程。

核心思路：

- 大小：慢速闭合时，记录每根手指第一次达到接触力阈值的角度。大球更早接触，闭合量更小；小球更晚接触，闭合量更大。
- 重量：优先接入腕部六维力或机械臂竖直方向力/力矩差。如果暂时没有这个传感器，`--weight-g` 只作为记录/分析字段；真正在线识别重量时应使用 `--lift-force-delta` 这类机器人能测到的输入。
- 分类：先采样，再训练一个轻量 nearest-centroid 模型。三类球差异明显时，这种方法比一开始上复杂模型更容易调试。

## 1. 采集样本

每类球建议先采 30-50 次。先少量试运行，确认力阈值不会夹太紧。

```bash
python3 ball_tactile_classifier/collect_samples.py --label A --repeats 10 --yes
python3 ball_tactile_classifier/collect_samples.py --label B --repeats 10 --yes
python3 ball_tactile_classifier/collect_samples.py --label C --repeats 10 --yes
```

如果已知重量，可以记录到 CSV 里便于后续分析：

```bash
python3 ball_tactile_classifier/collect_samples.py --label A --weight-g 80 --repeats 10 --yes
```

如果以后有腕部力/机械臂力矩估计结果，把抓起后的竖直力差传进来：

```bash
python3 ball_tactile_classifier/collect_samples.py --label A --lift-force-delta 0.78 --repeats 10 --yes
```

默认输出：

```text
ball_tactile_classifier/samples.csv
```

## 2. 训练模型

```bash
python3 ball_tactile_classifier/train_classifier.py
```

默认训练特征不会使用 `weight_g`，因为未知球在线预测时通常不知道真实重量。若你确认预测时也会提供同类数值，可以用 `--features` 显式指定特征。

默认生成：

```text
ball_tactile_classifier/model.json
```

## 2.5 当前位姿上抬/悬停采样

如果想加入“抓起来悬停时的大拇指压力变化”，先把机械臂移动到当前要抓/放球的位置。新脚本不会去固定坐标，只会沿当前位姿的 `ee.z` 上下移动：

```bash
python3 ball_tactile_classifier/collect_lift_samples.py --label A --repeats 10 --yes
python3 ball_tactile_classifier/collect_lift_samples.py --label B --repeats 10 --yes
python3 ball_tactile_classifier/collect_lift_samples.py --label C --repeats 10 --yes
```

默认动作：

```text
当前位姿闭合抓球 -> ee.z 上抬 50 mm -> 悬停 2 s 记录压力 -> 降回当前 z -> 张开
```

这条上抬/下降路径直接复用 `claw_machine/claw_init.py` 里的 `wait_for_movep_ready()` 和 `send_movep_for()`，也就是你之前能用的 Piper MOVE_P 代码。脚本默认每一轮会先张开手再闭合。如果接触不够导致不上抬，默认不会把这条失败样本写入 CSV。

采完悬停样本后，建议只用带悬停数据的行训练：

```bash
python3 ball_tactile_classifier/train_classifier.py --min-hover-samples 5
python3 ball_tactile_classifier/predict_from_csv.py --limit 0
```

如果你想调整动作幅度：

```bash
python3 ball_tactile_classifier/collect_lift_samples.py --label A --lift-height-mm 40 --hover-duration 3 --repeats 10 --yes
```

悬停会新增这些关键特征：

```text
hover_thumb_force_delta_mean
hover_thumb_force_delta_max
hover_thumb_force_delta_std
hover_thumb_force_delta_drift
hover_force_delta_sum_mean
hover_force_delta_sum_max
```

## 3. 离线检查预测

```bash
python3 ball_tactile_classifier/predict_from_csv.py --limit 30
```

## 4. 在线预测

先训练出 `model.json`，再放入未知球：

```bash
python3 ball_tactile_classifier/predict_live.py --yes
```

输出会给出预测类别、置信度和距离。

## 5. 可视化

这不是必须，但适合解释和调参：

```bash
python3 ball_tactile_classifier/visualize_samples.py
```

界面里会显示：

- 第一次接触时的平均闭合量，用来看大小差异。
- 最终手指力变化总和，用来看抓取受力差异。
- 最新一次样本的六通道“伪触觉热力图”。
- 类别间主要特征分布。

## 参数建议

- `--contact-threshold`：第一次接触阈值。默认 120。误触多就调高；接触检不出来就调低。
- `--max-force-delta`：保护阈值。默认 800。达到后停止本次触摸。
- `--hand-force`：RH56F2 力限制。默认 600，先保守。
- `--repeats`：每个类别至少 30 次更稳。

## 注意

RH56F2 的 `forceAct` 不是高密度触觉阵列，所以这里做的是“手指级触觉特征分类”，不是完整形状重建。真正区分重量时，最好接入腕部六维力传感器或可靠的机械臂力矩估计。
