# 同步感知—响应数据集

项目只优先考虑具有共同时间轴的输入与响应/动作数据。

## 1. ActionSense

真实人类厨房活动：第一视角和环境视频、双臂 EMG、身体姿态、眼动、触觉与活动标签同步。首个任务选择单人 `S04` 的第一视角视频到 16 通道 EMG。详见 [ACTIONSENSE.md](ACTIONSENSE.md)。

官方入口：https://action-sense.csail.mit.edu/data.html

## 2. LeRobot PushT

约 31.6 MB，包含 96×96 图像、二维状态和同步二维动作。适合在接入大型真实数据之前验证视觉控制映射。

官方入口：https://huggingface.co/datasets/lerobot/pusht_image

## 3. DROID-100

真实机器人多视角视频、机器人状态和 7 维动作。官方提供约 2 GB 的 100-episode 调试子集；完整 RLDS 数据约 1.7 TB。

官方入口：https://droid-dataset.github.io/droid/the-droid-dataset

## 统一任务约定

```text
observations: [episode, time, modality...]
responses:    [episode, time, response_dim]
timestamps:   [episode, time]
```

训练、验证和测试必须按完整 episode 切分，不能先切相邻窗口再随机分组。
