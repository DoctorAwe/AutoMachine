# 同步感知—响应数据集

## 0. Li et al. 2025（当前推荐）

4 段连续自然电影和小鼠上丘浅层双光子钙响应；公开下载、单成像平面处理与 Colab 训练见 [LI2025.md](LI2025.md)。它具有共享时间轴，适合检验流式因果状态，而不是把独立图像误当成连续视频。

官方数据：https://zenodo.org/records/14885567

## 1. ActionSense

人类厨房活动中的第一视角视频、双臂 EMG、姿态、眼动、触觉和标签。见 [ACTIONSENSE.md](ACTIONSENSE.md)。

## 2. Goldin et al. 2022（历史实验）

自然图像到视网膜神经节细胞响应。当前公开处理形式中的图像不构成可靠连续电影，因此不用于验证长时间流式因果状态。旧流程见 [GOLDIN2022.md](GOLDIN2022.md)。

## 切分规则

```text
observations: [episode, time, modality...]
responses:    [episode, time, response_dim]
```

训练、验证和测试应按完整 episode/电影切分，窗口不得跨越边界，也不能先生成重叠窗口再随机切分。
