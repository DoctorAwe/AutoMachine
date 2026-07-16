# AutoMachine / 神经处理机

这是一个基于 Transformer 思路的流式输入输出模型试作型项目。

当前代码落地的是第一版小原型：

```text
视频帧块 -> StreamEncoder -> 分层 Token 状态管线 -> StreamDecoder -> 下一帧预测
```

## 当前模块

- `StreamEncoder`: 将连续 `chunk_size` 帧压缩成固定 token 网格。
- `HierarchicalMemoryPipeline`: 多层持久 token 状态，每层 token 容量可以不同。
- `StateFusionBlock`: 用 cross-attention、self-attention 和门控更新融合输入与当前层状态。
- `StreamDecoder`: 将当前最高层状态读出并解码成输出帧。
- `NeuralStreamProcessor`: 端到端流式模型。

## 环境

已按本机 Python 3.11 创建虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Smoke Test

如果环境中已有 PyTorch：

```powershell
python -m tests.smoke_forward
```

## 最小训练闭环

先用合成的移动光斑视频验证模型可以前向、反向传播和保存权重：

```powershell
python -m automachine.train --steps 20 --batch-size 4
```

这个合成数据只是工程 bring-up 用；真实训练仍建议使用现实生物运动、神经信号、肌电信号或其他时域信号流。

## 在线训练

把项目上传到 Colab、Kaggle 或云 GPU 后，按 [CLOUD_TRAINING.md](CLOUD_TRAINING.md) 执行。
