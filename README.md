# AutoMachine / 神经处理机

基于 Transformer 思路的流式时域信号输入/输出试作模型。当前版本包含两条路径：

```text
视频帧块 -> 时空编码器 -> 分层持久 Token 记忆 -> 多尺度读出 -> 视频帧
特征序列 -> 时间编码器 -> 分层持久 Token 记忆 -> 多尺度读出 -> 特征向量
```

## 模型结构

- `StreamEncoder`：使用 3D 卷积显式保留帧顺序，再以内容相关权重压缩时间维。
- `HierarchicalMemoryPipeline`：不同容量的持久 Token 层，每步用交叉注意力、自注意力和门控残差更新。
- 多尺度读出：解码时同时读取全部记忆层；浅层提供近期细节，深层提供较长上下文。
- `FeatureStreamProcessor`：处理姿态、IMU、EMG、神经活动等 `[B,T,F]` 信号。

## 环境与验证

项目要求 Python 3.11+，建议重建本地虚拟环境后安装依赖：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m tests.smoke_forward
```

合成视频仅用于验证前向、反向传播和 checkpoint：

```powershell
python -m automachine.train --steps 20 --batch-size 4
```

## 第一阶段真实数据：UCI HAR

从 UCI 官方页面下载并解压 `UCI HAR Dataset.zip`，然后转换 50 Hz 九通道原始 IMU：

```powershell
python -m automachine.prepare_uci_har --source "D:\data\UCI HAR Dataset" --output data\uci_har
python -m automachine.train_features --data data\uci_har\train.npz --input-length 96 --chunk-size 8 --steps 1000 --device auto
```

转换器沿用官方按受试者划分的 train/test 集，并且只用训练集统计量标准化，避免数据泄漏。更多候选见 [DATASETS.md](DATASETS.md)。云端训练步骤见 [CLOUD_TRAINING.md](CLOUD_TRAINING.md)。

## Colab 在线预测展示

把训练 checkpoint 放在 `checkpoints/`，安装展示依赖并启动：

```bash
pip install -r requirements-demo.txt
python -m automachine.demo \
  --data data/uci_har/test.npz \
  --checkpoints checkpoints \
  --device auto \
  --share
```

终端会输出一个临时 `gradio.live` 公网地址。页面可以切换多个 checkpoint 和测试样本，对比九通道输入、真实后续信号、模型预测及“复制上一采样点”基线。
