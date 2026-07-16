# ActionSense 单人 S04：Colab 下载、处理与训练

目标任务：

```text
S04 第一视角 RGB 视频（约30 Hz）
    → 因果 Token 管线
    → 同步双臂16通道 EMG RMS 响应（约30 Hz）
```

模型学习的是当前感知到当前生物响应的同步映射，不是下一帧预测。

官方资料：

- 数据下载：https://action-sense.csail.mit.edu/data.html
- 文件结构：https://action-sense.csail.mit.edu/dataset_info.html
- 官方解析示例：https://github.com/delpreto/ActionNet/tree/master/parsing_data

## 1. 准备 Colab 项目

假设项目已经位于：

```text
/content/AutoMachine
```

进入项目并安装依赖：

```bash
cd /content/AutoMachine
pip install -r requirements-actionsense.txt
```

确认新入口存在：

```bash
python -m automachine.prepare_actionsense --help
python -m automachine.train_actionsense --help
```

如果提示模块不存在，说明 Colab 中还是旧版本项目，需要先同步最新代码。

## 2. 下载受试者 S04

S04 只有一个完整 recording split，适合首个单人实验。

```bash
cd /content/AutoMachine
mkdir -p data/actionsense/S04/raw
cd data/actionsense/S04/raw
```

下载 wearable HDF5。它包含双臂 EMG、活动标签和第一视角视频时间戳：

```bash
wget -c -O S04_wearables.hdf5 \
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S04/2022-06-14_16-38-18_actionNet-wearables_S04/2022-06-14_16-38-43_streamLog_actionNet-wearables_S04.hdf5"
```

下载第一视角无 gaze-overlay 视频：

```bash
wget -c -O S04_world.mp4 \
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S04/2022-06-14_16-38-18_actionNet-wearables_S04/2022-06-14_16-38-43_S04_eye-tracking-video-world_frame.mp4"
```

验证文件不是错误的 HTML 页面：

```bash
ls -lh S04_wearables.hdf5 S04_world.mp4
file S04_wearables.hdf5 S04_world.mp4
```

正确结果应分别包含类似：

```text
Hierarchical Data Format
ISO Media, MP4
```

若显示 `HTML document`，删除错误文件，再从官方 S04 页面复制最新下载链接。

## 3. 检查双臂 EMG

```bash
cd /content/AutoMachine

python - <<'PY'
import h5py
import numpy as np

path = "data/actionsense/S04/raw/S04_wearables.hdf5"
with h5py.File(path, "r") as h5:
    for device in ("myo-left", "myo-right"):
        signal = np.asarray(h5[device]["emg"]["data"])
        times = np.asarray(h5[device]["emg"]["time_s"]).squeeze()
        rate = (len(times) - 1) / (times[-1] - times[0])
        print(device, "signal=", signal.shape, "rate=", round(float(rate), 2), "Hz")
PY
```

每只 Myo 应有8通道，最终拼接为16通道。

## 4. 生成同步训练分片

```bash
cd /content/AutoMachine

python -m automachine.prepare_actionsense \
  --hdf5 data/actionsense/S04/raw/S04_wearables.hdf5 \
  --video data/actionsense/S04/raw/S04_world.mp4 \
  --output data/actionsense/S04/processed \
  --image-size 96 \
  --window 32 \
  --stride 32 \
  --shard-size 32
```

处理器会自动执行：

1. 查找 HDF5 中真实的第一视角视频时间戳路径。
2. 读取 `myo-left/emg` 和 `myo-right/emg`。
3. 在每个视频帧附近计算16通道 EMG RMS。
4. 只保留评级不是 `Bad` 或 `Maybe` 的完整活动实例。
5. 按完整活动实例的时间顺序切分：70%训练、15%验证、15%测试。
6. 仅用训练集计算 EMG 均值和标准差。
7. 生成32帧、约1.07秒的同步窗口。

处理不会把整段视频一次性载入内存；活动结束后立即写出临时 episode，再生成训练分片。

## 5. 验证处理结果

```bash
cd /content/AutoMachine

find data/actionsense/S04/processed -maxdepth 2 -type f | head -30
cat data/actionsense/S04/processed/manifest.json
```

目录应类似：

```text
processed/
  manifest.json
  train/shard_0000.npz
  validation/shard_0000.npz
  test/shard_0000.npz
```

检查首个分片：

```bash
python - <<'PY'
import numpy as np

path = "data/actionsense/S04/processed/train/shard_0000.npz"
with np.load(path) as data:
    print("frames   =", data["frames"].shape, data["frames"].dtype)
    print("responses=", data["responses"].shape, data["responses"].dtype)
    print("labels   =", data["labels"][:5])
PY
```

典型形状：

```text
frames    = [窗口数, 32, 96, 96, 3] uint8
responses = [窗口数, 32, 16] float32
```

## 6. 先做模型动态验证

```bash
cd /content/AutoMachine
python -m pytest -q
python -m tests.smoke_forward
```

只有测试通过后才开始真实训练。

## 7. 开始 S04 小模型训练

首轮验证配置：

```bash
cd /content/AutoMachine

python -m automachine.train_actionsense \
  --data data/actionsense/S04/processed \
  --steps 2000 \
  --batch-size 4 \
  --token-dim 64 \
  --spatial-grid 4 \
  --state-tokens 16 16 24 32 \
  --num-heads 4 \
  --dropout 0.05 \
  --lr 1e-4 \
  --derivative-weight 0.1 \
  --log-every 20 \
  --eval-every 100 \
  --save-every 200 \
  --device cuda \
  --checkpoint checkpoints/actionsense_S04.pt
```

训练输出包含：

```text
train_loss
val_mse
mean_baseline
improvement
```

其中 `mean_baseline` 是每个EMG通道始终输出训练集均值的基线。首个有效性标准是：

```text
val_mse < mean_baseline
improvement > 0%
```

这只能证明视频和时间上下文提供了超过常数输出的信息，不代表已经能安全控制硬件。

## 8. 中断与继续训练

训练器每200步保存一次；按 `Ctrl+C` 也会保存当前状态。

继续训练时，`--steps` 表示目标总步数：

```bash
python -m automachine.train_actionsense \
  --data data/actionsense/S04/processed \
  --steps 4000 \
  --batch-size 4 \
  --token-dim 64 \
  --spatial-grid 4 \
  --state-tokens 16 16 24 32 \
  --num-heads 4 \
  --device cuda \
  --resume checkpoints/actionsense_S04.pt \
  --checkpoint checkpoints/actionsense_S04.pt
```

恢复训练时必须保持模型结构参数一致。

## 9. 保存到 Google Drive

先在 Colab Python 单元格挂载：

```python
from google.colab import drive
drive.mount("/content/drive")
```

然后在 Bash 终端执行，不要加 `!`：

```bash
mkdir -p "/content/drive/MyDrive/AutoMachine/checkpoints"
mkdir -p "/content/drive/MyDrive/AutoMachine/data/actionsense/S04"

cp /content/AutoMachine/checkpoints/actionsense_S04.pt \
  "/content/drive/MyDrive/AutoMachine/checkpoints/"

cp -r /content/AutoMachine/data/actionsense/S04/processed \
  "/content/drive/MyDrive/AutoMachine/data/actionsense/S04/"
```

## 10. 训练任务的时间语义

当前配置 `frames_per_step=1`，因为视频和EMG RMS已经对齐到同一帧率：

```text
视频时刻 t 的空间Token
    → 当前第一Token层
    → 与历史Token管线共同读出
    → 同一时刻 t 的16通道EMG RMS
```

如果以后使用高帧率视频，可把多个原始帧组成一个内部时刻，同时把目标EMG按相同时间区间聚合。不能只降低视频频率而保持目标时间轴不变。
