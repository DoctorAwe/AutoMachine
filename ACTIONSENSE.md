# ActionSense 单受试者数据处理方案

本方案面向 AutoMachine 的核心任务：

```text
第一视角视频流 -> 分层持久记忆 -> 双臂 16 通道 EMG 响应流
```

第一阶段只使用 ActionSense 受试者 `S04`。S04 只有一个完整 recording split，包含全部活动，下载和时间同步比多 split 受试者简单。

官方入口：

- 数据下载：https://action-sense.csail.mit.edu/data.html
- 文件结构：https://action-sense.csail.mit.edu/dataset_info.html
- 官方解析代码：https://github.com/delpreto/ActionNet/tree/master/parsing_data

## 1. 数据范围

首轮实验只下载：

1. S04 wearable HDF5：双臂 EMG、活动标签、第一视角视频时间戳。
2. S04 第一视角无 gaze-overlay 视频。

暂不下载五个 1600×1200 环境相机、深度、音频、触觉和身体姿态。这能显著降低 Colab 存储与解码压力。

目标模态：

| 模态 | HDF5 路径 | 典型频率 | 用途 |
|---|---|---:|---|
| 左臂 EMG | `myo-left/emg/data` | 约 160–200 Hz | 输出 0–7 |
| 右臂 EMG | `myo-right/emg/data` | 约 160–200 Hz | 输出 8–15 |
| 视频时间戳 | `eye-tracking-video-world/frame_timestamp/data` | 约 30 Hz | 视频/EMG 对齐 |
| 活动标签 | `experiment-activities/activities` | 异步事件 | 切分 episode |

每个流的时间戳通常位于同组的 `time_s`；视频时间戳组的实际叶节点可能因 recording 版本略有差异，因此下载后必须先执行第 4 节的结构检查。

## 2. Colab 下载 S04

```bash
cd /content/AutoMachine
mkdir -p data/actionsense/S04/raw
cd data/actionsense/S04/raw
```

下载 wearable HDF5：

```bash
wget -c -O S04_wearables.hdf5 \
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S04/2022-06-14_16-38-18_actionNet-wearables_S04/2022-06-14_16-38-43_streamLog_actionNet-wearables_S04.hdf5"
```

下载第一视角无 gaze-overlay 视频：

```bash
wget -c -O S04_world.mp4 \
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S04/2022-06-14_16-38-18_actionNet-wearables_S04/2022-06-14_16-38-43_S04_eye-tracking-video-world_frame.mp4"
```

验证下载：

```bash
ls -lh /content/AutoMachine/data/actionsense/S04/raw
file /content/AutoMachine/data/actionsense/S04/raw/S04_wearables.hdf5
file /content/AutoMachine/data/actionsense/S04/raw/S04_world.mp4
```

如果 `wget` 得到 HTML 页面而不是数据文件，`file` 会显示 `HTML document`。此时删除该错误文件，再从官方 S04 页面右键复制实际下载链接。

## 3. 安装处理依赖

```bash
cd /content/AutoMachine

pip install \
  h5py \
  scipy \
  opencv-python-headless \
  tqdm
```

训练环境仍需要项目基础依赖：

```bash
pip install -r requirements.txt
```

## 4. 检查 HDF5 真实结构

不同 recording 的部分视频时间戳叶节点可能略有区别。先列出全部数据集路径：

```bash
cd /content/AutoMachine

python - <<'PY'
import h5py

path = "data/actionsense/S04/raw/S04_wearables.hdf5"

with h5py.File(path, "r") as h5:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name:75s} shape={obj.shape} dtype={obj.dtype}")
    h5.visititems(show)
PY
```

重点确认以下路径存在：

```text
myo-left/emg/data
myo-left/emg/time_s
myo-right/emg/data
myo-right/emg/time_s
experiment-activities/activities/data
experiment-activities/activities/time_s
```

查找第一视角视频时间戳：

```bash
python - <<'PY'
import h5py

path = "data/actionsense/S04/raw/S04_wearables.hdf5"
with h5py.File(path, "r") as h5:
    def show(name, obj):
        lower = name.lower()
        if isinstance(obj, h5py.Dataset) and "video-world" in lower:
            print(name, obj.shape, obj.dtype)
    h5.visititems(show)
PY
```

## 5. 快速验证双臂 EMG

```bash
python - <<'PY'
import h5py
import numpy as np

path = "data/actionsense/S04/raw/S04_wearables.hdf5"
with h5py.File(path, "r") as h5:
    for device in ("myo-left", "myo-right"):
        signal = np.asarray(h5[device]["emg"]["data"])
        time_s = np.asarray(h5[device]["emg"]["time_s"]).squeeze()
        rate = (len(time_s) - 1) / (time_s[-1] - time_s[0])
        print(device)
        print("  signal:", signal.shape, signal.dtype)
        print("  time  :", time_s.shape)
        print("  rate  :", round(float(rate), 2), "Hz")
        print("  finite:", bool(np.isfinite(signal).all()))
PY
```

正常情况下，每只 Myo 输出 8 通道，拼接后为 16 通道。

## 6. 同步策略

不要按数组索引直接拼接视频和 EMG。两个设备频率不同，且可能存在丢帧，必须使用 epoch 秒时间戳对齐。

推荐统一到视频帧频率，约 30 Hz。对于每个视频帧时间 `t`：

1. 取区间 `[t-1/30, t+1/30)` 内的左右臂原始 EMG。
2. 对每个通道计算 RMS 包络：`sqrt(mean(x²))`。
3. 左右臂拼接为 16 维输出。
4. 若区间内没有足够 EMG 样本，则丢弃该视频帧。

使用 RMS 而不是原始瞬时 EMG 的原因：视频只有约 30 Hz，无法表达 160–200 Hz 原始载波；RMS 包络更接近肌肉激活强度，也更适合作为第一阶段控制响应目标。

建议处理后频率：

```text
视频输入：30 Hz，96×96 RGB
EMG 输出：30 Hz，16维 RMS 包络
```

## 7. 活动片段切分

活动标签每行格式为：

```text
[Activity, Start/Stop, Valid, Notes]
```

只保留评级不是 `Bad` 或 `Maybe`、并且同时落在视频与双臂 EMG有效时间范围内的完整活动片段。

禁止把同一个活动片段切成窗口后再随机分配到训练和验证，否则相邻视频帧会泄漏。应先按完整活动 instance 切分：

```text
前 70% 活动实例 -> train
中间 15% 活动实例 -> validation
最后 15% 活动实例 -> test
```

如果活动实例数量很少，使用前 80% 训练、最后 20% 测试，并在边界丢弃至少 2 秒数据。

## 8. 建议的处理后格式

不要把全部视频帧直接塞进单个 NPZ。视频数据较大，应按 activity episode 保存：

```text
data/actionsense/S04/processed/
  train/
    episode_000.npz
    episode_001.npz
  validation/
    episode_000.npz
  test/
    episode_000.npz
  manifest.json
```

单个 episode：

```python
frames.shape        == [time, 3, 96, 96]  # uint8
emg.shape           == [time, 16]         # float32 RMS envelope
timestamps.shape    == [time]             # float64 epoch seconds
activity            == "slice_cucumber"  # string
subject             == "S04"
```

训练时再把 `frames` 转换为 `[0,1]` 浮点数。不要在磁盘中保存 float32 视频，否则数据体积会扩大约四倍。

EMG 标准化统计量只能用训练集计算：

```text
emg_normalized = (emg - train_mean) / max(train_std, 1e-6)
```

将 `train_mean` 和 `train_std` 写入 `manifest.json`，验证集和测试集必须复用训练统计量。

## 9. 模型训练任务

ActionSense 使用项目的同步控制主模型：

```text
[B,T,3,96,96] 视频
    -> CausalVisualEncoder
    -> TokenPipelineMemory
    -> ResponseDecoder
    -> [B,T,16] EMG
```

模型类为 `SynchronousControlProcessor`，输出与输入逐时刻对齐，并支持把 `ControlState` 跨 chunk 传递。数据转换器与真实数据训练入口仍需在完成第一次 HDF5 结构检查后接入，避免对官方 recording 中可能变化的视频时间戳叶节点做错误假设。

推荐首轮训练参数：

```text
subject: S04
camera: first-person world camera
input chunk: 4 frames
frames per internal step: 1（首轮视频与 EMG 都对齐到约 30 Hz）
image size: 96×96
token dim: 64
state tokens: 16, 16, 24, 32
output: synchronized 16-channel EMG RMS
batch size: 4
learning rate: 1e-4
steps: 2000
```

训练目标不是预测未来 EMG，而是对当前视觉上下文产生同步肌肉响应：

```text
Loss = SmoothL1(predicted_emg, synchronized_emg)
     + 0.1 * MSE(diff(predicted_emg), diff(synchronized_emg))
```

必须设置两个基线：

1. 每通道训练均值输出。
2. 只根据活动标签输出每类平均 EMG。

视频模型只有在测试集误差低于这两个基线时，才能说明视觉输入提供了有效信息。

## 10. Colab 存储建议

原始文件和处理后的 episode 都应复制到 Google Drive：

```bash
mkdir -p "/content/drive/MyDrive/AutoMachine/data/actionsense/S04"

cp -r /content/AutoMachine/data/actionsense/S04/* \
  "/content/drive/MyDrive/AutoMachine/data/actionsense/S04/"
```

checkpoint 单独保存：

```bash
mkdir -p "/content/drive/MyDrive/AutoMachine/checkpoints/actionsense_S04"
```

## 11. 当前实施状态

- 下载地址和官方数据结构已确认。
- S04 已选为首个单人实验对象。
- 视频与 EMG 对齐规则、episode 切分和目标格式已确定。
- 同步视频到 EMG 主模型已经实现。
- 下一步先在 Colab 完成第 2–5 节，把实际 HDF5 路径和形状输出保存下来。
- 随后实现 ActionSense 转换器和专用训练入口；合成 `automachine.train` 只用于工程验证，不能替代真实数据训练。
