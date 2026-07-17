# CRCNS pvc-3：单记录视觉刺激 → 神经响应训练

## 任务定义

pvc-3 包含猫初级视觉皮层（area 17）中 10 个同步记录神经元对视觉刺激的脉冲时间。本项目使用自然电影实验，把每个 64×64 灰度帧作为同步感知输入，把校正神经延迟后该帧时间窗内 10 个神经元的 spike count 作为响应输出。

这不是下一帧预测。模型输出当前刺激对应的神经响应 log-rate，训练损失为 Poisson NLL。

## 1. 在 Colab 获取数据

CRCNS 下载要求账户授权。先登录 CRCNS，从 pvc-3 下载页取得数据包，再上传到 Colab `/content/pvc3_download/` 或 Google Drive。不要用网站预览 AVI 代替原始刺激。

```bash
%cd /content
!mkdir -p pvc3_raw
!find pvc3_download -maxdepth 2 -type f -print
!unzip -q -o /content/pvc3_download/pvc-3.zip -d /content/pvc3_raw
# tar.gz 改用：!tar -xzf 文件.tar.gz -C /content/pvc3_raw
```

## 2. 检查实际文件布局

不同刺激实验不能混在一次对齐中。先检查：

```bash
%cd /content/AutoMachine
!python -m automachine.prepare_pvc3 --inspect-root /content/pvc3_raw
```

检查器会标出 `.spk` 的 uint64 时间戳数量，以及可能的 64×64 原始电影帧数。结合包内 README/用户指南，确认同一次 natural-movie 记录的一个 movie 文件、10 个同步神经元 `.spk` 文件、实际帧率、电影开始时刻和视觉反应延迟。不要只按大小猜实验对应关系。

## 3. 对齐并生成窗口

将占位路径换成真实路径：

```bash
%cd /content/AutoMachine
!python -m automachine.prepare_pvc3 \
  --movie "/content/pvc3_raw/自然电影文件" \
  --spike-files \
    "/content/pvc3_raw/neuron00.spk" \
    "/content/pvc3_raw/neuron01.spk" \
    "/content/pvc3_raw/neuron02.spk" \
    "/content/pvc3_raw/neuron03.spk" \
    "/content/pvc3_raw/neuron04.spk" \
    "/content/pvc3_raw/neuron05.spk" \
    "/content/pvc3_raw/neuron06.spk" \
    "/content/pvc3_raw/neuron07.spk" \
    "/content/pvc3_raw/neuron08.spk" \
    "/content/pvc3_raw/neuron09.spk" \
  --frame-rate 30 \
  --onset-us 0 \
  --response-delay-ms 50 \
  --window 32 \
  --stride 16 \
  --output data/pvc3/processed
```

`30 Hz / onset=0 / delay=50 ms` 只是初值，必须按下载包元数据修改。若没有 spike 落入区间，核对这三个参数、时间单位和所选记录。

## 4. 训练 16 层模型

默认 PVC-3 配置为：1 通道输入、10 维响应、128 维 token、8 个注意力头，16 层 token 数：

```text
16 16 16 16 16 16 16 16 24 24 24 24 32 32 48 64
```

先做短验证：

```bash
%cd /content/AutoMachine
!python -m automachine.train_pvc3 \
  --data data/pvc3/processed \
  --steps 300 \
  --batch-size 2 \
  --device cuda \
  --checkpoint checkpoints/pvc3_16layer_smoke.pt
```

确认 `val_poisson < mean_baseline` 后完整训练：

```bash
!python -m automachine.train_pvc3 \
  --data data/pvc3/processed \
  --steps 3000 \
  --batch-size 4 \
  --device cuda \
  --checkpoint checkpoints/pvc3_16layer.pt
```

显存不足先将 batch size 改为 2 或 1。恢复训练：

```bash
!python -m automachine.train_pvc3 \
  --data data/pvc3/processed --steps 3000 --batch-size 2 --device cuda \
  --resume checkpoints/pvc3_16layer.pt \
  --checkpoint checkpoints/pvc3_16layer.pt
```

## 5. 有效性判据

- 验证集 `val_poisson` 必须低于训练集均值响应基线；
- 再按神经元计算 held-out movie 上预测 firing rate 与真实重复试验 PSTH 的相关性；
- 若电影重复呈现，应按完整 trial 划分数据，不要把同一次呈现的相邻窗口随机拆开；
- 当前转换器针对“一段电影 + 10 个同步神经元文件”。重复 trial 必须保留边界，不能直接拼 spike 时间戳。
