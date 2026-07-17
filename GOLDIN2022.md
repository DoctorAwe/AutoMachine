# Goldin et al. 2022 单视网膜训练（Colab）

## 任务

使用一个 Goldin 2022 HDF5 session：

```text
自然视觉刺激序列 → 同步视网膜神经节细胞 binned response
```

每个 session 单独训练一个模型；输出维度由该文件记录的神经元数量自动确定。模型保持 128 维 token 和 16 层状态管线：

```text
16 16 16 16 16 16 16 16 24 24 24 24 32 32 48 64
```

## 1. 下载项目与依赖

```bash
%cd /content
!git clone https://github.com/DoctorAwe/AutoMachine.git
%cd /content/AutoMachine
!python -m pip install -q -r requirements-goldin2022.txt
!python -m pip install -q -U huggingface_hub
```

如果仓库已经存在：

```bash
%cd /content/AutoMachine
!git pull
```

## 2. 匿名下载 Goldin 2022

无需 Hugging Face 登录：

```bash
%cd /content
!hf download open-retina/open-retina \
  --repo-type dataset \
  --include "marre_lab/goldin_2022/*" \
  --local-dir open_retina_data
```

列出可用 session：

```bash
!find /content/open_retina_data/marre_lab/goldin_2022 \
  -type f -name "*.h5" -print
```

先选择一个名称包含 `mouse` 的 HDF5 文件。以下用环境变量保存路径，避免反复复制长文件名：

```python
from pathlib import Path

files = sorted(Path("/content/open_retina_data/marre_lab/goldin_2022").rglob("*mouse*.h5"))
for i, path in enumerate(files):
    print(i, path)
assert files, "没有找到 mouse HDF5；请查看实际下载目录"
SESSION = str(files[0])
print("selected:", SESSION)
```

## 3. 检查单个 session

在 Colab 中，`!` 命令要用 `$SESSION` 才能展开 Python 变量：

```bash
%cd /content/AutoMachine
!python -m automachine.prepare_goldin2022 \
  --session "$SESSION" \
  --inspect
```

必须至少看到：

```text
/train/stimulus
/train/response/binned
/test/stimulus
/test/response/binned
```

## 4. 转换数据

```bash
!python -m automachine.prepare_goldin2022 \
  --session "$SESSION" \
  --window 16 \
  --stride 8 \
  --validation-fraction 0.1 \
  --output data/goldin2022/processed
```

处理器执行以下操作：

- 自动识别刺激、响应的时间轴；
- 保留官方 test split；
- 从原 train 尾部切出 validation，避免随机窗口泄漏；
- 只使用最终 train 段计算刺激均值和标准差；
- 将每个连续窗口保存为 `[time,height,width,channel]`；
- 自动记录该 session 的神经元数量和训练均值基线。

## 5. 小规模有效性验证

```bash
!python -m automachine.train_goldin2022 \
  --data data/goldin2022/processed \
  --steps 300 \
  --batch-size 2 \
  --device cuda \
  --checkpoint checkpoints/goldin2022_smoke.pt
```

判断模型开始有效：

```text
val_poisson < mean_baseline
better=True
```

300 步只用于检查数据方向、损失和反向传播，不代表最终拟合质量。

## 6. 完整训练

```bash
!python -m automachine.train_goldin2022 \
  --data data/goldin2022/processed \
  --steps 3000 \
  --batch-size 4 \
  --device cuda \
  --checkpoint checkpoints/goldin2022_16layer.pt
```

显存不足时将 batch size 改成 2 或 1。程序在 Ctrl+C 时自动保存。恢复：

```bash
!python -m automachine.train_goldin2022 \
  --data data/goldin2022/processed \
  --steps 3000 \
  --batch-size 2 \
  --device cuda \
  --resume checkpoints/goldin2022_16layer.pt \
  --checkpoint checkpoints/goldin2022_16layer.pt
```

训练完成会报告官方 test split 的 `test_poisson`。最终 checkpoint 同时保存模型配置与数据 manifest，便于之后恢复正确的输入通道和神经元输出维度。
