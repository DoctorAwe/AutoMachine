# 在线训练环境使用步骤

这份说明用于把当前 AutoMachine 原型放到 Colab、Kaggle、云 GPU Notebook 或远程 Linux 训练机上跑起来。

## 1. 上传代码

任选一种方式：

```bash
git clone <your-repo-url> AutoMachine
cd AutoMachine
```

或者把整个 `D:\WorkSpace\project\Python\AutoMachine` 目录压缩上传，解压后进入目录。

不要上传本地 `.venv/`、`__pycache__/`、`checkpoints/`，这些已经在 `.gitignore` 里排除了。

## 2. 创建环境

云端一般已有 Python。建议使用 Python 3.11 或 3.12：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果平台已经预装 PyTorch，也可以跳过最后一行，直接验证。

## 3. 验证 GPU 和模型前向

```bash
python -c "import torch; print(torch.__version__); print('cuda=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
python -m tests.smoke_forward
```

看到 `OK output_shape=...` 就说明模型前向可用。

## 4. 跑一个最小训练闭环

```bash
python -m automachine.train \
  --steps 100 \
  --batch-size 8 \
  --token-dim 64 \
  --device auto \
  --checkpoint checkpoints/synthetic_smoke.pt \
  --save-every 50
```

这个任务使用合成移动光斑视频，只用于验证训练流程、显存和 checkpoint。

## 5. 断点恢复

```bash
python -m automachine.train \
  --steps 300 \
  --batch-size 8 \
  --token-dim 64 \
  --device auto \
  --resume checkpoints/synthetic_smoke_step_000100.pt \
  --checkpoint checkpoints/synthetic_smoke.pt \
  --save-every 50
```

`--steps` 是目标总步数，不是额外再跑多少步。

## 6. 云端建议参数

小显存 GPU：

```bash
python -m automachine.train --steps 500 --batch-size 4 --token-dim 64 --device auto
```

中等显存 GPU：

```bash
python -m automachine.train --steps 1000 --batch-size 8 --token-dim 96 --device auto
```

更大模型可以之后再把 `state_tokens`、层数和真实数据集接进训练脚本。

## 7. 下一步数据方向

当前代码只验证工程闭环。真正项目数据建议按优先级接入：

1. 动物运动视频或姿态序列，做下一帧/下一步预测。
2. 肌电、脑电、神经 spike train 等真实时域信号，做后续波形预测。
3. 输入视频或传感器流，输出控制信号流。
