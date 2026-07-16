# Colab 训练环境

## 安装与验证

```bash
cd /content/AutoMachine
python -m pip install -r requirements.txt
python -m tests.smoke_forward
```

确认 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 同步控制工程闭环

```bash
python -m automachine.train \
  --steps 200 \
  --batch-size 8 \
  --device cuda \
  --checkpoint checkpoints/control_smoke.pt
```

这个合成任务是“视频中的目标位置 -> 同步位置/速度响应”，只验证因果前向、反向传播和 checkpoint。真实数据按 [ACTIONSENSE.md](ACTIONSENSE.md) 准备。

## 保存到 Google Drive

先在 Colab Python 单元格挂载：

```python
from google.colab import drive
drive.mount('/content/drive')
```

再在 Bash 终端复制：

```bash
mkdir -p "/content/drive/MyDrive/AutoMachine/checkpoints"
cp checkpoints/*.pt "/content/drive/MyDrive/AutoMachine/checkpoints/"
```
