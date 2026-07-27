# Colab：固定容量联想记忆最小验证

这个实验只验证一种能力：

```text
本次episode中的视觉线索 + 辅助标签
→ 经过长时间干扰
→ 再次只看到视觉线索
→ 回忆本次episode随机绑定的标签
```

每个episode都会重新随机排列4种线索和4种标签，因此固定网络权重无法提前背下映射。学习阶段后默认有48个干扰时间步，超过16层短时Token管线。

## 1. Colab环境

在Colab选择“代码执行程序 → 更改运行时类型 → T4 GPU”，然后进入项目：

```bash
%cd /content/AutoMachine
!pip install -q -r requirements.txt
```

先确认当前代码包含实验入口：

```bash
!python -m automachine.train_associative --help
```

## 2. 200步快速冒烟测试

```bash
!python -m automachine.train_associative \
  --steps 200 \
  --batch-size 8 \
  --gap 32 \
  --memory-capacity 32 \
  --eval-every 50 \
  --device cuda \
  --checkpoint checkpoints/associative_quick.pt
```

这个阶段只确认程序、CUDA、记忆写入和三组对照能够运行，不要求达到最终准确率。

## 3. 正式最小验证

```bash
!python -m automachine.train_associative \
  --steps 800 \
  --batch-size 8 \
  --image-size 24 \
  --token-dim 32 \
  --cue-frames 2 \
  --gap 48 \
  --query-frames 2 \
  --memory-capacity 32 \
  --retrieval-threshold 0.40 \
  --eval-every 100 \
  --device cuda \
  --checkpoint checkpoints/associative_smoke.pt
```

输出示例：

```text
step=0800 recall=91.0% empty=26.2% wrong=24.6% gain=64.8% slots=4.0/32
```

指标含义：

- `recall`：正常长期记忆条件；
- `empty`：保留完全相同的16层短时状态，但清空长期记忆；
- `wrong`：保留检索Key，但打乱槽位Value；
- `gain`：`recall - empty`；
- `slots`：每个episode平均占用槽位数。

第一阶段通过标准：

```text
recall >= 80%
empty <= 40%
wrong <= 40%
gain >= 40%
slots <= memory_capacity
```

理想结果是 `recall >= 90%`，而 `empty` 和 `wrong` 接近四分类随机水平25%。

## 4. 更长间隔测试

如果48步通过，保持其他参数不变，把间隔增加到96：

```bash
!python -m automachine.train_associative \
  --steps 1000 \
  --batch-size 8 \
  --gap 96 \
  --memory-capacity 32 \
  --eval-every 100 \
  --device cuda \
  --checkpoint checkpoints/associative_gap96.pt
```

这会显著增加训练时间，但更能排除短时Token管线残留。

## 5. 查看最终指标

```bash
!cat checkpoints/associative_smoke_metrics.json
```

## 6. 保存到Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!mkdir -p /content/drive/MyDrive/AutoMachine/checkpoints
!cp checkpoints/associative_smoke.pt \
  /content/drive/MyDrive/AutoMachine/checkpoints/
!cp checkpoints/associative_smoke_metrics.json \
  /content/drive/MyDrive/AutoMachine/checkpoints/
```

## 结果判断

如果 `recall`、`empty` 和 `wrong` 都很高，模型很可能绕过了长期记忆对照；如果三者都在25%左右，说明检索或联想Token没有学会；只有正常记忆明显高、清空和错误记忆接近随机水平，才能说明固定槽位实现了内容相关的情景联想。
