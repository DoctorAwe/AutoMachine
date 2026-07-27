# AutoMachine / 同步神经处理机

当前默认真实生物任务已更改为 **Li et al. 2025 小鼠上丘自然电影—钙响应**。下载、单成像平面处理和 Colab 训练见 [LI2025.md](LI2025.md)。Goldin 2022 流程只作为历史静态刺激实验保留，不再作为连续流式任务的推荐数据。

AutoMachine 是面向“同步感知信号 → 同步响应/控制信号”的因果流式模型：

```text
连续视觉/传感器输入 → 因果编码器 → 16 层时间移位 Token 管线 → 注意力读出 → 同步响应
```

新输入只进入第一层；旧状态每个内部时间步向下一层移动一次。前层默认更易接受强输入，后层更倾向保留自身状态。`forward_chunk` 可在任意数据块边界持续携带状态。

模型现已加入固定容量的长期联想记忆。相似刺激可检索过去同一情景中的多感官潜在表示，并作为明确标记来源的联想Token参与注意力。槽位写入、融合、替换和跨进程保存方法见 [ASSOCIATIVE_MEMORY.md](ASSOCIATIVE_MEMORY.md)。

在使用真实生物数据前，建议先按 [ASSOCIATIVE_SMOKE.md](ASSOCIATIVE_SMOKE.md) 在Colab运行随机线索—标签配对实验，分别比较正常记忆、清空记忆和错误记忆。

```python
from automachine import SynchronousControlConfig, SynchronousControlProcessor

config = SynchronousControlConfig(response_dim=64)
model = SynchronousControlProcessor(config)
response, state = model.forward_chunk(video_chunk)
response_next, state = model.forward_chunk(next_video_chunk, state=state)
```

## Li 2025 快速入口

```bash
python -m automachine.prepare_li2025 --root /path/to/li_2025 --inspect
python -m automachine.prepare_li2025 --root /path/to/li_2025 --output data/li2025/plane_one
python -m automachine.train_li2025 --data data/li2025/plane_one --device cuda
```

## 验证

```bash
python -m pytest -q
python -m tests.smoke_forward
```

模型会检查逐时刻对齐、未来扰动不改变过去、整段与分块流式输出一致以及 Token 状态的逐层因果传播。
