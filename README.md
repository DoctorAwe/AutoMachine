# AutoMachine / 同步神经处理机

AutoMachine 是一个面向连续感知与控制的因果流式模型：

```text
同步视觉/传感器输入 -> 分层持久记忆 -> 同步响应/控制信号
```

它不以预测下一帧为核心任务。模型在每个输入时刻产生一个同频率响应，并允许状态跨数据 chunk 持续存在。

## 核心结构

```text
RGB 视频 [B,T,C,H,W]
    -> 因果频率适配器（每 K 帧压缩为一个内部时刻）
    -> 固定空间 Token 网格（保留局部信息）
    -> 可选辅助传感器融合 [B,T,F]
    -> 时间移位 Token 管线
       - 新输入只进入第一层
       - 旧 Layer i 状态在下一步流向 Layer i+1
       - 后层可以使用更大的 Token 容量
    -> 多尺度注意力读出
    -> 同步响应 [B,T,A]
```

核心接口：

```python
from automachine import SynchronousControlConfig, SynchronousControlProcessor

config = SynchronousControlConfig(
    image_size=96,
    frames_per_step=1,
    auxiliary_features=0,
    response_dim=16,
    state_tokens=(16, 16, 24, 32),
)
model = SynchronousControlProcessor(config)

response, state = model.forward_chunk(video_chunk)
next_response, state = model.forward_chunk(next_video_chunk, state=state)
```

`ControlState.detach()` 用于截断跨 chunk 的反向传播历史，同时保留在线记忆值。

## 验证

```powershell
python -m pytest -q
python -m tests.smoke_forward
```

测试会验证：

- 输入和输出严格逐时刻对齐；
- 修改未来帧不会改变过去响应；
- 整段处理与分 chunk 流式处理结果一致；
- 新输入每个内部时刻只向后传播一层；
- 不完整输入分组能跨 chunk 缓冲；
- 可选辅助传感器能与视觉融合。

## 最小训练闭环

合成任务用移动目标视频同步输出位置和速度，只用于验证工程闭环：

```powershell
python -m automachine.train --steps 200 --device auto
```

真实生物响应任务使用 ActionSense 的第一视角视频到双臂 16 通道 EMG，数据准备见 [ACTIONSENSE.md](ACTIONSENSE.md)。

## 输出安全边界

模型输出应解释为归一化动作目标、速度目标、肌肉激活或其他高层响应。真实硬件的电流、电压和 PWM 应由带限幅、速率限制、急停和故障检测的确定性低层控制器产生，不应直接把未经约束的神经网络输出接到执行器。
