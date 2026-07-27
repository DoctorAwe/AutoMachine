# 固定容量长期联想记忆

当前模型包含两种状态：

```text
ControlState.layers       16层短时工作记忆
ControlState.associative  固定容量长期情景记忆
```

长期记忆不是不断增长的样本列表，而是固定形状的槽位张量。默认配置为256个槽位、每槽8个128维联想Token、每次检索Top-4。

## 工作过程

每个内部时间步执行：

1. 当前视觉和辅助感官Token产生查询；
2. 在已占用槽位中计算相似度和可靠性；
3. 达到阈值的Top-K记忆生成“联想来源”Token；
4. 当前Token和联想Token共同进入16层管线及注意力读出；
5. 写入控制器在固定槽位内执行融合、写入或替换。

槽位分为候选、稳定和保护三级。重复且一致的经历提高使用次数、置信度与等级；低价值且可替换的记忆会让位于更显著的新经历。保护记忆不能被普通写入覆盖。

## 配置

```python
from automachine import SynchronousControlConfig, SynchronousControlProcessor

config = SynchronousControlConfig(
    associative_memory_enabled=True,
    associative_memory_capacity=256,
    associative_value_tokens=8,
    associative_top_k=4,
    associative_retrieval_threshold=0.72,
    associative_retrieval_temperature=0.10,
    associative_merge_key_threshold=0.88,
    associative_merge_value_threshold=0.80,
    associative_write_threshold=0.65,
    associative_replacement_margin=0.10,
)
model = SynchronousControlProcessor(config)
```

## 在线持续使用

必须持续传入返回的 `state`。调用普通 `model(video)` 会在调用结束后丢弃运行状态。

```python
state = None

response, state = model.forward_chunk(first_chunk, state=state)
response, state = model.forward_chunk(second_chunk, state=state)
```

如有外部奖励、疼痛或人工标注的显著度，可传入 `[batch, internal_time]` 的0到1数值：

```python
response, state = model.forward_chunk(
    video,
    auxiliary=sensors,
    state=state,
    memory_salience=salience,
)
```

`memory_salience` 与输入新颖度共同决定是否写入。推理或评估时可关闭写入但继续检索：

```python
response, state = model.forward_chunk(
    video,
    state=state,
    update_associative_memory=False,
)
```

## 跨进程保存和恢复

网络权重和长期记忆内容应分开保存：

```python
torch.save(model.state_dict(), "model.pt")
model.save_associative_memory("long_term_memory.pt", state)
```

恢复：

```python
model.load_state_dict(torch.load("model.pt", map_location=device))
state = model.init_state(batch=1, device=device)
state.associative = model.load_associative_memory(
    "long_term_memory.pt", device=device
)
```

长期记忆按batch保存。真正的在线单体模型建议固定 `batch=1`；训练batch中的每个样本拥有独立记忆池，不会相互写入。

## 与权重学习的边界

- 查询编码器、Value压缩器、来源嵌入和注意力是可训练参数；
- 槽位内容是显式状态，通过受控规则更新，不参与普通梯度更新；
- 在线刺激不会直接修改主模型权重；
- 后续可增加离线重放，让高价值槽位缓慢巩固到权重中。

这使记忆容量、来源、替换和持久化都可检查，也避免在线权重更新造成灾难性遗忘。
