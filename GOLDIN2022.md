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

### 默认融合行为

16 层融合器采用随深度变化的可训练先验：

```text
层位置                 第 1 层 ------------------------------ 第 16 层
初始候选接受率           0.95                                    0.10
刺激显著性阈值           0.05                                    0.20
默认倾向               快速接受输入                         长时间保持自身状态
```

每层实际融合门为：

```text
实际融合门 = 可训练状态门 × sigmoid((输入更新强度 - 当前层阈值) / 0.025)
新状态     = 实际融合门 × 候选状态 + (1 - 实际融合门) × 旧状态
```

所以弱刺激默认难以改变任何一层；强刺激更容易进入前层，但要继续进入深层必须产生更明显的更新。
接受率和阈值提供初始化行为，注意力、候选状态和状态门仍通过训练调整。

该融合初始化属于新模型行为。旧 checkpoint 中已经训练过的 gate bias 不代表这套默认曲线，不能用于
验证新设计；请使用新的 checkpoint 文件名从 step 0 开始训练。

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

新版验证会先从重叠窗口重建不重复的连续时间轴，再通过 `forward_chunk` 保持状态贯穿整个验证段。
默认使用 `8,16,7,13` 个时间步的不规则 chunk，并报告：

```text
response_r_mean/median  完整连续区间内逐神经元 Pearson 的均值和中位数
response_r_positive     有效神经元中 Pearson 大于零的比例
delta_r_mean            相邻时间步响应变化量的逐神经元平均 Pearson
delta_r_positive        动态相关为正的神经元比例
neurons_better          Poisson 损失优于各神经元均值基线的神经元比例
time_steps              去除重叠窗口后实际评估的唯一时间点数量
causal_max  修改后半段未来刺激后，前半段输出的最大变化
stream_max  流式分块输出与整段因果输出的最大差异
```

结构检查正常时，`causal_max` 和 `stream_max` 应接近浮点误差（通常约 `1e-6` 或更小）；
`response_r` 和 `delta_r` 越高越好。Poisson 超过均值基线但相关系数接近零，仍不能认为时间响应已经拟合成功。

Poisson、响应相关性和动态相关性均在重建后的完整连续验证段上一次性按神经元计算，不再先算
batch 相关再平均，也不会重复计算重叠窗口的时间点。每次结果还会保存到
`checkpoints/*_validation_metrics.json`，其中包含每个神经元的 correlation 和 Poisson 数组。

旧版没有 `starts` 字段的 shards 可以按照原始固定 stride 推导位置；下次重新处理数据时会直接写入
精确起点。因果性和流式一致性默认从连续验证段抽取两个区间诊断。可调整：

```bash
--stream-chunks 8 16 7 13 --diagnostic-batches 4
```

完整验证过慢时可暂时限制连续时间长度，例如 `--evaluation-max-steps 2000`；最终模型评估应使用
默认值 `0`，即完整区间。`--validation-batches` 仅为兼容旧命令保留，已不再控制新版评估范围。

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

## 7. 在 Colab 启动在线演示

安装界面依赖：

```bash
%cd /content/AutoMachine
!python -m pip install -q -r requirements-demo.txt
```

使用新的深度融合 checkpoint：

```bash
!python -m automachine.demo_goldin2022 \
  --checkpoint checkpoints/goldin2022_depth_fusion_v1.pt \
  --data data/goldin2022/processed \
  --device cuda \
  --share
```

Colab 输出中会出现一个 Gradio 公网地址。打开后可以：

- 在 validation/test 连续序列中选择起点和展示长度；
- 选择一个神经元查看真实响应、流式模型响应和均值基线；
- 设置 token 状态预热长度和不规则 chunk 序列；
- 查看全部神经元的真实/预测响应热图；
- 查看逐神经元 response correlation、delta correlation 和 Poisson；
- 同时检查未来扰动因果误差与整段/流式一致性误差。

演示会读取 checkpoint 内保存的模型配置，不会用当前代码默认值覆盖已经训练的 16 层深度融合模型。

## 8. 修复平均值直线输出

如果模型曲线与均值基线几乎重合，说明普通 Poisson 在稀疏响应上收敛到了平均 firing rate。
新版训练目标为：

```text
总损失 = 非零事件加权 Poisson
       + 0.25 × 逐神经元时间相关损失
       + 0.05 × 相邻响应差分损失
```

可以从深度融合 checkpoint 继续纠正，并保存为新文件：

```bash
!python -m automachine.train_goldin2022 \
  --data data/goldin2022/processed \
  --steps 4000 \
  --batch-size 4 \
  --lr 5e-5 \
  --correlation-weight 0.25 \
  --delta-weight 0.05 \
  --event-weight 1.0 \
  --device cuda \
  --resume checkpoints/goldin2022_depth_fusion_v1.pt \
  --checkpoint checkpoints/goldin2022_depth_fusion_v2.pt
```

`--steps` 是总步数，不是追加步数；如果原 checkpoint 已到 3000，上述命令会继续训练到 4000。
新版日志会分别报告 `poisson`、`corr_loss` 和 `delta_loss`。评估新增：

```text
modulation             逐神经元 prediction_std / target_std 的中位数
neurons_modulated      modulation ratio > 0.1 的神经元比例
```

修复有效时，`response_r_mean` 应上升、`modulation` 应离开零，同时 Poisson 不应长期显著差于基线。
