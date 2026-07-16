# 可用真实时序数据集

## 推荐顺序

### 1. UCI HAR（现在即可接入）

- 30 名受试者，手机腰部佩戴；三轴加速度与三轴陀螺仪，50 Hz。
- 官方压缩包约 58 MB，原始惯性窗口已经整理成 128×9，CC BY 4.0。
- 适合先做“过去信号 → 后续信号”预测，数据轻、下载和显存压力低。
- 官方页面：https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- 本仓库转换器：`python -m automachine.prepare_uci_har`。

这是当前首选：先证明模型能学习真实人体运动动力学，再扩大模型或处理高采样率信号。

### 2. PhysioNet GRABMyo（第二阶段）

- 43 名健康参与者、16 种手势、3 次采集；EMG 采样率 2048 Hz。
- 32 通道记录，完整数据解压约 9.4 GB，PhysioNet Credentialed Health Data License 1.5.0。
- 适合波形续写、动作分类辅助任务，以及输入 EMG → 输出运动状态。
- 官方页面：https://physionet.org/content/grabmyo/1.1.0/

建议在云端下载，先降采样到 256–512 Hz 并切成 1–2 秒窗口；不要直接把全部 2048 Hz 长记录送进注意力层。

### 3. DANDI 000728（神经活动路线）

- Allen Visual Coding 光学成像；NWB 文件中可直接取 `DfOverF`，形状为 `[时间, ROI]`。
- 同时含行为视频，后续可研究“神经活动/视觉输入 → 行为”的多模态任务。
- DANDI 文档示例：https://docs.dandiarchive.org/example-notebooks/tutorials/open_data_quick_start_2026/Get-to-know-a-Dandiset/

这条路线科学价值高，但需要 `dandi`、`pynwb` 和按文件流式读取，放在模型闭环验证之后。

## 暂不优先

- 原始动物视频/DeepLabCut：适合最终的视频输入目标，但下载、解码和标注格式差异较大。
- 大规模 Neuropixels：通道数与采样率都高，应先设计分块、降采样和稀疏读出。

## 统一数据约定

特征训练入口读取：

```python
sequences.shape == [samples, time, features]
```

训练时目标与模型输出自动按下一时刻对齐。正式实验必须按受试者或 session 切分，不能把同一人的相邻窗口随机分到训练和测试集。
