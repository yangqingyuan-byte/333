# TimeCMA_NodeEmbed 模型说明

## 概述

这是一个修改版的 TimeCMA 模型，主要变化是**在节点数维度上做嵌入，而不是在时间步维度上做嵌入**。

## 主要变化

### 原模型 (TimeCMA)
- 嵌入方式：在时间步维度上做嵌入
- 流程：`[B, L, N]` → permute → `[B, N, L]` → `Linear(L, C)` → `[B, N, C]`
- 序列长度 = 节点数 N，特征维度 = C
- 在节点之间做注意力，学习空间关系

### 新模型 (TimeCMA_NodeEmbed)
- 嵌入方式：在节点数维度上做嵌入
- 流程：`[B, L, N]` → `Linear(N, C)` → `[B, L, C]`
- 序列长度 = 时间步 L，特征维度 = C
- 在时间步之间做注意力，学习时间关系

## 文件说明

- `models/TimeCMA_NodeEmbed.py`: 新模型定义
- `train_nodeembed.py`: 训练脚本
- `test_nodeembed.py`: 快速测试脚本
- `scripts/train_nodeembed_etth1.sh`: ETTh1数据集训练脚本
- `scripts/test_nodeembed_etth1.sh`: ETTh1数据集测试脚本

## 使用方法

### 1. 快速测试

首先运行测试脚本验证模型是否能正常工作：

```bash
bash scripts/test_nodeembed_etth1.sh
```

或者直接运行：

```bash
python test_nodeembed.py --data_path ETTh1
```

### 2. 训练模型

运行训练脚本：

```bash
bash scripts/train_nodeembed_etth1.sh
```

或者直接运行：

```bash
python train_nodeembed.py \
  --data_path ETTh1 \
  --batch_size 16 \
  --seq_len 96 \
  --pred_len 96 \
  --epochs 10 \
  --channel 32 \
  --learning_rate 1e-4
```

### 3. 参数说明

主要参数：
- `--data_path`: 数据集名称（默认：ETTh1）
- `--seq_len`: 输入序列长度（默认：96）
- `--pred_len`: 预测长度（默认：96）
- `--batch_size`: 批次大小（默认：16）
- `--channel`: 特征维度（默认：32）
- `--learning_rate`: 学习率（默认：1e-4）
- `--epochs`: 训练轮数（默认：10，用于快速测试）

## 架构差异

### Embedding 处理

原模型：
- embeddings: `[B, E, N]` → permute → `[B, N, E]` → encoder
- 每个节点一个 embedding

新模型：
- embeddings: `[B, E, N]` → permute → `[B, N, E]` → expand → `[B, L, E]` → project → `[B, L, C]`
- 将节点维度的 embedding 扩展到时间步维度，然后投影到与时间序列特征相同的维度

### Cross-Modal Alignment

原模型：
- 使用 `CrossModal` 层，在节点维度上对齐
- 输入：`[B, C, N]` 和 `[B, E, N]`

新模型：
- 使用 `TransformerDecoderLayer` 做 cross-attention，在时间步维度上对齐
- 输入：`[B, L, C]` 和 `[B, L, C]`

## 注意事项

1. 新模型在时间步维度上学习，可能更适合捕捉时间依赖关系
2. embeddings 需要从节点维度扩展到时间步维度，这可能会丢失一些节点特定的信息
3. 训练时建议先用较少的 epochs 进行快速测试，确认模型能正常运行后再进行完整训练

## 实验建议

- 对比原模型和新模型在相同数据集上的性能
- 尝试不同的超参数组合
- 观察两种嵌入方式对模型性能的影响
