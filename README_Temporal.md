# TimeCMA_Temporal 模型使用指南

## 概述

`TimeCMA_Temporal` 是 TimeCMA 的一个变体，主要区别在于**嵌入的生成维度**：

- **原版 TimeCMA**: 在**节点维度**生成嵌入，每个节点一个嵌入 `[B, E, N]`
- **TimeCMA_Temporal**: 在**时间步维度**生成嵌入，每个时间步一个嵌入 `[B, L, E]`

## 文件结构

```
TimeCMA/
├── storage/
│   ├── gen_prompt_emb_temporal.py    # 时间步维度嵌入生成器
│   └── store_emb_temporal.py         # 存储时间步维度嵌入
├── data_provider/
│   └── data_loader_emb_temporal.py   # 加载时间步维度嵌入的数据加载器
├── models/
│   └── TimeCMA_Temporal.py           # 时间步维度嵌入模型
├── train_temporal.py                 # 训练脚本
└── scripts/
    ├── generate_temporal_embeddings.sh  # 生成嵌入脚本
    └── train_temporal.sh                 # 训练脚本
```

## 使用步骤

### 步骤 1: 生成时间步维度嵌入

首先需要为数据集生成时间步维度的嵌入：

```bash
bash scripts/generate_temporal_embeddings.sh
```

或者手动运行：

```bash
# 生成训练集嵌入
python storage/store_emb_temporal.py --data_path ETTh1 --divide train

# 生成验证集嵌入
python storage/store_emb_temporal.py --data_path ETTh1 --divide val

# 生成测试集嵌入
python storage/store_emb_temporal.py --data_path ETTh1 --divide test
```

嵌入文件会保存在 `./Embeddings_Temporal/ETTh1/{train|val|test}/` 目录下。

### 步骤 2: 训练模型

```bash
bash scripts/train_temporal.sh
```

或者手动运行：

```bash
python train_temporal.py \
  --data_path ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 16 \
  --channel 32 \
  --learning_rate 1e-4 \
  --dropout_n 0.2 \
  --e_layer 1 \
  --d_layer 2 \
  --head 8 \
  --epochs 100 \
  --seed 2024
```

## 模型架构差异

### 原版 TimeCMA
- 嵌入形状: `[B, E, N]` - 每个节点一个嵌入
- 跨模态对齐: 在节点维度对齐，时间序列特征 `[B, C, N]` 与文本嵌入 `[B, E, N]` 对齐
- 编码器: 在节点之间做注意力

### TimeCMA_Temporal
- 嵌入形状: `[B, L, E]` - 每个时间步一个嵌入
- 跨模态对齐: 在时间步维度对齐，时间序列特征 `[B, L, C]` 与文本嵌入 `[B, L, E]` 对齐
- 编码器: 在时间步之间做注意力

## 参数说明

训练参数与原版 TimeCMA 相同：

- `--data_path`: 数据集名称（目前支持 ETTh1, ETTh2）
- `--seq_len`: 输入序列长度
- `--pred_len`: 预测长度
- `--channel`: 特征维度
- `--num_nodes`: 节点数
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--dropout_n`: Dropout 率
- `--e_layer`: 编码器层数
- `--d_layer`: 解码器层数
- `--head`: 注意力头数
- `--epochs`: 训练轮数
- `--seed`: 随机种子

## 注意事项

1. **嵌入生成**: 时间步维度嵌入生成比节点维度慢，因为需要为每个时间步生成一个嵌入
2. **内存占用**: 时间步维度嵌入的内存占用与序列长度成正比
3. **数据支持**: 目前只支持 ETTh1 和 ETTh2，如需支持其他数据集，需要扩展 `data_loader_emb_temporal.py`

## 实验结果

训练完成后，结果会自动记录到 `experiment_results.log`，模型名称标记为 `TimeCMA_Temporal`。
