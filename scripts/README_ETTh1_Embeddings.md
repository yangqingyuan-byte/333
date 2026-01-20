# ETTh1 GPT-2 嵌入生成脚本使用说明

本目录包含用于生成ETTh1数据集的GPT-2嵌入的脚本，包括训练、验证和测试集。

## 文件说明

1. **generate_etth1_embeddings.sh** - Bash脚本，用于批量生成所有数据集的嵌入
2. **generate_etth1_embeddings.py** - Python脚本，提供更灵活的控制和进度显示

## 使用方法

### 方法1: 使用Bash脚本（推荐）

```bash
# 在项目根目录下运行
bash scripts/generate_etth1_embeddings.sh
```

该脚本会自动为训练、验证和测试集生成嵌入。

### 方法2: 使用Python脚本

```bash
# 生成所有数据集的嵌入
python scripts/generate_etth1_embeddings.py --data_path ETTh1

# 只生成训练集的嵌入
python scripts/generate_etth1_embeddings.py --data_path ETTh1 --divide train

# 只生成验证集的嵌入
python scripts/generate_etth1_embeddings.py --data_path ETTh1 --divide val

# 只生成测试集的嵌入
python scripts/generate_etth1_embeddings.py --data_path ETTh1 --divide test

# 自定义参数
python scripts/generate_etth1_embeddings.py \
    --data_path ETTh1 \
    --input_len 96 \
    --output_len 96 \
    --device cuda \
    --batch_size 1 \
    --model_name gpt2
```

### 方法3: 使用原始脚本（逐个生成）

```bash
# 生成训练集嵌入
python storage/store_emb.py --data_path ETTh1 --divide train

# 生成验证集嵌入
python storage/store_emb.py --data_path ETTh1 --divide val

# 生成测试集嵌入
python storage/store_emb.py --data_path ETTh1 --divide test
```

## 参数说明

- `--data_path`: 数据集名称，默认为 "ETTh1"
- `--input_len`: 输入序列长度，默认为 96
- `--output_len`: 输出序列长度，默认为 96
- `--device`: 设备类型，默认为 "cuda"
- `--batch_size`: 批次大小，默认为 1
- `--model_name`: GPT-2模型名称，默认为 "gpt2"
- `--d_model`: GPT-2模型维度，默认为 768
- `--l_layers`: GPT-2层数，默认为 12
- `--divide`: 数据集划分，可选 "train", "val", "test", "all"（默认）
- `--root_path`: 数据根路径，默认为项目内的 "./dataset/" 目录

## 输出位置

生成的嵌入文件保存在：
```
./Embeddings/ETTh1/
├── train/
│   ├── 0.h5
│   ├── 1.h5
│   └── ...
├── val/
│   ├── 0.h5
│   ├── 1.h5
│   └── ...
└── test/
    ├── 0.h5
    ├── 1.h5
    └── ...
```

每个 `.h5` 文件包含一个样本的嵌入，使用HDF5格式存储。

## 日志文件

日志文件保存在：
```
./Results/emb_logs/
├── ETTh1_train.log
├── ETTh1_val.log
└── ETTh1_test.log
```

## 注意事项

1. 确保已安装所需的依赖包：
   - torch
   - transformers
   - h5py
   - pandas
   - numpy

2. 确保数据文件位于正确的位置（默认：`./dataset/ETTh1.csv` 或项目根目录下的 `dataset/ETTh1.csv`）

3. 生成嵌入需要一定时间，建议使用GPU加速

4. 如果内存不足，可以减小 `batch_size` 或使用CPU模式

5. 嵌入文件会占用一定的磁盘空间，请确保有足够的存储空间

## 验证嵌入

生成嵌入后，可以使用以下代码验证：

```python
import h5py
import os

# 检查嵌入文件是否存在
embed_path = "./Embeddings/ETTh1/train/0.h5"
if os.path.exists(embed_path):
    with h5py.File(embed_path, 'r') as hf:
        embeddings = hf['embeddings'][:]
        print(f"嵌入形状: {embeddings.shape}")
        print(f"嵌入数据类型: {embeddings.dtype}")
else:
    print("嵌入文件不存在")
```
