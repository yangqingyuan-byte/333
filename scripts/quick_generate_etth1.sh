#!/bin/bash
# 快速生成ETTh1嵌入的简化脚本

# 设置环境
export PYTHONPATH=/root/0/TimeCMA:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# 获取项目根目录
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
root_path="${PROJECT_ROOT}/dataset"

# 运行Python脚本生成所有嵌入
python scripts/generate_etth1_embeddings.py \
    --data_path ETTh1 \
    --input_len 96 \
    --output_len 96 \
    --device cuda \
    --root_path $root_path \
    --divide all
