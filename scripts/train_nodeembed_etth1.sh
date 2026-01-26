#!/bin/bash
# 训练脚本：在ETTh1数据集上训练节点嵌入版本的模型
export PYTHONPATH=/root/0/TimeCMA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

data_path="ETTh1"
seq_len=96
pred_len=96
batch_size=16

log_path="./Results_NodeEmbed/${data_path}/"
mkdir -p "$log_path"

log_file="${log_path}train_seq${seq_len}_pred${pred_len}_bs${batch_size}.log"

echo ">>> Training NodeEmbed model on ${data_path}"
echo ">>> Log file: ${log_file}"

CUDA_VISIBLE_DEVICES=0 python train_nodeembed.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 10 \
  --seed 2024 \
  --channel 32 \
  --learning_rate 1e-4 \
  --dropout_n 0.2 \
  --e_layer 1 \
  --d_layer 1 \
  --head 8 \
  2>&1 | tee "$log_file"

echo "Training completed! Log saved to: ${log_file}"
