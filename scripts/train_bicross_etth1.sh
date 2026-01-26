#!/bin/bash
# 训练脚本：在ETTh1数据集上训练双向跨模态对齐模型
# 使用与原TimeCMA模型相同的参数
export PYTHONPATH=/root/0/TimeCMA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

data_path="ETTh1"
seq_len=96
pred_len=96
batch_size=16
channel=64
learning_rate=0.0001
dropout_n=0.7
e_layer=1
d_layer=2
d_ff=32
head=8
seed=2024
epochs=100

log_path="./Results_BiCross/${data_path}/"
mkdir -p "$log_path"

log_file="${log_path}train_seq${seq_len}_pred${pred_len}_c${channel}_lr${learning_rate}_dn${dropout_n}_el${e_layer}_dl${d_layer}_bs${batch_size}_seed${seed}.log"

echo ">>> Training TimeCMA_BiCross model on ${data_path}"
echo ">>> Bidirectional Cross-Modal Alignment"
echo ">>> Parameters:"
echo "    - seq_len: ${seq_len}"
echo "    - pred_len: ${pred_len}"
echo "    - channel: ${channel}"
echo "    - batch_size: ${batch_size}"
echo "    - learning_rate: ${learning_rate}"
echo "    - dropout_n: ${dropout_n}"
echo "    - e_layer: ${e_layer}"
echo "    - d_layer: ${d_layer}"
echo "    - d_ff: ${d_ff}"
echo "    - head: ${head}"
echo "    - seed: ${seed}"
echo ">>> Log file: ${log_file}"
echo ""

CUDA_VISIBLE_DEVICES=0 python train_bicross.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs $epochs \
  --seed $seed \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer \
  --d_ff $d_ff \
  --head $head \
  2>&1 | tee "$log_file"

echo ""
echo "Training completed! Log saved to: ${log_file}"
