#!/bin/bash
# 使用与 TimeCMA 相同的参数训练 TimeCMA_BiCross 模型
# 包括 pred_len=192, 336, 720 三个配置
export PYTHONPATH=/root/0/TimeCMA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

data_path="ETTh1"
seq_len=96
batch_size=16
seed=2024
epochs=100

log_path="./Results_BiCross/${data_path}/"
mkdir -p "$log_path"

run_task () {
  local gpu_id=$1
  local pred_len=$2
  local channel=$3
  local learning_rate=$4
  local dropout_n=$5
  local e_layer=$6
  local d_layer=$7

  log_file="${log_path}pred${pred_len}_c${channel}_lr${learning_rate}_dn${dropout_n}_el${e_layer}_dl${d_layer}_bs${batch_size}_seed${seed}.log"
  
  echo ">>> GPU ${gpu_id} | pred_len=${pred_len} | channel=${channel} | dropout=${dropout_n} | e_layer=${e_layer} | d_layer=${d_layer}"
  echo ">>> Log file: ${log_file}"
  
  CUDA_VISIBLE_DEVICES=${gpu_id} python train_bicross.py \
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
    --d_ff 32 \
    --head 8 \
    2>&1 | tee "$log_file"
}

# 按顺序运行三个任务（如果有多张GPU可以改成并行）
echo "=========================================="
echo "Training TimeCMA_BiCross on ETTh1"
echo "=========================================="

# pred_len=192: channel=64, dropout_n=0.7, e_layer=1, d_layer=2
echo ""
echo "[1/3] pred_len=192"
run_task 0 192 64 0.0001 0.7 1 2

# pred_len=336: channel=64, dropout_n=0.7, e_layer=1, d_layer=2
echo ""
echo "[2/3] pred_len=336"
run_task 0 336 64 0.0001 0.7 1 2

# pred_len=720: channel=32, dropout_n=0.8, e_layer=2, d_layer=2
echo ""
echo "[3/3] pred_len=720"
run_task 0 720 32 0.0001 0.8 2 2

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "Logs saved to: ${log_path}"
echo "=========================================="
