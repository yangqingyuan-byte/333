#!/bin/bash
# 同时在 4 张 GPU 上前台启动 4 个预测长度任务（后台但无 nohup，日志写文件且在当前终端可见）
export PYTHONPATH=/root/0/TimeCMA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

data_path="ETTh1"
seq_len=96
batch_size=16

log_path="./Results/${data_path}/"
mkdir -p "$log_path"

run_task () {
  local gpu_id=$1
  local pred_len=$2
  local learning_rate=$3
  local channel=$4
  local e_layer=$5
  local d_layer=$6
  local dropout_n=$7

  log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
  echo ">>> GPU ${gpu_id} | pred_len=${pred_len} | log=${log_file}"
  CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
    --data_path $data_path \
    --batch_size $batch_size \
    --num_nodes 7 \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --epochs 999 \
    --seed 2024 \
    --channel $channel \
    --learning_rate $learning_rate \
    --dropout_n $dropout_n \
    --e_layer $e_layer \
    --d_layer $d_layer \
    --head 8 \
    > "$log_file" 2>&1 &
}

# 四个任务分配到四张卡 0/1/2/3
run_task 0 96  1e-4 64 1 2 0.7
run_task 1 192 1e-4 64 1 2 0.7
run_task 2 336 1e-4 64 1 2 0.7
run_task 3 720 1e-4 32 2 2 0.8

echo "全部任务已启动（后台运行但无 nohup），日志输出到 ${log_path}"
