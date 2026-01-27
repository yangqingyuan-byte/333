#!/bin/bash
# ============================================================================
# ETTh1 数据集训练脚本 - 仅 pred_len=336
# - 11 个种子 (2020-2030)
# - 共 11 个实验，分配到 8 张 GPU
# ============================================================================

export PYTHONPATH=/root/0/TimeCMA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

data_path="ETTh1"
seq_len=96
batch_size=16
epochs=100
head=8

# pred_len=336 的参数
pred_len=336
channel=64
e_layer=1
d_layer=2
dropout_n=0.7

# 创建日志目录
log_path="./Results/${data_path}/"
mkdir -p "$log_path"

# 种子列表
SEEDS=(2020 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030)

echo "=========================================="
echo "ETTh1 pred_len=336 Multi-Seed Training"
echo "Seeds: ${SEEDS[*]}"
echo "Total tasks: ${#SEEDS[@]}"
echo "=========================================="

# 为每张 GPU 创建任务脚本
for gpu_id in 0 1 2 3 4 5 6 7; do
    cat > /tmp/etth1_pred336_gpu_${gpu_id}.sh << 'HEADER'
#!/bin/bash
export PYTHONPATH=/root/0/TimeCMA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/0/TimeCMA

HEADER
done

# 分配任务到 GPU（轮询方式）
task_id=0
for seed in "${SEEDS[@]}"; do
    gpu_id=$((task_id % 8))
    
    log_file="${log_path}pred${pred_len}_c${channel}_dn${dropout_n}_el${e_layer}_dl${d_layer}_seed${seed}.log"
    
    cat >> /tmp/etth1_pred336_gpu_${gpu_id}.sh << EOF

echo "[GPU ${gpu_id}] Starting: pred_len=${pred_len} seed=${seed}"
CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \\
    --data_path ${data_path} \\
    --batch_size ${batch_size} \\
    --num_nodes 7 \\
    --seq_len ${seq_len} \\
    --pred_len ${pred_len} \\
    --epochs ${epochs} \\
    --seed ${seed} \\
    --channel ${channel} \\
    --learning_rate 0.0001 \\
    --dropout_n ${dropout_n} \\
    --e_layer ${e_layer} \\
    --d_layer ${d_layer} \\
    --head ${head} \\
    > "${log_file}" 2>&1
echo "[GPU ${gpu_id}] Completed: pred_len=${pred_len} seed=${seed}"

EOF
    task_id=$((task_id + 1))
done

# 统计每张 GPU 的任务数
echo ""
echo "Task distribution:"
for gpu_id in 0 1 2 3 4 5 6 7; do
    count=$(grep -c "CUDA_VISIBLE_DEVICES" /tmp/etth1_pred336_gpu_${gpu_id}.sh 2>/dev/null || echo 0)
    [ "$count" -gt 0 ] && echo "  GPU ${gpu_id}: ${count} tasks"
done
echo "  Total: ${task_id} tasks"
echo ""

# 启动所有 GPU 的任务
echo "=========================================="
echo "Starting tasks..."
echo "=========================================="

for gpu_id in 0 1 2 3 4 5 6 7; do
    # 只启动有任务的 GPU
    if grep -q "CUDA_VISIBLE_DEVICES" /tmp/etth1_pred336_gpu_${gpu_id}.sh 2>/dev/null; then
        chmod +x /tmp/etth1_pred336_gpu_${gpu_id}.sh
        echo "Starting GPU ${gpu_id} tasks in background..."
        nohup bash /tmp/etth1_pred336_gpu_${gpu_id}.sh > ${log_path}gpu_${gpu_id}_pred336.log 2>&1 &
    fi
done

echo ""
echo "=========================================="
echo "All tasks started!"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "  for i in 0 1 2 3 4 5 6 7; do [ -f ${log_path}gpu_\${i}_pred336.log ] && echo \"=== GPU \$i ===\" && tail -1 ${log_path}gpu_\${i}_pred336.log; done"
echo ""
echo "Results saved to: ${log_path}"
echo "Experiment log: ./experiment_results.log"
