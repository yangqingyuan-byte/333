#!/bin/bash
# ============================================================================
# 多种子训练脚本：TimeCMA 和 TimeCMA_BiCross
# - 2 个模型
# - 4 种预测长度 (96, 192, 336, 720)
# - 11 个种子 (2020-2030)
# - 共 2 * 4 * 11 = 88 个实验
# - 分配到 8 张 GPU（每张 GPU 约 11 个实验）
# ============================================================================

export PYTHONPATH=/root/0/TimeCMA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

data_path="ETTh1"
seq_len=96
batch_size=16
epochs=100
head=8
d_ff=32

# 创建日志目录
mkdir -p ./Results_MultiSeed/TimeCMA/${data_path}/
mkdir -p ./Results_MultiSeed/TimeCMA_BiCross/${data_path}/

# 定义参数（来自 ETTh1.sh）
# pred_len  channel  e_layer  d_layer  dropout_n
declare -A PARAMS
PARAMS[96]="64 1 2 0.7"
PARAMS[192]="64 1 2 0.7"
PARAMS[336]="64 1 2 0.7"
PARAMS[720]="32 2 2 0.8"

# 种子列表
SEEDS=(2020 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030)

# 预测长度列表
PRED_LENS=(96 192 336 720)

# ============================================================================
# 生成所有任务并分配到 8 张 GPU
# ============================================================================

echo "=========================================="
echo "Generating task assignments..."
echo "=========================================="

# 创建任务分配文件
for gpu_id in 0 1 2 3 4 5 6 7; do
    echo "#!/bin/bash" > /tmp/gpu_${gpu_id}_tasks.sh
    echo "export PYTHONPATH=/root/0/TimeCMA:\$PYTHONPATH" >> /tmp/gpu_${gpu_id}_tasks.sh
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" >> /tmp/gpu_${gpu_id}_tasks.sh
    echo "" >> /tmp/gpu_${gpu_id}_tasks.sh
done

# 任务计数器
task_id=0

# 为每个模型、每个预测长度、每个种子生成任务
for model in "TimeCMA" "TimeCMA_BiCross"; do
    for pred_len in "${PRED_LENS[@]}"; do
        # 获取该预测长度的参数
        params=(${PARAMS[$pred_len]})
        channel=${params[0]}
        e_layer=${params[1]}
        d_layer=${params[2]}
        dropout_n=${params[3]}
        
        for seed in "${SEEDS[@]}"; do
            # 分配到 GPU（轮询分配）
            gpu_id=$((task_id % 8))
            
            # 确定训练脚本和日志路径
            if [ "$model" == "TimeCMA" ]; then
                train_script="train.py"
                log_path="./Results_MultiSeed/TimeCMA/${data_path}/"
            else
                train_script="train_bicross.py"
                log_path="./Results_MultiSeed/TimeCMA_BiCross/${data_path}/"
            fi
            
            log_file="${log_path}pred${pred_len}_c${channel}_dn${dropout_n}_el${e_layer}_dl${d_layer}_seed${seed}.log"
            
            # 添加任务到对应 GPU 的脚本
            cat >> /tmp/gpu_${gpu_id}_tasks.sh << EOF

echo "[GPU ${gpu_id}] ${model} pred_len=${pred_len} seed=${seed}"
CUDA_VISIBLE_DEVICES=${gpu_id} python ${train_script} \\
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
echo "[GPU ${gpu_id}] Completed: ${model} pred_len=${pred_len} seed=${seed}"

EOF
            
            task_id=$((task_id + 1))
        done
    done
done

# 统计每张 GPU 的任务数
echo ""
echo "Task distribution:"
for gpu_id in 0 1 2 3 4 5 6 7; do
    count=$(grep -c "CUDA_VISIBLE_DEVICES" /tmp/gpu_${gpu_id}_tasks.sh)
    echo "  GPU ${gpu_id}: ${count} tasks"
done
echo "  Total: ${task_id} tasks"
echo ""

# ============================================================================
# 启动所有 GPU 的任务（后台运行）
# ============================================================================

echo "=========================================="
echo "Starting all tasks on 8 GPUs..."
echo "=========================================="

for gpu_id in 0 1 2 3 4 5 6 7; do
    chmod +x /tmp/gpu_${gpu_id}_tasks.sh
    echo "Starting GPU ${gpu_id} tasks in background..."
    nohup bash /tmp/gpu_${gpu_id}_tasks.sh > ./Results_MultiSeed/gpu_${gpu_id}.log 2>&1 &
done

echo ""
echo "=========================================="
echo "All tasks started!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  tail -f ./Results_MultiSeed/gpu_0.log"
echo "  tail -f ./Results_MultiSeed/gpu_1.log"
echo "  ..."
echo ""
echo "Or check all GPUs:"
echo "  for i in 0 1 2 3 4 5 6 7; do echo \"=== GPU \$i ===\"; tail -1 ./Results_MultiSeed/gpu_\$i.log; done"
echo ""
echo "Results will be saved to:"
echo "  ./Results_MultiSeed/TimeCMA/${data_path}/"
echo "  ./Results_MultiSeed/TimeCMA_BiCross/${data_path}/"
