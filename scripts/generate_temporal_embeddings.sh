#!/bin/bash
# 生成时间步维度嵌入的脚本
# 使用方法: bash scripts/generate_temporal_embeddings.sh

# 设置项目路径
export PYTHONPATH=/root/0/TimeCMA:$PYTHONPATH

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 数据集配置
data_path="ETTh1"
num_nodes=7
input_len=96
output_len=96
model_name="gpt2"
d_model=768
l_layers=12
batch_size=1
device="cuda"

# 获取项目根目录
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
root_path="${PROJECT_ROOT}/dataset"

# 创建日志目录
log_dir="./Results/emb_logs"
mkdir -p $log_dir

# 创建嵌入保存目录
embed_dir="./Embeddings_Temporal/${data_path}"
mkdir -p $embed_dir

echo "=========================================="
echo "开始生成时间步维度的GPT-2嵌入"
echo "数据集: $data_path"
echo "输入长度: $input_len"
echo "输出长度: $output_len"
echo "模型: $model_name"
echo "=========================================="

# 为训练、验证、测试集分别生成嵌入
for divide in "train" "val" "test"; do
    echo ""
    echo "----------------------------------------"
    echo "正在处理: $divide 集"
    echo "----------------------------------------"
    
    log_file="${log_dir}/${data_path}_temporal_${divide}.log"
    save_path="${embed_dir}/${divide}/"
    
    # 创建保存目录
    mkdir -p $save_path
    
    echo "日志文件: $log_file"
    echo "保存路径: $save_path"
    
    # 运行嵌入生成脚本
    python storage/store_emb_temporal.py \
        --device $device \
        --data_path $data_path \
        --num_nodes $num_nodes \
        --input_len $input_len \
        --output_len $output_len \
        --batch_size $batch_size \
        --d_model $d_model \
        --l_layers $l_layers \
        --model_name $model_name \
        --divide $divide \
        --root_path $root_path \
        --num_workers 4 \
        2>&1 | tee $log_file
    
    if [ $? -eq 0 ]; then
        echo "✓ $divide 集嵌入生成完成"
    else
        echo "✗ $divide 集嵌入生成失败，请查看日志: $log_file"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "所有嵌入生成完成！"
echo "嵌入保存位置: $embed_dir"
echo "=========================================="
