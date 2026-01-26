#!/bin/bash
# 测试脚本：快速测试节点嵌入版本的模型
export PYTHONPATH=/root/0/TimeCMA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> Testing NodeEmbed model on ETTh1"
echo ">>> This is a quick test to verify the model works correctly"

CUDA_VISIBLE_DEVICES=0 python test_nodeembed.py \
  --data_path ETTh1 \
  --batch_size 4 \
  --num_nodes 7 \
  --seq_len 96 \
  --pred_len 96 \
  --channel 32 \
  --dropout_n 0.2 \
  --e_layer 1 \
  --d_layer 1 \
  --head 8

echo "Test completed!"
