#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成ETTh1数据集的GPT-2嵌入脚本
包含训练、验证和测试集的嵌入生成
"""
import torch
import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader
from data_provider.data_loader_save import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from storage.gen_prompt_emb import GenPromptEmb


def parse_args():
    parser = argparse.ArgumentParser(description='生成ETTh1数据集的GPT-2嵌入')
    parser.add_argument("--device", type=str, default="cuda", help="设备类型 (cuda/cpu)")
    parser.add_argument("--data_path", type=str, default="ETTh1", help="数据集路径")
    parser.add_argument("--num_nodes", type=int, default=7, help="节点数量")
    parser.add_argument("--input_len", type=int, default=96, help="输入序列长度")
    parser.add_argument("--output_len", type=int, default=96, help="输出序列长度")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--d_model", type=int, default=768, help="GPT-2模型维度")
    parser.add_argument("--l_layers", type=int, default=12, help="GPT-2层数")
    parser.add_argument("--model_name", type=str, default="gpt2", help="GPT-2模型名称")
    parser.add_argument("--divide", type=str, default="all", choices=["train", "val", "test", "all"], 
                       help="生成哪个数据集的嵌入 (all表示全部)")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    # 获取项目根目录，默认使用项目内的dataset目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_root_path = os.path.join(project_root, "dataset")
    parser.add_argument("--root_path", type=str, default=default_root_path, help="数据根路径")
    return parser.parse_args()


def get_dataset(data_path, flag, input_len, output_len, root_path):
    """获取数据集"""
    datasets = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    dataset_class = datasets.get(data_path, Dataset_Custom)
    return dataset_class(flag=flag, size=[input_len, 0, output_len], data_path=data_path, root_path=root_path)


def save_embeddings_for_divide(args, divide):
    """为指定的数据集划分生成并保存嵌入"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"开始处理 {divide} 集")
    print(f"{'='*60}")
    
    # 获取数据集
    dataset = get_dataset(args.data_path, divide, args.input_len, args.output_len, args.root_path)
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False, 
        num_workers=args.num_workers
    )
    
    # 初始化GPT-2嵌入生成器
    gen_prompt_emb = GenPromptEmb(
        device=device,
        input_len=args.input_len,
        data_path=args.data_path,
        model_name=args.model_name,
        d_model=args.d_model,
        layer=args.l_layers,
        divide=divide
    ).to(device)
    
    # 创建保存目录
    save_path = f"./Embeddings/{args.data_path}/{divide}/"
    os.makedirs(save_path, exist_ok=True)
    
    print(f"设备: {device}")
    print(f"数据集大小: {len(dataset)}")
    print(f"保存路径: {save_path}")
    print(f"开始生成嵌入...")
    
    start_time = time.time()
    total_samples = len(dataset)
    
    for i, (x, y, x_mark, y_mark) in enumerate(data_loader):
        # 生成嵌入
        embeddings = gen_prompt_emb.generate_embeddings(x.to(device), x_mark.to(device))
        
        # 保存嵌入到HDF5文件
        file_path = os.path.join(save_path, f"{i}.h5")
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('embeddings', data=embeddings.cpu().numpy())
        
        # 显示进度
        if (i + 1) % 100 == 0 or (i + 1) == total_samples:
            elapsed = time.time() - start_time
            progress = (i + 1) / total_samples * 100
            print(f"进度: [{i+1}/{total_samples}] ({progress:.1f}%) - 已用时间: {elapsed/60:.2f} 分钟")
    
    elapsed_time = time.time() - start_time
    print(f"\n{divide} 集嵌入生成完成！")
    print(f"总样本数: {total_samples}")
    print(f"总耗时: {elapsed_time/60:.2f} 分钟")
    print(f"平均每个样本: {elapsed_time/total_samples:.2f} 秒")
    print(f"嵌入保存位置: {save_path}")
    
    return elapsed_time


def main():
    args = parse_args()
    
    print("="*60)
    print("ETTh1 GPT-2 嵌入生成脚本")
    print("="*60)
    print(f"数据集: {args.data_path}")
    print(f"输入长度: {args.input_len}")
    print(f"输出长度: {args.output_len}")
    print(f"模型: {args.model_name}")
    print(f"设备: {args.device}")
    print("="*60)
    
    # 创建必要的目录
    os.makedirs("./Embeddings", exist_ok=True)
    os.makedirs("./Results/emb_logs", exist_ok=True)
    
    # 确定要处理的数据集划分
    if args.divide == "all":
        divides = ["train", "val", "test"]
    else:
        divides = [args.divide]
    
    total_start_time = time.time()
    total_times = {}
    
    # 为每个数据集划分生成嵌入
    for divide in divides:
        try:
            elapsed = save_embeddings_for_divide(args, divide)
            total_times[divide] = elapsed
        except Exception as e:
            print(f"\n错误: {divide} 集处理失败")
            print(f"错误信息: {str(e)}")
            import traceback
            traceback.print_exc()
            return
    
    # 输出总结
    total_elapsed = time.time() - total_start_time
    print("\n" + "="*60)
    print("所有嵌入生成完成！")
    print("="*60)
    print(f"总耗时: {total_elapsed/60:.2f} 分钟 ({total_elapsed/3600:.2f} 小时)")
    print("\n各数据集耗时:")
    for divide, elapsed in total_times.items():
        print(f"  {divide}: {elapsed/60:.2f} 分钟")
    print(f"\n嵌入保存位置: ./Embeddings/{args.data_path}/")
    print("="*60)


if __name__ == "__main__":
    main()
