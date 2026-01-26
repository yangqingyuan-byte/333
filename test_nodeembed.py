"""
快速测试脚本：验证新模型是否能正常运行
"""
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour
from models.TimeCMA_NodeEmbed import Dual_NodeEmbed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1", help="data path")
    parser.add_argument("--channel", type=int, default=32, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=96, help="out_len")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size for testing")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--d_layer", type=int, default=1, help="layers of transformer decoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载测试数据
    print("Loading test data...")
    test_set = Dataset_ETT_hour(flag='test', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)
    
    # 创建模型
    print("Creating model...")
    model = Dual_NodeEmbed(
        device=device, 
        channel=args.channel, 
        num_nodes=args.num_nodes, 
        seq_len=args.seq_len, 
        pred_len=args.pred_len, 
        dropout_n=args.dropout_n, 
        d_llm=args.d_llm, 
        e_layer=args.e_layer, 
        d_layer=args.d_layer, 
        head=args.head
    ).to(device)
    
    print(f"Model parameters: {model.count_trainable_params():,}")
    print(f"Total parameters: {model.param_num():,}")
    
    # 测试前向传播
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        for i, (x, y, x_mark, y_mark, embeddings) in enumerate(test_loader):
            print(f"\n--- Batch {i+1} ---")
            testx = torch.tensor(x, dtype=torch.float32).to(device)
            testy = torch.tensor(y, dtype=torch.float32).to(device)
            testx_mark = torch.tensor(x_mark, dtype=torch.float32).to(device)
            test_embedding = torch.tensor(embeddings, dtype=torch.float32).to(device)
            
            print(f"Input shapes:")
            print(f"  x: {testx.shape}")
            print(f"  y: {testy.shape}")
            print(f"  x_mark: {testx_mark.shape}")
            print(f"  embeddings: {test_embedding.shape}")
            
            try:
                # 关闭调试打印
                import sys
                import io
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                
                pred = model(testx, testx_mark, test_embedding)
                
                sys.stdout = old_stdout
                
                print(f"Output shape: {pred.shape}")
                print(f"Expected shape: {testy.shape}")
                
                if pred.shape == testy.shape:
                    print("✓ Shape match!")
                else:
                    print("✗ Shape mismatch!")
                
                # 检查NaN和Inf
                if torch.isnan(pred).any():
                    print("✗ Output contains NaN!")
                elif torch.isinf(pred).any():
                    print("✗ Output contains Inf!")
                else:
                    print("✓ Output is valid (no NaN/Inf)")
                
                print(f"Output range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
                print(f"Output mean: {pred.mean().item():.4f}, std: {pred.std().item():.4f}")
                
            except Exception as e:
                sys.stdout = old_stdout
                print(f"✗ Error during forward pass: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # 只测试第一个batch
            if i == 0:
                break
    
    print("\n" + "="*50)
    print("Test completed!")

if __name__ == "__main__":
    main()
