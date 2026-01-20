"""
实验结果日志记录工具
将结果统一追加到 JSONL 文件 experiment_results.log
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


def log_experiment_result(
    *,
    data_path: str,
    pred_len: int,
    model_name: str,
    seed: int,
    test_mse: float,
    test_mae: float,
    seq_len: Optional[int] = None,
    channel: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    dropout_n: Optional[float] = None,
    additional_info: Optional[Dict[str, Any]] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    将实验结果追加到统一日志文件（JSONL，一行一个 JSON）
    默认写入项目根目录的 experiment_results.log
    """
    if log_file is None:
        log_file = "./experiment_results.log"

    result = {
        "data_path": data_path,
        "pred_len": pred_len,
        "test_mse": round(float(test_mse), 6),
        "test_mae": round(float(test_mae), 6),
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
    }

    if seq_len is not None:
        result["seq_len"] = seq_len
    if channel is not None:
        result["channel"] = channel
    if batch_size is not None:
        result["batch_size"] = batch_size
    if learning_rate is not None:
        result["learning_rate"] = learning_rate
    if dropout_n is not None:
        result["dropout_n"] = dropout_n
    if additional_info:
        result.update(additional_info)

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"实验结果已记录到: {log_file}")
    print(
        f"数据集: {data_path}, 预测长度: {pred_len}, "
        f"Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}"
    )
    print(
        f"模型: {model_name}, 种子: {seed}, "
        f"seq_len: {seq_len}, channel: {channel}, batch: {batch_size}, lr: {learning_rate}"
    )
    if additional_info:
        print(f"额外信息: {additional_info}")
    print(f"{'='*60}\n")
