#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 experiment_results.log 导出 TimeCMA 和 TimeCMA_BiCross
在 ETTh1 数据集上、pred_len ∈ {96,192,336,720}、种子 2020-2030 的结果，
以 Markdown 表格形式输出，便于直接粘贴到论文/笔记中。
"""

import json
from collections import defaultdict

LOG_FILE = "experiment_results.log"
TARGET_MODELS = ["TimeCMA", "TimeCMA_BiCross"]
TARGET_DATASET = "ETTh1"
TARGET_PRED_LENS = [96, 192, 336, 720]
TARGET_SEEDS = list(range(2020, 2031))


def load_results(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(rec)
    return records


def main():
    records = load_results(LOG_FILE)

    # 按 (model, pred_len, seed) 索引结果，方便对齐
    by_key = {}
    for r in records:
        if r.get("data_path") != TARGET_DATASET:
            continue
        model = r.get("model")
        if model not in TARGET_MODELS:
            continue
        pred_len = r.get("pred_len")
        seed = r.get("seed")
        if pred_len not in TARGET_PRED_LENS:
            continue
        if seed not in TARGET_SEEDS:
            continue

        key = (model, pred_len, seed)
        by_key[key] = {
            "mse": float(r.get("test_mse")),
            "mae": float(r.get("test_mae")),
        }

    # 先整体说明
    print(f"**数据集**: {TARGET_DATASET}")
    print(f"**模型**: {', '.join(TARGET_MODELS)}")
    print(f"**预测长度**: {TARGET_PRED_LENS}")
    print(f"**种子**: {TARGET_SEEDS[0]}–{TARGET_SEEDS[-1]}")
    print("")

    # 对每个 pred_len 输出一个 Markdown 表格
    for pred in TARGET_PRED_LENS:
        print(f"### pred_len = {pred}")
        print("")
        print("| seed | TimeCMA MSE | TimeCMA MAE | BiCross MSE | BiCross MAE |")
        print("|------|------------:|------------:|------------:|------------:|")

        row_stats = defaultdict(list)  # 针对均值/方差等

        for seed in TARGET_SEEDS:
            base = by_key.get(("TimeCMA", pred, seed))
            bi = by_key.get(("TimeCMA_BiCross", pred, seed))

            if base is None and bi is None:
                # 该 seed 没有任一模型的记录，跳过
                continue

            if base is not None:
                row_stats["base_mse"].append(base["mse"])
                row_stats["base_mae"].append(base["mae"])
            if bi is not None:
                row_stats["bi_mse"].append(bi["mse"])
                row_stats["bi_mae"].append(bi["mae"])

            base_mse = f"{base['mse']:.6f}" if base else "-"
            base_mae = f"{base['mae']:.6f}" if base else "-"
            bi_mse = f"{bi['mse']:.6f}" if bi else "-"
            bi_mae = f"{bi['mae']:.6f}" if bi else "-"

            print(f"| {seed} | {base_mse} | {base_mae} | {bi_mse} | {bi_mae} |")

        # 汇总行（如果两边都有完整数据）
        if row_stats["base_mse"] and row_stats["bi_mse"]:
            import statistics as stats

            base_mse_mean = stats.mean(row_stats["base_mse"])
            base_mae_mean = stats.mean(row_stats["base_mae"])
            bi_mse_mean = stats.mean(row_stats["bi_mse"])
            bi_mae_mean = stats.mean(row_stats["bi_mae"])

            print("")
            print("> 平均表现（11 个种子）")
            print("")
            print("| model | MSE mean | MAE mean |")
            print("|-------|---------:|---------:|")
            print(f"| TimeCMA         | {base_mse_mean:.6f} | {base_mae_mean:.6f} |")
            print(f"| TimeCMA_BiCross | {bi_mse_mean:.6f} | {bi_mae_mean:.6f} |")
            print("")
            print("> BiCross 相对 TimeCMA 的提升 (BiCross - TimeCMA)")
            print("")
            print(f"- ΔMSE = {bi_mse_mean - base_mse_mean:+.6f}")
            print(f"- ΔMAE = {bi_mae_mean - base_mae_mean:+.6f}")

        print("\n")


if __name__ == "__main__":
    main()

