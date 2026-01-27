#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析 experiment_results.log 中 TimeCMA 与 TimeCMA_BiCross
在 ETTh1 数据集上、四个预测长度(96/192/336/720)、多种子下的表现。
"""

import json
from collections import defaultdict
from statistics import mean


LOG_FILE = "experiment_results.log"
TARGET_MODELS = ["TimeCMA", "TimeCMA_BiCross"]
TARGET_DATASET = "ETTh1"
TARGET_PRED_LENS = [96, 192, 336, 720]


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

    # 结构: stats[(model, pred_len)] = {"mse": [...], "mae": [...], "seeds": [...]}
    stats = defaultdict(lambda: {"mse": [], "mae": [], "seeds": []})

    for r in records:
        if r.get("data_path") != TARGET_DATASET:
            continue
        model = r.get("model")
        if model not in TARGET_MODELS:
            continue
        pred_len = r.get("pred_len")
        if pred_len not in TARGET_PRED_LENS:
            continue

        try:
            mse = float(r.get("test_mse"))
            mae = float(r.get("test_mae"))
        except (TypeError, ValueError):
            continue

        seed = r.get("seed")
        key = (model, pred_len)
        stats[key]["mse"].append(mse)
        stats[key]["mae"].append(mae)
        stats[key]["seeds"].append(seed)

    print("============================================================")
    print(f"Dataset: {TARGET_DATASET}")
    print(f"Models: {', '.join(TARGET_MODELS)}")
    print(f"Pred_lens: {TARGET_PRED_LENS}")
    print(f"Source log: {LOG_FILE}")
    print("============================================================\n")

    for pred in TARGET_PRED_LENS:
        print(f"-------------------- pred_len = {pred} --------------------")

        for model in TARGET_MODELS:
            key = (model, pred)
            if key not in stats or not stats[key]["mse"]:
                print(f"{model:16s}: 无记录")
                continue

            mse_list = stats[key]["mse"]
            mae_list = stats[key]["mae"]
            seeds = stats[key]["seeds"]

            print(f"{model:16s}:")
            print(f"  种子数量: {len(seeds)}")
            print(f"  种子列表: {sorted(seeds)}")
            print(
                f"  MSE: mean={mean(mse_list):.6f}, "
                f"min={min(mse_list):.6f}, max={max(mse_list):.6f}"
            )
            print(
                f"  MAE: mean={mean(mae_list):.6f}, "
                f"min={min(mae_list):.6f}, max={max(mae_list):.6f}"
            )

        # 对比 BiCross - TimeCMA
        key_base = ("TimeCMA", pred)
        key_bi = ("TimeCMA_BiCross", pred)
        if (
            key_base in stats
            and key_bi in stats
            and stats[key_base]["mse"]
            and stats[key_bi]["mse"]
        ):
            mse_base_mean = mean(stats[key_base]["mse"])
            mae_base_mean = mean(stats[key_base]["mae"])
            mse_bi_mean = mean(stats[key_bi]["mse"])
            mae_bi_mean = mean(stats[key_bi]["mae"])

            print("\n  对比 (TimeCMA_BiCross - TimeCMA):")
            print(f"    ΔMSE: {mse_bi_mean - mse_base_mean:+.6f}")
            print(f"    ΔMAE: {mae_bi_mean - mae_base_mean:+.6f}")

        print()


if __name__ == "__main__":
    main()

