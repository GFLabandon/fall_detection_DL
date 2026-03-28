"""
train.py — LSTM 跌倒分类器训练脚本

使用流程：
  1. python data/collect.py        # 采集训练数据
  2. python train.py               # 训练模型（约2-5分钟）
  3. python eval.py                # 评估指标（Precision/Recall/F1）
  4. python main.py                # 实时检测（自动加载训练好的模型）

训练完成后，权重保存至 weights/lstm_fall_classifier.pth
同时输出 logs/training_history.csv（损失曲线数据，可用于答辩展示）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_VAL_SPLIT,
    TRAIN_DATA_FILE, MODEL_WEIGHTS,
    SEQUENCE_LEN, FEATURE_DIM, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    LOG_DIR,
)
from models.lstm_fall_classifier import LSTMFallClassifier
from data.dataset import load_dataset


def train():
    # ---- 检查数据集 ----
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"❌ 未找到训练数据：{TRAIN_DATA_FILE}")
        print("   请先运行：python data/collect.py  采集训练数据")
        sys.exit(1)

    # ---- 设备选择（MPS for Apple Silicon / CUDA / CPU）----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 使用 Apple MPS 加速（M1/M2）")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ 使用 CUDA GPU 加速")
    else:
        device = torch.device("cpu")
        print("⚠️  使用 CPU 训练（较慢）")

    # ---- 加载数据集 ----
    train_ds, val_ds, class_weights = load_dataset(TRAIN_DATA_FILE, TRAIN_VAL_SPLIT)
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=TRAIN_BATCH_SIZE, shuffle=False)

    # ---- 构建模型 ----
    model = LSTMFallClassifier(
        input_dim  = FEATURE_DIM,
        hidden_dim = LSTM_HIDDEN,
        num_layers = LSTM_LAYERS,
        dropout    = LSTM_DROPOUT,
    ).to(device)
    model.summary(FEATURE_DIM, SEQUENCE_LEN)

    # ---- 损失函数（带类别权重处理样本不均衡）----
    pos_weight = class_weights[1].to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ---- 优化器 + 学习率调度 ----
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)

    # ---- 日志初始化 ----
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    history_file = os.path.join(LOG_DIR, "training_history.csv")
    with open(history_file, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr"])

    best_val_loss = float("inf")
    t_start       = time.time()

    print(f"\n开始训练：epochs={TRAIN_EPOCHS}  batch={TRAIN_BATCH_SIZE}  lr={TRAIN_LR}")
    print("-" * 60)

    for epoch in range(1, TRAIN_EPOCHS + 1):
        # ======== 训练阶段 ========
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(yb)

        train_loss /= len(train_ds)
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        # ======== 验证阶段 ========
        model.eval()
        val_loss = 0.0
        correct  = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb   = xb.to(device), yb.to(device)
                logits   = model(xb)
                val_loss += criterion(logits, yb).item() * len(yb)
                preds    = (torch.sigmoid(logits) > 0.5).float()
                correct  += (preds == yb).sum().item()

        val_loss /= len(val_ds)
        val_acc   = correct / len(val_ds) * 100

        # ---- 保存最优权重 ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "config": {
                    "input_dim":  FEATURE_DIM,
                    "hidden_dim": LSTM_HIDDEN,
                    "num_layers": LSTM_LAYERS,
                    "dropout":    LSTM_DROPOUT,
                    "seq_len":    SEQUENCE_LEN,
                }
            }, MODEL_WEIGHTS)
            mark = " ← 最优"
        else:
            mark = ""

        # ---- 写入训练日志 ----
        with open(history_file, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                                    f"{val_acc:.1f}", f"{cur_lr:.6f}"])

        # ---- 打印进度 ----
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"Epoch [{epoch:3d}/{TRAIN_EPOCHS}]  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.1f}%  "
                  f"lr={cur_lr:.5f}  "
                  f"elapsed={elapsed:.0f}s{mark}")

    total_time = time.time() - t_start
    print("-" * 60)
    print(f"\n✅ 训练完成！耗时 {total_time:.0f} 秒")
    print(f"   最优验证损失：{best_val_loss:.4f}")
    print(f"   模型权重：{MODEL_WEIGHTS}")
    print(f"   训练日志：{history_file}")
    print(f"\n下一步：python eval.py  查看完整评估指标")


if __name__ == "__main__":
    train()