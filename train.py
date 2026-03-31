# fall_detection_DL/train.py
# LSTM 跌倒分类器训练脚本
#
# 使用流程：
#   1. python data/preprocess.py   → 生成 data/processed/fall_sequences.npz
#   2. python train.py             → 训练 LSTM，约 3-8 分钟（Apple M2 MPS）
#   3. python eval.py              → 查看完整评估指标和消融实验
#   4. python main.py              → 实时检测演示（自动加载训练好的权重）
#
# 训练产出：
#   weights/lstm_fall.pth          — 最优 epoch 的模型权重
#   logs/training_history.csv      — 每 epoch 的 loss/acc 记录（答辩 loss 曲线数据源）

import os
import sys
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 确保从项目根目录导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_DATA_FILE,
    TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE,
    TRAIN_LR,
    TRAIN_VAL_SPLIT,
    TRAIN_TEST_SPLIT,
    FEATURE_DIM,
    SEQUENCE_LEN,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    MODEL_WEIGHTS,
    LOG_DIR,
)
from data.dataset          import load_dataset
from models.lstm_classifier import LSTMFallClassifier


# ============================================================
#  工具函数
# ============================================================

def get_device() -> torch.device:
    """按优先级选择计算设备：MPS（Apple M2）→ CUDA → CPU。"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  ✅ 使用 Apple MPS 加速（M1/M2）")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  ✅ 使用 CUDA GPU：{torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("  ⚠️  使用 CPU 训练（较慢，建议使用 GPU/MPS）")
    return device


def run_epoch_train(model, loader, criterion, optimizer, device):
    """
    执行一个训练 epoch。

    Returns:
        avg_loss: float，batch 加权平均 loss
    """
    model.train()
    total_loss  = 0.0
    total_count = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)                    # (batch,) logit
        loss   = criterion(logits, yb)
        loss.backward()

        # 梯度裁剪：防止 LSTM 训练中的梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss  += loss.item() * len(yb)
        total_count += len(yb)

    return total_loss / total_count


def run_epoch_eval(model, loader, criterion, device):
    """
    执行一个评估 epoch（验证集 / 测试集通用）。

    Returns:
        avg_loss: float
        accuracy: float（0~1）
    """
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_count   = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb   = xb.to(device), yb.to(device)
            logits   = model(xb)
            loss     = criterion(logits, yb)

            # 以 0.5 为阈值计算准确率（验证集快速监控用）
            preds    = (torch.sigmoid(logits) >= 0.5).float()
            correct  = (preds == yb).sum().item()

            total_loss    += loss.item() * len(yb)
            total_correct += correct
            total_count   += len(yb)

    avg_loss = total_loss    / total_count
    accuracy = total_correct / total_count
    return avg_loss, accuracy


def init_csv_log(log_path: str):
    """初始化 training_history.csv，写入表头。"""
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr"])


def append_csv_log(log_path: str, epoch: int, train_loss: float,
                   val_loss: float, val_acc: float, lr: float):
    """追加一行训练记录到 CSV。"""
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_loss:.4f}",
            f"{val_loss:.4f}",
            f"{val_acc:.4f}",
            f"{lr:.6f}",
        ])


# ============================================================
#  主训练流程
# ============================================================

def train():
    print()
    print("=" * 60)
    print("  LSTM 跌倒分类器训练")
    print("=" * 60)

    # ---- 前置检查 ----
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"\n❌ 数据集文件不存在：{TRAIN_DATA_FILE}")
        print("   请先运行：python data/preprocess.py")
        sys.exit(1)

    # ---- 创建必要目录 ----
    os.makedirs("weights", exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    # ---- 设备选择 ----
    device = get_device()

    # ---- 加载数据集 ----
    print()
    train_ds, val_ds, test_ds, pos_weight = load_dataset(
        TRAIN_DATA_FILE,
        val_split  = TRAIN_VAL_SPLIT,
        test_split = TRAIN_TEST_SPLIT,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = TRAIN_BATCH_SIZE,
        shuffle    = True,
        drop_last  = False,
        num_workers= 0,          # macOS 多进程 DataLoader 有时不稳定，使用 0
        pin_memory = False,      # MPS 不支持 pin_memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size = TRAIN_BATCH_SIZE,
        shuffle    = False,
        num_workers= 0,
        pin_memory = False,
    )

    # ---- 构建模型 ----
    print()
    model = LSTMFallClassifier(
        input_dim  = FEATURE_DIM,
        hidden_dim = LSTM_HIDDEN,
        num_layers = LSTM_LAYERS,
        dropout    = LSTM_DROPOUT,
    ).to(device)
    model.summary()

    # ---- 损失函数（带正例权重，处理类别不均衡）----
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # ---- 优化器 ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = TRAIN_LR,
        weight_decay = 1e-4,       # L2 正则，防止过拟合
    )

    # ---- 学习率调度：余弦退火（在 TRAIN_EPOCHS 内从 TRAIN_LR 平滑衰减到接近 0）----
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = TRAIN_EPOCHS,
    )

    # ---- 初始化日志 ----
    log_path = os.path.join(LOG_DIR, "training_history.csv")
    init_csv_log(log_path)

    # ---- 训练状态追踪 ----
    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    best_epoch     = 0

    print(f"  开始训练：epochs={TRAIN_EPOCHS}  batch={TRAIN_BATCH_SIZE}"
          f"  lr={TRAIN_LR}  设备={device}")
    print(f"  日志：{log_path}")
    print(f"  权重：{MODEL_WEIGHTS}")
    print()
    print(f"  {'Epoch':>10}  {'train_loss':>12}  {'val_loss':>10}"
          f"  {'val_acc':>9}  {'lr':>10}")
    print("  " + "─" * 58)

    t_start = time.time()

    # ============================================================
    #  训练循环
    # ============================================================
    for epoch in range(1, TRAIN_EPOCHS + 1):

        # 训练 epoch
        train_loss = run_epoch_train(model, train_loader, criterion, optimizer, device)

        # 验证 epoch
        val_loss, val_acc = run_epoch_eval(model, val_loader, criterion, device)

        # 学习率调度步进（每个 epoch 结束后调用）
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        # 写入 CSV 日志（每个 epoch 都记录，用于绘制 loss 曲线）
        append_csv_log(log_path, epoch, train_loss, val_loss, val_acc, cur_lr)

        # 保存最优模型（验证集 loss 最低的 checkpoint）
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_epoch    = epoch
            torch.save(
                {
                    "epoch":       best_epoch,
                    "model_state": model.state_dict(),
                    "val_loss":    best_val_loss,
                    "val_acc":     best_val_acc,
                    "config": {
                        "input_dim":  FEATURE_DIM,
                        "hidden_dim": LSTM_HIDDEN,
                        "num_layers": LSTM_LAYERS,
                        "dropout":    LSTM_DROPOUT,
                        "seq_len":    SEQUENCE_LEN,
                    },
                },
                MODEL_WEIGHTS,
            )

        # 控制台输出：每5个 epoch 打印一次，以及第1 epoch 和最后1 epoch
        should_print = (epoch % 5 == 0) or (epoch == 1) or (epoch == TRAIN_EPOCHS)
        if should_print or is_best:
            best_mark = "  ← 最优" if is_best else ""
            print(
                f"  Epoch [{epoch:3d}/{TRAIN_EPOCHS}]"
                f"  train_loss={train_loss:.4f}"
                f"  val_loss={val_loss:.4f}"
                f"  val_acc={val_acc*100:.1f}%"
                f"  lr={cur_lr:.5f}"
                f"{best_mark}"
            )

    # ============================================================
    #  训练结束
    # ============================================================
    total_time = time.time() - t_start
    mins, secs = divmod(int(total_time), 60)

    print()
    print("=" * 60)
    print("  🎉 训练完成！")
    print(f"   总耗时：{mins}分{secs}秒")
    print(f"   最优 Epoch: {best_epoch}"
          f"   val_loss={best_val_loss:.4f}"
          f"   val_acc={best_val_acc*100:.1f}%")
    print(f"   权重已保存：{MODEL_WEIGHTS}")
    print(f"   训练日志：{log_path}")
    print()
    print("   下一步：python eval.py  获取完整评估指标和消融实验")
    print("=" * 60)
    print()


# ============================================================
#  入口
# ============================================================
if __name__ == "__main__":
    train()