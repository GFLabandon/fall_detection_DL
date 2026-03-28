"""
eval.py — 模型评估 + 消融实验（答辩必备）

输出内容：
  1. LSTM 模型完整指标：Accuracy / Precision / Recall / F1 / AUC
  2. 混淆矩阵
  3. 消融实验：纯规则通道 vs LSTM 模型 对比
  4. 评估结果保存至 logs/eval_results.txt

使用：
  python eval.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    TRAIN_DATA_FILE, MODEL_WEIGHTS, TRAIN_VAL_SPLIT,
    SEQUENCE_LEN, FEATURE_DIM, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    LSTM_FALL_THRESHOLD, ASPECT_RATIO_THRESHOLD, BODY_ANGLE_THRESHOLD,
    LOG_DIR,
)
from models.lstm_fall_classifier import LSTMFallClassifier
from data.dataset import load_dataset


# ============================================================
#  指标计算工具
# ============================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict:
    """计算 Accuracy / Precision / Recall / F1 / AUC。"""
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())

    acc  = (TP + TN) / max(1, TP + FP + TN + FN)
    prec = TP / max(1, TP + FP)
    rec  = TP / max(1, TP + FN)
    f1   = 2 * prec * rec / max(1e-8, prec + rec)
    fpr  = FP / max(1, FP + TN)

    metrics = {
        "Accuracy":  acc,  "Precision": prec,
        "Recall":    rec,  "F1":        f1,
        "FPR":       fpr,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
    }

    # AUC（简易梯形法）
    if y_prob is not None:
        thresholds = np.linspace(0, 1, 50)
        tprs, fprs = [1.0], [1.0]
        for t in thresholds:
            pred_t = (y_prob >= t).astype(int)
            tp_ = int(((pred_t == 1) & (y_true == 1)).sum())
            fp_ = int(((pred_t == 1) & (y_true == 0)).sum())
            fn_ = int(((pred_t == 0) & (y_true == 1)).sum())
            tn_ = int(((pred_t == 0) & (y_true == 0)).sum())
            tprs.append(tp_ / max(1, tp_ + fn_))
            fprs.append(fp_ / max(1, fp_ + tn_))
        tprs.append(0.0); fprs.append(0.0)
        metrics["AUC"] = float(np.trapz(sorted(tprs), sorted(fprs)))

    return metrics


def print_metrics(name: str, m: dict):
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy:  {m['Accuracy']:.4f}  ({m['Accuracy']*100:.1f}%)")
    print(f"  Precision: {m['Precision']:.4f}")
    print(f"  Recall:    {m['Recall']:.4f}")
    print(f"  F1 Score:  {m['F1']:.4f}")
    if "AUC" in m:
        print(f"  AUC:       {m['AUC']:.4f}")
    print(f"  FPR(误报率): {m['FPR']:.4f}")
    print(f"\n  混淆矩阵：")
    print(f"              预测正常  预测跌倒")
    print(f"  实际正常 |   TN={m['TN']:3d}   |   FP={m['FP']:3d}   |")
    print(f"  实际跌倒 |   FN={m['FN']:3d}   |   TP={m['TP']:3d}   |")
    print(f"{'─'*50}")


# ============================================================
#  规则基线（消融实验对比用）
# ============================================================
def rule_predict(X: np.ndarray, ar_thr: float = 1.40,
                 angle_thr: float = 0.64) -> np.ndarray:
    """
    纯规则预测（对应原始版本的几何判断）。
    特征索引：[2]=纵横比  [4]=身体角度余弦
    """
    ar  = X[:, -1, 2]        # 最后一帧纵横比
    ang = X[:, -1, 4]        # 最后一帧角度余弦（<cos(50°)≈0.64 表示倾斜）
    pred = ((ar > ar_thr) | (ang < angle_thr)).astype(int)
    return pred


# ============================================================
#  主评估流程
# ============================================================
def evaluate():
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"❌ 未找到数据集：{TRAIN_DATA_FILE}")
        print("   请先运行：python data/collect.py  采集训练数据")
        sys.exit(1)

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ 未找到模型权重：{MODEL_WEIGHTS}")
        print("   请先运行：python train.py  训练模型")
        sys.exit(1)

    # ---- 加载数据 ----
    _, val_ds, _ = load_dataset(TRAIN_DATA_FILE, TRAIN_VAL_SPLIT)
    X_val  = val_ds.X.numpy()
    y_true = val_ds.y.numpy().astype(int)

    print(f"\n评估集：{len(y_true)} 条样本  "
          f"跌倒={int(y_true.sum())}  正常={int((y_true==0).sum())}")

    # ---- 加载 LSTM 模型 ----
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(MODEL_WEIGHTS, map_location=device)
    cfg    = ckpt.get("config", {})

    model  = LSTMFallClassifier(
        input_dim  = cfg.get("input_dim",  FEATURE_DIM),
        hidden_dim = cfg.get("hidden_dim", LSTM_HIDDEN),
        num_layers = cfg.get("num_layers", LSTM_LAYERS),
        dropout    = cfg.get("dropout",    LSTM_DROPOUT),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X_tensor = val_ds.X.to(device)
    with torch.no_grad():
        probs  = torch.sigmoid(model(X_tensor)).cpu().numpy()
    y_pred_lstm = (probs > LSTM_FALL_THRESHOLD).astype(int)

    # ---- 规则基线 ----
    y_pred_rule = rule_predict(X_val)

    # ---- 计算指标 ----
    m_lstm = compute_metrics(y_true, y_pred_lstm, probs)
    m_rule = compute_metrics(y_true, y_pred_rule)

    print_metrics("LSTM 深度学习模型（本项目核心）", m_lstm)
    print_metrics("纯规则基线（几何阈值法）", m_rule)

    # ---- 消融对比表格 ----
    print(f"\n{'='*60}")
    print(f"  消融实验对比（Ablation Study）")
    print(f"{'='*60}")
    print(f"  {'方法':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*60}")
    for name, m in [("LSTM 分类器（本项目）", m_lstm), ("纯规则几何判断（基线）", m_rule)]:
        print(f"  {name:<22} {m['Accuracy']:>10.4f} {m['Precision']:>10.4f} "
              f"{m['Recall']:>10.4f} {m['F1']:>10.4f}")
    print(f"{'='*60}")

    # ---- 关键提升 ----
    f1_gain  = m_lstm["F1"]   - m_rule["F1"]
    fpr_drop = m_rule["FPR"]  - m_lstm["FPR"]
    print(f"\n  ✅ LSTM vs 规则：F1 提升 {f1_gain:+.4f}  误报率降低 {fpr_drop:+.4f}")
    print(f"  （消融实验说明引入 LSTM 时序建模的有效性）\n")

    # ---- 保存结果 ----
    os.makedirs(LOG_DIR, exist_ok=True)
    result_file = os.path.join(LOG_DIR, "eval_results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("LSTM 模型评估结果\n")
        f.write(f"Accuracy={m_lstm['Accuracy']:.4f} Precision={m_lstm['Precision']:.4f} "
                f"Recall={m_lstm['Recall']:.4f} F1={m_lstm['F1']:.4f} "
                f"AUC={m_lstm.get('AUC', 0):.4f}\n\n")
        f.write("消融实验\n")
        f.write(f"LSTM:  F1={m_lstm['F1']:.4f}  FPR={m_lstm['FPR']:.4f}\n")
        f.write(f"规则:  F1={m_rule['F1']:.4f}  FPR={m_rule['FPR']:.4f}\n")
    print(f"✅ 评估结果已保存至 {result_file}")
    print(f"   （可将此数据写入答辩文档作为实验结果表格）\n")


if __name__ == "__main__":
    evaluate()