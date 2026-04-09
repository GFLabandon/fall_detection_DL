# fall_detection_DL/eval.py
# 模型评估脚本 + 消融实验
#
# 功能：
#   1. 在独立测试集上评估 LSTM 模型（Accuracy/Precision/Recall/F1/AUC/FPR）
#   2. 输出混淆矩阵
#   3. 消融实验：纯规则基线 vs LSTM 单独 vs LSTM+规则融合 三方对比
#   4. 将结果保存至 logs/eval_results.txt（供填写答辩文档）
#
# 使用：
#   python eval.py

import os
import sys
import math

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_DATA_FILE,
    TRAIN_VAL_SPLIT,
    TRAIN_TEST_SPLIT,
    FEATURE_DIM,
    SEQUENCE_LEN,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    LSTM_FALL_THRESHOLD,
    MODEL_WEIGHTS,
    LOG_DIR,
    ASPECT_RATIO_THRESHOLD,
    BODY_ANGLE_THRESHOLD,
)
from data.dataset           import load_dataset
from models.lstm_classifier import LSTMFallClassifier


# ============================================================
#  工具函数
# ============================================================

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict:
    """
    计算完整分类指标。

    Args:
        y_true: 真实标签 (N,) int
        y_pred: 预测标签 (N,) int（阈值化后的二值预测）
        y_prob: 预测概率 (N,) float（用于 AUC，可为 None）

    Returns:
        dict 包含 accuracy/precision/recall/f1/fpr/auc/TP/FP/TN/FN
    """
    # sklearn 指标
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    # 混淆矩阵：[[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # 误报率 FPR = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # AUC-ROC（需要概率值）
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.0   # 只有单一类别时无法计算

    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "fpr":       fpr,
        "auc":       auc,
        "TP": int(tp), "FP": int(fp),
        "TN": int(tn), "FN": int(fn),
    }


def rule_predict(X_test: np.ndarray,
                 ar_threshold:    float = ASPECT_RATIO_THRESHOLD,
                 angle_threshold: float = BODY_ANGLE_THRESHOLD) -> np.ndarray:
    """
    纯几何规则预测（方法A）。

    取每个样本最后一帧的特征（模拟实时单帧判断）：
      feat[3] = aspect_ratio
      feat[4] = body_angle_cos → 转换为角度

    规则：aspect_ratio > ar_threshold OR body_angle_deg > angle_threshold → 预测跌倒(1)

    注意：angle_threshold 在 config 中以度数表示（50.0），
    feat[4] 是余弦值，需要反余弦转换。
    cos(50°) ≈ 0.6428，当 angle_cos < 0.6428 时表示角度 > 50°。

    Args:
        X_test: shape (N, SEQUENCE_LEN, FEATURE_DIM)
        ar_threshold: 纵横比阈值
        angle_threshold: 角度阈值（度数）

    Returns:
        y_pred: shape (N,) int
    """
    cos_threshold = math.cos(math.radians(angle_threshold))  # cos(50°) ≈ 0.6428

    # 取每个样本的最后一帧特征
    last_feats = X_test[:, -1, :]   # (N, FEATURE_DIM)

    ar_vals  = last_feats[:, 3]   # aspect_ratio
    cos_vals = last_feats[:, 4]   # body_angle_cos

    # 规则判断：纵横比过大 OR 身体倾斜过大
    pred = ((ar_vals > ar_threshold) | (cos_vals < cos_threshold)).astype(int)
    return pred


def lstm_predict(model: LSTMFallClassifier, X_tensor: torch.Tensor,
                 device: torch.device, threshold: float = LSTM_FALL_THRESHOLD):
    """
    LSTM 模型批量推理（方法B）。

    Returns:
        y_pred: np.ndarray (N,) int
        y_prob: np.ndarray (N,) float
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor.to(device))
        probs  = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    return preds, probs


def fusion_predict(lstm_pred: np.ndarray, rule_pred: np.ndarray) -> np.ndarray:
    """
    LSTM + 规则融合预测（方法C）：任意一个触发 → 预测跌倒。

    Returns:
        y_pred: np.ndarray (N,) int
    """
    return ((lstm_pred == 1) | (rule_pred == 1)).astype(int)


# ============================================================
#  格式化输出工具
# ============================================================

def fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def build_report(y_true, y_pred_lstm, y_prob_lstm,
                 y_pred_rule, y_pred_fusion) -> str:
    """生成完整评估报告文本，同时用于控制台打印和文件保存。"""
    lines = []
    add   = lines.append

    m_lstm   = compute_metrics(y_true, y_pred_lstm,   y_prob_lstm)
    m_rule   = compute_metrics(y_true, y_pred_rule)
    m_fusion = compute_metrics(y_true, y_pred_fusion)

    # ── A. LSTM 完整指标 ──────────────────────────────────────────
    add("")
    add("╔══════════════════════════════════════════╗")
    add("║       LSTM 模型评估结果 (测试集)          ║")
    add("╠══════════════════════════════════════════╣")
    add(f"║  Accuracy  : {fmt_pct(m_lstm['accuracy']):<28}║")
    add(f"║  Precision : {fmt_pct(m_lstm['precision']):<28}║")
    add(f"║  Recall    : {fmt_pct(m_lstm['recall']):<28}║")
    add(f"║  F1 Score  : {fmt_pct(m_lstm['f1']):<28}║")
    auc_str = f"{m_lstm['auc']:.4f}" if m_lstm["auc"] is not None else "N/A"
    add(f"║  AUC-ROC   : {auc_str:<28}║")
    add(f"║  FPR       : {fmt_pct(m_lstm['fpr']):<22}（误报率）  ║")
    add("╚══════════════════════════════════════════╝")

    # ── B. 混淆矩阵 ───────────────────────────────────────────────
    add("")
    add("混淆矩阵：")
    add(f"              预测正常   预测跌倒")
    add(f"实际正常:     TN={m_lstm['TN']:<6}   FP={m_lstm['FP']}")
    add(f"实际跌倒:     FN={m_lstm['FN']:<6}   TP={m_lstm['TP']}")

    # ── C. 消融实验表格 ───────────────────────────────────────────
    add("")
    add("消融实验对比（测试集）：")
    add("┌────────────────┬──────────┬───────────┬────────┬────────┐")
    add("│ 方法           │ Accuracy │ Precision │ Recall │   F1   │")
    add("├────────────────┼──────────┼───────────┼────────┼────────┤")

    def row(label, m):
        return (f"│ {label:<14} │ {fmt_pct(m['accuracy']):>8} │"
                f" {fmt_pct(m['precision']):>9} │"
                f" {fmt_pct(m['recall']):>6} │"
                f" {fmt_pct(m['f1']):>6} │")

    add(row("A. 纯几何规则",  m_rule))
    add(row("B. LSTM 单独",   m_lstm))
    add(row("C. LSTM+规则",   m_fusion))
    add("└────────────────┴──────────┴───────────┴────────┴────────┘")

    # 结论：LSTM vs 规则 F1 提升
    f1_gain = (m_lstm["f1"] - m_rule["f1"]) * 100
    fpr_drop = (m_rule["fpr"] - m_lstm["fpr"]) * 100
    sign = "+" if f1_gain >= 0 else ""
    add(f"结论：LSTM (方法B) 相比纯规则 (方法A) F1 提升了 {sign}{f1_gain:.2f}%，"
        f"误报率降低了 {fpr_drop:.2f}%")

    return "\n".join(lines)


# ============================================================
#  主评估流程
# ============================================================

def evaluate():
    print()
    print("=" * 60)
    print("  LSTM 跌倒分类器评估 + 消融实验")
    print("=" * 60)

    # ---- 前置检查 ----
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"\n❌ 数据集文件不存在：{TRAIN_DATA_FILE}")
        print("   请先运行：python data/preprocess.py")
        sys.exit(1)

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"\n❌ 模型权重不存在：{MODEL_WEIGHTS}")
        print("   请先运行：python train.py")
        sys.exit(1)

    # ---- 加载数据集（只使用测试集）----
    print()
    _, _, test_ds, _ = load_dataset(
        TRAIN_DATA_FILE,
        val_split  = TRAIN_VAL_SPLIT,
        test_split = TRAIN_TEST_SPLIT,
    )

    X_test_np = test_ds.X.numpy()   # (N, 30, 12) — 用于规则方法
    y_true    = test_ds.y.numpy().astype(int)
    X_test_t  = test_ds.X           # Tensor，用于 LSTM 推理

    n_test   = len(y_true)
    n_fall   = int(y_true.sum())
    n_normal = n_test - n_fall
    print(f"  测试集：{n_test} 条样本  跌倒={n_fall}  正常={n_normal}")

    # ---- 加载 LSTM 模型 ----
    device = get_device()
    print(f"  推理设备：{device}")

    ckpt = torch.load(MODEL_WEIGHTS, map_location=device)
    cfg  = ckpt.get("config", {})

    model = LSTMFallClassifier(
        input_dim  = cfg.get("input_dim",  FEATURE_DIM),
        hidden_dim = cfg.get("hidden_dim", LSTM_HIDDEN),
        num_layers = cfg.get("num_layers", LSTM_LAYERS),
        dropout    = cfg.get("dropout",    LSTM_DROPOUT),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    saved_epoch    = ckpt.get("epoch", "?")
    saved_val_loss = ckpt.get("val_loss", float("nan"))
    saved_val_acc  = ckpt.get("val_acc",  float("nan"))
    print(f"  加载权重：Epoch={saved_epoch}"
          f"  训练集验证: loss={saved_val_loss:.4f}"
          f"  acc={saved_val_acc*100:.1f}%")

    # ---- 三种方法预测 ----
    print("\n  正在推理...")

    # 方法A：纯几何规则
    y_pred_rule   = rule_predict(X_test_np)

    # 方法B：LSTM 单独
    y_pred_lstm, y_prob_lstm = lstm_predict(model, X_test_t, device, LSTM_FALL_THRESHOLD)

    # 方法C：LSTM + 规则融合
    y_pred_fusion = fusion_predict(y_pred_lstm, y_pred_rule)

    # ---- 生成报告 ----
    report = build_report(
        y_true, y_pred_lstm, y_prob_lstm,
        y_pred_rule, y_pred_fusion,
    )

    # 控制台打印
    print(report)

    # ---- 保存到文件 ----
    os.makedirs(LOG_DIR, exist_ok=True)
    result_file = os.path.join(LOG_DIR, "eval_results.txt")
    header = (
        f"DLCV 大作业评估结果\n"
        f"模型：LSTMFallClassifier  权重：{MODEL_WEIGHTS}\n"
        f"数据：{TRAIN_DATA_FILE}  测试集大小：{n_test}\n"
        f"{'─' * 60}\n"
    )
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(report)
        f.write("\n")

    print()
    print("=" * 60)
    print(f"  ✅ 评估完成，结果已保存至 {result_file}")
    print("     请将以上指标填写到答辩文档中")
    print("=" * 60)
    print()


if __name__ == "__main__":
    evaluate()