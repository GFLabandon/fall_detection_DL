# fall_detection_DL/data/dataset.py
# PyTorch 数据集封装
#
# 提供：
#   FallDataset  — 继承 torch.utils.data.Dataset，支持数据增强
#   load_dataset — 分层划分训练/验证/测试集，返回 DataLoader 可用的 Dataset 实例

import os
import sys
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# 确保可以从项目根目录导入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEQUENCE_LEN, FEATURE_DIM


# ============================================================
#  FallDataset
# ============================================================

class FallDataset(Dataset):
    """
    跌倒检测时序数据集。

    每个样本由一段 SEQUENCE_LEN 帧的特征序列及其二分类标签组成：
      X[i]: shape (SEQUENCE_LEN, FEATURE_DIM)  float32
      y[i]: scalar float32   0.0=正常  1.0=跌倒

    标签使用 float32 以直接配合 BCEWithLogitsLoss（无需额外类型转换）。

    Args:
        X:       shape (N, SEQUENCE_LEN, FEATURE_DIM)，来自 preprocess.py 输出
        y:       shape (N,)，int 标签数组
        augment: 是否启用数据增强（训练集开启，验证/测试集关闭）
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        # 转换为 float32 tensor，避免训练时频繁类型转换
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        """
        返回 (x, y)：
          x: torch.Tensor shape (SEQUENCE_LEN, FEATURE_DIM) float32
          y: torch.Tensor scalar float32
        """
        x = self.X[idx].clone()   # clone 防止 in-place 修改影响原数据

        if self.augment:
            x = self._augment(x)

        return x, self.y[idx]

    # ----------------------------------------------------------
    #  数据增强（仅训练集启用）
    # ----------------------------------------------------------

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        三种轻量增强，模拟真实场景的变化：

        1. 高斯噪声注入：模拟 MediaPipe 关键点定位误差（约 1% 幅度）
        2. 随机时间偏移：模拟不同起始帧的序列（循环移位，不改变统计特性）
        3. 特征缩放抖动：模拟不同体型/摄像头距离（±5% 范围内缩放）

        以上增强在 numpy 层面操作后转回 tensor，或直接在 tensor 上操作。
        """
        # ---- 增强 1：高斯噪声（σ=0.01）----
        noise = torch.randn_like(x) * 0.01
        x = x + noise

        # ---- 增强 2：随机时间偏移（循环移位 0~5 帧）----
        offset = random.randint(0, 5)
        if offset > 0:
            # torch.roll 沿时间轴（dim=0）循环移位
            x = torch.roll(x, shifts=offset, dims=0)

        # ---- 增强 3：特征缩放抖动（[0.95, 1.05] 均匀分布）----
        scale = 0.95 + random.random() * 0.10   # [0.95, 1.05)
        x = x * scale

        return x


# ============================================================
#  load_dataset — 分层划分并构建数据集实例
# ============================================================

def load_dataset(data_file: str,
                 val_split:  float = 0.2,
                 test_split: float = 0.1,
                 seed:       int   = 42):
    """
    加载预处理好的 .npz 文件，按分层采样策略划分训练/验证/测试集。

    划分顺序：
      1. 先从全量数据中划分 test_split 比例作为测试集（独立保留，不参与任何训练）
      2. 剩余数据再划分 val_split 比例作为验证集
      3. 其余全部为训练集

    Args:
        data_file:  预处理输出的 .npz 文件路径（TRAIN_DATA_FILE）
        val_split:  验证集比例（相对于非测试集部分）
        test_split: 测试集比例（相对于全量数据）
        seed:       随机种子，保证实验可复现

    Returns:
        train_ds:    FallDataset，augment=True
        val_ds:      FallDataset，augment=False
        test_ds:     FallDataset，augment=False
        pos_weight:  torch.Tensor scalar，= n_negative / n_positive
                     传入 BCEWithLogitsLoss(pos_weight=...) 处理类别不均衡
    """
    # ---- 检查文件 ----
    if not os.path.exists(data_file):
        print(f"❌ 数据集文件不存在：{data_file}")
        print("   请先运行：python data/preprocess.py")
        sys.exit(1)

    # ---- 加载数据 ----
    data = np.load(data_file)
    X    = data["X"]   # (N, SEQUENCE_LEN, FEATURE_DIM)
    y    = data["y"]   # (N,)

    n_total  = len(y)
    n_fall   = int(y.sum())
    n_normal = n_total - n_fall

    # ---- 分层划分：先划出测试集 ----
    # stratify=y 保证跌倒/正常比例在各分集中一致
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size  = test_split,
        random_state = seed,
        stratify   = y,
    )

    # ---- 再划分验证集 ----
    # val_split 是相对于非测试集部分的比例
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size    = val_split,
        random_state = seed,
        stratify     = y_trainval,
    )

    # ---- 计算正例权重（处理类别不均衡）----
    n_train_fall   = int(y_train.sum())
    n_train_normal = len(y_train) - n_train_fall
    # 防止除零（极端情况下某类样本为0）
    n_train_fall   = max(n_train_fall,   1)
    n_train_normal = max(n_train_normal, 1)
    pos_weight = torch.tensor(n_train_normal / n_train_fall, dtype=torch.float32)

    # ---- 构建 Dataset 实例 ----
    train_ds = FallDataset(X_train, y_train, augment=True)
    val_ds   = FallDataset(X_val,   y_val,   augment=False)
    test_ds  = FallDataset(X_test,  y_test,  augment=False)

    # ---- 打印统计信息 ----
    print("📊 数据集加载完成")
    print(f"   总样本: {n_total}   跌倒: {n_fall}   正常: {n_normal}")
    print(f"   训练集: {len(train_ds)}   验证集: {len(val_ds)}   测试集: {len(test_ds)}")
    print(f"   pos_weight: {pos_weight.item():.2f}（用于处理类别不均衡）")

    return train_ds, val_ds, test_ds, pos_weight