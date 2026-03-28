# ============================================================
#  data/dataset.py — PyTorch Dataset 封装
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset


class FallDataset(Dataset):
    """
    跌倒检测时序数据集。

    样本格式：
      X[i]: (SEQUENCE_LEN, FEATURE_DIM) float32 特征序列
      y[i]: scalar int   0=正常  1=跌倒
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        """
        Args:
            X:       shape (N, T, F)
            y:       shape (N,)
            augment: 训练时启用数据增强
        """
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            x = self._augment(x)
        return x, self.y[idx]

    # ----------------------------------------------------------
    #  数据增强（小样本场景常用技巧）
    # ----------------------------------------------------------
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 添加高斯噪声（模拟关键点定位误差）
        x = x + torch.randn_like(x) * 0.01

        # 2. 随机时间偏移（截取不同起始帧）
        T = x.shape[0]
        shift = torch.randint(0, max(1, T // 5), (1,)).item()
        if shift > 0:
            x = torch.cat([x[shift:], x[:shift]], dim=0)

        # 3. 特征归一化抖动（模拟不同体型）
        scale = 0.95 + torch.rand(1).item() * 0.10   # [0.95, 1.05]
        x     = x * scale

        return x


def load_dataset(data_file: str, val_split: float = 0.2, seed: int = 42):
    """
    加载 npz 数据集，按比例划分训练集和验证集。

    Returns:
        train_dataset, val_dataset, class_weights
    """
    data    = np.load(data_file)
    X, y    = data["X"], data["y"]

    rng     = np.random.RandomState(seed)
    idx     = rng.permutation(len(y))
    n_val   = max(1, int(len(y) * val_split))
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    X_tr, y_tr = X[trn_idx], y[trn_idx]
    X_vl, y_vl = X[val_idx],  y[val_idx]

    # 类别权重（处理样本不均衡）
    n_pos = max(1, int(y_tr.sum()))
    n_neg = max(1, int((y_tr == 0).sum()))
    w_pos = (n_neg + n_pos) / (2.0 * n_pos)
    w_neg = (n_neg + n_pos) / (2.0 * n_neg)
    class_weights = torch.tensor([w_neg, w_pos])

    train_ds = FallDataset(X_tr, y_tr, augment=True)
    val_ds   = FallDataset(X_vl, y_vl, augment=False)

    print(f"数据集加载完毕：训练={len(train_ds)}  验证={len(val_ds)}")
    print(f"  训练集正例（跌倒）：{int(y_tr.sum())}  负例（正常）：{int((y_tr==0).sum())}")

    return train_ds, val_ds, class_weights