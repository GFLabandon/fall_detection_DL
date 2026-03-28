# ============================================================
#  models/lstm_fall_classifier.py
#  跌倒检测 LSTM 分类器（真正的 PyTorch 深度学习模型）
#
#  网络结构：
#    Input (batch, T=30, F=12)
#      → LSTM (hidden=64, layers=2, dropout=0.3)  ← 时序建模核心
#      → 取最后一帧隐状态 (batch, 64)
#      → Dropout(0.3)
#      → Linear(64, 32) + ReLU + Dropout(0.2)     ← 分类头
#      → Linear(32, 1) + Sigmoid
#      → 跌倒概率 P ∈ [0, 1]
#
#  训练细节：
#    Loss:      BCEWithLogitsLoss（带类别权重，处理样本不均衡）
#    Optimizer: Adam(lr=1e-3, weight_decay=1e-4)
#    Scheduler: CosineAnnealingLR（动态学习率）
#    Epochs:    60
#    Augment:   噪声注入 + 时间偏移 + 特征缩放
# ============================================================

import torch
import torch.nn as nn


class LSTMFallClassifier(nn.Module):
    """
    LSTM-based 跌倒检测分类器。

    Args:
        input_dim:   每帧特征维度（FEATURE_DIM=12）
        hidden_dim:  LSTM 隐层维度（LSTM_HIDDEN=64）
        num_layers:  LSTM 层数（LSTM_LAYERS=2）
        dropout:     Dropout 比例
    """

    def __init__(self,
                 input_dim:  int = 12,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout:    float = 0.3):
        super().__init__()

        # ---- LSTM 主体：时序建模 ----
        self.lstm = nn.LSTM(
            input_size   = input_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            bidirectional= False,        # 单向（实时推理不能用双向）
        )

        # ---- 分类头 ----
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),           # 输出 logit（配合 BCEWithLogitsLoss）
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 均匀初始化，加速收敛。"""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, F) float32 时序特征矩阵
        Returns:
            logits: (batch,) — 未经 sigmoid 的原始输出
        """
        # LSTM 前向传播：只取最后一帧的隐状态
        out, (hn, _) = self.lstm(x)      # out: (batch, T, hidden)
        last = out[:, -1, :]              # (batch, hidden) — 取序列末尾

        logits = self.classifier(last).squeeze(-1)   # (batch,)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回跌倒概率（经过 Sigmoid）。
        推理时使用此方法。
        """
        with torch.no_grad():
            logits = self.forward(x)
        return torch.sigmoid(logits)

    # ----------------------------------------------------------
    #  模型统计信息
    # ----------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self, input_dim: int, seq_len: int):
        print(f"\n{'='*50}")
        print(f"  LSTMFallClassifier 网络结构")
        print(f"{'='*50}")
        print(f"  输入:    (batch, {seq_len}, {input_dim})")
        print(f"  LSTM:    hidden={self.lstm.hidden_size}  layers={self.lstm.num_layers}")
        print(f"  分类头:  Linear({self.lstm.hidden_size}→32) → ReLU → Linear(32→1)")
        print(f"  输出:    (batch,) logit → Sigmoid → P(跌倒) ∈ [0,1]")
        print(f"  参数量:  {self.count_parameters():,} 个可学习参数")
        print(f"{'='*50}\n")