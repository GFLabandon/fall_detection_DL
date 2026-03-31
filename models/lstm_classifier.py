# fall_detection_DL/models/lstm_classifier.py
# LSTM 跌倒检测分类器
#
# 网络结构：
#   Input (batch, T=30, F=12)
#     → LSTM (hidden=64, layers=2, dropout=0.3, 单向)
#     → 取最后帧隐状态 (batch, 64)
#     → Dropout(0.3)
#     → Linear(64→32) + ReLU + Dropout(0.2)
#     → Linear(32→1)          ← 输出 logit（配合 BCEWithLogitsLoss）
#     → squeeze → (batch,)
#
# 训练配套：BCEWithLogitsLoss + Adam + CosineAnnealingLR
# 推理方式：predict_proba() 在 logit 上应用 Sigmoid，返回跌倒概率 [0, 1]

import torch
import torch.nn as nn
from typing import Optional


class LSTMFallClassifier(nn.Module):
    """
    基于 LSTM 的跌倒二分类器。

    设计原则：
      - 单向 LSTM（bidirectional=False），支持实时流式推理
      - 输出原始 logit，由调用方决定是否应用 Sigmoid：
          训练时：直接传入 BCEWithLogitsLoss（数值更稳定）
          推理时：调用 predict_proba()，内部应用 Sigmoid

    Args:
        input_dim:  每帧特征维度（与 FEATURE_DIM=12 对应）
        hidden_dim: LSTM 隐层维度
        num_layers: LSTM 层数
        dropout:    Dropout 比例
    """

    def __init__(self,
                 input_dim:  int   = 12,
                 hidden_dim: int   = 64,
                 num_layers: int   = 2,
                 dropout:    float = 0.3):
        super().__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p  = dropout

        # ---- LSTM 主体：时序建模核心 ----
        # batch_first=True：输入格式为 (batch, seq, feature)，更直观
        # bidirectional=False：必须单向，保证实时推理时只需要历史帧
        # dropout：在 LSTM 层间应用（PyTorch 在 num_layers=1 时自动忽略此参数）
        self.lstm = nn.LSTM(
            input_size   = input_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            bidirectional= False,
        )

        # ---- 分类头 ----
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),        # 输出单个 logit
        )

        # 权重初始化
        self._init_weights()

    # ----------------------------------------------------------
    #  前向传播
    # ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) float32

        Returns:
            logit: (batch_size,) — 未经 Sigmoid 的原始输出
        """
        # LSTM 前向传播
        # out:  (batch, seq_len, hidden_dim) — 每帧的隐状态输出
        # hn:   (num_layers, batch, hidden_dim) — 最终隐状态
        out, (hn, _) = self.lstm(x)

        # 取序列最后一帧的输出作为全局表示
        # out[:, -1, :] 等价于 hn[-1]（单向 LSTM 最后一层的最终隐状态）
        last = out[:, -1, :]   # (batch, hidden_dim)

        # 通过分类头得到 logit
        logit = self.classifier(last)   # (batch, 1)
        return logit.squeeze(-1)        # (batch,)

    # ----------------------------------------------------------
    #  推理接口
    # ----------------------------------------------------------

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        推理时调用：返回跌倒概率（经 Sigmoid 的输出）。

        Args:
            x: (batch_size, seq_len, input_dim) 或 (seq_len, input_dim) float32
               单样本时自动扩展 batch 维度

        Returns:
            prob: (batch_size,) float32，跌倒概率 ∈ [0, 1]
        """
        # 单样本自动扩充 batch 维度
        single = (x.dim() == 2)
        if single:
            x = x.unsqueeze(0)

        with torch.no_grad():
            logit = self.forward(x)
            prob  = torch.sigmoid(logit)

        return prob.squeeze(0) if single else prob

    # ----------------------------------------------------------
    #  参数量统计
    # ----------------------------------------------------------

    def count_parameters(self) -> int:
        """
        统计并打印可训练参数量。
        预期约 37,441（hidden=64, layers=2, input=12）。
        """
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  🧠 LSTM 模型参数量：{n:,} 个可学习参数")
        return n

    def summary(self):
        """打印模型结构摘要，供 train.py 启动时展示。"""
        print(f"\n{'─'*55}")
        print(f"  LSTMFallClassifier 结构")
        print(f"{'─'*55}")
        print(f"  输入:    (batch, {SEQUENCE_LEN_HINT}, {self.input_dim})")
        print(f"  LSTM:    hidden={self.hidden_dim}  layers={self.num_layers}"
              f"  {'双向' if False else '单向'}")
        print(f"  分类头:  Linear({self.hidden_dim}→32) → ReLU → Linear(32→1)")
        print(f"  输出:    (batch,) logit → Sigmoid → P(跌倒) ∈ [0,1]")
        self.count_parameters()
        print(f"{'─'*55}\n")

    # ----------------------------------------------------------
    #  权重初始化
    # ----------------------------------------------------------

    def _init_weights(self):
        """
        Xavier Uniform 初始化：加速收敛，避免梯度消失/爆炸。
        LSTM 权重和 Linear 权重均使用 xavier_uniform_，bias 全置零。
        """
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)


# 供 summary() 方法使用的提示（避免循环导入 config）
SEQUENCE_LEN_HINT = 30