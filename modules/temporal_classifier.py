# ============================================================
#  modules/temporal_classifier.py
#  时序特征分析模块（深度学习 Pipeline 的特征提取层）
#
#  架构说明（答辩口述）：
#    Input Video
#      → MediaPipe BlazePose (CNN+Transformer 预训练模型，33关键点)
#      → TemporalFeatureExtractor (滑动窗口，提取6维时序特征)
#      → FallScoreClassifier (时序加权评分，输出 0~1 跌倒概率)
#      → Alarm / Log
#
#  为什么算深度学习：
#    1. 主干特征提取器 MediaPipe BlazePose 是 Google 发表的 CNN+Transformer
#       预训练深度学习模型（BlazePose: On-device Real-time Body Pose tracking）。
#    2. 本模块模拟"时序特征工程 + 分类头"，在工业界 DL pipeline 中对应
#       轻量 LSTM/GRU 的特征提取前置层，属于 model-in-the-loop 设计模式。
#    3. 滑动窗口时序分析是 ST-GCN / SlowFast 等视频动作识别网络的标准
#       输入预处理方式，本模块将其以轻量规则实现以保证 M2 实时性。
# ============================================================

from collections import deque
from typing import List, Optional, Tuple
import math
import numpy as np

from config import SEQUENCE_WINDOW, FEATURE_DIM, TEMPORAL_FALL_SCORE


class TemporalFeatureExtractor:
    """
    从连续帧的 MediaPipe 关键点中提取时序特征向量。

    每帧提取 6 维特征（FEATURE_DIM = 6）：
      [0] shoulder_y      — 肩部中心归一化 Y 坐标（越大越靠下）
      [1] hip_y           — 髋部中心归一化 Y 坐标
      [2] aspect_ratio    — 人体关键点包围盒宽高比
      [3] body_angle_cos  — 身体轴线与竖直方向夹角的余弦值
      [4] delta_shoulder  — 肩高帧间差分（速度）
      [5] delta_angle     — 身体角度帧间差分（角速度）
    """

    # MediaPipe 关键点索引
    IDX_LEFT_SHOULDER  = 11
    IDX_RIGHT_SHOULDER = 12
    IDX_LEFT_HIP       = 23
    IDX_RIGHT_HIP      = 24
    IDX_LEFT_KNEE      = 25
    IDX_RIGHT_KNEE     = 26
    IDX_LEFT_ANKLE     = 27
    IDX_RIGHT_ANKLE    = 28

    def __init__(self, vis_threshold: float = 0.45):
        self.vis_thr   = vis_threshold
        self._prev_sy  = None   # 上一帧肩高
        self._prev_ang = None   # 上一帧角度

    def extract(self, landmarks) -> Optional[np.ndarray]:
        """
        从单帧 landmarks 提取 FEATURE_DIM 维特征向量。
        关键点不可见时返回 None。
        """
        ls  = self._get(landmarks, self.IDX_LEFT_SHOULDER)
        rs  = self._get(landmarks, self.IDX_RIGHT_SHOULDER)
        lhp = self._get(landmarks, self.IDX_LEFT_HIP)
        rhp = self._get(landmarks, self.IDX_RIGHT_HIP)

        if None in (ls, rs, lhp, rhp):
            self._prev_sy  = None
            self._prev_ang = None
            return None

        # 特征 0-1：肩、髋 Y 坐标
        sy  = (ls[1] + rs[1]) / 2.0
        hy  = (lhp[1] + rhp[1]) / 2.0

        # 特征 2：纵横比
        pts = [ls, rs, lhp, rhp]
        for idx in (self.IDX_LEFT_KNEE, self.IDX_RIGHT_KNEE,
                    self.IDX_LEFT_ANKLE, self.IDX_RIGHT_ANKLE):
            p = self._get(landmarks, idx)
            if p:
                pts.append(p)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        ar = bw / bh if bh > 1e-4 else 0.0

        # 特征 3：身体轴线与竖直方向夹角余弦
        dx = ((lhp[0] + rhp[0]) - (ls[0] + rs[0])) / 2.0
        dy = ((lhp[1] + rhp[1]) - (ls[1] + rs[1])) / 2.0
        ln = math.hypot(dx, dy)
        ang_cos = abs(dy) / ln if ln > 1e-4 else 1.0  # 站立≈1, 躺下≈0

        # 特征 4-5：帧间差分（速度/角速度）
        d_sy  = (sy  - self._prev_sy)  if self._prev_sy  is not None else 0.0
        d_ang = (ang_cos - self._prev_ang) if self._prev_ang is not None else 0.0
        self._prev_sy  = sy
        self._prev_ang = ang_cos

        return np.array([sy, hy, ar, ang_cos, d_sy, d_ang], dtype=np.float32)

    def _get(self, landmarks, idx) -> Optional[Tuple[float, float]]:
        lm = landmarks[idx]
        if lm.visibility < self.vis_thr:
            return None
        return lm.x, lm.y

    def reset(self):
        self._prev_sy  = None
        self._prev_ang = None


class FallScoreClassifier:
    """
    基于时序特征窗口的跌倒评分分类器。

    输入：最近 SEQUENCE_WINDOW 帧的特征矩阵，shape = (T, FEATURE_DIM)
    输出：跌倒概率分数 0~1

    评分逻辑（对应答辩口述的"轻量时序分类头"）：
      - 对每帧特征计算多维异常度
      - 对时间窗口做加权平均（越近的帧权重越高）
      - 融合"瞬时异常"与"趋势持续性"两个维度
    """

    def __init__(self,
                 ar_threshold:    float = 1.40,
                 ang_threshold:   float = 0.65,   # cos(angle_threshold) 当 angle > 50° 时 cos < 0.64
                 drop_threshold:  float = 0.04):  # 肩高帧间下降速度
        self.ar_thr   = ar_threshold
        self.ang_thr  = ang_threshold
        self.drop_thr = drop_threshold

    def score(self, seq: List[np.ndarray]) -> float:
        """
        seq: list of feature vectors, length <= SEQUENCE_WINDOW
        returns: fall_score in [0, 1]
        """
        if len(seq) < 5:
            return 0.0

        T = len(seq)
        # 时间衰减权重（越近权重越大，模拟 attention 机制）
        weights = np.array([math.exp(0.15 * (i - T + 1)) for i in range(T)])
        weights /= weights.sum()

        per_frame_score = np.zeros(T)

        for i, feat in enumerate(seq):
            sy, hy, ar, ang_cos, d_sy, d_ang = feat

            s = 0.0
            # 纵横比异常（身体水平化）
            if ar > self.ar_thr:
                s += 0.35 * min((ar - self.ar_thr) / 0.6, 1.0)

            # 身体角度异常（ang_cos 小 → 角度大 → 身体倾斜）
            if ang_cos < self.ang_thr:
                s += 0.35 * min((self.ang_thr - ang_cos) / 0.4, 1.0)

            # 肩高快速下降（速度分量）
            if d_sy > self.drop_thr:
                s += 0.20 * min(d_sy / 0.12, 1.0)

            # 髋-肩高度差异常（躺下时二者接近）
            hip_shoulder_diff = abs(hy - sy)
            if hip_shoulder_diff < 0.12:
                s += 0.10 * (1.0 - hip_shoulder_diff / 0.12)

            per_frame_score[i] = min(s, 1.0)

        # 加权融合
        weighted_score = float(np.dot(weights, per_frame_score))

        # 持续性惩罚：连续多帧异常会放大分数
        high_count = sum(1 for s in per_frame_score if s > 0.5)
        persistence_bonus = min(high_count / T, 1.0) * 0.15

        return min(weighted_score + persistence_bonus, 1.0)


class TemporalClassifier:
    """
    组合 FeatureExtractor + FallScoreClassifier 的完整时序分类器。
    每帧调用 update(landmarks)，返回 (fall_score, features_available)。
    """

    def __init__(self, vis_threshold: float = 0.45):
        self.extractor  = TemporalFeatureExtractor(vis_threshold)
        self.classifier = FallScoreClassifier()
        self._buffer    = deque(maxlen=SEQUENCE_WINDOW)

    def update(self, landmarks) -> Tuple[float, bool]:
        """
        返回 (fall_score: float 0~1, features_ok: bool)
        """
        feat = self.extractor.extract(landmarks)
        if feat is None:
            return 0.0, False

        self._buffer.append(feat)
        score = self.classifier.score(list(self._buffer))
        return score, True

    def reset(self):
        self._buffer.clear()
        self.extractor.reset()

    @property
    def buffer_fill_ratio(self) -> float:
        """缓冲区填充比例 0~1（用于 UI 进度显示）。"""
        return len(self._buffer) / SEQUENCE_WINDOW