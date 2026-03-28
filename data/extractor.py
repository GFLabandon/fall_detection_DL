# ============================================================
#  data/extractor.py — 从 MediaPipe 关键点提取特征向量
#
#  每帧提取 12 维特征（FEATURE_DIM = 12）：
#    [0]  shoulder_y     — 肩部中心归一化 Y（越大越靠下）
#    [1]  hip_y          — 髋部中心归一化 Y
#    [2]  ankle_y        — 踝部中心归一化 Y
#    [3]  aspect_ratio   — 人体包围盒宽/高比
#    [4]  body_angle_cos — 身体轴线与竖直方向夹角余弦（站立≈1，躺≈0）
#    [5]  hip_shoulder_gap — 髋-肩 Y 差（躺下时趋近 0）
#    [6]  delta_shoulder — 肩高帧间差分（速度，快速下落时大）
#    [7]  delta_hip      — 髋高帧间差分
#    [8]  delta_angle    — 角度帧间差分（角速度）
#    [9]  wrist_y_mean   — 双手腕 Y 均值（倒地时手臂位置）
#    [10] knee_y_mean    — 双膝 Y 均值
#    [11] head_body_ratio — 头部 Y / 髋部 Y（头脚相对位置）
# ============================================================

import math
import numpy as np
from typing import Optional, Tuple

# MediaPipe 关键点索引
IDX = {
    "nose":         0,
    "l_shoulder":   11, "r_shoulder": 12,
    "l_elbow":      13, "r_elbow":    14,
    "l_wrist":      15, "r_wrist":    16,
    "l_hip":        23, "r_hip":      24,
    "l_knee":       25, "r_knee":     26,
    "l_ankle":      27, "r_ankle":    28,
}

FEATURE_DIM = 12


class FeatureExtractor:
    """单帧特征提取器，含帧间差分计算。"""

    def __init__(self, vis_threshold: float = 0.45):
        self.vis_thr   = vis_threshold
        self._prev     = None          # 上一帧特征（用于差分）

    # ----------------------------------------------------------
    def extract(self, landmarks) -> Optional[np.ndarray]:
        """
        从单帧 MediaPipe Pose landmarks 提取 FEATURE_DIM 维特征。
        关键点不满足可见性条件时返回 None。
        """
        def get(name) -> Optional[Tuple[float, float]]:
            lm = landmarks[IDX[name]]
            if lm.visibility < self.vis_thr:
                return None
            return lm.x, lm.y

        ls  = get("l_shoulder")
        rs  = get("r_shoulder")
        lhp = get("l_hip")
        rhp = get("r_hip")

        # 肩、髋必须可见
        if None in (ls, rs, lhp, rhp):
            self._prev = None
            return None

        # ---- 基础坐标 ----
        sy  = (ls[1] + rs[1]) / 2.0
        hy  = (lhp[1] + rhp[1]) / 2.0

        la  = get("l_ankle")
        ra  = get("r_ankle")
        ay  = ((la[1] + ra[1]) / 2.0) if (la and ra) else hy + 0.3

        lw  = get("l_wrist")
        rw  = get("r_wrist")
        wy  = ((lw[1] + rw[1]) / 2.0) if (lw and rw) else sy

        lk  = get("l_knee")
        rk  = get("r_knee")
        ky  = ((lk[1] + rk[1]) / 2.0) if (lk and rk) else (hy + ay) / 2.0

        nose = get("nose")
        ny   = nose[1] if nose else max(0.0, sy - 0.15)

        # ---- 包围盒纵横比 ----
        pts  = [ls, rs, lhp, rhp]
        if la: pts.append(la)
        if ra: pts.append(ra)
        xs   = [p[0] for p in pts]
        ys   = [p[1] for p in pts]
        bw   = max(xs) - min(xs)
        bh   = max(ys) - min(ys)
        ar   = bw / bh if bh > 1e-4 else 0.0

        # ---- 身体轴线角度余弦（肩→髋向量与竖直方向）----
        dx   = ((lhp[0] + rhp[0]) - (ls[0] + rs[0])) / 2.0
        dy   = ((lhp[1] + rhp[1]) - (ls[1] + rs[1])) / 2.0
        ln   = math.hypot(dx, dy)
        ang  = abs(dy) / ln if ln > 1e-4 else 1.0   # 站立≈1，水平≈0

        # ---- 髋-肩高度差 ----
        gap  = abs(hy - sy)

        # ---- 头-髋相对位置 ----
        hbr  = ny / hy if hy > 1e-4 else 0.5

        # ---- 静态特征 ----
        static = np.array([sy, hy, ay, ar, ang, gap, 0.0, 0.0, 0.0, wy, ky, hbr],
                           dtype=np.float32)

        # ---- 差分特征（动态速度）----
        if self._prev is not None:
            static[6] = float(sy  - self._prev[0])   # Δ肩高
            static[7] = float(hy  - self._prev[1])   # Δ髋高
            static[8] = float(ang - self._prev[4])   # Δ角度
        self._prev = static.copy()

        return static

    def reset(self):
        self._prev = None