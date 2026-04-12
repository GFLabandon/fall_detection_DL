# fall_detection_DL/data/extractor.py
# 单帧骨架特征提取器
#
# 【重要】本模块同时被两个阶段调用，必须保持逻辑完全一致：
#   - 训练阶段：data/preprocess.py 批量处理 URFD 视频帧
#   - 运行阶段：modules/detector.py 实时处理摄像头帧
# 任何特征计算逻辑的修改都必须同时影响两个阶段，防止训练-推理不一致。
#
# ── v2 改动 ──────────────────────────────────────────────────
# Domain-shift fix: 所有Y坐标改为以肩部为原点、肩髋距离为单位的相对坐标。
# 原因: URFD数据集全身入画(shoulder_y≈0.30)与实时摄像头近距上半身
#       (shoulder_y≈0.55)存在约+0.25的绝对Y平移，导致站立被误判为跌倒。
#       相对坐标消除此偏差: 任意距离下站立者 shoulder_rel=0, hip_rel≈1。
# 注意: preprocess.py需重跑以用新特征重新训练模型。
# v3 新增修复：
#   - 倒置骨架过滤：body_angle_cos < -0.5（angle>120°）时返回 None
#     效果：彻底消除背光/快速移动导致的倒置骨架误触发 B-Static
#   - 保留 v2 的 Euclidean scale（欧氏距离）归一化方案

import math
import numpy as np
from typing import Optional, Tuple

IDX_NOSE           = 0
IDX_LEFT_WRIST     = 15
IDX_RIGHT_WRIST    = 16
IDX_LEFT_SHOULDER  = 11
IDX_RIGHT_SHOULDER = 12
IDX_LEFT_HIP       = 23
IDX_RIGHT_HIP      = 24
IDX_LEFT_KNEE      = 25
IDX_RIGHT_KNEE     = 26
IDX_LEFT_ANKLE     = 27
IDX_RIGHT_ANKLE    = 28

_BBOX_INDICES = [
    IDX_LEFT_SHOULDER, IDX_RIGHT_SHOULDER,
    IDX_LEFT_HIP,      IDX_RIGHT_HIP,
    IDX_LEFT_KNEE,     IDX_RIGHT_KNEE,
    IDX_LEFT_ANKLE,    IDX_RIGHT_ANKLE,
]

FEATURE_NAMES = [
    "shoulder_y_rel",   # 0  肩部相对Y（= 0.0，参考原点）
    "hip_y_rel",        # 1  髋部相对Y（站立≈1.0，躺下<0.5）
    "ankle_y_rel",      # 2  踝部相对Y（站立≈2.0+）
    "aspect_ratio",     # 3  人体关键点包围盒 宽/高
    "body_angle_cos",   # 4  身体轴线与竖直方向夹角余弦
    "hip_shoulder_gap", # 5  髋-肩 Y 相对距离（站立≈1.0）
    "delta_shoulder",   # 6  肩高帧间变化（相对单位）
    "delta_hip",        # 7  髋高帧间变化（相对单位）
    "delta_angle",      # 8  角度余弦帧间变化
    "wrist_y_rel",      # 9  双腕相对Y
    "knee_y_rel",       # 10 双膝相对Y
    "head_hip_ratio",   # 11 鼻尖相对Y / 髋部相对Y
]

FEATURE_DIM = len(FEATURE_NAMES)  # 12

# ── v3 新增常量 ──────────────────────────────────────────────
# body_angle_cos < 此值时认为骨架倒置（angle > 120°），直接丢弃该帧
# 正常跌倒最大角度约90°（cos≈0），躺平约85°~90°（cos≈0~0.09）
# 120°以上（cos<-0.5）说明MediaPipe把肩髋位置搞反，必须过滤
_INVERTED_COS_THRESHOLD = -0.5


class FeatureExtractor:
    """
    从单帧 MediaPipe Pose landmarks 提取 12 维特征向量。

    v2: 所有Y坐标改用以肩部为原点、肩髋欧氏距离为尺度的相对坐标。
    v3: 新增倒置骨架过滤（body_angle_cos < -0.5 → 返回 None）。
    """

    def __init__(self, vis_threshold: float = 0.45):
        self.vis_thr = vis_threshold
        self._prev_shoulder_y: Optional[float] = None
        self._prev_hip_y:      Optional[float] = None
        self._prev_angle_cos:  Optional[float] = None
        # v3: 追踪可见关键点数量，供 detector 判断骨架质量
        self.last_visible_count: int = 0

    def extract(self, landmarks) -> Optional[np.ndarray]:
        """
        从单帧 landmarks 提取 FEATURE_DIM=12 维特征向量。
        以下情况返回 None（detector 将判定为 insufficient）：
          1. 肩/髋不可见
          2. v3新增：骨架倒置（body_angle_cos < -0.5）
        """
        ls  = self._get(landmarks, IDX_LEFT_SHOULDER)
        rs  = self._get(landmarks, IDX_RIGHT_SHOULDER)
        lhp = self._get(landmarks, IDX_LEFT_HIP)
        rhp = self._get(landmarks, IDX_RIGHT_HIP)

        # 统计可见关键点数量
        self.last_visible_count = sum(
            1 for idx in _BBOX_INDICES
            if landmarks[idx].visibility >= self.vis_thr
        )

        if ls is None or rs is None or lhp is None or rhp is None:
            self._reset_prev()
            return None

        # ---- 绝对坐标 ----
        shoulder_y_abs = (ls[1] + rs[1]) / 2.0
        shoulder_x     = (ls[0] + rs[0]) / 2.0
        hip_y_abs      = (lhp[1] + rhp[1]) / 2.0
        hip_x          = (lhp[0] + rhp[0]) / 2.0

        # ── v3: 先计算 body_angle_cos，倒置骨架提前返回 None ───────
        # 用绝对坐标计算（在归一化之前），避免scale灾难
        vx = hip_x - shoulder_x
        vy = hip_y_abs - shoulder_y_abs
        torso_dist = math.hypot(vx, vy)
        scale = max(torso_dist, 0.10)
        body_angle_cos = float(np.clip(vy / scale, -1.0, 1.0))

        if body_angle_cos < _INVERTED_COS_THRESHOLD:
            # 骨架倒置（angle > 120°）：MediaPipe检测失误，丢弃此帧
            # 同时清空前帧状态，避免污染 delta 计算
            self._reset_prev()
            return None

        # ── 相对坐标归一化 ────────────────────────────────────────
        origin = shoulder_y_abs

        def rel(y_abs):
            return (y_abs - origin) / scale

        # ---- 踝部 ----
        la = self._get(landmarks, IDX_LEFT_ANKLE)
        ra = self._get(landmarks, IDX_RIGHT_ANKLE)
        if la is not None and ra is not None:
            ankle_y = rel((la[1] + ra[1]) / 2.0)
        elif la is not None:
            ankle_y = rel(la[1])
        elif ra is not None:
            ankle_y = rel(ra[1])
        else:
            ankle_y = rel(hip_y_abs) + 1.2

        # ---- 腕部 ----
        lw = self._get(landmarks, IDX_LEFT_WRIST)
        rw = self._get(landmarks, IDX_RIGHT_WRIST)
        if lw is not None and rw is not None:
            wrist_y = rel((lw[1] + rw[1]) / 2.0)
        elif lw is not None:
            wrist_y = rel(lw[1])
        elif rw is not None:
            wrist_y = rel(rw[1])
        else:
            wrist_y = 0.0

        # ---- 膝部 ----
        lk = self._get(landmarks, IDX_LEFT_KNEE)
        rk = self._get(landmarks, IDX_RIGHT_KNEE)
        if lk is not None and rk is not None:
            knee_y = rel((lk[1] + rk[1]) / 2.0)
        elif lk is not None:
            knee_y = rel(lk[1])
        elif rk is not None:
            knee_y = rel(rk[1])
        else:
            knee_y = rel(hip_y_abs) + 0.6

        # ---- 鼻子 ----
        nose = self._get(landmarks, IDX_NOSE)
        nose_y = rel(nose[1]) if nose is not None else -0.5

        # ---- 相对坐标的肩/髋 ----
        shoulder_y = rel(shoulder_y_abs)   # = 0.0（固定参考点）
        hip_y      = rel(hip_y_abs)        # ≈ 1.0 站立，< 0.5 跌倒

        # ---- 包围盒纵横比 ----
        aspect_ratio = self._calc_aspect_ratio(landmarks)

        # ---- 髋-肩高度差 ----
        hip_shoulder_gap = abs(hip_y - shoulder_y)

        # ---- 帧间差分 ----
        if self._prev_shoulder_y is not None:
            delta_shoulder = shoulder_y     - self._prev_shoulder_y
            delta_hip      = hip_y          - self._prev_hip_y
            delta_angle    = body_angle_cos - self._prev_angle_cos
        else:
            delta_shoulder = 0.0
            delta_hip      = 0.0
            delta_angle    = 0.0

        self._prev_shoulder_y = shoulder_y
        self._prev_hip_y      = hip_y
        self._prev_angle_cos  = body_angle_cos

        # ---- 头-髋比 ----
        head_hip_ratio = nose_y / (hip_y + 1e-6)

        feat = np.array([
            shoulder_y,       # 0
            hip_y,            # 1
            ankle_y,          # 2
            aspect_ratio,     # 3
            body_angle_cos,   # 4
            hip_shoulder_gap, # 5
            delta_shoulder,   # 6
            delta_hip,        # 7
            delta_angle,      # 8
            wrist_y,          # 9
            knee_y,           # 10
            head_hip_ratio,   # 11
        ], dtype=np.float32)

        return feat

    def reset(self):
        self._reset_prev()

    def _get(self, landmarks, idx: int) -> Optional[Tuple[float, float]]:
        lm = landmarks[idx]
        if lm.visibility < self.vis_thr:
            return None
        return float(lm.x), float(lm.y)

    def _calc_aspect_ratio(self, landmarks) -> float:
        xs, ys = [], []
        for idx in _BBOX_INDICES:
            lm = landmarks[idx]
            if lm.visibility >= self.vis_thr:
                xs.append(lm.x)
                ys.append(lm.y)
        if len(xs) < 2:
            return 0.0
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        return bbox_w / (bbox_h + 1e-6)

    def _reset_prev(self):
        self._prev_shoulder_y = None
        self._prev_hip_y      = None
        self._prev_angle_cos  = None

    @staticmethod
    def get_pixel(landmarks, idx, h, w, vis_threshold=0.45):
        lm = landmarks[idx]
        if lm.visibility < vis_threshold:
            return None
        return int(lm.x * w), int(lm.y * h)