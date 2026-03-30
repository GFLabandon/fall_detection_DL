# fall_detection_DL/data/extractor.py
# 单帧骨架特征提取器
#
# 【重要】本模块同时被两个阶段调用，必须保持逻辑完全一致：
#   - 训练阶段：data/preprocess.py 批量处理 URFD 视频帧
#   - 运行阶段：modules/detector.py 实时处理摄像头帧
# 任何特征计算逻辑的修改都必须同时影响两个阶段，防止训练-推理不一致。

import math
import numpy as np
from typing import Optional, Tuple


# ============================================================
#  MediaPipe Pose 关键点索引（33点模型）
# ============================================================
IDX_NOSE          = 0
IDX_LEFT_WRIST    = 15
IDX_RIGHT_WRIST   = 16
IDX_LEFT_SHOULDER = 11
IDX_RIGHT_SHOULDER= 12
IDX_LEFT_HIP      = 23
IDX_RIGHT_HIP     = 24
IDX_LEFT_KNEE     = 25
IDX_RIGHT_KNEE    = 26
IDX_LEFT_ANKLE    = 27
IDX_RIGHT_ANKLE   = 28

# 参与包围盒计算的所有关键点索引
_BBOX_INDICES = [
    IDX_LEFT_SHOULDER, IDX_RIGHT_SHOULDER,
    IDX_LEFT_HIP,      IDX_RIGHT_HIP,
    IDX_LEFT_KNEE,     IDX_RIGHT_KNEE,
    IDX_LEFT_ANKLE,    IDX_RIGHT_ANKLE,
]

# 特征向量各维度说明（FEATURE_DIM = 12）
FEATURE_NAMES = [
    "shoulder_y",       # 0  肩部中心归一化 Y 坐标
    "hip_y",            # 1  髋部中心归一化 Y 坐标
    "ankle_y",          # 2  踝部中心归一化 Y 坐标（不可见时估算）
    "aspect_ratio",     # 3  人体关键点包围盒 宽/高
    "body_angle_cos",   # 4  身体轴线与竖直方向夹角余弦（站立≈1，躺≈0）
    "hip_shoulder_gap", # 5  髋-肩 Y 距离绝对值（躺下时趋近 0）
    "delta_shoulder",   # 6  肩高帧间差分（快速下落时正值大）
    "delta_hip",        # 7  髋高帧间差分
    "delta_angle",      # 8  角度余弦帧间差分
    "wrist_y_mean",     # 9  双腕 Y 均值（倒地时手臂下落）
    "knee_y_mean",      # 10 双膝 Y 均值
    "head_hip_ratio",   # 11 鼻尖 Y / 髋部 Y（头部相对于髋的位置）
]

FEATURE_DIM = len(FEATURE_NAMES)  # 12


class FeatureExtractor:
    """
    从单帧 MediaPipe Pose landmarks 提取 12 维特征向量。

    设计原则：
      - 训练和推理必须调用同一份代码，保证特征分布一致
      - 维护帧间状态（prev_*），支持差分特征计算
      - 肩部或髋部不可见时返回 None，由调用方决定如何处理
    """

    def __init__(self, vis_threshold: float = 0.45):
        """
        Args:
            vis_threshold: 关键点可见性阈值，低于此值视为不可见
        """
        self.vis_thr = vis_threshold
        # 帧间差分所需的上一帧状态
        self._prev_shoulder_y: Optional[float] = None
        self._prev_hip_y:      Optional[float] = None
        self._prev_angle_cos:  Optional[float] = None

    # ----------------------------------------------------------
    #  公开接口
    # ----------------------------------------------------------

    def extract(self, landmarks) -> Optional[np.ndarray]:
        """
        从单帧 landmarks 提取 FEATURE_DIM=12 维特征向量。

        Args:
            landmarks: MediaPipe Pose 的 NormalizedLandmarkList（或 landmark 列表）

        Returns:
            np.ndarray shape (12,) float32，或 None（肩/髋不可见时）
        """
        # ---- 必须可见的关键点：肩部和髋部 ----
        ls  = self._get(landmarks, IDX_LEFT_SHOULDER)
        rs  = self._get(landmarks, IDX_RIGHT_SHOULDER)
        lhp = self._get(landmarks, IDX_LEFT_HIP)
        rhp = self._get(landmarks, IDX_RIGHT_HIP)

        # 肩部或髋部不可见 → 无法计算有效特征，返回 None
        if ls is None or rs is None or lhp is None or rhp is None:
            self._reset_prev()
            return None

        # ---- 基础坐标计算 ----
        shoulder_y = (ls[1] + rs[1]) / 2.0
        shoulder_x = (ls[0] + rs[0]) / 2.0
        hip_y      = (lhp[1] + rhp[1]) / 2.0
        hip_x      = (lhp[0] + rhp[0]) / 2.0

        # 踝部（可选，不可见时估算）
        la   = self._get(landmarks, IDX_LEFT_ANKLE)
        ra   = self._get(landmarks, IDX_RIGHT_ANKLE)
        if la is not None and ra is not None:
            ankle_y = (la[1] + ra[1]) / 2.0
        elif la is not None:
            ankle_y = la[1]
        elif ra is not None:
            ankle_y = ra[1]
        else:
            ankle_y = hip_y + 0.3  # 不可见时估算（屏幕坐标 Y 向下）

        # 腕部（可选）
        lw = self._get(landmarks, IDX_LEFT_WRIST)
        rw = self._get(landmarks, IDX_RIGHT_WRIST)
        if lw is not None and rw is not None:
            wrist_y = (lw[1] + rw[1]) / 2.0
        elif lw is not None:
            wrist_y = lw[1]
        elif rw is not None:
            wrist_y = rw[1]
        else:
            wrist_y = shoulder_y  # 不可见时用肩部代替

        # 膝部（可选）
        lk = self._get(landmarks, IDX_LEFT_KNEE)
        rk = self._get(landmarks, IDX_RIGHT_KNEE)
        if lk is not None and rk is not None:
            knee_y = (lk[1] + rk[1]) / 2.0
        elif lk is not None:
            knee_y = lk[1]
        elif rk is not None:
            knee_y = rk[1]
        else:
            knee_y = (hip_y + ankle_y) / 2.0  # 估算

        # 鼻子（可选）
        nose = self._get(landmarks, IDX_NOSE)
        nose_y = nose[1] if nose is not None else max(0.0, shoulder_y - 0.15)

        # ---- 特征 3：包围盒纵横比 ----
        aspect_ratio = self._calc_aspect_ratio(landmarks)

        # ---- 特征 4：身体轴线角度余弦 ----
        body_angle_cos = self._calc_body_angle_cos(
            shoulder_x, shoulder_y, hip_x, hip_y
        )

        # ---- 特征 5：髋-肩高度差 ----
        hip_shoulder_gap = abs(hip_y - shoulder_y)

        # ---- 特征 6-8：帧间差分 ----
        if self._prev_shoulder_y is not None:
            delta_shoulder = shoulder_y  - self._prev_shoulder_y
            delta_hip      = hip_y       - self._prev_hip_y
            delta_angle    = body_angle_cos - self._prev_angle_cos
        else:
            delta_shoulder = 0.0
            delta_hip      = 0.0
            delta_angle    = 0.0

        # 更新帧间状态
        self._prev_shoulder_y = shoulder_y
        self._prev_hip_y      = hip_y
        self._prev_angle_cos  = body_angle_cos

        # ---- 特征 11：头-髋相对位置 ----
        # 屏幕坐标 Y 向下，站立时 nose_y < hip_y，比值 < 1
        head_hip_ratio = nose_y / (hip_y + 1e-6)

        # ---- 组装特征向量 ----
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
        """重置帧间差分状态，切换视频序列或切换检测目标时调用。"""
        self._reset_prev()

    # ----------------------------------------------------------
    #  内部工具
    # ----------------------------------------------------------

    def _get(self, landmarks, idx: int) -> Optional[Tuple[float, float]]:
        """
        取关键点归一化坐标 (x, y)。
        可见性 < vis_threshold 时返回 None。
        """
        lm = landmarks[idx]
        if lm.visibility < self.vis_thr:
            return None
        return float(lm.x), float(lm.y)

    def _calc_aspect_ratio(self, landmarks) -> float:
        """
        计算所有可见关键点的包围盒宽高比。
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        aspect_ratio = bbox_w / (bbox_h + 1e-6)
        """
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

    @staticmethod
    def _calc_body_angle_cos(sx: float, sy: float,
                              hx: float, hy: float) -> float:
        """
        计算肩→髋向量与竖直方向（0, 1，Y轴正方向向下）的夹角余弦。
        站立时：向量 ≈ (0, +)，余弦 ≈ 1
        水平躺下时：向量 ≈ (±, 0)，余弦 ≈ 0
        """
        # 肩→髋向量
        vx = hx - sx
        vy = hy - sy  # 屏幕坐标 Y 向下为正

        length = math.hypot(vx, vy)
        if length < 1e-6:
            return 1.0  # 无法计算时默认竖直

        # 与竖直方向 (0, 1) 的点积
        cos_val = vy / length  # dot([vx,vy],[0,1]) / length
        return float(np.clip(cos_val, -1.0, 1.0))

    def _reset_prev(self):
        """清除帧间状态。"""
        self._prev_shoulder_y = None
        self._prev_hip_y      = None
        self._prev_angle_cos  = None

    # ----------------------------------------------------------
    #  工具方法：获取像素坐标（运行阶段骨架绘制使用）
    # ----------------------------------------------------------

    @staticmethod
    def get_pixel(landmarks, idx: int, h: int, w: int,
                  vis_threshold: float = 0.45) -> Optional[Tuple[int, int]]:
        """
        取关键点像素坐标 (px, py)。
        不可见时返回 None。
        被 modules/renderer.py 的骨架绘制逻辑调用。
        """
        lm = landmarks[idx]
        if lm.visibility < vis_threshold:
            return None
        return int(lm.x * w), int(lm.y * h)