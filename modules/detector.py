# ============================================================
#  modules/detector.py — 跌倒检测主控（三通道融合）
#
#  通道A（动态）: 肩高骤降 + 纵横比 → 快速跌倒
#  通道B（静态）: 纵横比/角度持续超标 N 秒 → 慢速/躺地
#  通道C（时序）: 时序特征评分 > 阈值 → 深度学习置信输出
#
#  最终判定 = 三通道 OR 融合（任一通道确认即报警）
# ============================================================

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from config import (
    VISIBILITY_THRESHOLD,
    FALL_DROP_RATIO,
    ASPECT_RATIO_THRESHOLD,
    BODY_ANGLE_THRESHOLD,
    STATIC_FALL_DURATION,
    DYNAMIC_CONFIRM_FRAMES,
    CHECK_INTERVAL,
    TEMPORAL_FALL_SCORE,
)
from modules.temporal_classifier import TemporalClassifier


# ---- MediaPipe Pose 关键点索引 ----
IDX_NOSE           = 0
IDX_LEFT_SHOULDER  = 11
IDX_RIGHT_SHOULDER = 12
IDX_LEFT_ELBOW     = 13
IDX_RIGHT_ELBOW    = 14
IDX_LEFT_WRIST     = 15
IDX_RIGHT_WRIST    = 16
IDX_LEFT_HIP       = 23
IDX_RIGHT_HIP      = 24
IDX_LEFT_KNEE      = 25
IDX_RIGHT_KNEE     = 26
IDX_LEFT_ANKLE     = 27
IDX_RIGHT_ANKLE    = 28

SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
    (IDX_NOSE, IDX_LEFT_SHOULDER),
    (IDX_NOSE, IDX_RIGHT_SHOULDER),
    (IDX_LEFT_SHOULDER,  IDX_RIGHT_SHOULDER),
    (IDX_LEFT_SHOULDER,  IDX_LEFT_ELBOW),
    (IDX_LEFT_ELBOW,     IDX_LEFT_WRIST),
    (IDX_RIGHT_SHOULDER, IDX_RIGHT_ELBOW),
    (IDX_RIGHT_ELBOW,    IDX_RIGHT_WRIST),
    (IDX_LEFT_SHOULDER,  IDX_LEFT_HIP),
    (IDX_RIGHT_SHOULDER, IDX_RIGHT_HIP),
    (IDX_LEFT_HIP,       IDX_RIGHT_HIP),
    (IDX_LEFT_HIP,       IDX_LEFT_KNEE),
    (IDX_LEFT_KNEE,      IDX_LEFT_ANKLE),
    (IDX_RIGHT_HIP,      IDX_RIGHT_KNEE),
    (IDX_RIGHT_KNEE,     IDX_RIGHT_ANKLE),
]


@dataclass
class DetectionResult:
    status:           str   = "safe"   # safe|fall_dynamic|fall_static|fall_temporal|warning_static|insufficient|no_person
    aspect_ratio:     float = 0.0
    body_angle:       float = 0.0
    drop_ratio:       float = 0.0
    temporal_score:   float = 0.0      # 时序分类器输出的跌倒概率
    reason:           str   = ""
    confirm_progress: float = 0.0
    channel:          str   = ""       # 触发通道标识，用于答辩演示


class FallDetector:

    def __init__(self):
        self._prev_shoulder_y:   Optional[float] = None
        self._dynamic_count:     int             = 0
        self._static_start:      Optional[float] = None
        self._last_check_time:   float           = 0.0
        self._cached_result:     DetectionResult = DetectionResult(status="no_person")

        # 通道C：时序分类器
        self.temporal = TemporalClassifier(vis_threshold=VISIBILITY_THRESHOLD)

    # ----------------------------------------------------------
    #  公开接口
    # ----------------------------------------------------------
    def update(self, landmarks, h: int, w: int) -> DetectionResult:
        now = time.time()

        # 时序分类器每帧都更新（保持滑动窗口连续）
        t_score, t_ok = self.temporal.update(landmarks)

        # 限频：几何检测每 CHECK_INTERVAL 秒一次
        if now - self._last_check_time < CHECK_INTERVAL:
            # 用上次几何结果合并最新时序分
            r = self._cached_result
            r.temporal_score = t_score
            # 时序分数独立触发
            if t_score >= TEMPORAL_FALL_SCORE and r.status not in ("fall_dynamic", "fall_static"):
                r.status  = "fall_temporal"
                r.channel = "C-Temporal"
                r.reason  = f"时序评分: {t_score:.2f}"
            return r

        self._last_check_time = now
        result = self._analyze(landmarks, h, w)
        result.temporal_score = t_score

        # 三通道 OR 融合：时序通道最高优先级
        if t_score >= TEMPORAL_FALL_SCORE and result.status not in ("fall_dynamic", "fall_static"):
            result.status  = "fall_temporal"
            result.channel = "C-Temporal"
            result.reason  = f"时序分类评分:{t_score:.2f}"

        self._cached_result = result
        return result

    def reset(self):
        self._prev_shoulder_y = None
        self._dynamic_count   = 0
        self._static_start    = None
        self.temporal.reset()

    # ----------------------------------------------------------
    #  几何通道分析
    # ----------------------------------------------------------
    def _analyze(self, landmarks, h: int, w: int) -> DetectionResult:
        ls  = self._get_norm(landmarks, IDX_LEFT_SHOULDER)
        rs  = self._get_norm(landmarks, IDX_RIGHT_SHOULDER)
        lhp = self._get_norm(landmarks, IDX_LEFT_HIP)
        rhp = self._get_norm(landmarks, IDX_RIGHT_HIP)

        if ls is None or rs is None:
            self._reset_counters()
            return DetectionResult(status="insufficient", reason="肩部不可见")

        if lhp is None or rhp is None:
            self._reset_counters()
            return DetectionResult(status="insufficient", reason="髋部不可见，请后退使全身入画")

        shoulder_y   = (ls[1] + rs[1]) / 2.0
        aspect_ratio = self._calc_aspect_ratio(ls, rs, lhp, rhp, landmarks)
        body_angle   = self._calc_body_angle(ls, rs, lhp, rhp)
        drop_ratio   = self._calc_drop_ratio(shoulder_y)
        self._prev_shoulder_y = shoulder_y

        # ---- 通道A：动态 ----
        dynamic_ok = (drop_ratio > FALL_DROP_RATIO and aspect_ratio > ASPECT_RATIO_THRESHOLD)
        if dynamic_ok:
            self._dynamic_count += 1
        else:
            self._dynamic_count = max(0, self._dynamic_count - 1)

        confirm = min(self._dynamic_count / DYNAMIC_CONFIRM_FRAMES, 1.0)

        if self._dynamic_count >= DYNAMIC_CONFIRM_FRAMES:
            self._static_start = None
            return DetectionResult(
                status="fall_dynamic", aspect_ratio=aspect_ratio,
                body_angle=body_angle, drop_ratio=drop_ratio,
                reason=f"肩降:{drop_ratio:.2f} 纵横比:{aspect_ratio:.2f}",
                confirm_progress=1.0, channel="A-Dynamic"
            )

        # ---- 通道B：静态 ----
        static_abnormal = (aspect_ratio > ASPECT_RATIO_THRESHOLD or
                           body_angle > BODY_ANGLE_THRESHOLD)
        now = time.time()
        if static_abnormal:
            if self._static_start is None:
                self._static_start = now
            elapsed = now - self._static_start
            if elapsed >= STATIC_FALL_DURATION:
                return DetectionResult(
                    status="fall_static", aspect_ratio=aspect_ratio,
                    body_angle=body_angle, drop_ratio=drop_ratio,
                    reason=f"持续{elapsed:.1f}s 纵横比:{aspect_ratio:.2f} 角度:{body_angle:.0f}°",
                    confirm_progress=1.0, channel="B-Static"
                )
            return DetectionResult(
                status="warning_static", aspect_ratio=aspect_ratio,
                body_angle=body_angle, drop_ratio=drop_ratio,
                reason=f"姿态异常 {elapsed:.1f}/{STATIC_FALL_DURATION:.0f}s",
                confirm_progress=elapsed / STATIC_FALL_DURATION,
            )
        else:
            self._static_start = None

        self._dynamic_count = max(0, self._dynamic_count - 1)
        return DetectionResult(
            status="safe", aspect_ratio=aspect_ratio,
            body_angle=body_angle, drop_ratio=drop_ratio,
            reason=f"纵横比:{aspect_ratio:.2f} 角度:{body_angle:.0f}°",
            confirm_progress=confirm,
        )

    # ----------------------------------------------------------
    #  特征计算
    # ----------------------------------------------------------
    def _calc_aspect_ratio(self, ls, rs, lhp, rhp, landmarks) -> float:
        pts = [ls, rs, lhp, rhp]
        for idx in (IDX_LEFT_KNEE, IDX_RIGHT_KNEE, IDX_LEFT_ANKLE, IDX_RIGHT_ANKLE):
            p = self._get_norm(landmarks, idx)
            if p:
                pts.append(p)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        return bw / bh if bh > 1e-4 else 0.0

    @staticmethod
    def _calc_body_angle(ls, rs, lhp, rhp) -> float:
        sx = (ls[0] + rs[0]) / 2.0
        sy = (ls[1] + rs[1]) / 2.0
        hx = (lhp[0] + rhp[0]) / 2.0
        hy = (lhp[1] + rhp[1]) / 2.0
        dx, dy = hx - sx, hy - sy
        ln = math.hypot(dx, dy)
        if ln < 1e-4:
            return 0.0
        return math.degrees(math.acos(min(1.0, abs(dy) / ln)))

    def _calc_drop_ratio(self, shoulder_y: float) -> float:
        if self._prev_shoulder_y is None or self._prev_shoulder_y < 1e-4:
            return 1.0
        return shoulder_y / self._prev_shoulder_y

    def _reset_counters(self):
        self._dynamic_count = 0
        self._static_start  = None

    @staticmethod
    def _get_norm(landmarks, idx) -> Optional[Tuple[float, float]]:
        lm = landmarks[idx]
        return (lm.x, lm.y) if lm.visibility >= VISIBILITY_THRESHOLD else None

    @staticmethod
    def get_pixel(landmarks, idx, h: int, w: int) -> Optional[Tuple[int, int]]:
        lm = landmarks[idx]
        if lm.visibility < VISIBILITY_THRESHOLD:
            return None
        return int(lm.x * w), int(lm.y * h)