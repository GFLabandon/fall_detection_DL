# ============================================================
#  modules/detector.py — 实时跌倒检测主控
#
#  优先使用 LSTM 深度学习模型（通道C）
#  模型未加载时自动回退到几何规则通道（A/B），保证演示不崩
# ============================================================

import math
import time
import os
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch

from config import (
    VISIBILITY_THRESHOLD, MODEL_WEIGHTS,
    FEATURE_DIM, SEQUENCE_LEN, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    LSTM_FALL_THRESHOLD,
    FALL_DROP_RATIO, ASPECT_RATIO_THRESHOLD, BODY_ANGLE_THRESHOLD,
    STATIC_FALL_DURATION, DYNAMIC_CONFIRM_FRAMES, CHECK_INTERVAL,
)
from data.extractor import FeatureExtractor

# 关键点索引
IDX_NOSE           = 0
IDX_LEFT_SHOULDER  = 11; IDX_RIGHT_SHOULDER = 12
IDX_LEFT_ELBOW     = 13; IDX_RIGHT_ELBOW    = 14
IDX_LEFT_WRIST     = 15; IDX_RIGHT_WRIST    = 16
IDX_LEFT_HIP       = 23; IDX_RIGHT_HIP      = 24
IDX_LEFT_KNEE      = 25; IDX_RIGHT_KNEE     = 26
IDX_LEFT_ANKLE     = 27; IDX_RIGHT_ANKLE    = 28

SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
    (IDX_NOSE, IDX_LEFT_SHOULDER), (IDX_NOSE, IDX_RIGHT_SHOULDER),
    (IDX_LEFT_SHOULDER, IDX_RIGHT_SHOULDER),
    (IDX_LEFT_SHOULDER, IDX_LEFT_ELBOW), (IDX_LEFT_ELBOW, IDX_LEFT_WRIST),
    (IDX_RIGHT_SHOULDER, IDX_RIGHT_ELBOW), (IDX_RIGHT_ELBOW, IDX_RIGHT_WRIST),
    (IDX_LEFT_SHOULDER, IDX_LEFT_HIP), (IDX_RIGHT_SHOULDER, IDX_RIGHT_HIP),
    (IDX_LEFT_HIP, IDX_RIGHT_HIP),
    (IDX_LEFT_HIP, IDX_LEFT_KNEE), (IDX_LEFT_KNEE, IDX_LEFT_ANKLE),
    (IDX_RIGHT_HIP, IDX_RIGHT_KNEE), (IDX_RIGHT_KNEE, IDX_RIGHT_ANKLE),
]


@dataclass
class DetectionResult:
    status:           str   = "safe"
    aspect_ratio:     float = 0.0
    body_angle:       float = 0.0
    lstm_prob:        float = 0.0    # LSTM 输出跌倒概率
    reason:           str   = ""
    confirm_progress: float = 0.0
    channel:          str   = ""
    model_loaded:     bool  = False  # LSTM 是否已加载


class FallDetector:
    def __init__(self):
        # ---- 特征提取器 ----
        self.extractor = FeatureExtractor(VISIBILITY_THRESHOLD)
        self._seq_buf  = deque(maxlen=SEQUENCE_LEN)

        # ---- 几何通道状态 ----
        self._prev_sy      = None
        self._dyn_count    = 0
        self._static_start = None
        self._last_check   = 0.0
        self._cached       = DetectionResult(status="no_person")

        # ---- 加载 LSTM 模型 ----
        self.device       = self._get_device()
        self.lstm_model   = self._load_model()
        self.model_loaded = self.lstm_model is not None

    # ----------------------------------------------------------
    #  设备 & 模型加载
    # ----------------------------------------------------------
    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self):
        if not os.path.exists(MODEL_WEIGHTS):
            print(f"⚠️  未找到 LSTM 权重 {MODEL_WEIGHTS}，将使用几何规则回退模式")
            print(f"   运行 python data/collect.py + python train.py 可训练模型")
            return None
        try:
            from models.lstm_classifier import LSTMFallClassifier
            ckpt  = torch.load(MODEL_WEIGHTS, map_location=self.device)
            cfg   = ckpt.get("config", {})
            model = LSTMFallClassifier(
                input_dim  = cfg.get("input_dim",  FEATURE_DIM),
                hidden_dim = cfg.get("hidden_dim", LSTM_HIDDEN),
                num_layers = cfg.get("num_layers", LSTM_LAYERS),
                dropout    = cfg.get("dropout",    LSTM_DROPOUT),
            ).to(self.device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            print(f"✅ LSTM 模型已加载：{MODEL_WEIGHTS}  设备:{self.device}")
            return model
        except Exception as e:
            print(f"⚠️  LSTM 加载失败 ({e})，使用规则回退")
            return None

    # ----------------------------------------------------------
    #  公开接口
    # ----------------------------------------------------------
    def update(self, landmarks, h: int, w: int) -> DetectionResult:
        now = time.time()

        # 特征提取（每帧）
        feat = self.extractor.extract(landmarks)
        if feat is not None:
            self._seq_buf.append(feat)

        # 限频
        if now - self._last_check < CHECK_INTERVAL:
            r = self._cached
            r.model_loaded = self.model_loaded
            # 实时更新 LSTM 概率（每帧）
            r.lstm_prob = self._lstm_infer()
            if r.lstm_prob >= LSTM_FALL_THRESHOLD and r.status not in ("fall_dynamic", "fall_static", "fall_lstm"):
                r.status  = "fall_lstm"
                r.channel = "C-LSTM"
                r.reason  = f"LSTM P={r.lstm_prob:.2f}"
            return r

        self._last_check = now
        result = self._analyze_geometry(landmarks, h, w)
        result.model_loaded = self.model_loaded

        # LSTM 推理
        lstm_prob = self._lstm_infer()
        result.lstm_prob = lstm_prob
        if lstm_prob >= LSTM_FALL_THRESHOLD and result.status not in ("fall_dynamic", "fall_static"):
            result.status  = "fall_lstm"
            result.channel = "C-LSTM"
            result.reason  = f"LSTM 时序分类 P={lstm_prob:.2f}"

        self._cached = result
        return result

    def reset(self):
        self._seq_buf.clear()
        self.extractor.reset()
        self._prev_sy      = None
        self._dyn_count    = 0
        self._static_start = None

    # ----------------------------------------------------------
    #  LSTM 推理
    # ----------------------------------------------------------
    def _lstm_infer(self) -> float:
        if self.lstm_model is None or len(self._seq_buf) < SEQUENCE_LEN // 2:
            return 0.0
        seq  = list(self._seq_buf)
        # 不足 SEQUENCE_LEN 帧时，用第一帧填充开头
        while len(seq) < SEQUENCE_LEN:
            seq.insert(0, seq[0])
        arr  = np.stack(seq[-SEQUENCE_LEN:])  # (30, 12)
        x    = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = torch.sigmoid(self.lstm_model(x)).item()
        return float(prob)

    # ----------------------------------------------------------
    #  几何规则通道（A动态 / B静态）
    # ----------------------------------------------------------
    def _analyze_geometry(self, landmarks, h: int, w: int) -> DetectionResult:
        ls  = self._gn(landmarks, IDX_LEFT_SHOULDER)
        rs  = self._gn(landmarks, IDX_RIGHT_SHOULDER)
        lhp = self._gn(landmarks, IDX_LEFT_HIP)
        rhp = self._gn(landmarks, IDX_RIGHT_HIP)

        if ls is None or rs is None:
            self._reset_geo()
            return DetectionResult(status="insufficient", reason="肩部不可见")
        if lhp is None or rhp is None:
            self._reset_geo()
            return DetectionResult(status="insufficient", reason="髋部不可见，请后退")

        sy  = (ls[1] + rs[1]) / 2.0
        ar  = self._aspect_ratio(ls, rs, lhp, rhp, landmarks)
        ang = self._body_angle(ls, rs, lhp, rhp)
        dr  = (sy / self._prev_sy) if (self._prev_sy and self._prev_sy > 1e-4) else 1.0
        self._prev_sy = sy

        # 通道A：动态
        dyn_ok = (dr > FALL_DROP_RATIO and ar > ASPECT_RATIO_THRESHOLD)
        self._dyn_count = (self._dyn_count + 1) if dyn_ok else max(0, self._dyn_count - 1)
        conf = min(self._dyn_count / DYNAMIC_CONFIRM_FRAMES, 1.0)

        if self._dyn_count >= DYNAMIC_CONFIRM_FRAMES:
            self._static_start = None
            return DetectionResult(
                status="fall_dynamic", aspect_ratio=ar, body_angle=ang,
                reason=f"肩降:{dr:.2f} 纵横比:{ar:.2f}",
                confirm_progress=1.0, channel="A-Dynamic")

        # 通道B：静态
        now = time.time()
        static_ok = (ar > ASPECT_RATIO_THRESHOLD or ang > BODY_ANGLE_THRESHOLD)
        if static_ok:
            if self._static_start is None:
                self._static_start = now
            elapsed = now - self._static_start
            if elapsed >= STATIC_FALL_DURATION:
                return DetectionResult(
                    status="fall_static", aspect_ratio=ar, body_angle=ang,
                    reason=f"持续{elapsed:.1f}s 纵横比:{ar:.2f}",
                    confirm_progress=1.0, channel="B-Static")
            return DetectionResult(
                status="warning_static", aspect_ratio=ar, body_angle=ang,
                reason=f"姿态异常 {elapsed:.1f}/{STATIC_FALL_DURATION:.0f}s",
                confirm_progress=elapsed / STATIC_FALL_DURATION)
        else:
            self._static_start = None

        return DetectionResult(
            status="safe", aspect_ratio=ar, body_angle=ang,
            reason=f"纵横比:{ar:.2f} 角度:{ang:.0f}°",
            confirm_progress=conf)

    # ---- 工具 ----
    def _aspect_ratio(self, ls, rs, lhp, rhp, landmarks) -> float:
        pts = [ls, rs, lhp, rhp]
        for idx in (IDX_LEFT_KNEE, IDX_RIGHT_KNEE, IDX_LEFT_ANKLE, IDX_RIGHT_ANKLE):
            p = self._gn(landmarks, idx)
            if p: pts.append(p)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        return bw / bh if bh > 1e-4 else 0.0

    @staticmethod
    def _body_angle(ls, rs, lhp, rhp) -> float:
        dx = ((lhp[0]+rhp[0]) - (ls[0]+rs[0])) / 2
        dy = ((lhp[1]+rhp[1]) - (ls[1]+rs[1])) / 2
        ln = math.hypot(dx, dy)
        return math.degrees(math.acos(min(1.0, abs(dy)/ln))) if ln > 1e-4 else 0.0

    def _gn(self, landmarks, idx) -> Optional[Tuple[float, float]]:
        lm = landmarks[idx]
        return (lm.x, lm.y) if lm.visibility >= VISIBILITY_THRESHOLD else None

    def _reset_geo(self):
        self._dyn_count = 0; self._static_start = None

    @staticmethod
    def get_pixel(landmarks, idx, h, w) -> Optional[Tuple[int, int]]:
        lm = landmarks[idx]
        if lm.visibility < VISIBILITY_THRESHOLD: return None
        return int(lm.x * w), int(lm.y * h)