# fall_detection_DL/modules/detector.py
# 跌倒检测主控模块
#
# 架构：三通道融合决策
#   通道 LSTM    : LSTMFallClassifier 实时推理（主通道，精度高）
#   通道 A-Dynamic: 肩高骤降 + 纵横比（几何动态，检测快速跌倒）
#   通道 B-Static : 异常姿态持续超标（几何静态，检测慢速倒地/躺地不动）
#
# 优先级：LSTM > A-Dynamic > B-Static > warning_static > safe
#
# 回退机制：若 weights/lstm_fall.pth 不存在，自动切换为纯几何规则模式，
#           保证 main.py 在未训练模型时也能正常运行。
# 跌倒检测主控模块 - v2: B-Static 加入 LSTM 联合门控

import math
import os
import sys
import time
import collections
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VISIBILITY_THRESHOLD,
    SEQUENCE_LEN,
    FEATURE_DIM,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    LSTM_FALL_THRESHOLD,
    MODEL_WEIGHTS,
    ASPECT_RATIO_THRESHOLD,
    BODY_ANGLE_THRESHOLD,
    FALL_DROP_RATIO,
    STATIC_FALL_DURATION,
    DYNAMIC_CONFIRM_FRAMES,
    STATIC_LSTM_GATE,   # v2: 新增 B-Static 联合门控阈值
    CHECK_INTERVAL,
)
from data.extractor import FeatureExtractor


@dataclass
class DetectionResult:
    """
    status 取值：
      'safe' / 'fall_lstm' / 'fall_dynamic' / 'fall_static'
      'warning_static' / 'insufficient' / 'no_person'
    """
    status:           str   = "safe"
    lstm_prob:        float = 0.0
    aspect_ratio:     float = 0.0
    body_angle:       float = 0.0
    reason:           str   = ""
    channel:          str   = ""
    confirm_progress: float = 0.0
    model_loaded:     bool  = False


class FallDetector:
    """
    跌倒检测器，封装特征提取、LSTM 推理和几何规则三通道逻辑。
    v2: B-Static 触发时增加 LSTM 概率联合确认门控，消除纯几何误报。
    """

    def __init__(self):
        self.extractor = FeatureExtractor(vis_threshold=VISIBILITY_THRESHOLD)
        self._buffer   = collections.deque(maxlen=SEQUENCE_LEN)
        self.device    = self._select_device()
        self.model        = None
        self.model_loaded = False
        self._try_load_model()

        self._prev_shoulder_y:   Optional[float] = None
        self._dynamic_count:     int             = 0
        self._static_start_time: Optional[float] = None
        self._last_check_time:   float           = 0.0
        self._last_geo_result:   Optional[dict]  = None

    def update(self, landmarks, frame_h: int, frame_w: int) -> DetectionResult:
        feat = self.extractor.extract(landmarks)

        if feat is None:
            self._reset_geo_state()
            self._buffer.clear()
            return DetectionResult(
                status="insufficient",
                reason="关键点不可见，请保持全身入画",
                model_loaded=self.model_loaded,
            )

        self._buffer.append(feat)
        lstm_prob = self._lstm_infer()

        now = time.time()
        if now - self._last_check_time >= CHECK_INTERVAL:
            self._last_check_time = now
            self._last_geo_result = self._geometry_detect(feat, now)

        geo = self._last_geo_result or {
            "status": "safe", "ar": 0.0, "angle": 0.0,
            "progress": 0.0, "channel": ""
        }

        ar    = geo["ar"]
        angle = geo["angle"]
        prog  = geo["progress"]

        # ── LSTM 主通道 ──────────────────────────────────────────
        if lstm_prob >= LSTM_FALL_THRESHOLD:
            return DetectionResult(
                status="fall_lstm",
                lstm_prob=lstm_prob,
                aspect_ratio=ar,
                body_angle=angle,
                reason=f"LSTM 时序分类 P={lstm_prob:.2f}",
                channel="LSTM",
                confirm_progress=1.0,
                model_loaded=self.model_loaded,
            )

        # ── A-Dynamic 通道 ───────────────────────────────────────
        if geo["status"] == "fall_dynamic":
            return DetectionResult(
                status="fall_dynamic",
                lstm_prob=lstm_prob,
                aspect_ratio=ar,
                body_angle=angle,
                reason=geo.get("reason", "肩高骤降+纵横比超标"),
                channel="A-Dynamic",
                confirm_progress=1.0,
                model_loaded=self.model_loaded,
            )

        # ── B-Static 通道（v2: 需要LSTM联合确认）────────────────
        if geo["status"] == "fall_static":
            if lstm_prob >= STATIC_LSTM_GATE:
                # LSTM 也认可 → 真正报警
                return DetectionResult(
                    status="fall_static",
                    lstm_prob=lstm_prob,
                    aspect_ratio=ar,
                    body_angle=angle,
                    reason=geo.get("reason", f"持续异常姿态 {STATIC_FALL_DURATION:.0f}s"),
                    channel="B-Static",
                    confirm_progress=1.0,
                    model_loaded=self.model_loaded,
                )
            else:
                # 几何触发但LSTM不认可 → 降级为warning，不报警不发邮件
                # （这修复了 lstm_prob=0.03 时的 B-Static 误报）
                return DetectionResult(
                    status="warning_static",
                    lstm_prob=lstm_prob,
                    aspect_ratio=ar,
                    body_angle=angle,
                    reason=f"几何异常但LSTM不认可(P={lstm_prob:.2f}<{STATIC_LSTM_GATE})，继续观察",
                    channel="",
                    confirm_progress=1.0,
                    model_loaded=self.model_loaded,
                )

        # ── 静态预警 ─────────────────────────────────────────────
        if geo["status"] == "warning_static":
            return DetectionResult(
                status="warning_static",
                lstm_prob=lstm_prob,
                aspect_ratio=ar,
                body_angle=angle,
                reason=geo.get("reason", "姿态异常，持续确认中"),
                channel="",
                confirm_progress=prog,
                model_loaded=self.model_loaded,
            )

        # ── 安全 ─────────────────────────────────────────────────
        return DetectionResult(
            status="safe",
            lstm_prob=lstm_prob,
            aspect_ratio=ar,
            body_angle=angle,
            reason=f"纵横比={ar:.2f}  角度={angle:.0f}°  LSTM_P={lstm_prob:.2f}",
            channel="",
            confirm_progress=0.0,
            model_loaded=self.model_loaded,
        )

    def reset(self):
        self._buffer.clear()
        self.extractor.reset()
        self._reset_geo_state()

    def _lstm_infer(self) -> float:
        if self.model is None:
            return 0.0
        buf_len = len(self._buffer)
        if buf_len < SEQUENCE_LEN // 2:
            return 0.0
        buf_list = list(self._buffer)
        if buf_len < SEQUENCE_LEN:
            pad  = [buf_list[0]] * (SEQUENCE_LEN - buf_len)
            seq  = pad + buf_list
        else:
            seq = buf_list
        arr = np.stack(seq, axis=0)
        x   = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.model.predict_proba(x).item()
        return float(prob)

    def _geometry_detect(self, feat: np.ndarray, now: float) -> dict:
        # feat[0] = shoulder_y_rel (≈0), feat[3] = aspect_ratio, feat[4] = body_angle_cos
        # v2: 使用相对坐标后 shoulder_y_rel ≈ 0，改用 hip_y_rel (feat[1]) 判断骤降
        hip_y_rel    = float(feat[1])
        ar           = float(feat[3])
        cos_val      = float(np.clip(feat[4], -1.0, 1.0))
        body_angle   = math.degrees(math.acos(cos_val))

        result = {"status": "safe", "ar": ar, "angle": body_angle,
                  "progress": 0.0, "reason": "", "channel": ""}

        # A-Dynamic: 使用 delta_hip (feat[7]) 判断骤降，更稳健
        delta_hip = float(feat[7])   # 负值=髋部在相对坐标下上升（跌倒初期）
        # 相对坐标下骤降: delta_hip < -0.15（髋相对肩快速上升）且纵横比超标
        if delta_hip < -0.15 and ar > ASPECT_RATIO_THRESHOLD:
            self._dynamic_count += 1
        else:
            self._dynamic_count = 0

        if self._dynamic_count >= DYNAMIC_CONFIRM_FRAMES:
            result["status"]  = "fall_dynamic"
            result["reason"]  = f"髋部骤降×{self._dynamic_count}帧 纵横比={ar:.2f}"
            result["channel"] = "A-Dynamic"
            self._static_start_time = None
            return result

        # B-Static
        static_abnormal = (ar > ASPECT_RATIO_THRESHOLD or
                           body_angle > BODY_ANGLE_THRESHOLD)

        if static_abnormal:
            if self._static_start_time is None:
                self._static_start_time = now
            elapsed  = now - self._static_start_time
            progress = min(elapsed / STATIC_FALL_DURATION, 1.0)

            if elapsed >= STATIC_FALL_DURATION:
                result["status"]   = "fall_static"
                result["reason"]   = (f"持续异常 {elapsed:.1f}s"
                                      f"  纵横比={ar:.2f}  角度={body_angle:.0f}°")
                result["channel"]  = "B-Static"
                result["progress"] = 1.0
            else:
                result["status"]   = "warning_static"
                result["reason"]   = (f"异常姿态 {elapsed:.1f}/{STATIC_FALL_DURATION:.0f}s"
                                      f"  纵横比={ar:.2f}  角度={body_angle:.0f}°")
                result["progress"] = progress
        else:
            self._static_start_time = None

        return result

    def _select_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _try_load_model(self):
        if not os.path.exists(MODEL_WEIGHTS):
            print(f"⚠️  未找到 LSTM 权重 [{MODEL_WEIGHTS}]，使用几何规则回退模式")
            return
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
            self.model        = model
            self.model_loaded = True
            n_params = model.count_parameters()
            epoch    = ckpt.get("epoch", "?")
            val_acc  = ckpt.get("val_acc", float("nan"))
            print(f"✅ LSTM 模型已加载：{MODEL_WEIGHTS}")
            print(f"   Epoch={epoch}  val_acc={val_acc*100:.1f}%"
                  f"  参数量={n_params:,}  设备={self.device}")
        except Exception as e:
            print(f"⚠️  LSTM 加载失败 ({e})，使用几何规则回退模式")
            self.model        = None
            self.model_loaded = False

    def _reset_geo_state(self):
        self._prev_shoulder_y   = None
        self._dynamic_count     = 0
        self._static_start_time = None
        self._last_geo_result   = None