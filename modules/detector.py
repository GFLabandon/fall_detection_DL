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
# v3 改动（最小修改，最大效果）：
#   1. EMA平滑 LSTM 概率（alpha=0.5）：消除快速移动/摄像头抖动导致的单帧噪声触发
#   2. 连续帧确认（streak >= 2）：在 EMA 基础上额外要求连续 2 帧超阈值才报警
#   3. 几何站立保护门（ar<0.55 且 angle<12°）：屏蔽几何明确为站立时的 LSTM 触发
#   4. B-Static 联合门控（STATIC_LSTM_GATE=0.25）：保留 v2 方案
#   5. 倒置骨架已在 extractor.py v3 层面过滤（返回None→insufficient），detector无需额外处理

import math
import os
import sys
import time
import collections
from dataclasses import dataclass
from typing import Optional

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
    STATIC_LSTM_GATE,
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
    跌倒检测器 v3。
    核心改动：LSTM 概率经 EMA 平滑 + 连续帧确认，消除快速移动噪声误报。
    """

    # ── v3 平滑参数 ──────────────────────────────────────────
    _EMA_ALPHA    = 0.50   # EMA 平滑系数：值越大越响应快，越小越平滑
                           # 0.5 时：单帧 P=0.97 → EMA=0.485（不触发）
                           #         连续2帧 P=0.97 → EMA=0.728（触发）
    _STREAK_MIN   = 2      # EMA 超阈值后还需要连续帧数才真正报警
                           # 配合 EMA，等效于需要连续约 3 帧高概率

    # ── v3 几何保护参数 ──────────────────────────────────────
    # 仅当 ar 极小且 angle 极小（几何上明确是站立）时才激活保护
    # 不要调太宽，否则跌倒过渡帧会被错误屏蔽
    # ── 站立保护门（仅屏蔽几何上明确为竖直站立的情形）──────────
    # ar < 0.30（比人体正常宽高比更窄）且 angle < 5°（几乎完全竖直）
    # 蹲坐地上时 ar ≈ 0.35~0.50，超出保护门，不会被屏蔽
    # 相比 v3.1 的唯一改动：
    #   _STANDING_AR_MAX    : 0.30 → 0.45   （能屏蔽蹲坐背对误报）
    #   _STANDING_ANGLE_MAX : 5.0 → 10.0    （配套调整）
    # 改动逻辑：
    #   蹲坐背对时 ar≈0.37 angle≈6° → 0.37<0.45 AND 6<10 → 屏蔽 ✓
    #   正常站立时 ar≈0.28 angle≈4° → 0.28<0.45 AND 4<10  → 屏蔽 ✓
    #   侧躺跌倒时 ar≈1.60 angle≈75° → ar>0.45           → 不屏蔽 ✓
    #   俯身靠近时 ar≈0.80 angle≈5° → ar>0.45            → 不屏蔽（靠近本身就是问题，演示保持距离）
    _STANDING_AR_MAX    = 0.45   # 纵横比小于此值（人体细高）0.55 →  缩小站立保护门，避免蹲坐被误屏蔽
    _STANDING_ANGLE_MAX = 10.0   # 身体角度小于此值（几乎竖直）12° →    同上，只保护真正竖直站立的情形

    def __init__(self):
        self.extractor = FeatureExtractor(vis_threshold=VISIBILITY_THRESHOLD)
        self._buffer   = collections.deque(maxlen=SEQUENCE_LEN)
        self.device    = self._select_device()
        self.model        = None
        self.model_loaded = False
        self._try_load_model()

        # 几何状态
        self._dynamic_count:     int           = 0
        self._static_start_time: Optional[float] = None
        self._last_check_time:   float           = 0.0
        self._last_geo_result:   Optional[dict]  = None

        # v3: EMA 平滑状态
        self._lstm_prob_ema: float = 0.0
        self._high_ema_streak: int = 0   # EMA 连续超阈值帧数

    def update(self, landmarks, frame_h: int, frame_w: int) -> DetectionResult:
        feat = self.extractor.extract(landmarks)

        if feat is None:
            # extractor 返回 None：肩/髋不可见 或 骨架倒置（v3新增过滤）
            self._reset_all()
            return DetectionResult(
                status="insufficient",
                reason="关键点不可见或骨架异常，请调整位置",
                model_loaded=self.model_loaded,
            )

        self._buffer.append(feat)

        # ── v3: LSTM 推理 + EMA 平滑 ────────────────────────
        prob_raw  = self._lstm_infer_raw()
        lstm_prob = self._update_ema(prob_raw)

        # ── 几何检测（限频）───────────────────────────────────
        now = time.time()
        if now - self._last_check_time >= CHECK_INTERVAL:
            self._last_check_time = now
            self._last_geo_result = self._geometry_detect(feat, now)

        geo   = self._last_geo_result or {
            "status": "safe", "ar": 0.0, "angle": 0.0,
            "progress": 0.0, "channel": "", "reason": ""
        }
        ar    = geo["ar"]
        angle = geo["angle"]
        prog  = geo["progress"]

        # ── LSTM 主通道（带 EMA + 连续帧 + 几何保护）────────────
        standing_guard = (
            ar > 0.0
            and ar < self._STANDING_AR_MAX
            and angle < self._STANDING_ANGLE_MAX
        )

        if lstm_prob >= LSTM_FALL_THRESHOLD:
            self._high_ema_streak += 1
        else:
            self._high_ema_streak = 0

        if (lstm_prob >= LSTM_FALL_THRESHOLD
                and self._high_ema_streak >= self._STREAK_MIN
                and not standing_guard):
            return DetectionResult(
                status="fall_lstm",
                lstm_prob=lstm_prob,
                aspect_ratio=ar,
                body_angle=angle,
                reason=f"LSTM 时序分类 EMA_P={lstm_prob:.2f} 连续{self._high_ema_streak}帧",
                channel="LSTM",
                confirm_progress=1.0,
                model_loaded=self.model_loaded,
            )

        # 几何保护激活（站立姿态，LSTM被屏蔽）→ 安全
        if lstm_prob >= LSTM_FALL_THRESHOLD and standing_guard:
            return DetectionResult(
                status="safe",
                lstm_prob=lstm_prob,
                aspect_ratio=ar,
                body_angle=angle,
                reason=f"LSTM_P={lstm_prob:.2f} 但几何为站立(ar={ar:.2f},ang={angle:.0f}°)，屏蔽",
                model_loaded=self.model_loaded,
            )

        # ── A-Dynamic 通道 ───────────────────────────────────
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

        # ── B-Static 通道（需要 LSTM 联合确认）───────────────
        if geo["status"] == "fall_static":
            if lstm_prob >= STATIC_LSTM_GATE:
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
                # 几何触发但 LSTM 不认可（EMA后仍低）→ 降级为 warning
                return DetectionResult(
                    status="warning_static",
                    lstm_prob=lstm_prob,
                    aspect_ratio=ar,
                    body_angle=angle,
                    reason=f"几何异常但LSTM_EMA不认可(P={lstm_prob:.2f}<{STATIC_LSTM_GATE})，继续观察",
                    model_loaded=self.model_loaded,
                )

        # ── 静态预警 ─────────────────────────────────────────
        if geo["status"] == "warning_static":
            return DetectionResult(
                status="warning_static",
                lstm_prob=lstm_prob,
                aspect_ratio=ar,
                body_angle=angle,
                reason=geo.get("reason", "姿态异常，持续确认中"),
                confirm_progress=prog,
                model_loaded=self.model_loaded,
            )

        # ── 安全 ─────────────────────────────────────────────
        return DetectionResult(
            status="safe",
            lstm_prob=lstm_prob,
            aspect_ratio=ar,
            body_angle=angle,
            reason=f"ar={ar:.2f} ang={angle:.0f}° EMA_P={lstm_prob:.2f}",
            model_loaded=self.model_loaded,
        )

    def reset(self):
        self._reset_all()

    # ----------------------------------------------------------
    #  v3: EMA 更新
    # ----------------------------------------------------------

    def _update_ema(self, prob_raw: float) -> float:
        """
        指数移动平均平滑 LSTM 原始概率。
        alpha=0.5：单帧噪声（P_raw=0.97）→ EMA=0.485（不触发 threshold=0.65）
                   连续2帧（P_raw=0.97）→ EMA=0.728（触发）
        """
        self._lstm_prob_ema = (
            self._EMA_ALPHA * prob_raw
            + (1.0 - self._EMA_ALPHA) * self._lstm_prob_ema
        )
        return self._lstm_prob_ema

    # ----------------------------------------------------------
    #  LSTM 原始推理（不含 EMA，由 _update_ema 包装）
    # ----------------------------------------------------------

    def _lstm_infer_raw(self) -> float:
        if self.model is None:
            return 0.0
        buf_len  = len(self._buffer)
        if buf_len < SEQUENCE_LEN // 2:
            return 0.0
        buf_list = list(self._buffer)
        if buf_len < SEQUENCE_LEN:
            pad = [buf_list[0]] * (SEQUENCE_LEN - buf_len)
            seq = pad + buf_list
        else:
            seq = buf_list
        arr = np.stack(seq, axis=0) #不足30帧就前填充
        x   = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device) #转 tensor (1, 30, 12)
        with torch.no_grad():
            prob = self.model.predict_proba(x).item() #输出概率
        return float(prob)

    # ----------------------------------------------------------
    #  几何检测
    # ----------------------------------------------------------

    def _geometry_detect(self, feat: np.ndarray, now: float) -> dict:
        hip_y_rel  = float(feat[1])
        ar         = float(feat[3])
        cos_val    = float(np.clip(feat[4], -1.0, 1.0))
        body_angle = math.degrees(math.acos(cos_val))

        result = {"status": "safe", "ar": ar, "angle": body_angle,
                  "progress": 0.0, "reason": "", "channel": ""}

        # A-Dynamic
        delta_hip = float(feat[7])
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
                result.update({"status": "fall_static",
                                "reason": f"持续异常 {elapsed:.1f}s ar={ar:.2f} ang={body_angle:.0f}°",
                                "channel": "B-Static", "progress": 1.0})
            else:
                result.update({"status": "warning_static",
                                "reason": f"异常姿态 {elapsed:.1f}/{STATIC_FALL_DURATION:.0f}s",
                                "progress": progress})
        else:
            self._static_start_time = None

        return result

    # ----------------------------------------------------------
    #  工具
    # ----------------------------------------------------------

    def _reset_all(self):
        self._buffer.clear()
        self.extractor.reset()
        self._dynamic_count     = 0
        self._static_start_time = None
        self._last_geo_result   = None
        # v3: 重置 EMA（当 buffer 清空时 EMA 失效）
        self._lstm_prob_ema   = 0.0
        self._high_ema_streak = 0

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