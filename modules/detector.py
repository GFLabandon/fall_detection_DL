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
    CHECK_INTERVAL,
)
from data.extractor import FeatureExtractor


# ============================================================
#  DetectionResult 数据类
# ============================================================

@dataclass
class DetectionResult:
    """
    单次检测的结构化结果，被 renderer.py 和 main.py 直接读取。

    status 取值：
      'safe'           — 正常姿态
      'fall_lstm'      — LSTM 通道触发跌倒
      'fall_dynamic'   — A-Dynamic 通道触发跌倒
      'fall_static'    — B-Static 通道触发跌倒
      'warning_static' — 静态异常姿态，倒计时进行中
      'insufficient'   — 关键点不可见，视野不足
      'no_person'      — 画面中无人
    """
    status:           str   = "safe"
    lstm_prob:        float = 0.0    # LSTM 输出的跌倒概率 [0, 1]
    aspect_ratio:     float = 0.0    # 当前帧人体纵横比（调试展示）
    body_angle:       float = 0.0    # 身体倾斜角度（度数，调试展示）
    reason:           str   = ""     # 触发原因文字描述
    channel:          str   = ""     # 触发通道标识：'LSTM'|'A-Dynamic'|'B-Static'|''
    confirm_progress: float = 0.0    # 进度 [0, 1]，用于 UI 静态倒计时进度条
    model_loaded:     bool  = False  # LSTM 权重是否已成功加载


# ============================================================
#  FallDetector 类
# ============================================================

class FallDetector:
    """
    跌倒检测器，封装特征提取、LSTM 推理和几何规则三通道逻辑。

    典型使用方式（main.py 中）：
        detector = FallDetector()
        # 每帧：
        result = detector.update(pose_landmarks, frame_h, frame_w)
        # 检测到跌倒（result.status 以 'fall_' 开头）时触发报警
    """

    def __init__(self):
        # ---- 特征提取器（训练和推理共用同一份代码）----
        self.extractor = FeatureExtractor(vis_threshold=VISIBILITY_THRESHOLD)

        # ---- 特征序列缓冲区（滑动窗口）----
        # maxlen=SEQUENCE_LEN 自动丢弃最旧帧，保证始终是最近 30 帧
        self._buffer = collections.deque(maxlen=SEQUENCE_LEN)

        # ---- 设备选择 ----
        self.device = self._select_device()

        # ---- 加载 LSTM 模型 ----
        self.model        = None
        self.model_loaded = False
        self._try_load_model()

        # ---- 几何规则状态变量 ----
        self._prev_shoulder_y:   Optional[float] = None   # 上一帧肩高（Y值）
        self._dynamic_count:     int             = 0      # A 通道连续确认帧数
        self._static_start_time: Optional[float] = None   # B 通道计时起点
        self._last_check_time:   float           = 0.0    # 几何规则上次执行时间

        # 缓存上次的几何通道结果（限频期间直接复用）
        self._last_geo_result: Optional[dict] = None

    # ----------------------------------------------------------
    #  公开接口
    # ----------------------------------------------------------

    def update(self, landmarks, frame_h: int, frame_w: int) -> DetectionResult:
        """
        每帧调用的主检测函数。

        Args:
            landmarks:  MediaPipe Pose 的 landmark 列表（33个关键点）
            frame_h:    帧高度（像素），供骨架坐标换算
            frame_w:    帧宽度（像素）

        Returns:
            DetectionResult
        """
        # ---- 1. 特征提取 ----
        feat = self.extractor.extract(landmarks)

        if feat is None:
            # 关键点不可见（人不在画面 / 距离太近只露上半身）
            self._reset_geo_state()
            self._buffer.clear()
            return DetectionResult(
                status="insufficient",
                reason="关键点不可见，请保持全身入画",
                model_loaded=self.model_loaded,
            )

        # 将特征向量加入滑动窗口缓冲区
        self._buffer.append(feat)

        # ---- 2. LSTM 推理（每帧执行，保持概率实时更新）----
        lstm_prob = self._lstm_infer()

        # ---- 3. 几何规则检测（限频，约每 CHECK_INTERVAL 秒执行一次）----
        now = time.time()
        if now - self._last_check_time >= CHECK_INTERVAL:
            self._last_check_time  = now
            self._last_geo_result  = self._geometry_detect(feat, now)

        geo = self._last_geo_result or {"status": "safe", "ar": 0.0, "angle": 0.0,
                                         "progress": 0.0, "channel": ""}

        # ---- 4. 融合决策（优先级：LSTM > A-Dynamic > B-Static > Warning > Safe）----
        ar    = geo["ar"]
        angle = geo["angle"]
        prog  = geo["progress"]

        # LSTM 通道（主通道）
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

        # 动态跌倒（A 通道）
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

        # 静态跌倒（B 通道）
        if geo["status"] == "fall_static":
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

        # 静态预警（B 通道倒计时中）
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

        # 安全
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
        """
        重置检测器所有状态。
        在以下情况调用：画面中人消失、手动重置报警（按 R 键）。
        """
        self._buffer.clear()
        self.extractor.reset()
        self._reset_geo_state()

    # ----------------------------------------------------------
    #  LSTM 推理
    # ----------------------------------------------------------

    def _lstm_infer(self) -> float:
        """
        用当前缓冲区内容推理跌倒概率。

        缓冲区不足 SEQUENCE_LEN // 2 帧时返回 0（数据太少，不可信）。
        不足 SEQUENCE_LEN 帧时，用第一帧重复填充到序列开头。

        Returns:
            float: 跌倒概率 ∈ [0, 1]，0.0 表示未加载模型或数据不足
        """
        if self.model is None:
            return 0.0

        buf_len = len(self._buffer)
        if buf_len < SEQUENCE_LEN // 2:
            return 0.0   # 数据不足，概率不可信，返回 0

        # 构建长度恰好为 SEQUENCE_LEN 的序列
        buf_list = list(self._buffer)
        if buf_len < SEQUENCE_LEN:
            # 不足时用第一帧填充开头
            pad    = [buf_list[0]] * (SEQUENCE_LEN - buf_len)
            seq    = pad + buf_list
        else:
            seq = buf_list   # deque 已限制 maxlen，长度恰好是 SEQUENCE_LEN

        arr = np.stack(seq, axis=0)   # (SEQUENCE_LEN, FEATURE_DIM)
        x   = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        # (1, SEQUENCE_LEN, FEATURE_DIM)

        with torch.no_grad():
            prob = self.model.predict_proba(x).item()

        return float(prob)

    # ----------------------------------------------------------
    #  几何规则检测（A-Dynamic + B-Static）
    # ----------------------------------------------------------

    def _geometry_detect(self, feat: np.ndarray, now: float) -> dict:
        """
        从特征向量执行几何规则判断，维护动态/静态通道状态。

        Args:
            feat: 当前帧特征向量 (FEATURE_DIM,)
            now:  当前时间戳（time.time()）

        Returns:
            dict 包含 status/ar/angle/progress/reason/channel
        """
        # 从特征向量读取关键值
        shoulder_y   = float(feat[0])                                 # 肩高 Y（归一化，越大越靠下）
        ar           = float(feat[3])                                 # 纵横比
        cos_val      = float(np.clip(feat[4], -1.0, 1.0))            # body_angle_cos
        body_angle   = math.degrees(math.acos(cos_val))              # 转换为度数（0°=竖直，90°=水平）

        result = {"status": "safe", "ar": ar, "angle": body_angle,
                  "progress": 0.0, "reason": "", "channel": ""}

        # ──────────────────────────────────────────────────────────
        #  通道 A：动态跌倒（肩高骤降 + 纵横比超标，连续确认）
        # ──────────────────────────────────────────────────────────
        if self._prev_shoulder_y is not None:
            # 肩高下降比：当前 Y / 上一帧 Y（Y 增大 = 身体下落）
            drop_ratio = shoulder_y / (self._prev_shoulder_y + 1e-6)

            if drop_ratio > FALL_DROP_RATIO and ar > ASPECT_RATIO_THRESHOLD:
                # 本帧触发动态条件，计数+1
                self._dynamic_count += 1
            else:
                # 未触发，计数归零（防止非连续假触发累加）
                self._dynamic_count = 0
        else:
            self._dynamic_count = 0

        # 更新肩高历史
        self._prev_shoulder_y = shoulder_y

        # 连续确认帧数达到阈值 → 动态跌倒
        if self._dynamic_count >= DYNAMIC_CONFIRM_FRAMES:
            result["status"]  = "fall_dynamic"
            result["reason"]  = f"肩高骤降×{self._dynamic_count}帧 纵横比={ar:.2f}"
            result["channel"] = "A-Dynamic"
            # 触发动态跌倒后，清空静态计时（避免重复触发 B 通道）
            self._static_start_time = None
            return result

        # ──────────────────────────────────────────────────────────
        #  通道 B：静态跌倒（异常姿态持续超过 STATIC_FALL_DURATION 秒）
        # ──────────────────────────────────────────────────────────
        static_abnormal = (ar > ASPECT_RATIO_THRESHOLD or
                           body_angle > BODY_ANGLE_THRESHOLD)

        if static_abnormal:
            if self._static_start_time is None:
                # 首次检测到异常，记录开始时间
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
            # 姿态恢复正常，重置静态计时器
            self._static_start_time = None

        return result

    # ----------------------------------------------------------
    #  初始化工具
    # ----------------------------------------------------------

    def _select_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _try_load_model(self):
        """
        尝试从 MODEL_WEIGHTS 加载训练好的 LSTM 权重。
        失败时静默回退到规则模式，不影响主程序运行。
        """
        if not os.path.exists(MODEL_WEIGHTS):
            print(f"⚠️  未找到 LSTM 权重 [{MODEL_WEIGHTS}]，使用几何规则回退模式")
            print(f"   运行 python data/preprocess.py + python train.py 可训练模型")
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
            model.eval()   # 关闭 Dropout，进入推理模式

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
        """重置几何规则相关的所有状态变量。"""
        self._prev_shoulder_y   = None
        self._dynamic_count     = 0
        self._static_start_time = None
        self._last_geo_result   = None