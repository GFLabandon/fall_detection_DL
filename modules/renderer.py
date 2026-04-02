# fall_detection_DL/modules/renderer.py
# UI 渲染模块
#
# 负责将检测结果绘制到摄像头帧上：
#   - 骨架连线 + 关节点（OpenCV）
#   - 中文状态文字（Pillow，避免 OpenCV 不支持中文）
#   - 数值调试信息（OpenCV putText，英文/数字）
#   - LSTM 概率进度条、静态通道倒计时进度条（OpenCV）
#   - 报警时屏幕红色边框

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw

from modules.font_utils import get_font
from modules.detector   import DetectionResult
from config import (
    FONT_SIZE_LARGE,
    FONT_SIZE_NORMAL,
    FONT_SIZE_SMALL,
    LSTM_FALL_THRESHOLD,
    STATIC_FALL_DURATION,
)


# ============================================================
#  颜色常量（BGR 格式）
# ============================================================
COLOR_SAFE       = (0, 200, 0)      # 绿色，安全状态
COLOR_FALL       = (0, 0, 220)      # 红色，跌倒状态
COLOR_WARN       = (0, 165, 255)    # 橙色，警告状态
COLOR_INSUF      = (0, 200, 220)    # 黄色，视野不足
COLOR_WHITE      = (255, 255, 255)
COLOR_BLACK      = (0, 0, 0)
COLOR_BLUE       = (220, 100, 0)    # 蓝色（BGR）
COLOR_GRAY       = (120, 120, 120)

# Pillow 使用 RGB，需要转换
def _bgr2rgb(bgr):
    return (bgr[2], bgr[1], bgr[0])


# ============================================================
#  UIRenderer 类
# ============================================================

class UIRenderer:
    """
    跌倒检测系统 UI 渲染器。

    draw() 接收原始帧和检测结果，返回渲染后的帧（不修改原始帧）。
    """

    def __init__(self):
        # 预加载三种字号的字体（lru_cache 保证只加载一次）
        self._font_large  = get_font(FONT_SIZE_LARGE)
        self._font_normal = get_font(FONT_SIZE_NORMAL)
        self._font_small  = get_font(FONT_SIZE_SMALL)

        # MediaPipe 连接（用于骨架绘制）
        self._pose_connections = mp.solutions.pose.POSE_CONNECTIONS

    # ----------------------------------------------------------
    #  主入口
    # ----------------------------------------------------------

    def draw(self,
             frame:          np.ndarray,
             result:         DetectionResult,
             fps:            float,
             pose_landmarks  = None,
             today_count:    int = 0,
             session_count:  int = 0) -> np.ndarray:
        """
        渲染整帧画面。

        Args:
            frame:          原始 BGR 帧（不会被修改）
            result:         FallDetector.update() 返回的检测结果
            fps:            当前帧率
            pose_landmarks: MediaPipe Pose landmarks（可为 None）
            today_count:    今日跌倒次数（由 main.py 传入）
            session_count:  本次运行跌倒次数

        Returns:
            渲染后的 BGR 帧（新数组）
        """
        # 复制一份，不修改原始帧
        canvas = frame.copy()
        h, w   = canvas.shape[:2]

        is_fall = result.status.startswith("fall_")

        # 1. 骨架绘制
        if pose_landmarks is not None:
            self._draw_skeleton(canvas, pose_landmarks, h, w, is_fall)

        # 2. 报警红框
        if is_fall:
            cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), COLOR_FALL, 14)

        # 3. 半透明调试信息区（中部）
        self._draw_debug_overlay(canvas, result, fps, h, w)

        # 4. 中文状态文字（左上区，需要 Pillow）
        canvas = self._draw_chinese_status(canvas, result, h, w)

        # 5. 右上区统计信息（OpenCV，数字/英文）
        self._draw_top_right(canvas, today_count, session_count, w)

        # 6. 底部 LSTM 概率进度条
        self._draw_lstm_bar(canvas, result.lstm_prob, h, w)

        # 7. 底部静态通道倒计时条（仅 warning_static 时显示）
        if result.status == "warning_static" and result.confirm_progress > 0:
            self._draw_static_bar(canvas, result.confirm_progress, h, w)

        # 8. 右下角操作提示
        cv2.putText(canvas, "ESC:quit  R:reset  S:stats",
                    (w - 250, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GRAY, 1, cv2.LINE_AA)

        return canvas

    # ----------------------------------------------------------
    #  骨架绘制
    # ----------------------------------------------------------

    def _draw_skeleton(self, canvas, landmarks, h: int, w: int, is_fall: bool):
        """绘制 MediaPipe Pose 骨架连线和关节点。"""
        line_color  = COLOR_FALL  if is_fall else COLOR_SAFE
        joint_color = (0, 80, 220) if is_fall else (0, 191, 255)

        # 获取所有关键点像素坐标
        pts = {}
        for idx, lm in enumerate(landmarks):
            if lm.visibility >= 0.45:
                pts[idx] = (int(lm.x * w), int(lm.y * h))

        # 绘制连线
        for conn in self._pose_connections:
            i, j = conn
            if i in pts and j in pts:
                cv2.line(canvas, pts[i], pts[j], line_color, 2, cv2.LINE_AA)

        # 绘制关节点（小圆形）
        for pt in pts.values():
            cv2.circle(canvas, pt, 4, joint_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 4, COLOR_WHITE,  1,  cv2.LINE_AA)

    # ----------------------------------------------------------
    #  中文状态文字（Pillow 渲染）
    # ----------------------------------------------------------

    def _draw_chinese_status(self, canvas: np.ndarray,
                              result: DetectionResult,
                              h: int, w: int) -> np.ndarray:
        """
        用 Pillow 将中文状态文字渲染到画面左上区。
        返回合并后的 BGR numpy 数组。
        """
        try:
            # 转换为 PIL Image（RGB）
            img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw    = ImageDraw.Draw(img_pil)

            # ── 主状态行 ──────────────────────────────────────────
            st = result.status

            if st.startswith("fall_"):
                ch       = f"[{result.channel}]" if result.channel else ""
                text     = f"⚠ 警报：检测到跌倒！{ch}"
                color    = _bgr2rgb(COLOR_FALL)
                font     = self._font_large
            elif st == "warning_static":
                text     = "⏱ 姿态异常，持续确认中..."
                color    = _bgr2rgb(COLOR_WARN)
                font     = self._font_normal
            elif st == "insufficient":
                text     = "📷 视野不足，请调整摄像头"
                color    = _bgr2rgb(COLOR_INSUF)
                font     = self._font_normal
            elif st == "no_person":
                text     = "🔍 未检测到人物"
                color    = _bgr2rgb(COLOR_GRAY)
                font     = self._font_normal
            else:
                text     = "✅ 状态：安全巡检中"
                color    = _bgr2rgb(COLOR_SAFE)
                font     = self._font_large

            # 绘制半透明背景矩形（增加可读性）
            if font is not None:
                try:
                    bbox = draw.textbbox((16, 12), text, font=font)
                    pad  = 6
                    bg_layer = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
                    bg_draw  = ImageDraw.Draw(bg_layer)
                    bg_draw.rectangle(
                        [bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
                        fill=(0, 0, 0, 160)
                    )
                    img_pil = Image.alpha_composite(img_pil.convert("RGBA"), bg_layer).convert("RGB")
                    draw    = ImageDraw.Draw(img_pil)
                except Exception:
                    pass   # 背景绘制失败不影响文字

                draw.text((16, 12), text, font=font, fill=color)

            # ── 模型状态行 ────────────────────────────────────────
            y_model = 12 + FONT_SIZE_LARGE + 8
            if result.model_loaded:
                model_text  = "🧠 LSTM 模型已加载"
                model_color = _bgr2rgb(COLOR_BLUE)
            else:
                model_text  = "📐 规则回退模式（未加载权重）"
                model_color = _bgr2rgb(COLOR_WARN)

            if self._font_small is not None:
                draw.text((16, y_model), model_text,
                           font=self._font_small, fill=model_color)

            # 转回 BGR numpy
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        except Exception as e:
            # Pillow 渲染失败，回退到 OpenCV putText（无中文，但不崩溃）
            label = "FALL" if result.status.startswith("fall_") else "SAFE"
            cv2.putText(canvas, label, (16, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        COLOR_FALL if label == "FALL" else COLOR_SAFE,
                        2, cv2.LINE_AA)
            return canvas

    # ----------------------------------------------------------
    #  调试信息（半透明背景 + 白色文字）
    # ----------------------------------------------------------

    def _draw_debug_overlay(self, canvas: np.ndarray,
                             result: DetectionResult,
                             fps: float, h: int, w: int):
        """
        在画面中部绘制半透明调试信息区。
        使用 cv2.addWeighted 实现半透明背景。
        """
        y_base = 12 + FONT_SIZE_LARGE + FONT_SIZE_SMALL + 28  # 状态行下方

        lines = [
            (f"LSTM P={result.lstm_prob:.2f}/{LSTM_FALL_THRESHOLD:.2f}   "
             f"aspect={result.aspect_ratio:.2f}   angle={result.body_angle:.0f}deg"),
            f"FPS:{fps:.0f}  MediaPipe BlazePose + LSTM  Edge-AI M2",
        ]

        line_h     = 26
        box_h      = len(lines) * line_h + 12
        box_w      = w - 32

        # 半透明黑色背景
        overlay = canvas.copy()
        cv2.rectangle(overlay, (16, y_base - 4), (16 + box_w, y_base + box_h),
                       COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

        # 文字
        for i, line in enumerate(lines):
            y = y_base + 10 + i * line_h
            cv2.putText(canvas, line, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        COLOR_WHITE, 1, cv2.LINE_AA)

    # ----------------------------------------------------------
    #  右上区统计信息
    # ----------------------------------------------------------

    def _draw_top_right(self, canvas: np.ndarray,
                         today_count: int, session_count: int, w: int):
        """在右上角绘制今日/本次统计（OpenCV，无需中文）。"""
        lines = [
            f"Today falls: {today_count}",
            f"Session: {session_count}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (w - 230, 30 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        COLOR_WHITE, 1, cv2.LINE_AA)

    # ----------------------------------------------------------
    #  LSTM 概率进度条（底部左侧）
    # ----------------------------------------------------------

    def _draw_lstm_bar(self, canvas: np.ndarray,
                        lstm_prob: float, h: int, w: int):
        """
        在底部左侧绘制 LSTM 跌倒概率进度条。
        概率 < 阈值时蓝色，>= 阈值时红色。
        """
        bar_x, bar_y = 16, h - 30
        bar_w_max    = 300
        bar_h        = 18

        # 外框
        cv2.rectangle(canvas,
                       (bar_x, bar_y),
                       (bar_x + bar_w_max, bar_y + bar_h),
                       COLOR_GRAY, 1)

        # 填充（按概率比例）
        fill_w = int(bar_w_max * min(lstm_prob, 1.0))
        if fill_w > 0:
            bar_color = COLOR_FALL if lstm_prob >= LSTM_FALL_THRESHOLD else COLOR_BLUE
            cv2.rectangle(canvas,
                           (bar_x, bar_y),
                           (bar_x + fill_w, bar_y + bar_h),
                           bar_color, -1)

        # 标签文字（英文数字，OpenCV 可直接绘制）
        label = f"LSTM {lstm_prob:.2f}/{LSTM_FALL_THRESHOLD:.2f}"
        cv2.putText(canvas, label, (bar_x, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_WHITE, 1, cv2.LINE_AA)

    # ----------------------------------------------------------
    #  静态通道倒计时进度条（底部中部）
    # ----------------------------------------------------------

    def _draw_static_bar(self, canvas: np.ndarray,
                          progress: float, h: int, w: int):
        """
        在底部中部绘制静态通道确认进度条（仅 warning_static 时调用）。
        橙色，满格后触发 B-Static 通道。
        """
        bar_x     = 340
        bar_y     = h - 30
        bar_w_max = 280
        bar_h     = 18

        # 外框
        cv2.rectangle(canvas,
                       (bar_x, bar_y),
                       (bar_x + bar_w_max, bar_y + bar_h),
                       COLOR_GRAY, 1)

        # 填充
        fill_w = int(bar_w_max * min(progress, 1.0))
        if fill_w > 0:
            cv2.rectangle(canvas,
                           (bar_x, bar_y),
                           (bar_x + fill_w, bar_y + bar_h),
                           COLOR_WARN, -1)

        # 标签（elapsed 秒数）
        elapsed_s = progress * STATIC_FALL_DURATION
        label     = f"Static {elapsed_s:.1f}/{STATIC_FALL_DURATION:.0f}s"
        cv2.putText(canvas, label, (bar_x, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_WARN, 1, cv2.LINE_AA)

    # ----------------------------------------------------------
    #  中文文字渲染工具（供外部可选调用）
    # ----------------------------------------------------------

    def _draw_chinese(self, frame: np.ndarray, text: str,
                       pos: tuple, font_size: int,
                       color_bgr: tuple) -> np.ndarray:
        """
        用 Pillow 将中文文字渲染到指定位置。

        Args:
            frame:     BGR numpy 数组
            text:      要绘制的文字（支持中文）
            pos:       左上角坐标 (x, y)
            font_size: 字体大小
            color_bgr: BGR 颜色元组

        Returns:
            渲染后的 BGR numpy 数组
        """
        try:
            font    = get_font(font_size)
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw    = ImageDraw.Draw(img_pil)
            rgb     = _bgr2rgb(color_bgr)

            if font is not None:
                draw.text(pos, text, font=font, fill=rgb)
            else:
                draw.text(pos, text, fill=rgb)

            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception:
            # 降级到 OpenCV（无法显示中文，但不崩溃）
            cv2.putText(frame, text, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
            return frame