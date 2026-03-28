# ============================================================
#  modules/renderer.py — UI 渲染
# ============================================================

import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import Optional

from modules.detector import DetectionResult, SKELETON_CONNECTIONS, FallDetector
from modules.font_utils import load_font
from config import FONT_SIZE_LARGE, FONT_SIZE_NORMAL, FONT_SIZE_SMALL, LSTM_FALL_THRESHOLD


class Renderer:
    def __init__(self):
        self.fl = load_font(FONT_SIZE_LARGE,  bold=True)
        self.fn = load_font(FONT_SIZE_NORMAL, bold=False)
        self.fs = load_font(FONT_SIZE_SMALL,  bold=False)

    def render(self, frame, result: DetectionResult, landmarks,
               fps: int, alarm_active: bool,
               today_stats: Optional[dict] = None) -> np.ndarray:
        h, w   = frame.shape[:2]
        is_fall = result.status in ("fall_dynamic", "fall_static", "fall_lstm")

        if landmarks:
            self._skeleton(frame, landmarks, h, w, is_fall)
        if is_fall or alarm_active:
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 220), 14)

        return self._text_overlay(frame, result, fps, h, w, alarm_active, today_stats)

    def _skeleton(self, frame, lms, h, w, is_fall):
        lc = (0,0,200) if is_fall else (160,160,160)
        uc = (0,80,255) if is_fall else (0,191,255)
        dc = (0,200,80) if is_fall else (0,230,100)
        for (i, j) in SKELETON_CONNECTIONS:
            p1 = FallDetector.get_pixel(lms, i, h, w)
            p2 = FallDetector.get_pixel(lms, j, h, w)
            if p1 and p2: cv2.line(frame, p1, p2, lc, 2, cv2.LINE_AA)
        for idx in range(33):
            pt = FallDetector.get_pixel(lms, idx, h, w)
            if pt:
                c = dc if idx >= 23 else uc
                cv2.circle(frame, pt, 5, c, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 5, (255,255,255), 1, cv2.LINE_AA)

    def _text_overlay(self, frame, result: DetectionResult, fps, h, w, alarm_active, today_stats):
        img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        st   = result.status
        is_fall = st in ("fall_dynamic","fall_static","fall_lstm")

        # 主状态
        if is_fall or alarm_active:
            ch = result.channel or "FALL"
            icon, txt, fg, bg, font = "⚠", f"警报：检测到跌倒！[{ch}]", (255,60,60),(140,0,0), self.fl
        elif st == "warning_static":
            icon, txt, fg, bg, font = "⏱", f"姿态异常 {result.reason}", (255,165,0),(80,50,0), self.fn
        elif st == "insufficient":
            icon, txt, fg, bg, font = "📷", "视野不足：全身入画", (80,180,255),(0,40,80), self.fn
        elif st == "no_person":
            icon, txt, fg, bg, font = "🔍", "未检测到人物", (180,180,180),(40,40,40), self.fn
        else:
            icon, txt, fg, bg, font = "✅", "状态：安全巡检中", (80,220,80),(0,60,0), self.fl

        self._box(draw, f"{icon}  {txt}", (18,14), font, fg, bg)

        # 模型状态徽章
        badge = "🧠 LSTM模型" if result.model_loaded else "📐 规则模式"
        self._box(draw, badge, (18, 14+FONT_SIZE_LARGE+8), self.fs, (220,220,80),(50,50,0))

        # 调试行
        y = 14 + FONT_SIZE_LARGE + 36
        for line in [
            f"LSTM P={result.lstm_prob:.2f}/{LSTM_FALL_THRESHOLD}  纵横比={result.aspect_ratio:.2f}  角度={result.body_angle:.0f}°",
            f"FPS: {fps}  |  MediaPipe BlazePose CNN+Transformer  |  Edge-AI M2",
        ]:
            self._box(draw, line, (18, y), self.fs, (200,200,200),(20,20,20,160))
            y += FONT_SIZE_SMALL + 5

        # 今日统计（右上）
        if today_stats:
            for i, sl in enumerate([f"今日跌倒:{today_stats['fall_count']}次",
                                     f"本次:{today_stats['session_count']}次"]):
                self._box(draw, sl, (w-220, 14 + i*(FONT_SIZE_SMALL+6)), self.fs, (180,220,255),(0,30,60,200))

        self._box(draw, "ESC退出  R重置  S统计", (w-200, h-35), self.fs, (140,140,140), None)

        # LSTM 概率条（底部蓝/红）
        prob  = result.lstm_prob
        bar_w = int((w-40) * min(prob/LSTM_FALL_THRESHOLD, 1.0))
        t_col = (200,50,50) if prob >= LSTM_FALL_THRESHOLD else (50,100,220)
        draw.rectangle([20, h-10, 20+bar_w, h-3], fill=t_col)
        draw.rectangle([20, h-10, w-20, h-3], outline=(80,80,80), width=1)
        self._box(draw, f"LSTM {prob:.2f}", (22, h-32), self.fs, (160,200,255), None)

        # 静态进度条
        if st == "warning_static" and result.confirm_progress > 0:
            pw = int((w-40)*result.confirm_progress)
            draw.rectangle([20, h-22, 20+pw, h-14], fill=(255,140,0))
            draw.rectangle([20, h-22, w-20, h-14], outline=(80,80,80), width=1)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _box(draw, text, pos, font, fg, bg):
        bbox = draw.textbbox(pos, text, font=font)
        p = 4
        if bg:
            draw.rectangle([bbox[0]-p, bbox[1]-p, bbox[2]+p, bbox[3]+p], fill=bg)
        draw.text(pos, text, font=font, fill=fg)