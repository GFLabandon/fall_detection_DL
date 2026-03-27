# ============================================================
#  modules/renderer.py — UI 渲染模块
#  新增：时序分类评分进度条、触发通道标识、今日统计
# ============================================================

import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import Optional

from modules.detector import DetectionResult, SKELETON_CONNECTIONS, FallDetector
from modules.font_utils import load_font
from config import (
    FONT_SIZE_LARGE, FONT_SIZE_NORMAL, FONT_SIZE_SMALL,
    TEMPORAL_FALL_SCORE, STATIC_FALL_DURATION,
)

C_WHITE = (255, 255, 255)


class Renderer:
    def __init__(self):
        self.font_large  = load_font(FONT_SIZE_LARGE,  bold=True)
        self.font_normal = load_font(FONT_SIZE_NORMAL, bold=False)
        self.font_small  = load_font(FONT_SIZE_SMALL,  bold=False)

    # ----------------------------------------------------------
    #  主渲染入口
    # ----------------------------------------------------------
    def render(self, frame: np.ndarray, result: DetectionResult,
               landmarks, fps: int, alarm_active: bool,
               today_stats: Optional[dict] = None) -> np.ndarray:
        h, w   = frame.shape[:2]
        is_fall = result.status in ("fall_dynamic", "fall_static", "fall_temporal")

        # 骨架
        if landmarks:
            self._draw_skeleton(frame, landmarks, h, w, is_fall)

        # 红框
        if is_fall or alarm_active:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 14)

        # Pillow 文字覆盖层
        frame = self._overlay(frame, result, fps, h, w, alarm_active, today_stats)
        return frame

    # ----------------------------------------------------------
    #  骨架
    # ----------------------------------------------------------
    def _draw_skeleton(self, frame, landmarks, h, w, is_fall: bool):
        lc = (0, 0, 200)   if is_fall else (180, 180, 180)
        uc = (0, 80, 255)  if is_fall else (0, 191, 255)
        dc = (0, 200, 100) if is_fall else (0, 240, 120)

        for (i, j) in SKELETON_CONNECTIONS:
            p1 = FallDetector.get_pixel(landmarks, i, h, w)
            p2 = FallDetector.get_pixel(landmarks, j, h, w)
            if p1 and p2:
                cv2.line(frame, p1, p2, lc, 2, cv2.LINE_AA)

        for idx in range(33):
            pt = FallDetector.get_pixel(landmarks, idx, h, w)
            if pt:
                c = dc if idx >= 23 else uc
                cv2.circle(frame, pt, 5, c, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 5, C_WHITE, 1,  cv2.LINE_AA)

    # ----------------------------------------------------------
    #  文字覆盖层
    # ----------------------------------------------------------
    def _overlay(self, frame, result: DetectionResult, fps: int,
                 h: int, w: int, alarm_active: bool,
                 today_stats: Optional[dict]) -> np.ndarray:
        img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        status  = result.status
        is_fall = status in ("fall_dynamic", "fall_static", "fall_temporal")

        # ---- 主状态 ----
        if is_fall or alarm_active:
            icon, text, fg, bg = "⚠", "警报：检测到跌倒！", (255,60,60), (140,0,0)
            font = self.font_large
        elif status == "warning_static":
            icon, text, fg, bg = "⏱", f"姿态异常 {result.reason}", (255,165,0), (80,50,0)
            font = self.font_normal
        elif status == "insufficient":
            icon, text, fg, bg = "📷", "视野不足：全身入画面", (80,180,255), (0,40,80)
            font = self.font_normal
        elif status == "no_person":
            icon, text, fg, bg = "🔍", "未检测到人物", (180,180,180), (40,40,40)
            font = self.font_normal
        else:
            icon, text, fg, bg = "✅", "状态：安全巡检中", (80,220,80), (0,60,0)
            font = self.font_large

        self._tbox(draw, f"{icon}  {text}", (18, 14), font, fg, bg)

        # ---- 触发通道标识（答辩展示用）----
        if result.channel:
            self._tbox(draw, f"触发通道: {result.channel}", (18, 14 + FONT_SIZE_LARGE + 8),
                       self.font_small, (255, 220, 80), (60, 50, 0))

        # ---- 调试信息 ----
        y = 14 + FONT_SIZE_LARGE + 36
        lines = [
            f"纵横比: {result.aspect_ratio:.2f}  角度: {result.body_angle:.0f}°  时序评分: {result.temporal_score:.2f}",
            f"FPS: {fps}  |  Engine: MediaPipe BlazePose (CNN+Transformer) | Edge-AI",
        ]
        for line in lines:
            self._tbox(draw, line, (18, y), self.font_small, (200,200,200), (20,20,20,160))
            y += FONT_SIZE_SMALL + 5

        # ---- 今日统计（右侧）----
        if today_stats:
            stat_lines = [
                f"今日跌倒: {today_stats['fall_count']} 次",
                f"本次运行: {today_stats['session_count']} 次",
            ]
            sy = 14
            for sl in stat_lines:
                self._tbox(draw, sl, (w - 230, sy), self.font_small, (180,220,255), (0,30,60,200))
                sy += FONT_SIZE_SMALL + 6

        # ---- 右上角操作提示 ----
        self._tbox(draw, "ESC退出  R重置", (w - 160, h - 40), self.font_small, (150,150,150), None)

        # ---- 时序评分进度条（底部，蓝色）----
        bar_y = h - 12
        bar_w = int((w - 40) * min(result.temporal_score / TEMPORAL_FALL_SCORE, 1.0))
        t_color = (220, 60, 60) if result.temporal_score >= TEMPORAL_FALL_SCORE else (60, 120, 255)
        draw.rectangle([20, bar_y - 6, 20 + bar_w, bar_y], fill=t_color)
        draw.rectangle([20, bar_y - 6, w - 20, bar_y], outline=(80,80,80), width=1)
        self._tbox(draw, f"时序评分 {result.temporal_score:.2f}/{TEMPORAL_FALL_SCORE}",
                   (22, bar_y - 28), self.font_small, (160,200,255), None)

        # ---- 静态通道进度条 ----
        if status == "warning_static" and result.confirm_progress > 0:
            py = h - 35
            pw = int((w - 40) * result.confirm_progress)
            draw.rectangle([20, py - 6, 20 + pw, py], fill=(255, 140, 0))
            draw.rectangle([20, py - 6, w - 20, py], outline=(80,80,80), width=1)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _tbox(draw, text, pos, font, fg, bg):
        bbox = draw.textbbox(pos, text, font=font)
        pad  = 4
        if bg:
            draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad], fill=bg)
        draw.text(pos, text, font=font, fill=fg)