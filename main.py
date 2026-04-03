# fall_detection_DL/main.py
# 跌倒检测系统主程序入口
#
# 使用流程：
#   1. python data/preprocess.py   → 从 URFD 数据集提取特征（约 15 分钟）
#   2. python train.py             → 训练 LSTM 分类器（约 3-8 分钟）
#   3. python eval.py              → 查看 F1/混淆矩阵/消融实验
#   4. python main.py              → 实时检测演示（本文件）
#
# 多线程架构：
#   CaptureThread — 独立线程持续采集摄像头帧，放入 Queue
#   主线程        — 从 Queue 取帧 → MediaPipe → LSTM → 渲染 → 显示

import os
import sys
import cv2
import time
import queue
import threading

import mediapipe as mp
import numpy as np

# 确保从项目根目录导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CAMERA_INDEX,
    WINDOW_W,
    WINDOW_H,
    WINDOW_TITLE,
    MP_MODEL_COMPLEXITY,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
)
from modules.detector    import FallDetector, DetectionResult
from modules.renderer    import UIRenderer
from modules.alarm       import AlarmSystem
from modules.email_alert import send_fall_alert
from modules.logger      import EventLogger


# ============================================================
#  摄像头采集线程
# ============================================================

class CaptureThread(threading.Thread):
    """
    独立线程持续从摄像头读帧，放入有界队列。

    设计要点：
      - Queue(maxsize=2) 保证队列内始终只有最新帧，防止延迟累积
      - 队列满时丢弃旧帧（get_nowait + put），保持实时性
      - 主线程通过 stop() → join() 优雅退出
    """

    def __init__(self, cap: cv2.VideoCapture, frame_queue: queue.Queue):
        super().__init__(daemon=True)   # 守护线程，主线程退出时自动终止
        self._cap      = cap
        self._queue    = frame_queue
        self._running  = True

    def run(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                # 读帧失败，短暂等待后重试（避免 CPU 空转）
                time.sleep(0.05)
                continue

            # 若队列已满，丢弃最旧帧，保持实时性
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

            self._queue.put(frame)

    def stop(self):
        """请求线程停止。调用后应 join() 等待线程退出。"""
        self._running = False


# ============================================================
#  FPS 计算器
# ============================================================

class FPSCounter:
    """
    基于滑动窗口的 FPS 计算器。
    记录最近 N 帧的时间戳，FPS = N / 时间跨度。
    """

    def __init__(self, window: int = 30):
        self._window    = window
        self._timestamps = []

    def tick(self) -> float:
        """
        记录当前帧时间，返回当前 FPS 估计值。
        """
        now = time.time()
        self._timestamps.append(now)

        # 只保留最近 window 个时间戳
        if len(self._timestamps) > self._window:
            self._timestamps = self._timestamps[-self._window:]

        if len(self._timestamps) < 2:
            return 0.0

        # FPS = (帧数 - 1) / 时间跨度
        span = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / span if span > 1e-6 else 0.0


# ============================================================
#  主程序
# ============================================================

def print_banner(model_loaded: bool):
    """打印启动 Banner，显示系统架构信息。"""
    if model_loaded:
        model_line = "✅ LSTM 模型已加载（详见上方初始化信息）"
    else:
        model_line = "⚠  LSTM 权重未找到，使用规则回退模式"

    print()
    print(model_line)
    print()
    print("  系统架构：")
    print("    特征提取  MediaPipe BlazePose (CNN+Transformer，谷歌预训练)")
    print("    分类器    LSTM 二分类 (自训练，URFD 数据集)" if model_loaded
          else "    分类器    几何规则（回退模式）")
    print("    备用通道  几何规则（肩高+纵横比+角度）")
    print("    报警方式  本地语音 + SMTP 邮件")
    print()
    print("  按键：ESC=退出  R=重置报警  S=今日统计")
    print()


def print_today_stats(logger: EventLogger):
    """打印今日统计信息（按 S 键触发）。"""
    stats = logger.get_today_stats()
    print()
    print("  ─── 今日跌倒统计 ───────────────")
    print(f"  总计：{stats['total']} 次")
    for ch, n in stats["by_channel"].items():
        print(f"    {ch}: {n} 次")
    print("  ────────────────────────────────")
    print()


def main():
    print("🚀 正在初始化系统...")

    # ---- 初始化 MediaPipe Pose ----
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode       = False,   # 视频流模式（连续追踪）
        model_complexity        = MP_MODEL_COMPLEXITY,
        smooth_landmarks        = True,
        enable_segmentation     = False,
        min_detection_confidence= MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence = MP_MIN_TRACKING_CONFIDENCE,
    )

    # ---- 初始化各功能模块 ----
    # FallDetector 在 __init__ 内部自动尝试加载 LSTM 权重
    detector = FallDetector()
    renderer = UIRenderer()
    alarm    = AlarmSystem()
    logger   = EventLogger()
    fps_ctr  = FPSCounter(window=30)

    # 打印 Banner（使用 detector.model_loaded 判断权重是否加载成功）
    print_banner(detector.model_loaded)

    # ---- 打开摄像头 ----
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ 摄像头无法打开（索引 {CAMERA_INDEX}）")
        print("   请检查：")
        print("   1. macOS → 系统设置 → 隐私与安全 → 摄像头，允许终端访问")
        print("   2. config.py 中 CAMERA_INDEX 是否正确（默认 0 为内置摄像头）")
        mp_pose.close()
        sys.exit(1)

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WINDOW_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ---- 创建显示窗口 ----
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, WINDOW_W, WINDOW_H)

    # ---- 启动摄像头采集线程 ----
    frame_queue    = queue.Queue(maxsize=2)
    capture_thread = CaptureThread(cap, frame_queue)
    capture_thread.start()
    print("✅ 摄像头采集线程已启动（多线程模式）")
    print("✅ 系统初始化完成")

    # ---- 运行状态变量 ----
    session_fall_count = 0       # 本次运行检测到的跌倒次数
    last_log_time      = 0.0     # 上次记录日志的时间（防止重复记录同一次跌倒）
    LOG_COOLDOWN       = 10.0    # 日志记录冷却时间（秒），与报警冷却一致

    try:
        # ============================================================
        #  主循环
        # ============================================================
        while True:
            # ── 取帧 ──────────────────────────────────────────────
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                # 摄像头暂时无响应，继续等待
                continue

            # 镜像翻转（投影仪演示友好，避免左右镜像混淆）
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # ── FPS 计算 ──────────────────────────────────────────
            fps = fps_ctr.tick()

            # ── MediaPipe 推理 ────────────────────────────────────
            # BGR → RGB（MediaPipe 接受 RGB 格式）
            rgb                 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False   # 避免不必要的数组复制
            pose_results        = mp_pose.process(rgb)
            rgb.flags.writeable = True

            # 提取 landmarks 对象（可能为 None，即画面中无人）
            landmarks_obj = pose_results.pose_landmarks

            # ── 跌倒检测 ─────────────────────────────────────────
            if landmarks_obj is not None:
                # 将 landmarks 列表传入 detector
                result = detector.update(
                    landmarks_obj.landmark,   # landmark 列表（33个关键点）
                    h, w,
                )
            else:
                # 画面中无人，重置检测器状态
                detector.reset()
                result = DetectionResult(
                    status="no_person",
                    model_loaded=detector.model_loaded,
                )

            # ── 跌倒触发响应链 ────────────────────────────────────
            is_fall = result.status in ("fall_lstm", "fall_dynamic", "fall_static")
            if is_fall:
                now = time.time()

                # 语音报警（内置冷却，trigger 返回 True 表示本次实际触发）
                triggered = alarm.trigger()

                # 日志记录（独立冷却，防止同一次跌倒重复写入 CSV）
                if now - last_log_time >= LOG_COOLDOWN:
                    last_log_time = now
                    logger.log_fall(
                        channel      = result.channel,
                        aspect_ratio = result.aspect_ratio,
                        body_angle   = result.body_angle,
                        lstm_prob    = result.lstm_prob,
                    )
                    session_fall_count += 1

                    # 邮件报警（仅在本次实际触发语音报警时发送，避免重复）
                    if triggered:
                        send_fall_alert(
                            channel      = result.channel,
                            aspect_ratio = result.aspect_ratio,
                            body_angle   = result.body_angle,
                            lstm_prob    = result.lstm_prob,
                        )

            # ── 渲染 ─────────────────────────────────────────────
            today_stats = logger.get_today_stats()
            rendered    = renderer.draw(
                frame         = frame,
                result        = result,
                fps           = fps,
                pose_landmarks= landmarks_obj,   # 可为 None（renderer 内部处理）
                today_count   = today_stats["total"],
                session_count = session_fall_count,
            )

            # ── 显示 ─────────────────────────────────────────────
            cv2.imshow(WINDOW_TITLE, rendered)

            # ── 按键处理 ─────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == 27:   # ESC → 退出
                print("\n  收到退出指令（ESC）...")
                break

            elif key in (ord('r'), ord('R')):   # R → 重置
                detector.reset()
                alarm.reset()
                last_log_time = 0.0
                print("  ✅ 系统已重置（检测器、报警冷却已清除）")

            elif key in (ord('s'), ord('S')):   # S → 今日统计
                print_today_stats(logger)

    except KeyboardInterrupt:
        print("\n  收到中断信号（Ctrl+C）...")

    finally:
        # ============================================================
        #  优雅退出：释放所有资源
        # ============================================================
        print("\n  正在安全退出...")

        # 停止采集线程
        capture_thread.stop()
        capture_thread.join(timeout=2.0)

        # 释放摄像头和窗口
        cap.release()
        cv2.destroyAllWindows()

        # 关闭 MediaPipe
        mp_pose.close()

        # 打印本次运行摘要
        print(logger.session_summary())
        print("\n  👋 系统已安全退出")


# ============================================================
#  入口
# ============================================================
if __name__ == "__main__":
    main()