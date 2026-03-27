"""
全时域居家安防：独居老人跌倒实时预警与状态分析器  v4.0
DLCV 大作业最终版

项目目录结构：
  fall_detection_DL/
  ├── main.py                   ← 入口（多线程主循环）
  ├── config.py                 ← 全局配置
  ├── requirements.txt
  ├── logs/
  │   └── fall_events.csv       ← 跌倒事件日志（自动生成）
  └── modules/
      ├── temporal_classifier.py ← 时序特征分析器（DL Pipeline核心）
      ├── detector.py            ← 三通道跌倒检测（A动态/B静态/C时序）
      ├── renderer.py            ← UI渲染（中文+骨架+评分条）
      ├── alarm.py               ← 本地语音报警（跨平台）
      ├── email_alert.py         ← 邮件远程报警（配置后启用）
      ├── logger.py              ← 事件日志+统计分析
      └── font_utils.py          ← 中文字体加载

核心技术架构（答辩口述）：
  Input Video
    → MediaPipe BlazePose [CNN+Transformer 预训练模型]
        ↓ 33个关键点坐标
    → TemporalFeatureExtractor [6维时序特征 · 30帧滑动窗口]
        ↓ 特征矩阵 (30, 6)
    → FallScoreClassifier [时序加权评分分类头]
        ↓ 跌倒概率 0~1
    ┌─── 通道C (时序) ────┐
    │ 评分 > 0.60 → 报警  │  并行
    └─────────────────────┘
    ┌─── 通道A (动态) ────┐
    │ 肩降+纵横比 → 报警  │  并行（快速跌倒）
    └─────────────────────┘
    ┌─── 通道B (静态) ────┐
    │ 持续异常姿态 → 报警  │  并行（躺地不动）
    └─────────────────────┘
    → AlarmSystem [本地语音 + 邮件推送]
    → EventLogger [CSV日志 + 统计分析]

运行：
  conda activate fall_det
  python main.py

按键：
  ESC — 退出
  R   — 重置报警状态
  S   — 显示今日统计
"""

import sys
import time
import queue
import threading
import cv2
import mediapipe as mp

from config import (
    CAMERA_INDEX, WINDOW_W, WINDOW_H, WINDOW_TITLE,
    MP_MODEL_COMPLEXITY,
    MP_MIN_DETECTION_CONFIDENCE, MP_MIN_TRACKING_CONFIDENCE,
    ALARM_COOLDOWN,
)
from modules.detector      import FallDetector, DetectionResult
from modules.renderer      import Renderer
from modules.alarm         import AlarmSystem
from modules.logger        import EventLogger
from modules.email_alert   import send_fall_alert


# ============================================================
#  多线程帧队列（生产者-消费者模式）
#  采集线程 → frame_queue → 主推理线程
#  解决 MediaPipe 推理阻塞导致的画面卡顿
# ============================================================
class CaptureThread(threading.Thread):
    def __init__(self, cap, frame_queue: queue.Queue):
        super().__init__(daemon=True)
        self.cap   = cap
        self.queue = frame_queue
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            # 只保留最新帧，丢弃积压（避免延迟累积）
            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(frame)

    def stop(self):
        self._stop.set()


# ============================================================
#  主应用
# ============================================================
class FallDetectionApp:

    def __init__(self):
        print("🚀 正在初始化系统...")

        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose    = self.mp_pose.Pose(
            model_complexity=MP_MODEL_COMPLEXITY,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        )

        # 各模块
        self.detector = FallDetector()
        self.renderer = Renderer()
        self.alarm    = AlarmSystem()
        self.logger   = EventLogger()

        # FPS
        self._fps_times = []

        print("✅ 系统初始化完成")
        print("   按键：ESC=退出  R=重置报警  S=查看今日统计")
        self._print_architecture()

    def _print_architecture(self):
        print("""
  ╔══════════════════════════════════════════════╗
  ║  系统架构：三通道并行跌倒检测                 ║
  ║  通道A: 动态（肩高骤降+纵横比）               ║
  ║  通道B: 静态（持续异常姿态 3s）               ║
  ║  通道C: 时序（6维特征×30帧 加权评分）         ║
  ║  底层: MediaPipe BlazePose CNN+Transformer    ║
  ╚══════════════════════════════════════════════╝
        """)

    # ----------------------------------------------------------
    #  主运行循环
    # ----------------------------------------------------------
    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("❌ 无法打开摄像头！请检查摄像头权限（macOS→系统设置→隐私→摄像头）")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, WINDOW_W, WINDOW_H)

        # 启动采集线程
        frame_queue   = queue.Queue(maxsize=2)
        capture_thread = CaptureThread(cap, frame_queue)
        capture_thread.start()
        print("📷 摄像头采集线程已启动（多线程模式）")

        last_fall_time   = 0.0
        last_logged_time = 0.0
        today_stats      = self.logger.get_today_stats()
        stats_refresh_t  = time.time()

        while True:
            # 从队列取帧（超时 0.5s）
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                print("⚠️  摄像头无响应，等待中...")
                continue

            # 镜像翻转
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # FPS
            now = time.time()
            self._fps_times.append(now)
            self._fps_times = [t for t in self._fps_times if now - t < 1.0]
            fps = len(self._fps_times)

            # 刷新今日统计（每 30 秒）
            if now - stats_refresh_t > 30:
                today_stats     = self.logger.get_today_stats()
                stats_refresh_t = now

            # ---- MediaPipe 推理 ----
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_results = self.pose.process(rgb)
            rgb.flags.writeable = True

            # ---- 跌倒检测 ----
            landmarks = None
            result    = DetectionResult(status="no_person")

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                result    = self.detector.update(landmarks, h, w)

                if result.status in ("fall_dynamic", "fall_static", "fall_temporal"):
                    last_fall_time = now

                    # 触发本地语音报警
                    self.alarm.trigger()

                    # 记录日志（同一次跌倒只记录一次，冷却 10s）
                    if now - last_logged_time > 10.0:
                        last_logged_time = now
                        self.logger.log_fall(
                            channel       = result.channel,
                            aspect_ratio  = result.aspect_ratio,
                            body_angle    = result.body_angle,
                            temporal_score= result.temporal_score,
                        )
                        # 发送邮件报警（EMAIL_ENABLED=True 时生效）
                        send_fall_alert(
                            channel       = result.channel,
                            aspect_ratio  = result.aspect_ratio,
                            body_angle    = result.body_angle,
                            temporal_score= result.temporal_score,
                        )
            else:
                self.detector.reset()

            # 报警状态超时自动解除
            alarm_active = self.alarm.is_active and (now - last_fall_time < ALARM_COOLDOWN)
            if not alarm_active:
                self.alarm.reset()

            # ---- 渲染 ----
            frame = self.renderer.render(
                frame        = frame,
                result       = result,
                landmarks    = landmarks,
                fps          = fps,
                alarm_active = alarm_active,
                today_stats  = today_stats,
            )

            cv2.imshow(WINDOW_TITLE, frame)

            # ---- 按键 ----
            key = cv2.waitKey(1) & 0xFF
            if key == 27:                          # ESC → 退出
                break
            elif key in (ord('r'), ord('R')):      # R → 重置报警
                self.alarm.reset()
                self.detector.reset()
                last_fall_time = 0.0
                print("✅ 报警状态已手动重置")
            elif key in (ord('s'), ord('S')):      # S → 今日统计
                stats = self.logger.get_today_stats()
                print(f"\n📊 今日统计 ({stats['date']})：")
                print(f"   跌倒事件：{stats['fall_count']} 次")
                print(f"   触发通道：{stats['by_channel']}")
                print(f"   本次运行：{stats['session_count']} 次\n")

        # ---- 清理 ----
        capture_thread.stop()
        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
        print("\n" + self.logger.session_summary())
        print("✅ 系统已安全关闭。")


# ============================================================
#  入口
# ============================================================
if __name__ == "__main__":
    app = FallDetectionApp()
    app.run()