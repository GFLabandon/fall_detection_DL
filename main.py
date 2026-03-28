"""
全时域居家安防：独居老人跌倒实时预警与状态分析器  v5.0
DLCV 大作业最终版

技术架构（答辩核心口述）：
  Input Video
    → MediaPipe BlazePose [CNN+Transformer 预训练深度学习模型]
        ↓ 33个人体关键点
    → FeatureExtractor [12维时序特征工程]
        ↓ 特征矩阵 (30, 12)
    → LSTMFallClassifier [PyTorch LSTM nn.Module，可训练参数≈37K]
        ↓ 跌倒概率 P ∈ [0,1]  Loss=BCEWithLogitsLoss  Opt=Adam
    → 三通道融合判决：LSTM / 动态规则 / 静态规则
    → AlarmSystem [本地语音 + 邮件推送]
    → EventLogger [CSV日志 + 统计分析]

使用流程：
  1. python data/collect.py   → 采集训练数据（跌倒+正常各50片段）
  2. python train.py          → 训练 LSTM（约3分钟）
  3. python eval.py           → 查看 Precision/Recall/F1/消融实验
  4. python main.py           → 实时检测演示
"""

import sys, os, time, queue, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import mediapipe as mp

from config import (
    CAMERA_INDEX, WINDOW_W, WINDOW_H, WINDOW_TITLE,
    MP_MODEL_COMPLEXITY, MP_MIN_DETECTION_CONFIDENCE, MP_MIN_TRACKING_CONFIDENCE,
    ALARM_COOLDOWN,
)
from modules.detector    import FallDetector, DetectionResult
from modules.renderer    import Renderer
from modules.alarm       import AlarmSystem
from modules.logger      import EventLogger
from modules.email_alert import send_fall_alert


class CaptureThread(threading.Thread):
    def __init__(self, cap, q):
        super().__init__(daemon=True)
        self.cap = cap; self.q = q; self._stop = threading.Event()
    def run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret: time.sleep(0.05); continue
            if not self.q.empty():
                try: self.q.get_nowait()
                except: pass
            self.q.put(frame)
    def stop(self): self._stop.set()


class FallDetectionApp:
    def __init__(self):
        print("🚀 正在初始化系统...")
        self.mp_pose = mp.solutions.pose
        self.pose    = self.mp_pose.Pose(
            model_complexity=MP_MODEL_COMPLEXITY,
            enable_segmentation=False, smooth_landmarks=True,
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        )
        self.detector = FallDetector()   # 内部自动加载 LSTM
        self.renderer = Renderer()
        self.alarm    = AlarmSystem()
        self.logger   = EventLogger()
        self._fps_t   = []
        print("✅ 初始化完成  |  ESC=退出  R=重置  S=统计")

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("❌ 摄像头无法打开！请检查权限（macOS→系统设置→隐私→摄像头）")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, WINDOW_W, WINDOW_H)

        fq = queue.Queue(maxsize=2)
        ct = CaptureThread(cap, fq); ct.start()

        last_fall_t = 0.0; last_log_t = 0.0
        today_stats = self.logger.get_today_stats(); stats_t = time.time()

        while True:
            try: frame = fq.get(timeout=0.5)
            except queue.Empty: print("⚠️  摄像头无响应..."); continue

            frame = cv2.flip(frame, 1); h, w = frame.shape[:2]

            now = time.time()
            self._fps_t.append(now)
            self._fps_t = [t for t in self._fps_t if now-t < 1.0]
            fps = len(self._fps_t)

            if now - stats_t > 30: today_stats = self.logger.get_today_stats(); stats_t = now

            # MediaPipe 推理
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb.flags.writeable = False
            pr  = self.pose.process(rgb); rgb.flags.writeable = True

            lms    = None
            result = DetectionResult(status="no_person")

            if pr.pose_landmarks:
                lms    = pr.pose_landmarks.landmark
                result = self.detector.update(lms, h, w)
                if result.status in ("fall_dynamic","fall_static","fall_lstm"):
                    last_fall_t = now; self.alarm.trigger()
                    if now - last_log_t > 10.0:
                        last_log_t = now
                        self.logger.log_fall(result.channel, result.aspect_ratio,
                                             result.body_angle, result.lstm_prob)
                        send_fall_alert(result.channel, result.aspect_ratio,
                                        result.body_angle, result.lstm_prob)
            else:
                self.detector.reset()

            alarm_active = self.alarm.is_active and (now-last_fall_t < ALARM_COOLDOWN)
            if not alarm_active: self.alarm.reset()

            frame = self.renderer.render(frame, result, lms, fps, alarm_active, today_stats)
            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            elif key in (ord('r'),ord('R')):
                self.alarm.reset(); self.detector.reset(); last_fall_t = 0.0
                print("✅ 报警已重置")
            elif key in (ord('s'),ord('S')):
                s = self.logger.get_today_stats()
                print(f"\n📊 今日({s['date']}) 跌倒:{s['fall_count']} 本次:{s['session_count']} 通道:{s['by_channel']}\n")

        ct.stop(); cap.release(); cv2.destroyAllWindows(); self.pose.close()
        print("\n" + self.logger.session_summary())
        print("✅ 系统已安全关闭。")


if __name__ == "__main__":
    app = FallDetectionApp()
    app.run()