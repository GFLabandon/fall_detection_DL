import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
import numpy as np
import threading
import queue
import time
from collections import deque

print(f'OpenCV: {cv2.__version__} | MediaPipe: {mp.__version__}')


class FallDetectionSystem:
    def __init__(self, model_path='pose_landmarker.task', seq_length=30):
        # 1. 深度学习相关的时序队列初始化
        self.seq_length = seq_length
        self.pose_sequence = deque(maxlen=self.seq_length)  # 存储最近 N 帧的姿态数据

        # 2. 配置 MediaPipe 姿态检测器
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # 3. 企业级多线程视频读取队列
        self.frame_queue = queue.Queue(maxsize=5)
        self.running = True

    def _video_capture_thread(self, src=0):
        """独立线程：专门负责无阻塞读取摄像头（Mac M2 兼容）"""
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 保持队列更新，丢弃旧帧防止延迟堆积
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
            time.sleep(0.01)  # 让出 CPU 资源

        cap.release()

    def deep_learning_classifier(self, sequence_data):
        """
        [答辩加分项] 这里是你需要自己接入深度学习模型的地方！
        sequence_data shape: (30_frames, 33_landmarks * 3_coords)
        你可以用 PyTorch/TensorFlow 训练一个简单的 LSTM。
        """
        if len(sequence_data) < self.seq_length:
            return "Collecting Data..."

        # 伪代码演示：将数据送入你的自研 DL 模型
        # tensor_input = torch.tensor(sequence_data).float()
        # prediction = my_lstm_model(tensor_input)

        # 临时占位：简单的规则模拟
        latest_pose = sequence_data[-1]
        # 假设提取头部和脚踝的 Y 坐标对比 (简化逻辑，实际应由你的DL模型输出)
        return "Normal"

    def run(self):
        """主线程：负责推理与渲染"""
        print("✅ 系统初始化完成，正在启动安防监控...")
        print("💡 答辩亮点：已启用多线程流水线与时序缓冲队列！")

        capture_thread = threading.Thread(target=self._video_capture_thread, daemon=True)
        capture_thread.start()

        start_time = time.time()

        while self.running:
            if self.frame_queue.empty():
                continue

            frame = self.frame_queue.get()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 计算正确的相对时间戳（MediaPipe 视频模式强要求）
            timestamp_ms = int((time.time() - start_time) * 1000)

            # 执行 MediaPipe 姿态检测
            try:
                detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                print(f"推理错误: {e}")
                continue

            # 提取关键点并存入时序队列（为深度学习模型准备特征）
            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]  # 取单人
                # 展平 33 个点的 (x, y, z) 为一维特征向量
                pose_features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                self.pose_sequence.append(pose_features)

                # 兼容MediaPipe Tasks的骨架绘制（修复list无landmark属性报错）
                h, w, c = frame.shape
                # 绘制关键点
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                # 绘制骨骼连线
                for connection in solutions.pose.POSE_CONNECTIONS:
                    p1 = landmarks[connection[0]]
                    p2 = landmarks[connection[1]]
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 调用深度学习分类器评估当前序列
            status = self.deep_learning_classifier(list(self.pose_sequence))

            # UI 渲染
            cv2.putText(frame, f"DL Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Buffer: {len(self.pose_sequence)}/{self.seq_length}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Enterprise Fall Detection (ESC 退出)', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False
                break

        print("✅ 安防监控系统已安全关闭。")
        cv2.destroyAllWindows()
        self.detector.close()


if __name__ == "__main__":
    # 运行前请确保目录下存在 pose_landmarker.task 文件！
    app = FallDetectionSystem()
    app.run()