"""
data/collect.py — 交互式训练数据采集工具

使用方式：
  python data/collect.py

操作说明：
  按 F — 开始录制"跌倒"片段（5秒）
  按 N — 开始录制"正常"片段（5秒）
  按 Q — 退出并保存数据集

保存文件：data/fall_sequences.npz
  X: shape (N, SEQUENCE_LEN, FEATURE_DIM) — 特征序列
  y: shape (N,)                           — 标签 (1=跌倒, 0=正常)

采集建议：
  跌倒动作：侧倒、前倒、快速下蹲、躺地不起（各采集10+次）
  正常动作：站立、坐下、弯腰、慢蹲（各采集10+次）
  总计：建议每类至少采集50个片段（各250帧以上）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

from config import (
    CAMERA_INDEX, SEQUENCE_LEN, FEATURE_DIM,
    MP_MODEL_COMPLEXITY, MP_MIN_DETECTION_CONFIDENCE, MP_MIN_TRACKING_CONFIDENCE,
    VISIBILITY_THRESHOLD, TRAIN_DATA_FILE,
)
from data.extractor import FeatureExtractor


def collect():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    pose = mp.solutions.pose.Pose(
        model_complexity=MP_MODEL_COMPLEXITY,
        min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
    )

    extractor    = FeatureExtractor(VISIBILITY_THRESHOLD)
    buffer       = deque(maxlen=SEQUENCE_LEN)

    X_all, y_all = [], []

    RECORD_SECS  = 5      # 每次录制秒数
    fps_target   = 15     # 采样帧率（从原始帧中采样）

    state        = "idle"
    label        = -1
    record_start = 0.0
    recorded_seq = []
    frame_count  = 0

    print("=" * 50)
    print("  跌倒检测训练数据采集工具")
    print("=" * 50)
    print("  F — 录制跌倒动作片段（5秒）")
    print("  N — 录制正常动作片段（5秒）")
    print("  Q — 退出并保存数据集")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        feat = None
        if results.pose_landmarks:
            feat = extractor.extract(results.pose_landmarks.landmark)
            # 绘制简单骨架
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )

        # 录制逻辑
        if state == "recording":
            elapsed = time.time() - record_start
            remain  = max(0, RECORD_SECS - elapsed)

            if feat is not None:
                recorded_seq.append(feat)

            # 进度条
            prog = elapsed / RECORD_SECS
            cv2.rectangle(frame, (20, h - 30), (20 + int((w - 40) * min(prog, 1.0)), h - 10),
                           (0, 0, 220) if label == 1 else (0, 220, 0), -1)

            tag_text = f"录制中：{'跌倒' if label==1 else '正常'} 剩余 {remain:.1f}s"
            cv2.putText(frame, tag_text, (20, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if label == 1 else (0, 200, 0), 2)

            if elapsed >= RECORD_SECS:
                # 从录制帧中均匀采样 SEQUENCE_LEN 帧
                if len(recorded_seq) >= SEQUENCE_LEN:
                    idx = np.linspace(0, len(recorded_seq) - 1, SEQUENCE_LEN, dtype=int)
                    seq = np.stack([recorded_seq[i] for i in idx])  # (30, 12)
                    X_all.append(seq)
                    y_all.append(label)
                    lname = "跌倒" if label == 1 else "正常"
                    print(f"✅ 已保存第 {len(X_all)} 个片段（{lname}）"
                          f"，共 {len(recorded_seq)} 原始帧 → {SEQUENCE_LEN} 采样帧")
                else:
                    print(f"⚠️  关键点数据不足（{len(recorded_seq)}/{SEQUENCE_LEN} 帧），片段丢弃，请重试")

                state        = "idle"
                label        = -1
                recorded_seq = []
                extractor.reset()

        else:
            # 统计
            fall_n   = sum(1 for yy in y_all if yy == 1)
            normal_n = sum(1 for yy in y_all if yy == 0)
            cv2.putText(frame,
                        f"已采集: 跌倒={fall_n}  正常={normal_n}  |  F=跌倒  N=正常  Q=退出",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("数据采集 - 独居老人跌倒检测", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('f') or key == ord('F'):
            if state == "idle":
                state        = "recording"
                label        = 1
                record_start = time.time()
                recorded_seq = []
                extractor.reset()
                print("🔴 开始录制【跌倒】动作，请立即做跌倒动作...")
        elif key == ord('n') or key == ord('N'):
            if state == "idle":
                state        = "recording"
                label        = 0
                record_start = time.time()
                recorded_seq = []
                extractor.reset()
                print("🟢 开始录制【正常】动作，请做正常动作（站立/坐下/弯腰等）...")
        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    if len(X_all) == 0:
        print("⚠️  未采集到任何数据，退出。")
        return

    X = np.stack(X_all)  # (N, 30, 12)
    y = np.array(y_all)  # (N,)
    os.makedirs("data", exist_ok=True)
    np.savez(TRAIN_DATA_FILE, X=X, y=y)

    fall_n   = int(y.sum())
    normal_n = int((y == 0).sum())
    print(f"\n✅ 数据集已保存至 {TRAIN_DATA_FILE}")
    print(f"   总样本：{len(y)}  跌倒：{fall_n}  正常：{normal_n}")
    print(f"   特征维度：{X.shape}  → 可直接运行 python train.py 开始训练")


if __name__ == "__main__":
    collect()