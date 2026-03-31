# fall_detection_DL/data/preprocess.py
# URFD 数据集预处理脚本
#
# 功能：遍历 data/raw/ 下的 URFD 帧序列文件夹，
#       对每帧运行 MediaPipe BlazePose 提取关键点特征，
#       切分为滑动窗口，保存为 data/processed/fall_sequences.npz
#
# 使用方式：
#   python data/preprocess.py                    # 处理全部序列
#   python data/preprocess.py --max_fall 10 --max_adl 10  # 快速验证用
# URFD 数据集预处理脚本（已全面优化：路径配置化 + 跨平台 + 自定义参数 + 任意目录运行）

import argparse
import os
import sys

# 确保可以从项目根目录导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from config import (
    VISIBILITY_THRESHOLD,
    SEQUENCE_LEN,
    FEATURE_DIM,
    MP_MODEL_COMPLEXITY,
    TRAIN_DATA_FILE,
    RAW_DIR,           # ← 新增
    PROCESSED_DIR,     # ← 新增
)
from data.extractor import FeatureExtractor


# ============================================================
#  常量（路径已移到 config.py，仅保留非路径常量）
# ============================================================
#RAW_DIR        = "data/raw"
#PROCESSED_DIR  = "data/processed"
FRAME_STRIDE   = 2     # 帧采样步长（每隔1帧取1帧，约15fps）
WINDOW_STRIDE  = 5     # 滑动窗口步长（每5帧生成一个新样本）
MAX_FILL_FRAMES= 10    # 连续前向填充超过此帧数则丢弃该窗口


# ============================================================
#  工具函数（全部使用 os.path.join，跨平台）
# ============================================================

def scan_sequences(raw_dir: str):
    """
    扫描 data/raw/ 下所有 fall-xx-cam0 和 adl-xx-cam0 文件夹。

    Returns:
        fall_dirs: list of (folder_name, full_path)
        adl_dirs:  list of (folder_name, full_path)
    """
    fall_dirs, adl_dirs = [], []

    if not os.path.isdir(raw_dir):
        print(f"❌ 数据目录不存在：{raw_dir}")
        print("   请先下载 URFD 数据集，参见 data/raw/README.md")
        sys.exit(1)

    for name in sorted(os.listdir(raw_dir)):
        full_path = os.path.join(raw_dir, name)
        if not os.path.isdir(full_path):
            continue
        if name.startswith("fall-") and "cam0" in name:
            fall_dirs.append((name, full_path))
        elif name.startswith("adl-") and "cam0" in name:
            adl_dirs.append((name, full_path))

    return fall_dirs, adl_dirs


def load_frames(seq_dir: str):
    """
    从序列文件夹加载所有图像帧（按文件名排序）。
    支持 .png / .jpg / .jpeg 格式。

    Returns:
        list of np.ndarray (BGR)，按 FRAME_STRIDE 采样后的帧列表
    """
    exts    = {".png", ".jpg", ".jpeg"}
    all_files = sorted([
        f for f in os.listdir(seq_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    # 按 FRAME_STRIDE 采样
    sampled = all_files[::FRAME_STRIDE]

    frames = []
    for fname in sampled:
        img = cv2.imread(os.path.join(seq_dir, fname))
        if img is not None:
            frames.append(img)
    return frames


def frames_to_features(frames, pose, extractor: FeatureExtractor):
    """
    对帧列表逐帧运行 MediaPipe，提取特征向量序列。

    处理策略：
      - 若当前帧肩/髋不可见（extract 返回 None）：前向填充（复制上一帧特征）
      - 前向填充次数用计数器追踪，供滑动窗口丢弃判断

    Returns:
        features: list of np.ndarray(12,)，长度与有效帧数对应
        fill_counts: list of int，记录每帧是否是前向填充（0=正常，>0=连续填充次数）
    """
    features    = []
    fill_counts = []
    fill_streak = 0    # 当前连续前向填充计数
    last_feat   = None # 上一个有效特征（用于前向填充）

    for frame in frames:
        # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = pose.process(rgb)
        rgb.flags.writeable = True

        if result.pose_landmarks:
            feat = extractor.extract(result.pose_landmarks.landmark)
        else:
            feat = None

        if feat is not None:
            # 正常帧
            features.append(feat)
            fill_counts.append(0)
            fill_streak = 0
            last_feat   = feat
        else:
            # 不可见 → 前向填充
            fill_streak += 1
            if last_feat is not None:
                features.append(last_feat.copy())
            else:
                # 序列开头就不可见：用零向量填充
                features.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            fill_counts.append(fill_streak)

    return features, fill_counts


def extract_windows(features, fill_counts, label_func):
    """
    从特征序列中用滑动窗口切分训练样本。

    Args:
        features:    list of np.ndarray(12,)
        fill_counts: list of int（前向填充计数）
        label_func:  callable(window_idx, total_windows) → int(0或1)
                     根据窗口在序列中的位置决定标签

    Returns:
        windows: list of np.ndarray(SEQUENCE_LEN, FEATURE_DIM)
        labels:  list of int
    """
    windows = []
    labels  = []
    total_len = len(features)
    n_windows = 0  # 统计有效窗口数

    start = 0
    win_idx = 0
    while start + SEQUENCE_LEN <= total_len:
        end = start + SEQUENCE_LEN
        window_fills = fill_counts[start:end]

        # 丢弃策略：窗口内任意位置连续前向填充超过 MAX_FILL_FRAMES
        max_consecutive = 0
        current_streak  = 0
        for fc in window_fills:
            if fc > 0:
                current_streak = fc  # fill_count 本身记录的就是连续计数
                max_consecutive = max(max_consecutive, current_streak)
            else:
                current_streak = 0

        if max_consecutive > MAX_FILL_FRAMES:
            # 数据质量不足，跳过该窗口
            start += WINDOW_STRIDE
            win_idx += 1
            continue

        window_feat = np.stack(features[start:end])  # (SEQUENCE_LEN, FEATURE_DIM)
        label       = label_func(win_idx, None)       # total_windows 在切分时未知，传 None

        windows.append(window_feat)
        labels.append(label)
        n_windows += 1

        start += WINDOW_STRIDE
        win_idx += 1

    return windows, labels


def fall_label_func(win_idx: int, total_wins):
    """
    跌倒序列的标签规则：
    - 后 60% 的窗口标注为 1（跌倒动作发生阶段）
    - 前 40% 的窗口标注为 0（跌倒前正常站立阶段）
    注意：由于 total_wins 在切分时未知，此函数使用懒计算闭包方式调用。
    """
    # 占位符，由外部调用时动态计算（见下方 process_sequence）
    pass


def process_sequence(seq_dir: str, is_fall: bool,
                     pose, extractor: FeatureExtractor):
    """
    处理单个序列目录，返回 (windows_list, labels_list)。

    Args:
        seq_dir:  序列文件夹路径
        is_fall:  True=跌倒序列，False=日常活动序列
        pose:     MediaPipe Pose 实例
        extractor: FeatureExtractor 实例（每个序列前调用 reset()）
    """
    extractor.reset()

    frames = load_frames(seq_dir)
    if len(frames) < SEQUENCE_LEN:
        # 帧数不足，无法生成任何窗口
        return [], []

    features, fill_counts = frames_to_features(frames, pose, extractor)

    # 预先计算总窗口数（用于确定跌倒标签的切割点）
    total_possible = max(0, (len(features) - SEQUENCE_LEN) // WINDOW_STRIDE + 1)
    fall_start_idx = int(total_possible * 0.4)  # 后60%的起始窗口索引

    windows, labels = [], []
    total_len = len(features)
    win_idx   = 0
    start     = 0

    while start + SEQUENCE_LEN <= total_len:
        end          = start + SEQUENCE_LEN
        window_fills = fill_counts[start:end]

        # 检查连续前向填充是否超限
        max_consec = 0
        cur_streak = 0
        for fc in window_fills:
            if fc > 0:
                cur_streak  = fc
                max_consec  = max(max_consec, cur_streak)
            else:
                cur_streak = 0

        if max_consec <= MAX_FILL_FRAMES:
            window_feat = np.stack(features[start:end])

            if is_fall:
                # 跌倒序列：后60%窗口标注为1
                label = 1 if win_idx >= fall_start_idx else 0
            else:
                label = 0

            windows.append(window_feat)
            labels.append(label)

        start   += WINDOW_STRIDE
        win_idx += 1

    return windows, labels


# ============================================================
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="URFD 数据集预处理：提取关键点特征并保存为 .npz"
    )
    parser.add_argument(
        "--max_fall", type=int, default=None,
        help="最多处理多少个跌倒序列（不指定则全部处理，调试用）"
    )
    parser.add_argument(
        "--max_adl", type=int, default=None,
        help="最多处理多少个日常活动序列（不指定则全部处理，调试用）"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  URFD 数据集预处理")
    print(f"  SEQUENCE_LEN={SEQUENCE_LEN}  WINDOW_STRIDE={WINDOW_STRIDE}  FRAME_STRIDE={FRAME_STRIDE}")
    print("=" * 60)

    # ---- 扫描序列目录 ----
    fall_dirs, adl_dirs = scan_sequences(RAW_DIR)

    if not fall_dirs and not adl_dirs:
        print(f"❌ 在 {RAW_DIR} 中未找到任何 fall-xx-cam0 或 adl-xx-cam0 文件夹")
        print("   请参阅 data/raw/README.md 了解数据集下载方式")
        sys.exit(1)

    # 限制处理数量（调试模式）
    if args.max_fall is not None:
        fall_dirs = fall_dirs[:args.max_fall]
    if args.max_adl is not None:
        adl_dirs  = adl_dirs[:args.max_adl]

    print(f"  跌倒序列：{len(fall_dirs)} 个  |  日常活动序列：{len(adl_dirs)} 个")
    print()

    # ---- 初始化 MediaPipe（static_image_mode=True，逐帧处理）----
    pose = mp.solutions.pose.Pose(
        static_image_mode      = True,   # 批量处理图像时使用
        model_complexity       = MP_MODEL_COMPLEXITY,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )

    extractor = FeatureExtractor(vis_threshold=VISIBILITY_THRESHOLD)

    all_X, all_y = [], []

    # ---- 处理跌倒序列 ----
    print("【处理跌倒序列】")
    for name, path in tqdm(fall_dirs, desc="跌倒序列", unit="seq"):
        windows, labels = process_sequence(path, is_fall=True, pose=pose, extractor=extractor)
        n_fall   = sum(labels)
        n_normal = len(labels) - n_fall
        tqdm.write(f"  {name}: {len(load_frames(path))}帧 → {len(windows)}个窗口 "
                   f"(跌倒={n_fall}, 起身={n_normal})")
        all_X.extend(windows)
        all_y.extend(labels)

    # ---- 处理日常活动序列 ----
    print("\n【处理日常活动序列】")
    for name, path in tqdm(adl_dirs, desc="日常活动", unit="seq"):
        windows, labels = process_sequence(path, is_fall=False, pose=pose, extractor=extractor)
        tqdm.write(f"  {name}: {len(load_frames(path))}帧 → {len(windows)}个窗口 (全部正常=0)")
        all_X.extend(windows)
        all_y.extend(labels)

    pose.close()

    if len(all_X) == 0:
        print("\n❌ 未能生成任何训练样本，请检查数据集格式")
        sys.exit(1)

    # ---- 转换为 numpy 数组 ----
    X = np.stack(all_X).astype(np.float32)  # (N, SEQUENCE_LEN, FEATURE_DIM)
    y = np.array(all_y, dtype=np.int32)      # (N,)

    n_fall   = int(y.sum())
    n_normal = int((y == 0).sum())

    # ---- 保存 ----
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = TRAIN_DATA_FILE
    np.savez(output_path, X=X, y=y)

    print()
    print("=" * 60)
    print(f"  ✅ 数据集构建完成：跌倒样本={n_fall}  正常样本={n_normal}")
    print(f"  特征维度：{X.shape}  (样本数, 帧数, 特征维度)")
    print(f"  保存至 {PROCESSED_DIR}/")
    print()
    print("  下一步：python train.py  开始训练 LSTM 分类器")
    print("=" * 60)


if __name__ == "__main__":
    main()