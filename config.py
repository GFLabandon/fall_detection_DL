# fall_detection_DL/config.py
# 全局配置中心
# 非敏感参数直接定义；敏感参数（邮件账号等）通过 .env 文件加载
# 全局配置中心 - v2 改动: 阈值调整 + 新增 STATIC_LSTM_GATE
# v3 改动（仅数值调整，无结构变化）：
#   LSTM_FALL_THRESHOLD : 0.82 → 0.65  确保演示跌倒能触发
#   STATIC_FALL_DURATION: 保持 5.0     防止短暂弯腰误报
#   STATIC_LSTM_GATE    : 0.30 → 0.25  稍宽松，防止 B-Static 漏报
#   TRAIN_EPOCHS        : 保持 60      training_history 显示 ep56 是最优

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
#  邮件报警配置
# ============================================================
EMAIL_ENABLED   = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_SENDER    = os.getenv("EMAIL_SENDER",    "")
EMAIL_PASSWORD  = os.getenv("EMAIL_PASSWORD",  "")
EMAIL_RECEIVER  = os.getenv("EMAIL_RECEIVER",  "")
EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "smtp.qq.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "465"))

# ============================================================
#  摄像头 / 窗口
# ============================================================
CAMERA_INDEX  = 0
WINDOW_W      = 1280
WINDOW_H      = 720
WINDOW_TITLE  = "独居老人跌倒实时预警与状态分析器"

# ============================================================
#  MediaPipe BlazePose
# ============================================================
MP_MODEL_COMPLEXITY          = 1
MP_MIN_DETECTION_CONFIDENCE  = 0.6
MP_MIN_TRACKING_CONFIDENCE   = 0.5
VISIBILITY_THRESHOLD         = 0.45

# ============================================================
#  时序特征 / 滑动窗口
# ============================================================
SEQUENCE_LEN  = 30
FEATURE_DIM   = 12

# ============================================================
#  LSTM 分类器
# ============================================================
LSTM_HIDDEN         = 64
LSTM_LAYERS         = 2
LSTM_DROPOUT        = 0.3
# v3: 0.82 → 0.65
# detector v3 使用 EMA(alpha=0.5) 平滑后的概率与此阈值比较
# EMA平滑后单帧噪声不会触发，真实跌倒2帧内触发
# 调整建议: 误报多→调高至0.70；漏报多→调低至0.60
LSTM_FALL_THRESHOLD = 0.65
MODEL_WEIGHTS       = "weights/lstm_fall.pth"

# ============================================================
#  几何规则通道
# ============================================================
ASPECT_RATIO_THRESHOLD = 1.40
BODY_ANGLE_THRESHOLD   = 50.0
FALL_DROP_RATIO        = 1.35
# 5.0s：防止短暂弯腰/取物的 B-Static 误报
STATIC_FALL_DURATION   = 5.0
DYNAMIC_CONFIRM_FRAMES = 4
# B-Static 联合门控：EMA平滑后的LSTM概率也必须 > 此值
# v3: 0.30 → 0.25，稍宽松以防漏报
STATIC_LSTM_GATE       = 0.25
CHECK_INTERVAL         = 0.35

# ============================================================
#  报警
# ============================================================
ALARM_COOLDOWN = 10.0
ALARM_VOICE    = "Ting-Ting"
ALARM_TEXT_ZH  = "跌倒预警！请立即查看！"

# ============================================================
#  数据集 / 训练超参
# ============================================================
TRAIN_EPOCHS     = 60   # 保持60！training_history显示ep56是最优(val_acc=95.6%)
TRAIN_BATCH_SIZE = 32
TRAIN_LR         = 1e-3
TRAIN_VAL_SPLIT  = 0.2
TRAIN_TEST_SPLIT = 0.1

# ============================================================
#  日志
# ============================================================
LOG_DIR  = "logs"
LOG_FILE = "logs/fall_events.csv"

# ============================================================
#  UI 字体大小
# ============================================================
FONT_SIZE_LARGE  = 40
FONT_SIZE_NORMAL = 30
FONT_SIZE_SMALL  = 22

# ============================================================
#  项目路径
# ============================================================
PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_DATA_FILE = os.path.join(PROCESSED_DIR, "fall_sequences.npz")