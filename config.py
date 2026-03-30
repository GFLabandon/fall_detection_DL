# fall_detection_DL/config.py
# 全局配置中心
# 非敏感参数直接定义；敏感参数（邮件账号等）通过 .env 文件加载

import os
from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件（若不存在则静默跳过）
load_dotenv()

# ============================================================
#  邮件报警配置（从 .env 读取，保护隐私）
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
MP_MODEL_COMPLEXITY          = 1      # 0=快速  1=均衡  2=高精度
MP_MIN_DETECTION_CONFIDENCE  = 0.6
MP_MIN_TRACKING_CONFIDENCE   = 0.5
VISIBILITY_THRESHOLD         = 0.45  # 关键点可见性最低阈值，低于此值视为不可见

# ============================================================
#  时序特征 / 滑动窗口
# ============================================================
SEQUENCE_LEN  = 30   # 每个输入样本的帧数（约 1 秒，15fps 采样）
FEATURE_DIM   = 12   # 每帧特征维度，详见 data/extractor.py

# ============================================================
#  LSTM 分类器
# ============================================================
LSTM_HIDDEN         = 64     # LSTM 隐层维度
LSTM_LAYERS         = 2      # LSTM 层数
LSTM_DROPOUT        = 0.3    # Dropout 比例（仅在 num_layers > 1 时生效）
LSTM_FALL_THRESHOLD = 0.55   # 跌倒概率阈值，超过此值触发 LSTM 通道报警
MODEL_WEIGHTS       = "weights/lstm_fall.pth"  # 训练后权重保存路径

# ============================================================
#  几何规则通道（备用，LSTM 不可用时自动回退）
# ============================================================
ASPECT_RATIO_THRESHOLD = 1.40   # 人体宽/高比超过此值 → 身体趋水平
BODY_ANGLE_THRESHOLD   = 50.0   # 身体轴线与竖直方向夹角（度）超过此值 → 倾斜
FALL_DROP_RATIO        = 1.35   # 肩高骤降比例，超过此值 → 快速下落
STATIC_FALL_DURATION   = 3.0    # 静态异常姿态持续超过此秒数 → 判定跌倒
DYNAMIC_CONFIRM_FRAMES = 4      # 动态通道连续确认帧数，防单帧抖动
CHECK_INTERVAL         = 0.35   # 几何规则检测周期（秒），降低 CPU 负担

# ============================================================
#  报警
# ============================================================
ALARM_COOLDOWN = 10.0          # 报警冷却时间（秒），防止连续触发
ALARM_VOICE    = "Ting-Ting"  # macOS say 命令的声音名称
ALARM_TEXT_ZH  = "跌倒预警！请立即查看！"

# ============================================================
#  数据集 / 训练超参
# ============================================================
TRAIN_DATA_FILE  = "data/processed/fall_sequences.npz"  # 预处理输出文件
TRAIN_EPOCHS     = 60
TRAIN_BATCH_SIZE = 32
TRAIN_LR         = 1e-3
TRAIN_VAL_SPLIT  = 0.2    # 验证集比例（从训练集中划分）
TRAIN_TEST_SPLIT = 0.1    # 测试集比例（独立保留，不参与训练）

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