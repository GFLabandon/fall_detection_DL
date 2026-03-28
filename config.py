# ============================================================
#  config.py — 全局配置中心
# ============================================================

# -------- 摄像头 / 窗口 --------
CAMERA_INDEX   = 0
WINDOW_W       = 1280
WINDOW_H       = 720
WINDOW_TITLE   = "独居老人跌倒实时预警与状态分析器 v5.0"

# -------- MediaPipe --------
MP_MODEL_COMPLEXITY          = 1
MP_MIN_DETECTION_CONFIDENCE  = 0.6
MP_MIN_TRACKING_CONFIDENCE   = 0.5
VISIBILITY_THRESHOLD         = 0.45

# -------- LSTM 模型（核心DL模块）--------
SEQUENCE_LEN        = 30        # 每个样本帧数（时序窗口）
FEATURE_DIM         = 12        # 每帧特征维度（见 data/extractor.py）
LSTM_HIDDEN         = 64        # LSTM 隐层维度
LSTM_LAYERS         = 2         # LSTM 层数
LSTM_DROPOUT        = 0.3       # Dropout 防过拟合
LSTM_FALL_THRESHOLD = 0.55      # LSTM 输出概率 > 此值 → 判定跌倒（0~1）
MODEL_WEIGHTS       = "weights/lstm_fall_classifier.pth"

# -------- 几何规则通道（备用 / 消融对比）--------
ASPECT_RATIO_THRESHOLD = 1.40
BODY_ANGLE_THRESHOLD   = 50.0
FALL_DROP_RATIO        = 1.35
STATIC_FALL_DURATION   = 3.0
DYNAMIC_CONFIRM_FRAMES = 4
CHECK_INTERVAL         = 0.35

# -------- 报警 --------
ALARM_COOLDOWN  = 10.0
ALARM_VOICE     = "Ting-Ting"
ALARM_TEXT_ZH   = "跌倒预警！请立即查看！"

# -------- 邮件报警（可选）--------
EMAIL_ENABLED   = False
EMAIL_SENDER    = "your@163.com"
EMAIL_PASSWORD  = "your_smtp_auth_code"
EMAIL_RECEIVER  = "family@example.com"
EMAIL_SMTP_HOST = "smtp.163.com"
EMAIL_SMTP_PORT = 465

# -------- 训练超参 --------
TRAIN_EPOCHS      = 60
TRAIN_BATCH_SIZE  = 32
TRAIN_LR          = 1e-3
TRAIN_VAL_SPLIT   = 0.2        # 验证集比例
TRAIN_DATA_FILE   = "data/fall_sequences.npz"

# -------- 日志 --------
LOG_DIR   = "logs"
LOG_FILE  = "logs/fall_events.csv"

# -------- UI 字体 --------
FONT_SIZE_LARGE   = 40
FONT_SIZE_NORMAL  = 30
FONT_SIZE_SMALL   = 22