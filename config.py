# ============================================================
#  config.py — 全局配置中心
#  所有可调参数集中在此，无需修改其他文件
# ============================================================
import os
from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件
load_dotenv()

# -------- 摄像头 / 窗口 --------
CAMERA_INDEX   = 0
WINDOW_W       = 1280
WINDOW_H       = 720
WINDOW_TITLE   = "独居老人跌倒实时预警与状态分析器 v4.0"

# -------- MediaPipe --------
MP_MODEL_COMPLEXITY          = 1
MP_MIN_DETECTION_CONFIDENCE  = 0.6
MP_MIN_TRACKING_CONFIDENCE   = 0.5
VISIBILITY_THRESHOLD         = 0.45

# -------- 时序分析窗口（深度学习时序特征模块）--------
SEQUENCE_WINDOW   = 30    # 滑动窗口帧数（约 1 秒）
FEATURE_DIM       = 6     # 每帧特征维度（见 temporal_classifier.py）

# -------- 几何阈值 --------
ASPECT_RATIO_THRESHOLD  = 1.40   # 人体宽/高比
BODY_ANGLE_THRESHOLD    = 50.0   # 身体轴线与竖直方向夹角（度）
FALL_DROP_RATIO         = 1.35   # 肩高骤降比例

# -------- 时序分类器阈值 --------
# 综合评分 > 此值 → 判定跌倒（0~1）
TEMPORAL_FALL_SCORE     = 0.60
# 静态异常持续秒数
STATIC_FALL_DURATION    = 3.0

# -------- 防抖 --------
DYNAMIC_CONFIRM_FRAMES  = 4
CHECK_INTERVAL          = 0.35   # 检测周期（秒）

# -------- 报警 --------
ALARM_COOLDOWN     = 10.0
ALARM_VOICE        = "Ting-Ting"
ALARM_TEXT_ZH      = "跌倒预警！请立即查看！"

# -------- 邮件报警（可选，填写后生效）--------
# 从环境变量读取，如果不存在则使用默认值
EMAIL_ENABLED      = os.getenv("EMAIL_ENABLED", "False").lower() in ('true', '1', 't')  # ← 改为 True 并填写下面的参数后启用
EMAIL_SENDER       = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD     = os.getenv("EMAIL_PASSWORD", "")   # 163/QQ 邮箱授权码，不是登录密码
EMAIL_RECEIVER     = os.getenv("EMAIL_RECEIVER", "")    # 家人邮箱
EMAIL_SMTP_HOST    = os.getenv("EMAIL_SMTP_HOST", "smtp.qq.com")
EMAIL_SMTP_PORT    = int(os.getenv("EMAIL_SMTP_PORT", 465))

# -------- 日志 --------
LOG_DIR            = "logs"
LOG_FILE           = "logs/fall_events.csv"

# -------- UI 字体大小 --------
FONT_SIZE_LARGE    = 40
FONT_SIZE_NORMAL   = 30
FONT_SIZE_SMALL    = 22