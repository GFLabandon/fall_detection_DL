# fall_detection_DL/modules/email_alert.py
# SMTP 邮件报警模块
#
# EMAIL_ENABLED=False 时静默跳过，不影响主程序运行。
# 发送在独立线程中执行，不阻塞检测循环。
# 账号信息通过 .env 加载，不硬编码在代码中。

import ssl
import smtplib
import threading
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config import (
    EMAIL_ENABLED,
    EMAIL_SENDER,
    EMAIL_PASSWORD,
    EMAIL_RECEIVER,
    EMAIL_SMTP_HOST,
    EMAIL_SMTP_PORT,
)


def send_fall_alert(channel: str, aspect_ratio: float,
                    body_angle: float, lstm_prob: float):
    """
    发送跌倒报警邮件（非阻塞）。

    EMAIL_ENABLED=False 时直接返回，不做任何操作。
    发送在独立守护线程中执行，失败时只打印警告。

    Args:
        channel:      触发通道（'LSTM' / 'A-Dynamic' / 'B-Static'）
        aspect_ratio: 人体纵横比
        body_angle:   身体倾斜角度（度数）
        lstm_prob:    LSTM 跌倒概率
    """
    if not EMAIL_ENABLED:
        return   # 功能未启用，静默跳过

    # 在独立线程中发送，不阻塞主检测循环
    t = threading.Thread(
        target=_send_email,
        args=(channel, aspect_ratio, body_angle, lstm_prob),
        daemon=True,
    )
    t.start()


def _send_email(channel: str, aspect_ratio: float,
                body_angle: float, lstm_prob: float):
    """
    实际发送邮件的内部函数（在后台线程中运行）。
    任何异常只打印警告，不向主线程传播。
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- 邮件主题 ----
    subject = f"⚠ 跌倒预警 - {now_str}"

    # ---- 邮件正文（纯文本，确保兼容性）----
    body = f"""【跌倒预警通知】

检测时间：{now_str}
触发通道：{channel}
LSTM 跌倒概率：{lstm_prob:.2%}
身体纵横比：{aspect_ratio:.2f}
身体角度：{body_angle:.1f}°

请立即查看被监护人情况。

本邮件由居家老人跌倒实时监测系统自动发送
"""

    # ---- 构建邮件对象 ----
    msg               = MIMEMultipart("alternative")
    msg["Subject"]    = subject
    msg["From"]       = EMAIL_SENDER
    msg["To"]         = EMAIL_RECEIVER
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # ---- 发送 ----
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT,
                               context=context) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print(f"  ✅ 邮件报警已发送至 {EMAIL_RECEIVER}")
    except Exception as e:
        print(f"  ⚠️  邮件发送失败: {e}")
        print(f"     请检查 .env 中的 EMAIL_* 配置是否正确")