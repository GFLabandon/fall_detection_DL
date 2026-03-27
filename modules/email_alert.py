# ============================================================
#  modules/email_alert.py — 邮件远程报警
#  解决"本地语音报警对独居老人毫无意义"的核心短板
#  配置方式：在 config.py 中设置 EMAIL_ENABLED=True 并填写账号
# ============================================================

import smtplib
import ssl
import threading
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config import (
    EMAIL_ENABLED, EMAIL_SENDER, EMAIL_PASSWORD,
    EMAIL_RECEIVER, EMAIL_SMTP_HOST, EMAIL_SMTP_PORT,
)


def send_fall_alert(channel: str, aspect_ratio: float,
                    body_angle: float, temporal_score: float):
    """
    非阻塞发送邮件报警。
    EMAIL_ENABLED=False 时静默跳过。
    """
    if not EMAIL_ENABLED:
        return
    threading.Thread(
        target=_send,
        args=(channel, aspect_ratio, body_angle, temporal_score),
        daemon=True,
    ).start()


def _send(channel: str, ar: float, angle: float, score: float):
    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"【跌倒预警】{dt} 检测到跌倒事件！"

    body = f"""
    您好，

    独居老人跌倒预警系统检测到异常，请立即查看！

    ━━━━━━━━━━━━━━━━━━━━━━
    事件时间：{dt}
    触发通道：{channel}
    纵横比：{ar:.2f}（阈值 1.40）
    身体角度：{angle:.1f}°（阈值 50°）
    时序评分：{score:.2f}（阈值 0.60）
    ━━━━━━━━━━━━━━━━━━━━━━

    请尽快确认老人安全状况！

    ——独居老人跌倒实时预警与状态分析器
    """

    msg = MIMEMultipart()
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, context=context) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print(f"✅ 邮件报警已发送至 {EMAIL_RECEIVER}")
    except Exception as e:
        print(f"⚠️  邮件发送失败：{e}（请检查 config.py 中邮件配置）")