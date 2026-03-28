# ============================================================
#  modules/email_alert.py — 邮件远程报警
# ============================================================
import smtplib, ssl, threading, datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import EMAIL_ENABLED,EMAIL_SENDER,EMAIL_PASSWORD,EMAIL_RECEIVER,EMAIL_SMTP_HOST,EMAIL_SMTP_PORT

def send_fall_alert(channel, aspect_ratio, body_angle, lstm_prob):
    if not EMAIL_ENABLED: return
    threading.Thread(target=_send,args=(channel,aspect_ratio,body_angle,lstm_prob),daemon=True).start()

def _send(ch, ar, ang, prob):
    dt  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = MIMEMultipart()
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECEIVER
    msg["Subject"] = f"【跌倒预警】{dt}"
    msg.attach(MIMEText(
        f"事件时间：{dt}\n触发通道：{ch}\n纵横比：{ar:.2f}\n角度：{ang:.1f}°\nLSTM概率：{prob:.2f}\n\n请立即确认老人安全！",
        "plain","utf-8"))
    try:
        with smtplib.SMTP_SSL(EMAIL_SMTP_HOST,EMAIL_SMTP_PORT,context=ssl.create_default_context()) as s:
            s.login(EMAIL_SENDER,EMAIL_PASSWORD)
            s.sendmail(EMAIL_SENDER,EMAIL_RECEIVER,msg.as_string())
        print(f"✅ 邮件报警已发送")
    except Exception as e:
        print(f"⚠️  邮件失败：{e}")