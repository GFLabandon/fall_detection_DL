# ============================================================
#  modules/logger.py — 事件日志 + 统计分析
#  让"状态分析器"名副其实：自动保存跌倒事件到 CSV，支持统计查询
# ============================================================

import os, csv, time, datetime
from config import LOG_DIR, LOG_FILE

FIELDS = ["timestamp","datetime","event_type","channel","aspect_ratio","body_angle","lstm_prob"]

class EventLogger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE,"w",newline="",encoding="utf-8") as f:
                csv.DictWriter(f,fieldnames=FIELDS).writeheader()
        self._session = []

    def log_fall(self, channel, aspect_ratio, body_angle, lstm_prob):
        now = time.time()
        dt  = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
        row = {"timestamp":now,"datetime":dt,"event_type":"FALL","channel":channel,
               "aspect_ratio":f"{aspect_ratio:.2f}","body_angle":f"{body_angle:.1f}",
               "lstm_prob":f"{lstm_prob:.2f}"}
        with open(LOG_FILE,"a",newline="",encoding="utf-8") as f:
            csv.DictWriter(f,fieldnames=FIELDS).writerow(row)
        self._session.append(row)
        print(f"[LOG] 跌倒事件 {dt}  通道:{channel}  LSTM_P:{lstm_prob:.2f}")

    def get_today_stats(self):
        today = datetime.date.today().strftime("%Y-%m-%d")
        events = []
        try:
            with open(LOG_FILE,"r",encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["datetime"].startswith(today): events.append(r)
        except: pass
        chs = {}
        for e in events: chs[e.get("channel","?")] = chs.get(e.get("channel","?"),0)+1
        return {"date":today,"fall_count":len(events),"by_channel":chs,"session_count":len(self._session)}

    def session_summary(self):
        n = len(self._session)
        return f"本次运行：{'无跌倒' if n==0 else f'共{n}次跌倒，见{LOG_FILE}'}"