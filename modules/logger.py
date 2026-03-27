# ============================================================
#  modules/logger.py — 事件日志 + 统计分析
#  让"状态分析器"名副其实：自动保存跌倒事件到 CSV，支持统计查询
# ============================================================

import os
import csv
import time
import datetime
from typing import List, Dict

from config import LOG_DIR, LOG_FILE

FIELDS = ["timestamp", "datetime", "event_type", "channel",
          "aspect_ratio", "body_angle", "temporal_score", "duration_s"]


class EventLogger:

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self._init_csv()
        self._session_events: List[dict] = []
        self._fall_start: float = 0.0

    def _init_csv(self):
        """首次运行创建 CSV 文件并写入表头。"""
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def log_fall(self, channel: str, aspect_ratio: float,
                 body_angle: float, temporal_score: float,
                 duration_s: float = 0.0):
        """记录一次跌倒事件。"""
        now = time.time()
        dt  = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "timestamp":     now,
            "datetime":      dt,
            "event_type":    "FALL",
            "channel":       channel,
            "aspect_ratio":  f"{aspect_ratio:.2f}",
            "body_angle":    f"{body_angle:.1f}",
            "temporal_score":f"{temporal_score:.2f}",
            "duration_s":    f"{duration_s:.1f}",
        }
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        self._session_events.append(row)
        print(f"[LOG] 跌倒事件已记录 {dt}  通道:{channel}")

    # ----------------------------------------------------------
    #  统计分析（答辩展示"分析器"功能）
    # ----------------------------------------------------------
    def get_today_stats(self) -> Dict:
        """读取今日日志，返回统计摘要。"""
        today = datetime.date.today().strftime("%Y-%m-%d")
        events = []
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["datetime"].startswith(today):
                        events.append(row)
        except Exception:
            pass

        count    = len(events)
        channels = {}
        for e in events:
            ch = e.get("channel", "unknown")
            channels[ch] = channels.get(ch, 0) + 1

        return {
            "date":           today,
            "fall_count":     count,
            "by_channel":     channels,
            "session_count":  len(self._session_events),
        }

    def session_summary(self) -> str:
        """本次运行摘要（关机时打印）。"""
        n = len(self._session_events)
        if n == 0:
            return "本次运行：未检测到跌倒事件"
        return f"本次运行：共检测到 {n} 次跌倒事件，详见 {LOG_FILE}"