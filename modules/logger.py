# fall_detection_DL/modules/logger.py
# 跌倒事件日志模块
#
# 自动将每次跌倒事件追加写入 logs/fall_events.csv，
# 支持今日统计查询和本次运行摘要，让"状态分析器"名副其实。

import os
import csv
import time
import datetime
from typing import Dict

from config import LOG_DIR, LOG_FILE


# CSV 表头字段
_CSV_FIELDS = [
    "timestamp",      # Unix 时间戳（float）
    "datetime",       # 人类可读时间（YYYY-MM-DD HH:MM:SS）
    "event_type",     # 固定为 "FALL"
    "channel",        # 触发通道：LSTM / A-Dynamic / B-Static
    "aspect_ratio",   # 人体纵横比
    "body_angle",     # 身体角度（度数）
    "lstm_prob",      # LSTM 跌倒概率
]


class EventLogger:
    """
    跌倒事件日志记录器。

    CSV 文件格式示例：
        timestamp,datetime,event_type,channel,aspect_ratio,body_angle,lstm_prob
        1711958400.0,2026-04-01 10:00:00,FALL,LSTM,1.82,72.3,0.91
    """

    def __init__(self, log_file: str = LOG_FILE):
        self.log_file      = log_file
        self._session_start = time.time()  # 本次运行开始时间
        self._session_count = 0            # 本次运行检测到的跌倒次数
        self._channel_counts: Dict[str, int] = {}  # 各通道触发次数

        # 确保 logs/ 目录存在
        os.makedirs(LOG_DIR, exist_ok=True)

        # 若 CSV 文件不存在，创建并写入表头
        if not os.path.exists(self.log_file):
            self._write_header()

    # ----------------------------------------------------------
    #  公开接口
    # ----------------------------------------------------------

    def log_fall(self, channel: str, aspect_ratio: float,
                 body_angle: float, lstm_prob: float):
        """
        记录一次跌倒事件，追加写入 CSV。

        Args:
            channel:      触发通道标识
            aspect_ratio: 人体纵横比
            body_angle:   身体角度（度数）
            lstm_prob:    LSTM 跌倒概率
        """
        now      = time.time()
        now_str  = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

        row = {
            "timestamp":    f"{now:.1f}",
            "datetime":     now_str,
            "event_type":   "FALL",
            "channel":      channel,
            "aspect_ratio": f"{aspect_ratio:.2f}",
            "body_angle":   f"{body_angle:.1f}",
            "lstm_prob":    f"{lstm_prob:.2f}",
        }

        # 追加写入 CSV（utf-8-sig 保证 Excel 正确显示中文）
        with open(self.log_file, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writerow(row)

        # 更新本次运行统计
        self._session_count += 1
        self._channel_counts[channel] = self._channel_counts.get(channel, 0) + 1

        print(f"  📝 事件记录: {now_str}  通道={channel}  P_lstm={lstm_prob:.2f}")

    def get_today_stats(self) -> dict:
        """
        读取 CSV，统计今日跌倒情况。

        Returns:
            {
                'total':      int,
                'by_channel': {'LSTM': n, 'A-Dynamic': n, 'B-Static': n}
            }
        """
        today     = datetime.date.today().strftime("%Y-%m-%d")
        total     = 0
        by_channel: Dict[str, int] = {}

        if not os.path.exists(self.log_file):
            return {"total": 0, "by_channel": {}}

        try:
            with open(self.log_file, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("datetime", "").startswith(today):
                        total += 1
                        ch    = row.get("channel", "Unknown")
                        by_channel[ch] = by_channel.get(ch, 0) + 1
        except Exception as e:
            print(f"  ⚠️  读取日志失败: {e}")

        return {"total": total, "by_channel": by_channel}

    def session_summary(self) -> str:
        """
        返回本次运行摘要字符串，程序退出时打印。
        """
        elapsed   = time.time() - self._session_start
        mins, secs = divmod(int(elapsed), 60)

        # 通道分布字符串
        ch_parts = [f"{ch}={cnt}" for ch, cnt in self._channel_counts.items()]
        ch_str   = ", ".join(ch_parts) if ch_parts else "无"

        lines = [
            "",
            "本次运行摘要",
            "─────────────",
            f"运行时长: {mins}分{secs}秒",
            f"检测跌倒: {self._session_count} 次",
            f"触发通道分布: {ch_str}",
            f"日志文件: {self.log_file}",
        ]
        return "\n".join(lines)

    # ----------------------------------------------------------
    #  内部工具
    # ----------------------------------------------------------

    def _write_header(self):
        """创建 CSV 文件并写入表头。"""
        with open(self.log_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()