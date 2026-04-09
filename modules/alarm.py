# fall_detection_DL/modules/alarm.py
# 本地语音报警模块（跨平台）
#
# 触发时在独立线程中执行系统 TTS 命令，不阻塞主检测循环。
# 内置冷却机制，防止连续报警干扰演示。

import sys
import time
import threading
import subprocess

from config import ALARM_COOLDOWN, ALARM_VOICE, ALARM_TEXT_ZH


class AlarmSystem:
    """
    本地语音报警系统。

    设计原则：
      - trigger() 立即返回，语音播报在后台线程执行
      - 冷却期内重复调用 trigger() 直接返回 False，不重复播报
      - reset() 清除冷却状态，供手动复位（按 R 键）时调用
    """

    def __init__(self, cooldown: float = ALARM_COOLDOWN):
        """
        Args:
            cooldown: 触发后的冷却时间（秒），冷却期内不重复播报
        """
        self.cooldown           = cooldown
        self._last_trigger_time = 0.0   # 上次触发的时间戳（0 = 从未触发）
        self._lock              = threading.Lock()   # 保证多线程安全

    # ----------------------------------------------------------
    #  公开接口
    # ----------------------------------------------------------

    def trigger(self) -> bool:
        """
        触发语音报警。

        Returns:
            True  — 成功触发（已启动后台播报）
            False — 冷却期内，跳过（不重复播报）
        """
        with self._lock:
            now = time.time()
            if now - self._last_trigger_time < self.cooldown:
                return False   # 还在冷却中

            self._last_trigger_time = now

        # 在独立守护线程中执行（不阻塞主线程）
        t = threading.Thread(target=self._speak, daemon=True)
        t.start()
        return True

    def reset(self):
        """
        清除冷却状态，立即允许下一次触发。
        供手动复位（按 R 键）时调用。
        """
        with self._lock:
            self._last_trigger_time = 0.0

    def cooldown_remaining(self) -> float:
        """
        返回剩余冷却时间（秒）。
        0.0 表示当前可以触发。
        """
        elapsed = time.time() - self._last_trigger_time
        return max(0.0, self.cooldown - elapsed)

    def is_cooling_down(self) -> bool:
        """是否处于冷却期。"""
        return self.cooldown_remaining() > 0.0

    # ----------------------------------------------------------
    #  内部语音播报（在后台线程执行）
    # ----------------------------------------------------------

    def _speak(self):
        """
        根据平台选择 TTS 实现。
        任何异常只打印警告，不向主线程传播。
        """
        try:
            platform = sys.platform

            if platform == "darwin":
                # macOS：使用内置 say 命令
                # -v 指定声音（Ting-Ting 为普通话女声）
                subprocess.Popen(
                    ["say", "-v", ALARM_VOICE, ALARM_TEXT_ZH],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            elif platform.startswith("win"):
                # Windows：优先尝试 PowerShell TTS，失败则用系统提示音
                try:
                    ps_cmd = (
                        f"Add-Type -AssemblyName System.Speech; "
                        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        f"$s.Speak('{ALARM_TEXT_ZH}')"
                    )
                    subprocess.Popen(
                        ["powershell", "-Command", ps_cmd],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    import winsound
                    winsound.MessageBeep(winsound.MB_ICONHAND)

            else:
                # Linux：使用 espeak
                subprocess.Popen(
                    ["espeak", "-vzh", ALARM_TEXT_ZH],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        except Exception as e:
            print(f"  ⚠️  语音报警失败：{e}")