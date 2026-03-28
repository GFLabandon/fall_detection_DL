# ============================================================
#  modules/alarm.py — 报警模块（跨平台）
#  macOS: say 命令；Windows: winsound；Linux: espeak
# ============================================================

import sys, time, subprocess, threading
from config import ALARM_COOLDOWN, ALARM_VOICE, ALARM_TEXT_ZH

class AlarmSystem:
    def __init__(self):
        self._last = 0.0; self._active = False
    @property
    def is_active(self): return self._active
    def trigger(self):
        now = time.time()
        if now - self._last < ALARM_COOLDOWN: return
        self._last = now; self._active = True
        threading.Thread(target=self._speak, daemon=True).start()
    def reset(self): self._active = False
    def cooldown_remaining(self): return max(0.0, ALARM_COOLDOWN-(time.time()-self._last))
    def _speak(self):
        try:
            if sys.platform == "darwin":
                subprocess.run(["say","-v",ALARM_VOICE,ALARM_TEXT_ZH],
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=12)
            elif sys.platform.startswith("win"):
                import winsound; winsound.MessageBeep(winsound.MB_ICONHAND)
            else:
                subprocess.run(["espeak","-v","zh",ALARM_TEXT_ZH],
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=12)
        except Exception as e: print(f"⚠️  语音失败：{e}")