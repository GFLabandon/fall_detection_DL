# ============================================================
#  modules/font_utils.py — 中文字体加载工具
#  覆盖 macOS Sequoia / Ventura / Monterey 全部常见路径
# ============================================================

import os
from PIL import ImageFont

_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/STSong.ttf",
    "/System/Library/Fonts/Supplemental/华文黑体.ttf",
    "/Library/Fonts/Arial Unicode MS.ttf",
    "/opt/homebrew/share/fonts/noto-fonts/NotoSansCJK-Regular.ttc",
    os.path.expanduser("~/Library/Fonts/NotoSansSC-Regular.otf"),
    os.path.expanduser("~/Library/Fonts/SourceHanSansSC-Regular.otf"),
]

_BOLD_PATHS = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    search = (_BOLD_PATHS if bold else []) + _PATHS
    for path in search:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    print("⚠️  未找到中文字体，中文将显示为方块")
    return ImageFont.load_default()