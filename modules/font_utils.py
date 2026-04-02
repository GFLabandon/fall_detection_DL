# fall_detection_DL/modules/font_utils.py
# 中文字体加载工具
#
# 按平台优先级依次尝试加载字体，全部失败时回退到 PIL 默认字体。
# 使用 lru_cache 缓存，同一 size 只加载一次，避免重复 IO。

import os
import sys
import functools
from PIL import ImageFont


# ============================================================
#  各平台字体候选路径
# ============================================================

_MACOS_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/STSong.ttf",
    "/Library/Fonts/Arial Unicode MS.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    # Homebrew 安装路径
    "/opt/homebrew/share/fonts/noto-fonts/NotoSansCJK-Regular.ttc",
    # 用户目录
    os.path.expanduser("~/Library/Fonts/NotoSansSC-Regular.otf"),
    os.path.expanduser("~/Library/Fonts/SourceHanSansSC-Regular.otf"),
]

_WINDOWS_PATHS = [
    "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",     # 黑体
    "C:/Windows/Fonts/simsun.ttc",     # 宋体
]

_LINUX_PATHS = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJKsc-Regular.otf",
]


def _candidate_paths():
    """根据当前平台返回字体候选路径列表。"""
    if sys.platform == "darwin":
        return _MACOS_PATHS
    elif sys.platform.startswith("win"):
        return _WINDOWS_PATHS
    else:
        return _LINUX_PATHS


# ============================================================
#  公开接口
# ============================================================

@functools.lru_cache(maxsize=16)
def get_font(size: int) -> ImageFont.FreeTypeFont:
    """
    按平台优先级加载中文字体。
    同一 size 只加载一次（lru_cache 缓存）。

    Args:
        size: 字体大小（像素）

    Returns:
        PIL.ImageFont.FreeTypeFont（或降级为 load_default）
    """
    for path in _candidate_paths():
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                # 只在首次加载时打印（lru_cache 保证只执行一次）
                print(f"  🔤 字体已加载：{os.path.basename(path)}  size={size}")
                return font
            except Exception:
                continue  # 当前路径失败，尝试下一个

    # 所有路径均失败，回退到 PIL 默认字体（不支持中文，但不会崩溃）
    print(f"  ⚠️  未找到中文字体（size={size}），使用 PIL 默认字体（中文将显示为方块）")
    try:
        return ImageFont.load_default()
    except Exception:
        return None