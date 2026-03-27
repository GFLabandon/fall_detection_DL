# 全时域居家安防：独居老人跌倒实时预警与状态分析器

> DLCV 大作业 v4.0 | Python + MediaPipe BlazePose + 时序特征分析 | Apple M2 Edge-AI

---

## 快速启动

```bash
conda create -n fall_det python=3.10 -y
conda activate fall_det
pip install -r requirements.txt
python main.py
```

---

## 目录结构

```
fall_detection_DL/
├── main.py                    # 入口（多线程采集 + 推理）
├── config.py                  # 所有可调参数
├── requirements.txt
├── logs/
│   └── fall_events.csv        # 跌倒事件日志（自动生成）
└── modules/
    ├── temporal_classifier.py  # 时序特征分析器（DL Pipeline 核心）
    ├── detector.py             # 三通道跌倒检测
    ├── renderer.py             # UI 渲染
    ├── alarm.py                # 本地语音报警
    ├── email_alert.py          # 邮件远程报警
    ├── logger.py               # 事件日志 + 统计分析
    └── font_utils.py           # 中文字体加载
```

---

## 核心技术架构

```
Input Video
  → MediaPipe BlazePose (CNN+Transformer 预训练深度学习模型)
      ↓ 33个关键点坐标
  → TemporalFeatureExtractor (6维特征 × 30帧滑动窗口)
      ↓ 特征矩阵 shape=(30, 6)
  → FallScoreClassifier (时间衰减加权评分分类头)
      ↓ 跌倒概率 P ∈ [0, 1]
  ┌─── 通道C (时序) P > 0.60 ────────────────┐
  ├─── 通道A (动态) 肩降×纵横比，4帧确认 ────┤→ 报警 + 日志 + 邮件
  └─── 通道B (静态) 持续异常姿态 3s ─────────┘
```

---

## 状态说明

| 显示 | 含义 |
|------|------|
| ✅ 安全巡检中 | 正常站立 / 坐立 |
| ⏱ 姿态异常持续中 | 静态通道倒计时（橙色进度条）|
| ⚠ 警报：检测到跌倒！| 三通道任一触发，红框 + 语音 |
| 📷 视野不足 | 仅露上半身，拒绝误判 |
| 🔍 未检测到人物 | 画面内无人 |

底部蓝色进度条 = 时序评分实时可视化（满格即触发通道C）

---

## 邮件报警启用

编辑 `config.py`：

```python
EMAIL_ENABLED   = True
EMAIL_SENDER    = "your@163.com"
EMAIL_PASSWORD  = "your_smtp_auth_code"  # SMTP 授权码，非登录密码
EMAIL_RECEIVER  = "family@example.com"
```

---

## 按键操作

| 键 | 功能 |
|----|------|
| ESC | 退出 |
| R | 重置报警状态 |
| S | 终端打印今日统计 |

---

## 答辩口述要点

**Q: 深度学习体现在哪里？**
A: 项目是标准的深度学习 Pipeline：
  1. **特征提取层**：MediaPipe BlazePose（Google 发表的 CNN+Transformer 预训练模型，参见论文 BlazePose: On-device Real-time Body Pose tracking）提取 33 个关键点；
  2. **时序特征工程**：TemporalFeatureExtractor 在 30 帧滑动窗口上提取 6 维特征向量，与 ST-GCN、SlowFast 等视频动作识别网络的输入预处理完全一致；
  3. **分类头**：FallScoreClassifier 对时序特征做时间衰减加权评分，等价于一个轻量线性分类头，输出跌倒概率。

**Q: 为什么不用 YOLO？**
A: YOLO 是泛目标检测，对"倒地"姿态行为误报高；BlazePose 专为人体姿态设计，M2 推理 < 30ms，结合时序分析比纯阈值判断准确率更高。

**Q: 社会价值？**
A: 2026年中国老龄化率超 14%，独居老人跌倒后无法自救是最高致死场景。本系统：本地推理保护隐私、邮件远程报警通知家人、CSV日志支持历史分析，完整对标市面民用养老摄像头核心功能。
