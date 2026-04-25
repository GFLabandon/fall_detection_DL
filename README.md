# 基于时空骨架特征的居家老人跌倒实时监测系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch)
![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazePose-green?logo=google)
![Platform](https://img.shields.io/badge/Platform-macOS_M2_MPS-lightgrey?logo=apple)
![License](https://img.shields.io/badge/License-MIT-yellow)

**DLCV 课程大作业 | 深度学习 + 计算机视觉**

*用公开跌倒数据集训练 PyTorch LSTM 分类器，在 Apple M2 上实时检测跌倒并触发语音 + 邮件报警*

</div>

---

## 🎯 项目简介

本系统针对独居老人的居家安全场景，将传统的**几何规则阈值检测**升级为**深度学习时序分类**。系统使用 Google MediaPipe BlazePose 提取人体骨架关键点，再由自训练的**双层 LSTM 神经网络**分析30帧时序特征窗口，实时判断是否发生跌倒，并触发本地语音与 SMTP 邮件双重报警。

### 与原始仓库的核心区别

```
原始仓库 (barkhaaroraa/fall_detection_DL):
  关键点提取 → if shoulder_y > threshold → 报警
  （传统规则系统，无训练过程，无量化指标）

本项目:
  关键点提取 → 12维特征序列 → [LSTM 神经网络] → 跌倒概率 → 报警
                                       ↑
                                 在 URFD 数据集上训练
                                 有 Loss 曲线 / F1 / 混淆矩阵 / 消融实验
```

---

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│  BLOCK A：训练阶段（离线，只做一次）                           │
│                                                              │
│  UR Fall Detection Dataset (URFD, 公开 RGB 视频)             │
│    → data/preprocess.py   提取 MediaPipe 关键点序列          │
│    → data/dataset.py      PyTorch Dataset 封装              │
│    → train.py             LSTM 训练 (60 epochs) + 保存 .pth │
│    → eval.py              F1 / AUC / 混淆矩阵 / 消融实验    │
└──────────────────────────────────────────────────────────────┘
                           ↓ weights/lstm_fall.pth
┌──────────────────────────────────────────────────────────────┐
│  BLOCK B：运行阶段（每次演示直接启动）                         │
│                                                              │
│  摄像头输入 (多线程采集)                                       │
│    → MediaPipe BlazePose  实时提取 33 关键点                 │
│    → data/extractor.py    12 维特征提取                      │
│    → modules/detector.py  主通道: LSTM 推理 (每帧)           │
│                           备用通道: 几何规则 (限频 0.35s)    │
│    → modules/alarm.py     macOS 本地语音报警 (冷却 10s)      │
│    → modules/email_alert.py  SMTP 邮件报警 (独立线程)        │
│    → modules/logger.py    CSV 事件日志                       │
│    → modules/renderer.py  中文 UI 渲染 (Pillow + OpenCV)    │
└──────────────────────────────────────────────────────────────┘
```

### 跌倒检测三通道决策

| 通道 | 触发条件 | 优先级 |
|------|----------|--------|
| **LSTM 主通道** | `lstm_prob ≥ 0.55`（可在 config.py 调整） | 最高 |
| **A-Dynamic 动态通道** | 肩高骤降比 > 1.35 且纵横比 > 1.40，连续 4 帧确认 | 次高 |
| **B-Static 静态通道** | 身体水平姿态持续 3 秒以上 | 备用 |

---

## ✨ 主要特性

- 🧠 **真实深度学习**：基于 URFD 公开数据集自训练两层 LSTM 分类器，有完整的训练 Loss 曲线和测试集量化指标
- ⚡ **实时推理**：Apple M2 MPS 加速，LSTM 推理延迟 < 5ms，系统整体 FPS > 20
- 🔄 **多线程架构**：摄像头采集独立线程，主线程专注推理和渲染，无延迟累积
- 🛡️ **鲁棒降级**：LSTM 权重丢失时自动降级为几何规则模式，系统不崩溃
- 🔔 **双重报警**：macOS `say` 命令本地语音 + SMTP 邮件通知家人
- 🈯 **中文 UI**：Pillow + ImageDraw 绕过 OpenCV 中文渲染限制
- 📊 **事件日志**：CSV 格式记录每次跌倒的时间戳、通道、各项指标

---

## 📁 目录结构

```
fall_detection_DL/
│
├── .env                        # 敏感配置（不提交 Git，自行创建）
├── .env.example                # .env 模板
├── .gitignore
├── config.py                   # 所有非敏感参数（阈值、超参、路径）
├── requirements.txt
│
├── main.py                     # ▶ 运行阶段入口
├── train.py                    # ▶ 训练阶段入口
├── eval.py                     # ▶ 评估阶段入口
│
├── data/
│   ├── preprocess.py           # 从 URFD 视频提取关键点序列 → .npz
│   ├── extractor.py            # FeatureExtractor（训练与推理共用）
│   ├── dataset.py              # PyTorch Dataset + DataLoader
│   └── raw/                    # 原始 URFD 视频（不提交 Git）
│       └── README.md           # 数据集下载说明
│
├── models/
│   └── lstm_classifier.py      # LSTMFallClassifier (nn.Module)
│
├── weights/
│   └── lstm_fall.pth           # 训练后生成（不提交 Git）
│
├── logs/
│   ├── fall_events.csv         # 跌倒事件日志（运行时自动生成）
│   └── training_history.csv    # 训练 Loss 记录（ Loss 曲线数据源）
│
└── modules/
    ├── detector.py             # FallDetector（三通道决策融合）
    ├── renderer.py             # 中文 UI 渲染
    ├── alarm.py                # 本地语音报警
    ├── email_alert.py          # SMTP 邮件报警
    ├── logger.py               # CSV 事件日志
    └── font_utils.py           # macOS 中文字体加载
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建并激活 conda 环境
conda create -n fall_det python=3.10 -y
conda activate fall_det

# 安装依赖（Apple M2 会自动选 MPS 版 PyTorch）
pip install -r requirements.txt
```

### 2. 配置邮件报警（可选）

复制 `.env.example` 为 `.env`，填入实际配置：

```bash
cp .env.example .env
# 编辑 .env，填入 qq 邮箱的 SMTP 授权码
```

```env
EMAIL_ENABLED=true
EMAIL_SENDER=your@qq.com
EMAIL_PASSWORD=your_smtp_auth_code
EMAIL_RECEIVER=family@example.com
EMAIL_SMTP_HOST=smtp.qq.com
EMAIL_SMTP_PORT=465
```

> 不配置邮件时保持 `EMAIL_ENABLED=false`，语音报警仍正常工作。

### 3. 下载数据集

从以下任一来源下载 URFD 数据集，解压到 `data/raw/`：

- **官方地址**：https://fenix.ur.edu.pl/~mkepski/ds/uf.html
- **Kaggle 镜像**（推荐，需登录）：
  ```bash
  pip install kaggle
  kaggle datasets download -d shahliza27/ur-fall-detection-dataset
  unzip ur-fall-detection-dataset.zip -d data/raw/
  ```

解压后 `data/raw/` 应包含 `fall-01-cam0/`…`fall-30-cam0/` 和 `adl-01-cam0/`…`adl-40-cam0/` 文件夹。

### 4. 数据预处理

```bash
python data/preprocess.py
# M2 上约需 10~20 分钟
# 输出：data/processed/fall_sequences.npz

# 快速验证流程（只处理部分数据）：
python data/preprocess.py --max_fall 10 --max_adl 10
```

### 5. 训练模型

```bash
python train.py
# M2 MPS 加速下约 3~8 分钟（60 epochs）
# 输出：weights/lstm_fall.pth + logs/training_history.csv
```

### 6. 评估（获取答辩指标）

```bash
python eval.py
# 输出：完整指标 + 消融实验表格 + 混淆矩阵
# 保存：logs/eval_results.txt
```

### 7. 运行实时检测

```bash
python main.py
```

---

## 🎮 按键操作

| 按键 | 功能 |
|------|------|
| `ESC` | 退出，打印本次运行摘要 |
| `R` | 重置报警状态和检测器（演示时用） |
| `S` | 打印今日跌倒统计 |

---

## 🧠 深度学习模型

### LSTM 分类器结构

```
输入: (batch_size, 30帧, 12维特征)
  ↓
LSTM(input=12, hidden=64, layers=2, dropout=0.3, bidirectional=False)
  ↓ 取最后帧隐状态
Dropout(0.3) → Linear(64→32) → ReLU → Dropout(0.2) → Linear(32→1)
  ↓
输出: 跌倒概率 [0, 1]
```

- **参数量**：~55,361（轻量，M2 上推理 < 5ms）
- **单向设计**：支持实时流式推理（双向 LSTM 需要完整序列）
- **损失函数**：`BCEWithLogitsLoss(pos_weight=...)` 处理类别不均衡
- **优化器**：Adam + CosineAnnealingLR + 梯度裁剪（max_norm=1.0）

### 12 维骨架特征

| 索引 | 特征 | 物理含义 |
|------|------|----------|
| 0 | shoulder_y | 肩部高度（归一化坐标） |
| 1 | hip_y | 髋部高度 |
| 2 | ankle_y | 踝部高度 |
| 3 | aspect_ratio | 人体包围盒宽/高（躺下时 >1） |
| 4 | body_angle_cos | 肩→髋向量与竖直方向夹角余弦 |
| 5 | hip_shoulder_gap | \|hip_y - shoulder_y\|（躺下时趋近0） |
| 6 | delta_shoulder | 肩高帧间变化速度 |
| 7 | delta_hip | 髋高帧间变化速度 |
| 8 | delta_angle | 身体角度帧间变化速度 |
| 9 | wrist_y_mean | 手腕平均高度 |
| 10 | knee_y_mean | 膝部平均高度 |
| 11 | head_hip_ratio | 头部相对于髋部的位置 |

---

## 📊 评估结果

> 训练完成后，运行 `python eval.py` 获取实际指标并填入下表。

| 指标 | 纯几何规则（基线） | LSTM（本项目） | LSTM + 规则融合 |
|------|------------------|---------------|----------------|
| Accuracy | - | - | - |
| Precision | - | - | - |
| Recall | - | - | - |
| **F1 Score** | - | - | - |
| AUC-ROC | - | - | - |
| FPR（误报率）| - | - | - |

---

## 🔧 参数调优

在 `config.py` 中调整，无需修改代码：

```python
# 答辩演示时 LSTM 触发灵敏度调节
LSTM_FALL_THRESHOLD = 0.55   # 调高(0.65) → 减少误报；调低(0.50) → 减少漏检

# 静态通道持续时间
STATIC_FALL_DURATION = 3.0   # 秒，越大越保守

# 报警冷却时间
ALARM_COOLDOWN = 10.0        # 秒，防止重复报警
```

---

## 🖥️ UI 界面说明

```
┌─────────────────────────────────────────────────────────────┐
│ ✅ 状态：安全巡检中          [今日跌倒: 0次  本次: 0次]       │
│ 🧠 LSTM 模型已加载                                           │
│                                                             │
│              [实时骨架叠加画面]                              │
│                                                             │
│  LSTM P=0.12 / 阈值0.55  纵横比=0.42  角度=87°             │
│  FPS: 28  |  MediaPipe BlazePose + LSTM  |  Edge-AI M2     │
│ ▓▓▓░░░░░░░░░░░░░░ LSTM 0.12             ESC退出 R重置 S统计 │
└─────────────────────────────────────────────────────────────┘

检测到跌倒时：
  - 状态栏变红："⚠ 警报：检测到跌倒！[LSTM]"
  - 屏幕四周出现红色边框
  - 骨架连线变红色
  - LSTM 进度条变红色
  - 自动触发语音 + 邮件报警
```

---

## 📝 事件日志格式

`logs/fall_events.csv`：

```csv
timestamp,datetime,event_type,channel,aspect_ratio,body_angle,lstm_prob
1711958400.0,2026-04-01 10:00:00,FALL,LSTM,1.82,72.3,0.91
1711958450.0,2026-04-01 10:00:50,FALL,B-Static,1.67,48.2,0.31
```

---

## 🛠️ 开发环境

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | ≥ 2.1.0（MPS 加速） |
| MediaPipe | ≥ 0.10.0 |
| OpenCV | ≥ 4.8.0 |
| Pillow | ≥ 10.0.0（中文渲染） |
| scikit-learn | ≥ 1.3.0（指标计算） |
| 平台 | macOS Sequoia, Apple M2 |

---

## 📖 数据集说明

使用 **UR Fall Detection Dataset (URFD)**：
- 包含 **30 个跌倒序列** + **40 个日常活动序列**，真实室内场景
- 每序列为按帧命名的 RGB PNG 图像
- 原始帧率约 30fps，预处理时降采样到 15fps（stride=2）
- 滑动窗口：30帧/窗口，步长5（数据增强，提高样本量）
- 官方下载：https://fenix.ur.edu.pl/~mkepski/ds/uf.html

---

## 🙏 致谢

- 原始仓库：[barkhaaroraa/fall_detection_DL](https://github.com/barkhaaroraa/fall_detection_DL)（提供基础摄像头框架）
- 数据集：[UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)（Michal Kepski, Bogdan Kwolek）
- 骨架提取：[Google MediaPipe BlazePose](https://github.com/google/mediapipe)

---

## 📄 License

MIT License — 仅供学术学习使用
