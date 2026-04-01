# fall_detection_DL/visualize.py
# 训练曲线可视化工具（读取 logs/training_history.csv，生成 PNG 供答辩 PPT）

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

LOG_CSV = "logs/training_history.csv"
OUTPUT_PNG = "logs/training_curves.png"

if not os.path.exists(LOG_CSV):
    print("❌ 找不到训练日志，请先运行 python train.py")
    exit(1)

df = pd.read_csv(LOG_CSV)

plt.figure(figsize=(12, 5))

# Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#d62728")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss", color="#1f77b4")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

# Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(df["epoch"], df["val_acc"], label="Val Accuracy", color="#2ca02c", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
print(f"✅ 训练曲线已保存至 {OUTPUT_PNG}（可直接放入答辩 PPT）")
print(f"   最优 Epoch: {df.loc[df['val_loss'].idxmin(), 'epoch']}  val_acc={df['val_acc'].max():.1f}%")