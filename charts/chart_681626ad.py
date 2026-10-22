OUTPUT_PATH = 'C:\\Users\\jon\\Desktop\\AQI_predict_system\\charts\\chart_681626ad.png'
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字型與負號設定
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 資料
months_full = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
avg_aqi = [56.77, 55.62, 53.59, 53.22, 42.49, 38.46, 33.68, 40.47, 48.37]
sample_counts = [744, 671, 743, 720, 737, 718, 736, 743, 720]

# 繪圖
fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
bars = ax.bar(month_labels, avg_aqi, color="steelblue", edgecolor="black")

# Y 軸標籤與標題
ax.set_ylabel("平均實際 AQI", fontsize=12)
ax.set_title("2026 年 1~9 月木柵站每月平均實際 AQI", fontsize=14, pad=15)

# 網格線
ax.yaxis.grid(True, linestyle="--", color="lightgray", alpha=0.7)
ax.set_axisbelow(True)

# 在每根柱子上方標註數值（保留兩位小數）
for bar, value in zip(bars, avg_aqi):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1.5,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="black",
    )

# 樣本數說明文字（右下角）
sample_text = "樣本數為有效配對樣本"
ax.text(
    0.99,
    -0.12,
    sample_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    color="dimgray",
)

# 調整版面
plt.tight_layout()

# 儲存圖檔
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.close()
