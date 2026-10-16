OUTPUT_PATH = 'C:\\Users\\jon\\Desktop\\DL_AQI\\charts\\chart_71e8af60.png'
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 中文字型與負號設定
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 資料定義
data = {
    "date": [
        "2026-03-01","2026-03-02","2026-03-03","2026-03-04","2026-03-05","2026-03-06","2026-03-07",
        "2026-03-08","2026-03-09","2026-03-10","2026-03-11","2026-03-12","2026-03-13","2026-03-14",
        "2026-03-15","2026-03-16","2026-03-17","2026-03-18","2026-03-19","2026-03-20","2026-03-21",
        "2026-03-22","2026-03-23","2026-03-24","2026-03-25","2026-03-26","2026-03-27","2026-03-28",
        "2026-03-29","2026-03-30","2026-03-31"
    ],
    "avg_actual_aqi": [
        36.88,46.46,39.79,34.21,30.67,33.83,75.38,
        44.96,38.71,52.79,45.04,50.25,74.13,61.46,
        47.5,40.75,58.13,61.54,47.5,53.0,39.13,
        37.25,59.33,67.38,75.13,58.63,48.29,44.83,
        88.5,91.29,78.63
    ]
}
df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"])

# 建立圖形
fig, ax = plt.subplots(figsize=(12, 6))

# 繪製折線圖與圓形標記
ax.plot(df["date"], df["avg_actual_aqi"], color="steelblue", marker="o", linewidth=2, markersize=6)

# 每點上方註記樣本數 (此處示意為 N=1)
for x, y in zip(df["date"], df["avg_actual_aqi"]):
    ax.text(x, y + 2, "N=1", ha="center", va="bottom", fontsize=8, color="dimgray")

# X 軸設定：每週顯示一次
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.set_xticks(df["date"][::7])  # 每 7 天一個刻度
ax.set_xlabel("日期", fontsize=12)
ax.set_ylabel("AQI（單位）", fontsize=12)

# 標題與說明
ax.set_title("2026 年 3 月木柵站每日平均 AQI 趨勢", fontsize=14, pad=15)
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

# 註記資料來源或說明
fig.text(0.99, 0.01, "註：樣本數為示意資料，非實際數據", ha="right", va="bottom", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.close()
