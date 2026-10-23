OUTPUT_PATH = 'C:\\Users\\jon\\Desktop\\AQI_predict_system\\charts\\chart_43223b38.png'
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
        "2026-06-01","2026-06-02","2026-06-03","2026-06-04","2026-06-05","2026-06-06","2026-06-07",
        "2026-06-08","2026-06-09","2026-06-10","2026-06-11","2026-06-12","2026-06-13","2026-06-14",
        "2026-06-15","2026-06-16","2026-06-17","2026-06-18","2026-06-19","2026-06-20","2026-06-21",
        "2026-06-22","2026-06-23","2026-06-24","2026-06-25","2026-06-26","2026-06-27","2026-06-28",
        "2026-06-29","2026-06-30"
    ],
    "avg_actual_aqi": [
        60.54,64.13,64.46,51.13,26.92,33.46,41.04,34.21,25.67,34.42,
        27.08,27.88,30.29,32.42,38.17,36.25,33.67,33.54,35.54,52.04,
        50.54,51.32,54.42,43.54,36.54,26.21,21.83,26.33,29.92,31.5
    ],
    "sample_count": [
        24,24,24,24,24,24,24,24,24,24,
        24,24,24,24,24,24,24,24,24,24,
        24,22,24,24,24,24,24,24,24,24
    ]
}
df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"])

# 繪圖
plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["avg_actual_aqi"], color="blue", marker="o", linewidth=2, label="每日平均 AQI")

# 在每個點旁標示樣本數
for x, y, cnt in zip(df["date"], df["avg_actual_aqi"], df["sample_count"]):
    plt.text(x, y + 1.5, f"n={cnt}", ha="center", va="bottom", fontsize=8, color="gray")

# 標題與軸標籤
plt.title("2026 年 6 月木柵站每日平均 AQI 趨勢圖", fontsize=14, fontweight="bold")
plt.xlabel("日期", fontsize=12)
plt.ylabel("平均 AQI", fontsize=12)

# X 軸日期格式化與旋轉
plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d"))
plt.xticks(rotation=45, ha="right")

# 網格與圖例
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.close()
