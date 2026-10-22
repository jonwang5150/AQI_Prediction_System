OUTPUT_PATH = 'C:\\Users\\jon\\Desktop\\AQI_predict_system\\charts\\chart_e5e513fb.png'
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 中文字型與負號設定
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 資料
data = """日期,avg_actual_aqi,sample_count
2026-05-01,47.29,24
2026-05-02,43.13,24
2026-05-03,45.96,24
2026-05-04,58.58,24
2026-05-05,35.88,24
2026-05-06,27.5,24
2026-05-07,35.92,24
2026-05-08,37.25,24
2026-05-09,37.88,24
2026-05-10,28.88,24
2026-05-11,34.88,24
2026-05-12,40.54,24
2026-05-13,32.59,17
2026-05-14,31.79,24
2026-05-15,32.83,24
2026-05-16,37.29,24
2026-05-17,44.83,24
2026-05-18,33.63,24
2026-05-19,44.33,24
2026-05-20,44.83,24
2026-05-21,40.83,24
2026-05-22,38.5,24
2026-05-23,49.33,24
2026-05-24,59.75,24
2026-05-25,57.13,24
2026-05-26,48.17,24
2026-05-27,45.96,24
2026-05-28,54.83,24
2026-05-29,50.29,24
2026-05-30,51.5,24
2026-05-31,42.29,24"""

# 讀入資料
df = pd.read_csv(pd.compat.StringIO(data), parse_dates=["日期"])

# 繪圖
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df["日期"], df["avg_actual_aqi"],
        color="steelblue", marker="o", linestyle="-", linewidth=2, markersize=6)

# 在每個點上顯示數值
for x, y in zip(df["日期"], df["avg_actual_aqi"]):
    ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8, color="black")

# 設定標題與軸標籤
ax.set_title("2026年5月木柵站每日平均實際 AQI 趨勢圖", fontsize=14, pad=15)
ax.set_xlabel("日期", fontsize=12)
ax.set_ylabel("每日平均實際 AQI", fontsize=12)

# X 軸日期格式化與旋轉
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
fig.autofmt_xdate(rotation=45, ha="right")

# 格線
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

# 版面調整
plt.tight_layout()

# 儲存圖檔
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.close()
