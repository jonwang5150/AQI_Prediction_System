OUTPUT_PATH = 'C:\\Users\\jon\\Desktop\\AQI_predict_system\\charts\\chart_33875cf1.png'
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# 中文字型與負號設定
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK TC",
    "Microsoft JhengHei",
    "PingFang TC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 資料
date_str = [
    "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
    "2026-05-06", "2026-05-07", "2026-05-08", "2026-05-09", "2026-05-10",
    "2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15",
    "2026-05-16", "2026-05-17", "2026-05-18", "2026-05-19", "2026-05-20",
    "2026-05-21", "2026-05-22", "2026-05-23", "2026-05-24", "2026-05-25",
    "2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29", "2026-05-30",
    "2026-05-31",
]
aqi_vals = [
    47.29, 43.13, 45.96, 58.58, 35.88, 27.5, 35.92, 37.25, 37.88, 28.88,
    34.88, 40.54, 32.59, 31.79, 32.83, 37.29, 44.83, 33.63, 44.33, 44.83,
    40.83, 38.5, 49.33, 59.75, 57.13, 48.17, 45.96, 54.83, 50.29, 51.5,
    42.29,
]

df = pd.DataFrame({
    "日期": pd.to_datetime(date_str),
    "avg_actual_aqi": aqi_vals,
})

# 繪圖
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    df["日期"],
    df["avg_actual_aqi"],
    color="blue",
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=6,
)

# 在每個點上顯示數值
for x, y in zip(df["日期"], df["avg_actual_aqi"]):
    ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8, color="black")

# 標題與軸標籤
ax.set_title("2026年5月木柵站每日平均實際 AQI 趨勢圖", fontsize=14, pad=15)
ax.set_xlabel("日期", fontsize=12)
ax.set_ylabel("每日平均實際 AQI", fontsize=12)

# X 軸日期格式化與旋轉
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
fig.autofmt_xdate(rotation=45, ha="right")

# 格線
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

# 版面調整
plt.tight_layout()

# 儲存圖檔
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.close()
