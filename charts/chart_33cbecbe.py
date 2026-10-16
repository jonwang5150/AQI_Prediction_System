OUTPUT_PATH = 'C:\\Users\\jon\\Desktop\\DL_AQI\\charts\\chart_33cbecbe.png'
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

# 中文字型與負號設定
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 資料
data = {
    "日期": [
        "2026-06-01","2026-06-02","2026-06-03","2026-06-04","2026-06-05","2026-06-06","2026-06-07",
        "2026-06-08","2026-06-09","2026-06-10","2026-06-11","2026-06-12","2026-06-13","2026-06-14",
        "2026-06-15","2026-06-16","2026-06-17","2026-06-18","2026-06-19","2026-06-20","2026-06-21",
        "2026-06-22","2026-06-23","2026-06-24","2026-06-25","2026-06-26","2026-06-27","2026-06-28",
        "2026-06-29","2026-06-30"
    ],
    "avg_actual_aqi": [
        60.54,64.13,64.46,51.13,26.92,33.46,41.04,34.21,25.67,34.42,27.08,27.88,30.29,32.42,
        38.17,36.25,33.67,33.54,35.54,52.04,50.54,51.32,54.42,43.54,36.54,26.21,21.83,26.33,
        29.92,31.5
    ]
}
df = pd.DataFrame(data)
df["日期"] = pd.to_datetime(df["日期"])

# 繪圖
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df["日期"], df["avg_actual_aqi"], marker='o', color='#1f77b4', label="每日平均實際 AQI（有效配對樣本）")

# X 軸設定：每 5 天一個標籤
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45, ha='right')

# 標題與軸標籤
ax.set_title("2026 年 6 月木柵每日平均 AQI 趨勢", fontsize=14, fontweight='bold')
ax.set_xlabel("日期", fontsize=12)
ax.set_ylabel("平均 AQI", fontsize=12)

# 圖例
ax.legend(loc="upper left")

# 格線
ax.grid(True, linestyle='--', alpha=0.6)

# 資料來源註記（視需求可自行調整）
plt.text(0.01, -0.12, "資料來源：示意資料，非實際數據", transform=ax.transAxes, fontsize=9, ha='left')

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.close()
