OUTPUT_PATH = 'C:\\Users\\jon\\Desktop\\DL_AQI\\charts\\chart_e8666ffa.png'
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
data = {
    "日期": [
        "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05",
        "2026-09-06", "2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10",
        "2026-09-11", "2026-09-12", "2026-09-13", "2026-09-14", "2026-09-15",
        "2026-09-16", "2026-09-17", "2026-09-18", "2026-09-19", "2026-09-20",
        "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25",
        "2026-09-26", "2026-09-27", "2026-09-28", "2026-09-29", "2026-09-30",
    ],
    "avg_actual_aqi": [
        23.21, 29.71, 24.46, 34.29, 38.46, 35.29, 49.79, 52.04, 45.29, 49.83,
        38.17, 27.04, 30.58, 47.46, 66.88, 60.67, 47.46, 48.58, 52.5, 56.0,
        51.21, 53.25, 48.33, 47.79, 51.71, 62.13, 76.83, 73.17, 63.29, 65.75,
    ],
}
df = pd.DataFrame(data)
df["日期"] = pd.to_datetime(df["日期"])

# 繪圖
plt.figure(figsize=(12, 6), facecolor="white")
plt.plot(df["日期"], df["avg_actual_aqi"], color="blue", marker="o", linewidth=2)

# X 軸刻度：每 5 天一標籤
xticks = df["日期"][::5]
xtick_labels = xticks.dt.strftime("%m-%d")
plt.xticks(ticks=xticks, labels=xtick_labels, rotation=45)

# 標示每個點的數值
for x, y in zip(df["日期"], df["avg_actual_aqi"]):
    plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8, color="black")

# 標題與軸標籤
plt.title("2026 年 9 月木柵站每日平均 AQI 趨勢", fontsize=14, pad=15)
plt.xlabel("日期", fontsize=12)
plt.ylabel("AQI (單位)", fontsize=12)

# 格線
plt.grid(True, linestyle="--", alpha=0.5)

# 資料來源註記
plt.text(
    1,
    -0.12,
    "資料來源：木柵站",
    transform=plt.gca().transAxes,
    ha="right",
    fontsize=8,
    color="gray",
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.close()
