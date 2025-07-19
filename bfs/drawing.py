import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 读取所有 bfs_*.csv 文件
csv_dir = "../build/bfs"  # 上级的 build/bfs 目录
csv_files = [f for f in os.listdir(csv_dir) if f.startswith("bfs_") and f.endswith(".csv")]

# 合并所有文件并标记数据集名
all_data = []
for file in csv_files:
    path = os.path.join(csv_dir, file)
    try:
        df = pd.read_csv(path)
        if df.empty or df.shape[1] == 0:
            print(f"Skipping empty or invalid file: {file}")
            continue
        dataset = file.replace("bfs_", "").replace(".csv", "")
        df["Dataset"] = dataset
        all_data.append(df)
    except pd.errors.EmptyDataError:
        print(f"Skipping empty file: {file}")
    except Exception as e:
        print(f"Failed to read {file}: {e}")

df = pd.concat(all_data, ignore_index=True)

# 清洗时间字段
df["KernelTime(ms)"] = pd.to_numeric(df.get("KernelTime(ms)", 0), errors="coerce").fillna(0)
df["EndToEndTime(ms)"] = pd.to_numeric(df["EndToEndTime(ms)"], errors="coerce").fillna(0)
df["OtherTime(ms)"] = df["EndToEndTime(ms)"] - df["KernelTime(ms)"]

# 相对加速比图：以 SimpleBFS 为基准
baseline_algo = "SimpleBFS"
baseline = df[df["Algorithm"] == baseline_algo][["Dataset", "EndToEndTime(ms)"]].set_index("Dataset")
baseline.columns = ["BaselineTime(ms)"]
df = df.merge(baseline, on="Dataset", how="left")
df["RelativeSpeedup"] = df["BaselineTime(ms)"] / df["EndToEndTime(ms)"]

# 图1：相对加速比
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Dataset", y="RelativeSpeedup", hue="Algorithm")
plt.title("Relative Speedup vs SimpleBFS")
plt.ylabel("Relative Speedup")
plt.xlabel("Dataset")
plt.grid(True, axis='y')
plt.legend(title="Algorithm")
plt.tight_layout()
plt.savefig("relative_speedup_vs_simplebfs.png", dpi=300)
plt.show()

# 图2：端到端时间组成（每个算法一个堆叠柱子）
df_stacked = df.copy()
df_stacked = df_stacked[["Dataset", "Algorithm", "KernelTime(ms)", "OtherTime(ms)"]]
df_stacked = df_stacked.melt(
    id_vars=["Dataset", "Algorithm"],
    value_vars=["KernelTime(ms)", "OtherTime(ms)"],
    var_name="TimeComponent",
    value_name="Time(ms)"
)
df_stacked["Group"] = df_stacked["Dataset"] + " / " + df_stacked["Algorithm"]

plt.figure(figsize=(14, 7))
sns.barplot(data=df_stacked, x="Group", y="Time(ms)", hue="TimeComponent")
plt.title("End-to-End Time Breakdown by Algorithm and Dataset (Log Scale)")
plt.ylabel("Execution Time (ms, log scale)")
plt.xlabel("Dataset / Algorithm")
plt.yscale("log")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Component")
plt.tight_layout()
plt.savefig("end_to_end_breakdown_stacked_log.png", dpi=300)
plt.show()
