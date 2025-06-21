import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV
df = pd.read_csv('sgemm_benchmark.csv')

# 选择一个 K 维度固定，绘制不同 M/N 下的性能对比
# 也可以按照 M*N 做横坐标，展示 TFLOPS

df['MN'] = df['M'] * df['N']

# 画图
for version in sorted(df['Version'].unique()):
    df_v = df[df['Version'] == version]
    plt.plot(df_v['MN'], df_v['TFLOPS'], marker='o', label=version)

plt.xscale('log', base=2)
plt.xlabel('Matrix Size (M*N, log scale)')
plt.ylabel('TFLOPS')
plt.title('SGEMM Performance Comparison')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()
