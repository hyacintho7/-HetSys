import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('reduce_benchmark.csv')

for bsize in sorted(df['BlockSize'].unique()):
    df_b = df[df['BlockSize'] == bsize]
    plt.plot(df_b['InputSize'], df_b['KernelTime(ms)'], marker='o', label=f'BlockSize={bsize}')

plt.xscale('log', base=2)
plt.yscale('log')  # 添加对数Y轴
plt.xlabel('Input Size (log2)')
plt.ylabel('Kernel Time (ms, log scale)')
plt.title('CUDA Reduce Performance vs BlockSize (Log Scale)')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()
