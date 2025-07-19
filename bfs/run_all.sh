#!/bin/bash

# 切换到 ../build/bfs 目录
cd ../build/bfs || { echo "Directory ../build/bfs not found"; exit 1; }

# 遍历当前目录下所有的 .mtx 文件
for file in *.mtx; do
    if [[ -f "$file" ]]; then
        echo "Running ./bfs_exec $file"
        ./bfs_exec "$file"
    fi
done

# 回到原始目录（假设脚本位于上一级目录）
cd - 

# 全部执行完后运行 Python 绘图脚本
echo "All .mtx files processed. Running drawing.py..."
python3 drawing.py
