import random

def generate_sparse_graph(n, m, filename):
    edge_set = set()
    with open(filename, 'w') as f:
        f.write(f"{n} {m}\n")
        while len(edge_set) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            # 使用无向边判重
            edge = (min(u, v), max(u, v))
            if edge in edge_set:
                continue
            edge_set.add(edge)
            f.write(f"{u} {v}\n")  # 单向边；若需双向可写两行

if __name__ == '__main__':
    n = 100000000      # 顶点数
    m = 100000000      # 边数（单向边）
    generate_sparse_graph(n, m, "graph.txt")
