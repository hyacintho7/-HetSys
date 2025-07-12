#include <unordered_set>
#include <utility>
#include <cstdlib>
#include <ctime>
#include "graph.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

// 自定义 pair<int,int> 的哈希函数
// struct pair_hash
// {
//     std::size_t operator()(const std::pair<int, int> &p) const
//     {
//         return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
//     }
// };

// void readGraph(Graph &G, int argc, char **argv)
// {
//     int n, m;

//     bool fromStdin = argc <= 2;
//     if (fromStdin)
//     {
//         scanf("%d %d", &n, &m);
//     }
//     else
//     {
//         srand(12345);
//         n = atoi(argv[2]);
//         m = atoi(argv[3]);
//     }

//     std::vector<std::vector<int>> adjacencyLists(n);
//     // std::unordered_set<std::pair<int, int>, pair_hash> edgeSet;

//     for (int i = 0; i < m; i++)
//     {
//         int u = rand() % n;
//         int v = rand() % n;

//         // if (u == v)
//         //     continue;
//         // if (edgeSet.count({u, v}) || edgeSet.count({v, u}))
//         //     continue;

//         // edgeSet.insert({u, v});
//         adjacencyLists[u].push_back(v);
//     }

//     for (int i = 0; i < n; i++)
//     {
//         G.edgesOffset.push_back(G.adjacencyList.size());
//         G.edgesSize.push_back(adjacencyLists[i].size());
//         for (int neighbor : adjacencyLists[i])
//         {
//             G.adjacencyList.push_back(neighbor);
//         }
//     }

//     G.numVertices = n;
//     G.numEdges = G.adjacencyList.size();
// }

void readGraph(Graph &G, int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <mtx file path>" << std::endl;
        exit(1);
    }

    std::ifstream infile(argv[1]);
    if (!infile)
    {
        std::cerr << "Error: Cannot open file " << argv[1] << std::endl;
        exit(1);
    }

    std::string line;
    // 跳过注释行
    while (std::getline(infile, line))
    {
        if (line[0] != '%')
            break;
    }

    // 读取头部的 n, n, m
    std::istringstream header(line);
    int n, _, m; // n 行 n 列 m 条非零项（即边）
    header >> n >> _ >> m;

    std::vector<std::vector<int>> adjacencyLists(n);

    int u, v;
    while (infile >> u >> v)
    {
        // MatrixMarket 是从 1 开始计数，需减 1
        u -= 1;
        v -= 1;
        if (u < 0 || v < 0 || u >= n || v >= n)
            continue;

        adjacencyLists[u].push_back(v);
        adjacencyLists[v].push_back(u); // 假设无向图（可根据需要去掉）
    }

    // 构建 CSR 格式
    for (int i = 0; i < n; i++)
    {
        G.edgesOffset.push_back(G.adjacencyList.size());
        G.edgesSize.push_back(adjacencyLists[i].size());
        for (int neighbor : adjacencyLists[i])
        {
            G.adjacencyList.push_back(neighbor);
        }
    }

    G.numVertices = n;
    G.numEdges = G.adjacencyList.size();
}