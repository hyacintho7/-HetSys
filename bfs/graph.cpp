#include <unordered_set>
#include <utility>
#include <cstdlib>
#include <ctime>
#include "graph.h"

// 自定义 pair<int,int> 的哈希函数
struct pair_hash
{
    std::size_t operator()(const std::pair<int, int> &p) const
    {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

void readGraph(Graph &G, int argc, char **argv)
{
    int n, m;

    bool fromStdin = argc <= 2;
    if (fromStdin)
    {
        scanf("%d %d", &n, &m);
    }
    else
    {
        srand(12345);
        n = atoi(argv[2]);
        m = atoi(argv[3]);
    }

    std::vector<std::vector<int>> adjacencyLists(n);
    // std::unordered_set<std::pair<int, int>, pair_hash> edgeSet;

    for (int i = 0; i < m; i++)
    {
        int u = rand() % n;
        int v = rand() % n;

        // if (u == v)
        //     continue;
        // if (edgeSet.count({u, v}) || edgeSet.count({v, u}))
        //     continue;

        // edgeSet.insert({u, v});
        adjacencyLists[u].push_back(v);
    }

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
