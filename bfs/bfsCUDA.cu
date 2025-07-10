#include <device_launch_parameters.h>
#include <cstdio>

extern "C"
{

    __global__ void simpleBfs(int N, int level, int *d_adjacencyList, int *d_edgesOffset,
                              int *d_edgesSize, int *d_distance, int *d_parent, int *changed)
    {
        int thid = blockIdx.x * blockDim.x + threadIdx.x;
        int valueChange = 0;

        if (thid < N && d_distance[thid] == level)
        {
            int u = thid;
            for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++)
            {
                int v = d_adjacencyList[i];
                if (level + 1 < d_distance[v])
                {
                    d_distance[v] = level + 1;
                    d_parent[v] = i;
                    valueChange = 1;
                }
            }
        }

        if (valueChange)
        {
            *changed = valueChange;
        }
    }

    __global__ void queueBfs(int level, int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_distance, int *d_parent,
                             int queueSize, int *nextQueueSize, int *d_currentQueue, int *d_nextQueue)
    {
        int thid = blockIdx.x * blockDim.x + threadIdx.x;

        if (thid < queueSize)
        {
            int u = d_currentQueue[thid];
            for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++)
            {
                int v = d_adjacencyList[i];
                if (d_distance[v] == INT_MAX && atomicMin(&d_distance[v], level + 1) == INT_MAX)
                {
                    d_parent[v] = i;
                    int position = atomicAdd(nextQueueSize, 1);
                    d_nextQueue[position] = v;
                }
            }
        }
    }

    // Scan bfs
    __global__ void nextLayer(int level, int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_distance, int *d_parent,
                              int queueSize, int *d_currentQueue)
    {
        int thid = blockIdx.x * blockDim.x + threadIdx.x;

        if (thid < queueSize)
        {
            int u = d_currentQueue[thid];
            for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++)
            {
                int v = d_adjacencyList[i];
                if (level + 1 < d_distance[v])
                {
                    d_distance[v] = level + 1;
                    d_parent[v] = i;
                }
            }
        }
    }

    __global__ void countDegrees(int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_parent,
                                 int queueSize, int *d_currentQueue, int *d_degrees)
    {
        int thid = blockIdx.x * blockDim.x + threadIdx.x;

        if (thid < queueSize)
        {
            int u = d_currentQueue[thid];
            int degree = 0;
            for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++)
            {
                int v = d_adjacencyList[i];
                if (d_parent[v] == i && v != u)
                {
                    ++degree;
                }
            }
            d_degrees[thid] = degree;
        }
    }

    __global__ void scanDegrees(int size, int *d_degrees, int *incrDegrees)
    {
        int thid = blockIdx.x * blockDim.x + threadIdx.x;

        if (thid < size)
        {
            // write initial values to shared memory
            __shared__ int prefixSum[1024];
            int modulo = threadIdx.x;
            prefixSum[modulo] = d_degrees[thid];
            __syncthreads();

            // calculate scan on this block
            // go up
            for (int nodeSize = 2; nodeSize <= 1024; nodeSize <<= 1)
            {
                if ((modulo & (nodeSize - 1)) == 0)
                {
                    if (thid + (nodeSize >> 1) < size)
                    {
                        int nextPosition = modulo + (nodeSize >> 1);
                        prefixSum[modulo] += prefixSum[nextPosition];
                    }
                }
                __syncthreads();
            }

            // write information for increment prefix sums
            if (modulo == 0)
            {
                int block = thid >> 10;
                incrDegrees[block + 1] = prefixSum[modulo];
            }

            // go down
            for (int nodeSize = 1024; nodeSize > 1; nodeSize >>= 1)
            {
                if ((modulo & (nodeSize - 1)) == 0)
                {
                    if (thid + (nodeSize >> 1) < size)
                    {
                        int next_position = modulo + (nodeSize >> 1);
                        int tmp = prefixSum[modulo];
                        prefixSum[modulo] -= prefixSum[next_position];
                        prefixSum[next_position] = tmp;
                    }
                }
                __syncthreads();
            }
            d_degrees[thid] = prefixSum[modulo];
        }
    }

    __global__ void assignVerticesNextQueue(int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_parent, int queueSize,
                                            int *d_currentQueue, int *d_nextQueue, int *d_degrees, int *incrDegrees,
                                            int nextQueueSize)
    {
        int thid = blockIdx.x * blockDim.x + threadIdx.x;

        if (thid < queueSize)
        {
            __shared__ int sharedIncrement;
            if (!threadIdx.x)
            {
                sharedIncrement = incrDegrees[thid >> 10];
            }
            __syncthreads();

            int sum = 0;
            if (threadIdx.x)
            {
                sum = d_degrees[thid - 1];
            }

            int u = d_currentQueue[thid];
            int counter = 0;
            for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++)
            {
                int v = d_adjacencyList[i];
                if (d_parent[v] == i && v != u)
                {
                    int nextQueuePlace = sharedIncrement + sum + counter;
                    d_nextQueue[nextQueuePlace] = v;
                    counter++;
                }
            }
        }
    }
    __global__ void cuGunrockStyleBfsKernel(
        const int *frontier, // 当前前沿队列
        int frontier_size,
        const int *adjacencyList,
        const int *edgesOffset,
        const int *edgesSize,
        int *distance,
        int *parent,
        int *next_frontier,
        int *next_frontier_size,
        int level)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= frontier_size)
            return;

        int u = frontier[tid];
        int start = edgesOffset[u];
        int degree = edgesSize[u];

        for (int i = 0; i < degree; ++i)
        {
            int v = adjacencyList[start + i];
            if (atomicCAS(&distance[v], -1, level + 1) == -1)
            {
                parent[v] = u;

                // 获取下一个 frontier 写入位置
                int pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = v;
            }
        }
    }
}
