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
    __global__ void fusedNextLayerCountDegrees(
        const int *__restrict__ currentQueue,
        int queueSize,
        const int *__restrict__ adjacencyList,
        const int *__restrict__ edgesOffset,
        const int *__restrict__ edgesSize,
        int *distance,
        int *parent,
        int *degrees,
        int level)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= queueSize)
            return;

        int v = currentQueue[tid];
        int start = edgesOffset[v];
        int deg = edgesSize[v];
        int local_count = 0;

        for (int i = 0; i < deg; ++i)
        {
            int u = adjacencyList[start + i];
            if (parent[u] == INT_MAX)
            {
                if (atomicCAS(&parent[u], INT_MAX, v) == INT_MAX)
                {
                    distance[u] = level + 1;
                    local_count++;
                }
            }
        }

        degrees[tid] = local_count;
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

    __global__ void scanBfsFusedKernel(
        const int *__restrict__ currentQueue,
        int queueSize,
        const int *__restrict__ adjacencyList,
        const int *__restrict__ edgesOffset,
        const int *__restrict__ edgesSize,
        int *distance,
        int *parent,
        int *nextQueue,
        int *nextQueueSize,
        int level)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int laneId = threadIdx.x % 32;
        int warpId = threadIdx.x / 32;

        if (tid >= queueSize)
            return;

        extern __shared__ int shared[];
        int *threadCounts = shared;
        int *threadOffsets = threadCounts + blockDim.x;
        int *sharedOffset = threadOffsets + blockDim.x;
        int *sharedGlobalOffset = sharedOffset + 32;

        int v = currentQueue[tid];
        int start = edgesOffset[v];
        int degree = edgesSize[v];

        int localCount = 0;

        // 临时数组：每线程最多收集 32 个新邻居（假设不会太大）
        int localNeighbors[32];
        int outputCount = 0;

        for (int i = 0; i < degree; ++i)
        {
            int u = adjacencyList[start + i];
            if (atomicCAS(&parent[u], INT_MAX, v) == INT_MAX)
            {
                distance[u] = level + 1;
                int pos = atomicAdd(nextQueueSize, 1);
                nextQueue[pos] = u;
            }
        }

        threadCounts[threadIdx.x] = localCount;

        __syncthreads();

        // warp-level prefix sum
        int val = localCount;
        for (int offset = 1; offset < 32; offset *= 2)
        {
            int n = __shfl_up_sync(0xffffffff, val, offset);
            if (laneId >= offset)
                val += n;
        }

        // 每个warp最后一个线程写入warp总和
        if (laneId == 31)
            sharedOffset[warpId] = val;

        threadOffsets[threadIdx.x] = val - localCount;
        __syncthreads();

        // block-level prefix sum on warp totals
        if (warpId == 0 && threadIdx.x < 32)
        {
            int sum = sharedOffset[threadIdx.x];
            for (int i = 1; i < 32; ++i)
            {
                if (threadIdx.x == i)
                    sharedOffset[i] += sharedOffset[i - 1];
                __syncthreads();
            }
        }
        __syncthreads();

        int finalOffset = threadOffsets[threadIdx.x];
        if (warpId > 0)
            finalOffset += sharedOffset[warpId - 1];

        // 获取全局 nextQueue 偏移
        // __shared__ int globalOffset;
        if (threadIdx.x == 0)
        {
            int sum = 0;
            for (int i = 0; i < blockDim.x; ++i)
                sum += threadCounts[i];
            sharedGlobalOffset[0] = atomicAdd(nextQueueSize, sum);
        }
        __syncthreads();

        int baseOffset = sharedGlobalOffset[0]; // 避免共享内存竞争
        int writePos = baseOffset + finalOffset;
        if (localCount > 0)
        {
            for (int i = 0; i < localCount; ++i)
            {
                int pos = writePos + i;
                nextQueue[pos] = localNeighbors[i];
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
            if (atomicCAS(&distance[v], INT_MAX, level + 1) == INT_MAX)
            {
                parent[v] = u;

                // 获取下一个 frontier 写入位置
                int pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = v;
            }
        }
    }
}
