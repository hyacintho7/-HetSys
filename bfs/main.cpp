#include <cstdio>
#include <cuda.h>
#include <string>
#include <thread>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <regex>
#include "graph.h"
#include "bfsCPU.h"
#include <cuda.h> // CUDA Driver API
#include <cuda_runtime.h>
#include <iostream>
#define NUM_THREADS 1024

// #include <gunrock/graphio/graphio.cuh>
// #include <gunrock/app/bfs/bfs_app.cuh>

float runGswitchbfs(int argc, char **argv);
void my_scanDegrees(int queueSize);
// extern "C" void launchScanDegrees(int *d_input, int *d_output, int n);
//  void runVC_CM_BFS(Graph &G, int src);
void printLaunchConfig(const std::string &kernelName, int gridDim, int blockDim)
{
    printf("[%s] Launch config: gridDim=(%d,1,1), blockDim=(%d,1,1)\n",
           kernelName.c_str(), gridDim, blockDim);
}

void runCpu(int startVertex, Graph &G, std::vector<int> &distance,
            std::vector<int> &parent, std::vector<bool> &visited)
{
    printf("Starting sequential bfs.\n");
    auto start = std::chrono::steady_clock::now();
    bfsCPU(startVertex, G, distance, parent, visited);
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n\n", duration);
}

void checkError(CUresult error, std::string msg)
{
    if (error != CUDA_SUCCESS)
    {
        printf("%s: %d\n", msg.c_str(), error);
        exit(1);
    }
}

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;

CUfunction cuSimpleBfs;
CUfunction cuQueueBfs;
CUfunction cuNextLayer;
CUfunction cuCountDegrees;
CUfunction cuFusedNextCount;
CUfunction cuScanBfsFusedKernel;
CUfunction cuScanDegrees;
CUfunction cuAssignVerticesNextQueue;
CUfunction cuGunrockStyleBfsKernel;
CUfunction cuScanBfsFusedKernel_WM;
CUfunction cuScanBfsFusedKernel_CM;

CUdeviceptr d_adjacencyList;
CUdeviceptr d_edgesOffset;
CUdeviceptr d_edgesSize;
CUdeviceptr d_distance;
CUdeviceptr d_parent;
CUdeviceptr d_currentQueue;
CUdeviceptr d_nextQueue;
CUdeviceptr d_degrees;
int *incrDegrees;
CUdeviceptr d_nextQueueSizeDevice;

void initCuda(Graph &G)
{
    // initialize CUDA
    cuInit(0);
    checkError(cuDeviceGet(&cuDevice, 0), "cannot get device 0");
    checkError(cuCtxCreate(&cuContext, 0, cuDevice), "cannot create context");
    checkError(cuModuleLoad(&cuModule, "bfsCUDA.ptx"), "cannot load module");
    checkError(cuModuleGetFunction(&cuSimpleBfs, cuModule, "simpleBfs"), "cannot get kernel handle");
    checkError(cuModuleGetFunction(&cuQueueBfs, cuModule, "queueBfs"), "cannot get kernel handle");
    checkError(cuModuleGetFunction(&cuNextLayer, cuModule, "nextLayer"), "cannot get kernel handle");
    checkError(cuModuleGetFunction(&cuCountDegrees, cuModule, "countDegrees"), "cannot get kernel handle");
    checkError(cuModuleGetFunction(&cuFusedNextCount, cuModule, "fusedNextLayerCountDegrees"), "cannot get fused kernel handle");
    checkError(cuModuleGetFunction(&cuScanBfsFusedKernel, cuModule, "scanBfsFusedKernel"), "cannot get kernel scanBfsFusedKernel handle");
    checkError(cuModuleGetFunction(&cuScanDegrees, cuModule, "scanDegrees"), "cannot get kernel handle");
    checkError(cuModuleGetFunction(&cuAssignVerticesNextQueue, cuModule, "assignVerticesNextQueue"),
               "cannot get kernel handle");
    checkError(cuModuleGetFunction(&cuScanBfsFusedKernel_WM, cuModule, "scanBfsFusedKernel_WM"), "cannot get scanBfsFusedKernel_WM handle");
    checkError(cuModuleGetFunction(&cuScanBfsFusedKernel_CM, cuModule, "scanBfsFusedKernel_CM"), "cannot get scanBfsFusedKernel_CM handle");

    checkError(cuModuleGetFunction(&cuGunrockStyleBfsKernel, cuModule, "cuGunrockStyleBfsKernel"), "cannot get kernel handle");

    // copy memory to device
    checkError(cuMemAlloc(&d_adjacencyList, G.numEdges * sizeof(int)), "cannot allocate d_adjacencyList");
    checkError(cuMemAlloc(&d_edgesOffset, G.numVertices * sizeof(int)), "cannot allocate d_edgesOffset");
    checkError(cuMemAlloc(&d_edgesSize, G.numVertices * sizeof(int)), "cannot allocate d_edgesSize");
    checkError(cuMemAlloc(&d_distance, G.numVertices * sizeof(int)), "cannot allocate d_distance");
    checkError(cuMemAlloc(&d_parent, G.numVertices * sizeof(int)), "cannot allocate d_parent");
    checkError(cuMemAlloc(&d_currentQueue, G.numVertices * sizeof(int)), "cannot allocate d_currentQueue");
    checkError(cuMemAlloc(&d_nextQueue, 4 * G.numVertices * sizeof(int)), "cannot allocate d_nextQueue");
    checkError(cuMemAlloc(&d_degrees, G.numVertices * sizeof(int)), "cannot allocate d_degrees");
    checkError(cuMemAllocHost((void **)&incrDegrees, sizeof(int) * G.numVertices), "cannot allocate memory");
    checkError(cuMemAlloc(&d_nextQueueSizeDevice, sizeof(int)), "cannot allocate d_nextQueueSizeDevice");

    checkError(cuMemcpyHtoD(d_adjacencyList, G.adjacencyList.data(), G.numEdges * sizeof(int)),
               "cannot copy to d_adjacencyList");
    checkError(cuMemcpyHtoD(d_edgesOffset, G.edgesOffset.data(), G.numVertices * sizeof(int)),
               "cannot copy to d_edgesOffset");
    checkError(cuMemcpyHtoD(d_edgesSize, G.edgesSize.data(), G.numVertices * sizeof(int)),
               "cannot copy to d_edgesSize");
}

void finalizeCuda()
{
    // free memory
    checkError(cuMemFree(d_adjacencyList), "cannot free memory for d_adjacencyList");
    checkError(cuMemFree(d_edgesOffset), "cannot free memory for d_edgesOffset");
    checkError(cuMemFree(d_edgesSize), "cannot free memory for d_edgesSize");
    checkError(cuMemFree(d_distance), "cannot free memory for d_distance");
    checkError(cuMemFree(d_parent), "cannot free memory for d_parent");
    checkError(cuMemFree(d_currentQueue), "cannot free memory for d_parent");
    checkError(cuMemFree(d_nextQueue), "cannot free memory for d_parent");
    checkError(cuMemFreeHost(incrDegrees), "cannot free memory for incrDegrees");
}

void checkOutput(std::vector<int> &distance, std::vector<int> &expectedDistance, Graph &G)
{
    for (int i = 0; i < G.numVertices; i++)
    {
        if (distance[i] != expectedDistance[i])
        {
            printf("%d %d %d\n", i, distance[i], expectedDistance[i]);
            printf("Wrong output!\n");
            exit(1);
        }
    }

    printf("Output OK!\n");
}

void initializeCudaBfs(int startVertex, std::vector<int> &distance, std::vector<int> &parent, Graph &G)
{
    // initialize values
    std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
    std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
    distance[startVertex] = 0;
    parent[startVertex] = 0;

    checkError(cuMemcpyHtoD(d_distance, distance.data(), G.numVertices * sizeof(int)),
               "cannot copy to d)distance");
    checkError(cuMemcpyHtoD(d_parent, parent.data(), G.numVertices * sizeof(int)),
               "cannot copy to d_parent");

    int firstElementQueue = startVertex;
    cuMemcpyHtoD(d_currentQueue, &firstElementQueue, sizeof(int));
}

void finalizeCudaBfs(std::vector<int> &distance, std::vector<int> &parent, Graph &G)
{
    // copy memory from device
    checkError(cuMemcpyDtoH(distance.data(), d_distance, G.numVertices * sizeof(int)),
               "cannot copy d_distance to host");
    checkError(cuMemcpyDtoH(parent.data(), d_parent, G.numVertices * sizeof(int)), "cannot copy d_parent to host");
}

void runCudaSimpleBfs(int startVertex, Graph &G, std::vector<int> &distance,
                      std::vector<int> &parent, float &kernelTimeUs)
{
    initializeCudaBfs(startVertex, distance, parent, G);

    int *changed;
    checkError(cuMemAllocHost((void **)&changed, sizeof(int)), "cannot allocate changed");

    // launch kernel
    printf("Starting simple parallel bfs.\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    *changed = 1;
    int level = 0;
    while (*changed)
    {
        *changed = 0;
        void *args[] = {&G.numVertices, &level, &d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_distance, &d_parent,
                        &changed};
        int gridSize = G.numVertices / NUM_THREADS + 1;
        printLaunchConfig("simpleBfs", gridSize, NUM_THREADS);
        checkError(cuLaunchKernel(cuSimpleBfs, G.numVertices / NUM_THREADS + 1, 1, 1,
                                  NUM_THREADS, 1, 1, 0, 0, args, 0),
                   "cannot run kernel simpleBfs");
        cuCtxSynchronize();
        level++;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    // printf("Kernel execution time: %.3f us\n", ms * 1000.0f);
    kernelTimeUs = ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    finalizeCudaBfs(distance, parent, G);
}

void runCudaQueueBfs(int startVertex, Graph &G, std::vector<int> &distance,
                     std::vector<int> &parent, float &kernelTimeUs)
{
    initializeCudaBfs(startVertex, distance, parent, G);

    int *nextQueueSize;
    checkError(cuMemAllocHost((void **)&nextQueueSize, sizeof(int)), "cannot allocate nextQueueSize");

    // launch kernel
    printf("Starting queue parallel bfs.\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int queueSize = 1;
    *nextQueueSize = 0;
    int level = 0;
    while (queueSize)
    {
        void *args[] = {&level, &d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_distance, &d_parent, &queueSize,
                        &nextQueueSize, &d_currentQueue, &d_nextQueue};
        int gridSize = queueSize / NUM_THREADS + 1;
        printLaunchConfig("queueBfs", gridSize, NUM_THREADS);
        checkError(cuLaunchKernel(cuQueueBfs, queueSize / NUM_THREADS + 1, 1, 1,
                                  NUM_THREADS, 1, 1, 0, 0, args, 0),
                   "cannot run kernel queueBfs");
        cuCtxSynchronize();
        level++;
        queueSize = *nextQueueSize;
        *nextQueueSize = 0;
        std::swap(d_currentQueue, d_nextQueue);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    // printf("Kernel execution time: %.3f us\n", ms * 1000.0f);
    kernelTimeUs = ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    finalizeCudaBfs(distance, parent, G);
}

void nextLayer(int level, int queueSize)
{
    void *args[] = {&level, &d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_distance, &d_parent, &queueSize,
                    &d_currentQueue};
    int gridSize = queueSize / NUM_THREADS + 1;
    printLaunchConfig("nextLayer", gridSize, NUM_THREADS);
    checkError(cuLaunchKernel(cuNextLayer, queueSize / NUM_THREADS + 1, 1, 1,
                              NUM_THREADS, 1, 1, 0, 0, args, 0),
               "cannot run kernel cuNextLayer");
    cuCtxSynchronize();
}

void countDegrees(int level, int queueSize)
{
    void *args[] = {&d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_parent, &queueSize,
                    &d_currentQueue, &d_degrees};
    int gridSize = queueSize / NUM_THREADS + 1;
    printLaunchConfig("countDegrees", gridSize, NUM_THREADS);
    checkError(cuLaunchKernel(cuCountDegrees, queueSize / NUM_THREADS + 1, 1, 1,
                              NUM_THREADS, 1, 1, 0, 0, args, 0),
               "cannot run kernel cuNextLayer");
    cuCtxSynchronize();
}

void fusedNextLayerCountDegrees(int level, int queueSize)
{
    void *args[] = {
        &d_currentQueue,
        &queueSize,
        &d_adjacencyList,
        &d_edgesOffset,
        &d_edgesSize,
        &d_distance,
        &d_parent,
        &d_degrees,
        &level};
    int gridSize = (queueSize + NUM_THREADS - 1) / NUM_THREADS;
    printLaunchConfig("fusedNextLayerCountDegrees", gridSize, NUM_THREADS);
    checkError(cuLaunchKernel(cuFusedNextCount,
                              gridSize, 1, 1,
                              NUM_THREADS, 1, 1,
                              0, 0, args, 0),
               "cannot run fusedNextLayerCountDegrees kernel");
}

void scanDegrees(int queueSize)
{
    // run kernel so every block in d_currentQueue has prefix sums calculated
    void *args[] = {&queueSize, &d_degrees, &incrDegrees};
    int gridSize = queueSize / NUM_THREADS + 1;
    printLaunchConfig("scanDegrees", gridSize, NUM_THREADS);
    checkError(cuLaunchKernel(cuScanDegrees, queueSize / NUM_THREADS + 1, 1, 1,
                              NUM_THREADS, 1, 1, 0, 0, args, 0),
               "cannot run kernel scanDegrees");
    cuCtxSynchronize();

    // count prefix sums on CPU for ends of blocks exclusive
    // already written previous block sum
    // launchScanDegrees((int *)d_degrees, (int *)incrDegrees, queueSize);
    incrDegrees[0] = 0;
    for (int i = 1024; i < queueSize + 1024; i += 1024)
    {
        incrDegrees[i / 1024] += incrDegrees[i / 1024 - 1];
    }
}

void assignVerticesNextQueue(int queueSize, int nextQueueSize)
{
    void *args[] = {&d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_parent, &queueSize, &d_currentQueue,
                    &d_nextQueue, &d_degrees, &incrDegrees, &nextQueueSize};
    int gridSize = queueSize / NUM_THREADS + 1;
    printLaunchConfig("assignVerticesNextQueue", gridSize, NUM_THREADS);
    checkError(cuLaunchKernel(cuAssignVerticesNextQueue, queueSize / NUM_THREADS + 1, 1, 1,
                              NUM_THREADS, 1, 1, 0, 0, args, 0),
               "cannot run kernel assignVerticesNextQueue");
    cuCtxSynchronize();
}

void launchScanBfsFusedKernel(
    int queueSize, int level,
    CUdeviceptr d_currentQueue,
    CUdeviceptr d_adjacencyList,
    CUdeviceptr d_edgesOffset,
    CUdeviceptr d_edgesSize,
    CUdeviceptr d_distance,
    CUdeviceptr d_parent,
    CUdeviceptr d_nextQueue,
    CUdeviceptr d_nextQueueSize)
{
    int blockSize = 256;
    size_t sharedMemBytes = (2 * blockSize + 64) * sizeof(int); // 动态共享内存大小
    int gridSize = (queueSize + blockSize - 1) / blockSize;

    void *args[] = {
        &d_currentQueue,
        &queueSize,
        &d_adjacencyList,
        &d_edgesOffset,
        &d_edgesSize,
        &d_distance,
        &d_parent,
        &d_nextQueue,
        &d_nextQueueSize,
        &level};

    printLaunchConfig("scanBfsFusedKernel", gridSize, blockSize);
    checkError(cuLaunchKernel(cuScanBfsFusedKernel,
                              gridSize, 1, 1,
                              blockSize, 1, 1,
                              sharedMemBytes,
                              0,
                              args,
                              0),
               "cannot run kernel scanBfsFusedKernel");
}

void runCudaScanBfs(int startVertex, Graph &G, std::vector<int> &distance,
                    std::vector<int> &parent, float &kernelTimeUs)
{
    initializeCudaBfs(startVertex, distance, parent, G);

    // launch kernel
    printf("Starting scan parallel bfs.\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int queueSize = 1;
    int nextQueueSize = 0;
    int level = 0;
    while (queueSize)
    {
        // next layer phase
        nextLayer(level, queueSize);
        // counting degrees phase
        countDegrees(level, queueSize);
        // doing scan on degrees
        scanDegrees(queueSize);
        nextQueueSize = incrDegrees[(queueSize - 1) / NUM_THREADS + 1];
        // assigning vertices to nextQueue
        assignVerticesNextQueue(queueSize, nextQueueSize);

        level++;
        queueSize = nextQueueSize;
        std::swap(d_currentQueue, d_nextQueue);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    // printf("Kernel execution time: %.3f us\n", ms * 1000.0f);
    kernelTimeUs = ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    finalizeCudaBfs(distance, parent, G);
}

void runCudaScanfusedBfs(int startVertex, Graph &G, std::vector<int> &distance,
                         std::vector<int> &parent, float &kernelTimeUs)
{
    initializeCudaBfs(startVertex, distance, parent, G);

    printf("Starting scanfused parallel bfs.\n");
    int queueSize = 1;
    int nextQueueSize = 0;
    int level = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    while (queueSize)
    {
        // 初始化 nextQueueSize = 0
        int zero = 0;
        checkError(cuMemcpyHtoD(d_nextQueueSizeDevice, &zero, sizeof(int)),
                   "reset d_nextQueueSizeDevice");

        // 启动融合 BFS 内核
        launchScanBfsFusedKernel(queueSize, level,
                                 d_currentQueue,
                                 d_adjacencyList,
                                 d_edgesOffset,
                                 d_edgesSize,
                                 d_distance,
                                 d_parent,
                                 d_nextQueue,
                                 d_nextQueueSizeDevice);

        // 等待 kernel 完成
        cudaDeviceSynchronize();

        // 获取本轮生成的新前沿队列大小
        cuMemcpyDtoH(&nextQueueSize, d_nextQueueSizeDevice, sizeof(int));

        // 准备下一轮
        std::swap(d_currentQueue, d_nextQueue);
        queueSize = nextQueueSize;
        level++;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTimeUs, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //  关键步骤：把 device 上的结果拷回 host
    cudaMemcpy(distance.data(), (void *)d_distance, sizeof(int) * G.numVertices, cudaMemcpyDeviceToHost);
    cudaMemcpy(parent.data(), (void *)d_parent, sizeof(int) * G.numVertices, cudaMemcpyDeviceToHost);

    // 输出调试信息
}

void launchScanBfsFusedKernel_WM(
    int queueSize, int level,
    CUdeviceptr d_currentQueue,
    CUdeviceptr d_adjacencyList,
    CUdeviceptr d_edgesOffset,
    CUdeviceptr d_edgesSize,
    CUdeviceptr d_distance,
    CUdeviceptr d_parent,
    CUdeviceptr d_nextQueue,
    CUdeviceptr d_nextQueueSize)
{
    int blockSize = 256;
    int warpsPerBlock = blockSize / 32;
    int numWarps = (queueSize + 31) / 32;
    int gridSize = (numWarps + warpsPerBlock - 1) / warpsPerBlock;
    int warpSize = 32;
    int maxWarpsPerBlock = blockSize / warpSize;

    size_t sharedMemBytes = sizeof(int) * (4 * maxWarpsPerBlock * warpSize + // vList, vStartOffsets, vDegrees, prefixDegrees
                                           2 * blockSize +                   // threadCounts, threadOffsets
                                           warpSize +                        // sharedOffset
                                           1                                 // sharedGlobalOffset
                                          );

    void *args[] = {
        &d_currentQueue,
        &queueSize,
        &d_adjacencyList,
        &d_edgesOffset,
        &d_edgesSize,
        &d_distance,
        &d_parent,
        &d_nextQueue,
        &d_nextQueueSize,
        &level};

    printLaunchConfig("scanBfsFusedKernel_WM", gridSize, blockSize);
    checkError(cuLaunchKernel(cuScanBfsFusedKernel_WM,
                              gridSize, 1, 1,
                              blockSize, 1, 1,
                              sharedMemBytes,
                              0,
                              args,
                              0),
               "cannot run kernel scanBfsFusedKernel_WM");
}

void runCudaScanfusedBfs_WM(int startVertex, Graph &G, std::vector<int> &distance,
                            std::vector<int> &parent, float &kernelTimeUs)
{
    initializeCudaBfs(startVertex, distance, parent, G);

    printf("Starting scanfused WM-style parallel BFS.\n");

    int queueSize = 1;
    int nextQueueSize = 0;
    int level = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    while (queueSize > 0)
    {
        int zero = 0;
        checkError(cuMemcpyHtoD(d_nextQueueSizeDevice, &zero, sizeof(int)),
                   "reset d_nextQueueSizeDevice");
        // printf("Level %d: queueSize = %d\n", level, queueSize);
        launchScanBfsFusedKernel_WM(queueSize, level,
                                    d_currentQueue,
                                    d_adjacencyList,
                                    d_edgesOffset,
                                    d_edgesSize,
                                    d_distance,
                                    d_parent,
                                    d_nextQueue,
                                    d_nextQueueSizeDevice);

        cudaDeviceSynchronize();

        checkError(cuMemcpyDtoH(&nextQueueSize, d_nextQueueSizeDevice, sizeof(int)),
                   "read d_nextQueueSizeDevice");

        std::swap(d_currentQueue, d_nextQueue);
        queueSize = nextQueueSize;
        level++;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTimeUs, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(distance.data(), (void *)d_distance,
               sizeof(int) * G.numVertices, cudaMemcpyDeviceToHost);
    cudaMemcpy(parent.data(), (void *)d_parent,
               sizeof(int) * G.numVertices, cudaMemcpyDeviceToHost);

    // std::cout << "distance[0-100]: " << std::endl;

    // for (int i = 0; i < 100; i++)
    // {
    //     std::cout << distance[i] << " ";
    // }
}

// void runGunrockBFS(int startVertex, Graph &G, std::vector<int> &distance)
// {
//     gunrock::graph::Csr<int, int, int> csr(false);
//     csr.NumberOfVertices() = G.numVertices;
//     csr.NumberOfEdges() = G.numEdges;
//     csr.row_offsets = G.edgesOffset.data();
//     csr.column_indices = G.adjacencyList.data();

//     gunrock::app::bfs::Enactor bfs;
//     bfs.Enact(csr, startVertex, distance.data());
// }

void launchScanBfsFusedKernel_CM(
    int queueSize, int level,
    CUdeviceptr d_currentQueue,
    CUdeviceptr d_adjacencyList,
    CUdeviceptr d_edgesOffset,
    CUdeviceptr d_edgesSize,
    CUdeviceptr d_distance,
    CUdeviceptr d_parent,
    CUdeviceptr d_nextQueue,
    CUdeviceptr d_nextQueueSize)
{
    int blockSize = 256;
    int gridSize = (queueSize + blockSize - 1) / blockSize;

    // 为每个 block 准备 3 倍 blockSize 的共享内存：vList, startOffsets, degrees
    size_t sharedMemBytes = 3 * blockSize * sizeof(int);

    void *args[] = {
        &d_currentQueue,  // inputQueue
        &queueSize,       // inputQueueSize
        &d_adjacencyList, // adjacencyList
        &d_edgesOffset,   // edgesOffset
        &d_edgesSize,     // edgesSize
        &d_nextQueue,     // nextQueue
        &d_nextQueueSize, // nextQueueSize
        &d_parent,        // parent
        &d_distance,      // distance
        &level            // level
    };

    printLaunchConfig("scanBfsFusedKernel_CM", gridSize, blockSize);
    checkError(cuLaunchKernel(cuScanBfsFusedKernel_CM,
                              gridSize, 1, 1,
                              blockSize, 1, 1,
                              sharedMemBytes,
                              0,
                              args,
                              0),
               "cannot run kernel scanBfsFusedKernel_CM");
}

void runCudaScanfusedBfs_CM(int startVertex, Graph &G, std::vector<int> &distance,
                            std::vector<int> &parent, float &kernelTimeUs)
{
    initializeCudaBfs(startVertex, distance, parent, G);

    printf("Starting scanfused CM-style parallel BFS.\n");
    int queueSize = 1;
    int nextQueueSize = 0;
    int level = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    while (queueSize)
    {
        int zero = 0;
        checkError(cuMemcpyHtoD(d_nextQueueSizeDevice, &zero, sizeof(int)),
                   "reset d_nextQueueSizeDevice");

        launchScanBfsFusedKernel_CM(queueSize, level,
                                    d_currentQueue,
                                    d_adjacencyList,
                                    d_edgesOffset,
                                    d_edgesSize,
                                    d_distance,
                                    d_parent,
                                    d_nextQueue,
                                    d_nextQueueSizeDevice);

        cudaDeviceSynchronize();

        cuMemcpyDtoH(&nextQueueSize, d_nextQueueSizeDevice, sizeof(int));

        std::swap(d_currentQueue, d_nextQueue);
        queueSize = nextQueueSize;
        level++;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTimeUs, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(distance.data(), (void *)d_distance, sizeof(int) * G.numVertices, cudaMemcpyDeviceToHost);
    cudaMemcpy(parent.data(), (void *)d_parent, sizeof(int) * G.numVertices, cudaMemcpyDeviceToHost);
}

void runCudaGunrockBfs(int startVertex, Graph &G, std::vector<int> &distance,
                       std::vector<int> &parent, float &kernelTimeUs)
{
    initializeCudaBfs(startVertex, distance, parent, G);

    int *d_frontier, *d_next_frontier;
    int *d_frontier_size, *d_next_frontier_size;

    // 最大可能大小分配
    size_t max_size = G.numVertices * sizeof(int);
    checkError(cuMemAlloc((CUdeviceptr *)&d_frontier, max_size), "alloc frontier");
    checkError(cuMemAlloc((CUdeviceptr *)&d_next_frontier, max_size), "alloc next frontier");
    checkError(cuMemAlloc((CUdeviceptr *)&d_frontier_size, sizeof(int)), "alloc frontier size");
    checkError(cuMemAlloc((CUdeviceptr *)&d_next_frontier_size, sizeof(int)), "alloc next frontier size");

    // 初始化 frontier 为 startVertex
    int zero = 0;
    int one = 1;
    checkError(cuMemcpyHtoD((CUdeviceptr)d_frontier, &startVertex, sizeof(int)), "copy start vertex");
    checkError(cuMemcpyHtoD((CUdeviceptr)d_frontier_size, &one, sizeof(int)), "init frontier size");

    printf("Starting gunrock-style bfs.\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int level = 0;
    while (true)
    {
        // 清空 next frontier size
        checkError(cuMemcpyHtoD((CUdeviceptr)d_next_frontier_size, &zero, sizeof(int)), "clear next size");

        void *args[] = {
            &d_frontier, &d_frontier_size,
            &d_adjacencyList, &d_edgesOffset, &d_edgesSize,
            &d_distance, &d_parent,
            &d_next_frontier, &d_next_frontier_size,
            &level};

        checkError(cuLaunchKernel(cuGunrockStyleBfsKernel,
                                  G.numVertices / NUM_THREADS + 1, 1, 1,
                                  NUM_THREADS, 1, 1, 0, 0, args, 0),
                   "kernel launch failed"); //(G.numVertices + 255) / 256  G.numVertices / 1024 + 1

        checkError(cuCtxSynchronize(), "sync");

        // 拷贝 next frontier size 回来判断是否继续
        int host_next_size = 0;
        checkError(cuMemcpyDtoH(&host_next_size, (CUdeviceptr)d_next_frontier_size, sizeof(int)), "get next size");

        if (host_next_size == 0)
            break;

        // 交换 frontier
        std::swap(d_frontier, d_next_frontier);
        std::swap(d_frontier_size, d_next_frontier_size);
        level++;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    // printf("Kernel execution time: %.3f us\n", ms * 1000.0f);
    kernelTimeUs = ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    finalizeCudaBfs(distance, parent, G);

    // 清理
    cuMemFree((CUdeviceptr)d_frontier);
    cuMemFree((CUdeviceptr)d_next_frontier);
    cuMemFree((CUdeviceptr)d_frontier_size);
    cuMemFree((CUdeviceptr)d_next_frontier_size);
}

int main(int argc, char **argv)
{
    std::string datasetName = "unknown";
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg.size() >= 4 && arg.substr(arg.size() - 4) == ".mtx")
        {
            // 提取最后一个 '/' 后的文件名
            size_t lastSlash = arg.find_last_of("/\\");
            std::string filename = (lastSlash == std::string::npos) ? arg : arg.substr(lastSlash + 1);
            size_t dotPos = filename.find_last_of(".");
            if (dotPos != std::string::npos)
            {
                datasetName = filename.substr(0, dotPos); // 去掉 .mtx
            }
            else
            {
                datasetName = filename;
            }
            break;
        }
    }
    std::string csvFilename = "bfs_" + datasetName + ".csv";
    std::ofstream csvFile(csvFilename);
    csvFile << "Algorithm,KernelTime(ms),EndToEndTime(ms)\n";

    Graph G;
    int startVertex = atoi(argv[1]);
    auto start_total_load = std::chrono::high_resolution_clock::now();
    readGraph(G, argc, argv);
    auto end_total_load = std::chrono::high_resolution_clock::now();
    double total_time_ms_load = std::chrono::duration<double, std::milli>(end_total_load - start_total_load).count();

    std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
    std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());
    std::vector<bool> visited(G.numVertices, false);

    runCpu(startVertex, G, distance, parent, visited);
    std::vector<int> expectedDistance(distance);
    std::vector<int> expectedParent(parent);

    initCuda(G);

    float kernel_time = 0.0f;
    auto start_total = std::chrono::high_resolution_clock::now();
    runCudaSimpleBfs(startVertex, G, distance, parent, kernel_time);

    auto end_total = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    std::cout << "SimpleBFS Kernel execution time: " << kernel_time << " ms\n";
    std::cout << "SimpleBFS End-to-end time: " << total_time_ms << " ms\n\n";
    csvFile << "SimpleBFS," << kernel_time << "," << total_time_ms << "\n";
    checkOutput(distance, expectedDistance, G);

    // runVC_CM_BFS(G, startVertex);

    // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // kernel_time = 0.0f;
    // start_total = std::chrono::high_resolution_clock::now();
    // runCudaQueueBfs(startVertex, G, distance, parent, kernel_time);
    // end_total = std::chrono::high_resolution_clock::now();
    // total_time_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    // std::cout << "QueueBFS Kernel execution time: " << kernel_time << " ms\n";
    // std::cout << "QueueBFS End-to-end time: " << total_time_ms << " ms\n\n";
    // csvFile << "QueueBFS," << kernel_time << "," << total_time_ms << "\n";
    // checkOutput(distance, expectedDistance, G);

    // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // kernel_time = 0.0f;
    // start_total = std::chrono::high_resolution_clock::now();
    // runCudaScanBfs(startVertex, G, distance, parent, kernel_time);
    // end_total = std::chrono::high_resolution_clock::now();
    // total_time_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    // std::cout << "ScanBFS Kernel execution time: " << kernel_time << " ms\n";
    // std::cout << "ScanBFS End-to-end time: " << total_time_ms << " ms\n\n";
    // csvFile << "ScanBFS," << kernel_time << "," << total_time_ms << "\n";
    // checkOutput(distance, expectedDistance, G);

    // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // kernel_time = 0.0f;
    // start_total = std::chrono::high_resolution_clock::now();
    // runCudaScanfusedBfs(startVertex, G, distance, parent, kernel_time);
    // end_total = std::chrono::high_resolution_clock::now();
    // total_time_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    // std::cout << "ScanBFSfused Kernel execution time: " << kernel_time << " ms\n";
    // std::cout << "ScanBFSfused End-to-end time: " << total_time_ms << " ms\n\n";
    // csvFile << "ScanBFSfused," << kernel_time << "," << total_time_ms << "\n";
    // checkOutput(distance, expectedDistance, G);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    kernel_time = 0.0f;
    start_total = std::chrono::high_resolution_clock::now();
    runCudaScanfusedBfs_WM(startVertex, G, distance, parent, kernel_time);
    end_total = std::chrono::high_resolution_clock::now();
    total_time_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    std::cout << "ScanBFSfused_WM Kernel execution time: " << kernel_time << " ms\n";
    std::cout << "ScanBFSfused_WM End-to-end time: " << total_time_ms << " ms\n\n";
    csvFile << "ScanBFSfused_WM," << kernel_time << "," << total_time_ms << "\n";
    checkOutput(distance, expectedDistance, G);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    kernel_time = 0.0f;
    start_total = std::chrono::high_resolution_clock::now();
    runCudaScanfusedBfs_CM(startVertex, G, distance, parent, kernel_time);
    end_total = std::chrono::high_resolution_clock::now();
    total_time_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    std::cout << "ScanBFSfused_CM Kernel execution time: " << kernel_time << " ms\n";
    std::cout << "ScanBFSfused_CM End-to-end time: " << total_time_ms << " ms\n\n";
    csvFile << "ScanBFSfused_CM," << kernel_time << "," << total_time_ms << "\n";
    // std::cout << "expectedDistance[0-100]: " << std::endl;
    //  for (int i = 0; i < 100; i++)
    //  {
    //      std::cout << expectedDistance[i] << " ";
    //  }
    //  printf("\n");
    checkOutput(distance, expectedDistance, G);

    // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // kernel_time = 0.0f;
    // start_total = std::chrono::high_resolution_clock::now();
    // runCudaGunrockBfs(startVertex, G, distance, parent, kernel_time);
    // checkOutput(distance, expectedDistance, G);
    // end_total = std::chrono::high_resolution_clock::now();
    // total_time_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    // std::cout << "GunrockBFS Kernel execution time: " << kernel_time << " ms\n";
    // std::cout << "GunrockBFS End-to-end time: " << total_time_ms + total_time_ms_load << " ms\n\n";
    // csvFile << "GunrockBFS," << kernel_time << "," << total_time_ms + total_time_ms_load << "\n";

    finalizeCuda();
    // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // kernel_time = 0.0f;
    // std::cout << "Starting Gswitch bfs." << std::endl;
    // start_total = std::chrono::high_resolution_clock::now();
    // kernel_time = runGswitchbfs(argc, argv);
    // end_total = std::chrono::high_resolution_clock::now();
    // total_time_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    // std::cout << "GswitchBFS End-to-end time: " << total_time_ms << " ms\n";
    // csvFile << "GswitchBFS," << kernel_time << "," << total_time_ms << "\n";

    csvFile.close();
    return 0;
}