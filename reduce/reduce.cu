#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// reduce 核函数（优化版）
__global__ void reduceSum(int *input, int *output, int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int x = (i < n) ? input[i] : 0;
    int y = (i + blockDim.x < n) ? input[i + blockDim.x] : 0;
    sdata[tid] = x + y;
    __syncthreads();

    // 归约（tree-reduction）
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// host 端封装
void benchmarkReduce(int N, int blockSize = 256) {
    int *h_input = new int[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1;

    int *d_input, *d_output;
    int numBlocks = (N + blockSize * 2 - 1) / (blockSize * 2);
    cudaMalloc(&d_input, sizeof(int) * N);
    cudaMalloc(&d_output, sizeof(int) * numBlocks);
    cudaMemcpy(d_input, h_input, sizeof(int) * N, cudaMemcpyHostToDevice);

    // 创建时间事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduceSum<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Input size: " << N << ", blocks: " << numBlocks
              << ", time: " << ms << " ms" << std::endl;

    // 清理
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char* argv[]) {
    // 设定不同输入规模（2^10 到 2^22）
    int sizes[] = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    int blocks[] = {128, 256, 512};

    std::cout << "InputSize,BlockSize,KernelTime(ms)" << std::endl;

    for (int i = 0; i < sizeof(sizes) / sizeof(int); ++i) {
        for (int j = 0; j < sizeof(blocks) / sizeof(int); ++j) {
            benchmarkReduce(sizes[i], blocks[j]);
        }
    }

    return 0;
}
