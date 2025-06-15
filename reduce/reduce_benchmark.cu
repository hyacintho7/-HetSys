#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <fstream> 

// Warp-level reduction using shuffle
template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (blockSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (blockSize >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (blockSize >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (blockSize >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

// Main reduction kernel with shuffle instructions
template <unsigned int blockSize>
__global__ void reduce7(float *d_in, float *d_out, unsigned int n) {
    float sum = 0;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    // Each thread loads one element from global memory to shared memory
    while (i < n) {
        sum += d_in[i] + d_in[i + blockSize];
        i += gridSize;
    }

    // Shared memory for partial sums (one per warp in the block)
    static __shared__ float warpLevelSums[32];  // Maximum number of warps (assuming warp size = 32)
    const int laneId = threadIdx.x % 32;
    const int warpId = threadIdx.x / 32;

    // Perform warp-level reduction using shuffle
    sum = warpReduceSum<blockSize>(sum);

    // Store the result from each warp in shared memory
    if (laneId == 0) warpLevelSums[warpId] = sum;
    __syncthreads();

    // Final reduction across warps in the block
    sum = (threadIdx.x < blockDim.x / 32) ? warpLevelSums[laneId] : 0;

    // Perform final reduction on the first warp
    if (warpId == 0) sum = warpReduceSum<blockSize / 32>(sum);

    // Write the result for this block to global memory
    if (tid == 0) d_out[blockIdx.x] = sum;
}

// Host function to benchmark the reduction
void benchmarkReduce(int N, int blockSize, std::ofstream &outfile) {
    int *h_input = new int[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1;

    float *d_input, *d_output;
    int numBlocks = (N + blockSize * 2 - 1) / (blockSize * 2);
    cudaMalloc(&d_input, sizeof(int) * N);
    cudaMalloc(&d_output, sizeof(float) * numBlocks);
    cudaMemcpy(d_input, h_input, sizeof(int) * N, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce7<256><<<numBlocks, blockSize>>>(d_input, d_output, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // 输出到标准输出
    std::cout << "Input size: " << N << ", blocks: " << numBlocks
              << ", time: " << ms << " ms" << std::endl;

    // 输出到CSV文件
    outfile << N << "," << blockSize << "," << ms << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main(int argc, char* argv[]) {
    int sizes[] = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22};
    int blocks[] = {128, 256, 512};

    std::ofstream outfile("reduce_benchmark.csv");
    outfile << "InputSize,BlockSize,KernelTime(ms)" << std::endl;

    for (int i = 0; i < sizeof(sizes) / sizeof(int); ++i) {
        for (int j = 0; j < sizeof(blocks) / sizeof(int); ++j) {
            benchmarkReduce(sizes[i], blocks[j], outfile);
        }
    }

    outfile.close();
    return 0;
}

