#include <iostream>
#include <cuda_runtime.h>

__global__ void reduceSum(int *input, int *output, int n) {
    extern __shared__ int sdata[];

    // 每个线程的全局索引
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 将两个元素累加，减少访问次数
    sdata[tid] = (i < n ? input[i] : 0) + (i + blockDim.x < n ? input[i + blockDim.x] : 0);
    __syncthreads();

    // 进行归约操作（线程数缩减为1）
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // 将每个 block 的部分和写回
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

void reduce_host(int *input, int N) {
    int *d_input, *d_output;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    cudaMalloc(&d_input, N * sizeof(int));
    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, blocksPerGrid * sizeof(int));

    reduceSum<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_input, d_output, N);

    int *h_output = new int[blocksPerGrid];
    cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

    // 最终归约
    int sum = 0;
    for (int i = 0; i < blocksPerGrid; ++i)
        sum += h_output[i];

    std::cout << "Reduced sum: " << sum << std::endl;

    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    const int N = 1 << 20; // 1048576
    int *h_input = new int[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1;

    reduce_host(h_input, N);
    delete[] h_input;
    return 0;
}
