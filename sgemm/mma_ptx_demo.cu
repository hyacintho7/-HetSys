#include <iostream>
#include <cstdint>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_M 8
#define TILE_N 8
#define TILE_K 4

__global__ void mma_gemm_kernel(const __half *A, const __half *B, float *C, int M, int N, int K)
{
    int warp_id = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // warp tile coordinates
    int tile_m = warp_id / (N / TILE_N);
    int tile_n = warp_id % (N / TILE_N);

    if (tile_m * TILE_M >= M || tile_n * TILE_N >= N)
        return;

    float c_frag[8] = {0.f}; // 8 element accumulator (8×8 tile, row-major flattened)

    for (int tile_k = 0; tile_k < K; tile_k += TILE_K)
    {
        const __half *a_tile = A + tile_m * TILE_M * K + tile_k;
        const __half *b_tile = B + tile_k * N + tile_n * TILE_N;

        uint32_t a_frag[2], b_frag[2];
        a_frag[0] = *(const uint32_t *)(a_tile);
        a_frag[1] = *(const uint32_t *)(a_tile + 2);
        b_frag[0] = *(const uint32_t *)(b_tile);
        b_frag[1] = *(const uint32_t *)(b_tile + 2);

        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, "
            "{%8,%9}, "
            "{%10,%11}, "
            "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
            : "+f"(c_frag[0]), "+f"(c_frag[1]), "+f"(c_frag[2]), "+f"(c_frag[3]),
              "+f"(c_frag[4]), "+f"(c_frag[5]), "+f"(c_frag[6]), "+f"(c_frag[7])
            : "r"(a_frag[0]), "r"(a_frag[1]),
              "r"(b_frag[0]), "r"(b_frag[1]));
    }

    // write back to global memory
    float *c_ptr = C + tile_m * TILE_M * N + tile_n * TILE_N;
    for (int i = 0; i < 8; ++i)
    {
        c_ptr[i] = c_frag[i]; // 每行顺序写回（flattened）
    }
}

void launch_tensor_gemm(int M, int N, int K)
{
    std::vector<__half> h_A(M * K, __float2half(1.0f));
    std::vector<__half> h_B(K * N, __float2half(1.0f));
    std::vector<float> h_C(M * N, 0.0f);

    __half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(__half));
    cudaMalloc(&d_B, K * N * sizeof(__half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice);

    int num_warps = (M / TILE_M) * (N / TILE_N);
    int threads_per_block = 32;
    int blocks = (num_warps + threads_per_block - 1) / threads_per_block;

    dim3 gridDim(blocks, 1);
    mma_gemm_kernel<<<gridDim, threads_per_block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Sample C[0:8]: ";
    for (int i = 0; i < 8; ++i)
        std::cout << h_C[i] << " ";
    std::cout << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    int M = 12288, N = 12288, K = 1024; // 尽量为 TILE 的倍数
    launch_tensor_gemm(M, N, K);
    return 0;
}
