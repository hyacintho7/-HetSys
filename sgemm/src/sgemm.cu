/*
    A -- [M, K]
    B -- [K, N]
    C -- [M, N] = A * B
*/

#include "sgemm.cuh"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cublas_v2.h>
#include <random>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(float_var) (reinterpret_cast<float4 *>(&(float_var))[0])
#define CUDA_CHECK(call)                                             \
    do                                                               \
    {                                                                \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess)                                      \
        {                                                            \
            printf("CUDA Error: \n");                                \
            printf("    File:       %s\n", __FILE__);                \
            printf("    Line:       %d\n", __LINE__);                \
            printf("    Error Code: %d\n", err);                     \
            printf("    Error Text: %s\n", cudaGetErrorString(err)); \
            exit(1);                                                 \
        }                                                            \
    } while (0)
#define CUBLAS_CHECK(call)                                              \
    do                                                                  \
    {                                                                   \
        cublasStatus_t err = call;                                      \
        if (err != CUBLAS_STATUS_SUCCESS)                               \
        {                                                               \
            printf("cuBLAS Error: \n");                                 \
            printf("    File:       %s\n", __FILE__);                   \
            printf("    Line:       %d\n", __LINE__);                   \
            printf("    Error Code: %d\n", err);                        \
            printf("    Error Text: %s\n", cublasGetStatusString(err)); \
            exit(1);                                                    \
        }                                                               \
    } while (0)

void data_init(float *data, const int num)
{
    std::uniform_real_distribution<float> float_gen(-1.0f, 1.0f);
    std::default_random_engine rand_engine(time(nullptr));
    for (int i = 0; i < num; i++)
    {
        data[i] = float_gen(rand_engine);
    }
}

class TotalTimer
{
    using Clock = std::chrono::high_resolution_clock;

private:
    Clock::time_point m_start_point, m_end_point;

public:
    void start() { m_start_point = Clock::now(); };
    void end() { m_end_point = Clock::now(); };
    float cost()
    {
        std::chrono::duration<float, std::milli> dur =
            m_end_point - m_start_point;
        return dur.count();
    };
};

class KernelTimer
{
private:
    cudaEvent_t m_start_event, m_end_event;

public:
    KernelTimer()
    {
        CUDA_CHECK(cudaEventCreate(&m_start_event));
        CUDA_CHECK(cudaEventCreate(&m_end_event));
    };
    ~KernelTimer()
    {
        CUDA_CHECK(cudaEventDestroy(m_start_event));
        CUDA_CHECK(cudaEventDestroy(m_end_event));
    };
    void start() { CUDA_CHECK(cudaEventRecord(m_start_event)); };
    void end()
    {
        CUDA_CHECK(cudaEventRecord(m_end_event));
        CUDA_CHECK(cudaEventSynchronize(m_end_event));
    };
    float cost()
    {
        float kernel_cost;
        CUDA_CHECK(
            cudaEventElapsedTime(&kernel_cost, m_start_event, m_end_event));
        return kernel_cost;
    };
};

float test_error(SgemmFunc func)
{
    const int M = 512, N = 1024, K = 128;

    float *A, *B, *C1, *C2;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    A = (float *)malloc(size_A);
    B = (float *)malloc(size_B);
    C1 = (float *)malloc(size_C);
    C2 = (float *)malloc(size_C);

    data_init(A, M * K);
    data_init(B, K * N);

    sgemm_cpu(A, B, C1, M, N, K);
    func(A, B, C2, M, N, K);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++)
    {
        float this_error = std::abs(C1[i] - C2[i]);
        max_error = std::max(max_error, this_error);
    }

    free(A);
    free(B);
    free(C1);
    free(C2);

    return max_error;
}

Performance test_performance(SgemmFunc func, const int M, const int N,
                             const int K, const int test_num)
{
    float *A, *B, *C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    A = (float *)malloc(size_A);
    B = (float *)malloc(size_B);
    C = (float *)malloc(size_C);
    data_init(A, M * K);
    data_init(B, K * N);

    CostTime avg_cost_time;
    for (int i = 0; i < test_num; i++)
    {
        CostTime cost_time = func(A, B, C, M, N, K);
        avg_cost_time.total += cost_time.total;
        avg_cost_time.kernel += cost_time.kernel;
    }
    avg_cost_time.total /= test_num;
    avg_cost_time.kernel /= test_num;
    float flops = 2.0f * M * N * K / (avg_cost_time.kernel / 1e3);

    Performance performance;
    performance.cost_time = avg_cost_time;
    performance.tflops = flops / 1e12;

    free(A);
    free(B);
    free(C);

    return performance;
}

CostTime sgemm_cpu(float *A, float *B, float *C, const int M, const int N,
                   const int K)
{
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float value = 0.0f;
            for (int k = 0; k < K; k++)
            {
                value += A[OFFSET(m, k, K)] * B[OFFSET(k, n, N)];
            }
            C[OFFSET(m, n, N)] = value;
        }
    }

    total_timer.end();
    cost_time.total = total_timer.cost();
    cost_time.kernel = cost_time.total;
    return cost_time;
}

__global__ void sgemm_gpu_kernel_v1(float *__restrict__ A,
                                    float *__restrict__ B,
                                    float *__restrict__ C, const int M,
                                    const int N, const int K)
{
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;
    for (int k = 0; k < K; k++)
    {
        value += A[OFFSET(m, k, K)] * B[OFFSET(k, n, N)];
    }
    C[OFFSET(m, n, N)] = value;
}

CostTime sgemm_gpu_v1(float *A, float *B, float *C, const int M, const int N,
                      const int K)
{
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    const int BM = 16, BN = 16; // 受线程块最大线程数限制

    assert(M % BM == 0 && N % BN == 0); // 核函数不处理边界情况
    const dim3 block_size(BN, BM);
    const dim3 grid_size(N / BN, M / BM);

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    sgemm_gpu_kernel_v1<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

__global__ void sgemm_gpu_kernel_v2(float *__restrict__ A,
                                    float *__restrict__ B,
                                    float *__restrict__ C, const int M,
                                    const int N, const int K)
{
    const int BM = 16, BN = 16;
    const int BK = 64;
    __shared__ float s_a[BM][BK], s_b[BK][BN];
    float c = 0.0f;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // 每次从全局内存加载到共享内存，每个线程都负责一个float4。以下是当前线程负责的这个float4的索引
    const int row_s_a = tid / 16;
    const int col_s_a = (tid % 16) * 4;
    const int row_s_b = tid / 4;
    const int col_s_b = (tid % 4) * 4;
    // 每个线程从读取的全局内存的位置，在A上的行是固定不变的，在B上列是固定不变的
    const int row_A = blockIdx.y * BM + row_s_a;
    const int col_B = blockIdx.x * BN + col_s_b;

    for (int step = 0; step < K / BK; step++)
    {
        // 从A加载到s_a
        const int col_A = step * BK + col_s_a;
        const int index_A = OFFSET(row_A, col_A, K);
        FETCH_FLOAT4(s_a[row_s_a][col_s_a]) = FETCH_FLOAT4(A[index_A]);
        // 从B加载到s_b
        const int row_B = step * BK + row_s_b;
        const int index_B = OFFSET(row_B, col_B, N);
        FETCH_FLOAT4(s_b[row_s_b][col_s_b]) = FETCH_FLOAT4(B[index_B]);
        __syncthreads();
        // 计算
        for (int k = 0; k < BK; k++)
        {
            const float a = s_a[threadIdx.y][k];
            const float b = s_b[k][threadIdx.x];
            c += a * b;
        }
        __syncthreads();
    }
    // 写入C
    const int row_C = blockIdx.y * BM + threadIdx.y;
    const int col_C = blockIdx.x * BN + threadIdx.x;
    const int index_C = OFFSET(row_C, col_C, N);
    C[index_C] = c;
}

CostTime sgemm_gpu_v2(float *A, float *B, float *C, const int M, const int N,
                      const int K)
{
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    const int BM = 16, BN = 16; // 受线程块最大线程数限制
    // 理论上其大小不影响计算速度。为了每个线程刚好加载一个float4
    const int BK = 64;

    assert(M % BM == 0 && N % BN == 0 && K % BK == 0); // 核函数不处理边界情况
    const dim3 block_size(BN, BM);
    const dim3 grid_size(N / BN, M / BM);

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    sgemm_gpu_kernel_v2<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

__global__ void sgemm_gpu_kernel_v3(float *__restrict__ A,
                                    float *__restrict__ B,
                                    float *__restrict__ C, const int M,
                                    const int N, const int K)
{
    const int TM = 8, TN = 8;
    const int BM = 128, BN = 128;
    const int BK = 8;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];
    float r_a[TM];
    float r_b[TN];
    float r_c[TM][TN] = {0.0f};

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // 每次从全局内存加载到共享内存，每个线程都负责一个float4。以下是当前线程负责的这个float4的索引
    const int row_s_a = tid / 2;
    const int col_s_a = (tid % 2) * 4;
    const int row_s_b = tid / 32;
    const int col_s_b = (tid % 32) * 4;
    // 每个线程从读取的全局内存的位置，在A上的行是固定不变的，在B上列是固定不变的
    const int row_A = blockIdx.y * BM + row_s_a;
    const int col_B = blockIdx.x * BN + col_s_b;

    for (int step = 0; step < K / BK; step++)
    {
        // 从A加载到s_a
        const int col_A = step * BK + col_s_a;
        const int index_A = OFFSET(row_A, col_A, K);
        FETCH_FLOAT4(s_a[row_s_a][col_s_a]) = FETCH_FLOAT4(A[index_A]);
        // 从B加载到s_b
        const int row_B = step * BK + row_s_b;
        const int index_B = OFFSET(row_B, col_B, N);
        FETCH_FLOAT4(s_b[row_s_b][col_s_b]) = FETCH_FLOAT4(B[index_B]);
        __syncthreads();
        for (int k = 0; k < BK; k++)
        {
            // 从s_a加载到r_a
            const int row_start = threadIdx.y * TM;
            for (int i = 0; i < TM; i++)
            {
                r_a[i] = s_a[row_start + i][k];
            }
            // 从s_b加载到r_b
            const int col_start = threadIdx.x * TN;
            for (int i = 0; i < TN; i++)
            {
                r_b[i] = s_b[k][col_start + i];
            }
            // 计算
            for (int m = 0; m < TM; m++)
            {
                for (int n = 0; n < TN; n++)
                {
                    r_c[m][n] += r_a[m] * r_b[n];
                }
            }
        }
        __syncthreads();
    }

    // 写入C
    for (int m = 0; m < TM; m++)
    {
        for (int n = 0; n < TN; n += 4)
        {
            const int row = blockIdx.y * BM + threadIdx.y * TM + m;
            const int col = blockIdx.x * BN + threadIdx.x * TN + n;
            const int index_C = OFFSET(row, col, N);
            FETCH_FLOAT4(C[index_C]) = FETCH_FLOAT4(r_c[m][n]);
        }
    }
}

CostTime sgemm_gpu_v3(float *A, float *B, float *C, const int M, const int N,
                      const int K)
{
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    const int TM = 8, TN = 8;     // 受线程最大寄存器数限制
    const int BM = 128, BN = 128; // 受线程块最大线程数限制
    // 理论上其大小不影响计算速度。为了每个线程刚好加载一个float4
    const int BK = 8;

    assert(M % BM == 0 && N % BN == 0 && K % BK == 0); // 核函数不处理边界情况
    const dim3 block_size(BN / TN, BM / TM);
    const dim3 grid_size((N + BN - 1) / BN, (M + BM - 1) / BM);

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    sgemm_gpu_kernel_v3<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

__global__ void sgemm_gpu_kernel_v3_16_8(float *__restrict__ A,
                                         float *__restrict__ B,
                                         float *__restrict__ C,
                                         const int M, const int N, const int K)
{
    const int TM = 16, TN = 8;
    const int BM = 128, BN = 128;
    const int BK = 8;

    __shared__ float s_a[BM][BK]; // 128 x 8
    __shared__ float s_b[BK][BN]; // 8 x 128

    float r_a[TM];
    float r_b[TN];
    float r_c[TM][TN] = {0.0f};

    const int thread_row = threadIdx.y * TM;
    const int thread_col = threadIdx.x * TN;

    const int global_row = blockIdx.y * BM + thread_row;
    const int global_col = blockIdx.x * BN + thread_col;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int step = 0; step < K / BK; step++)
    {
        // 每个线程加载2个float4 → 2x4 = 8个float，总共128线程共加载1024 float
        for (int load = 0; load < 2; load++)
        {
            int t = tid * 2 + load;

            if (t < (BM * BK / 4)) // 128 * 8 / 4 = 256
            {
                int row = (t * 4) / BK;
                int col = (t * 4) % BK;
                int global_a_row = blockIdx.y * BM + row;
                int global_a_col = step * BK + col;
                int index_a = OFFSET(global_a_row, global_a_col, K);
                FETCH_FLOAT4(s_a[row][col]) = FETCH_FLOAT4(A[index_a]);
            }

            if (t < (BN * BK / 4)) // 128 * 8 / 4 = 256
            {
                int row = (t * 4) / BN;
                int col = (t * 4) % BN;
                int global_b_row = step * BK + row;
                int global_b_col = blockIdx.x * BN + col;
                int index_b = OFFSET(global_b_row, global_b_col, N);
                FETCH_FLOAT4(s_b[row][col]) = FETCH_FLOAT4(B[index_b]);
            }
        }

        __syncthreads();

        for (int k = 0; k < BK; ++k)
        {
            for (int i = 0; i < TM; ++i)
                r_a[i] = s_a[thread_row + i][k];

            for (int j = 0; j < TN; ++j)
                r_b[j] = s_b[k][thread_col + j];

            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    r_c[i][j] += r_a[i] * r_b[j];
        }

        __syncthreads();
    }

    // 写回 C，每线程写回 r_c[16][8]
    for (int i = 0; i < TM; ++i)
    {
        for (int j = 0; j < TN; j += 4)
        {
            int row = global_row + i;
            int col = global_col + j;
            int index_c = OFFSET(row, col, N);
            FETCH_FLOAT4(C[index_c]) = FETCH_FLOAT4(r_c[i][j]);
        }
    }
}

CostTime sgemm_gpu_v3_16_8(float *A, float *B, float *C, const int M, const int N,
                           const int K)
{
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    const int TM = 16, TN = 8;    // 受线程最大寄存器数限制
    const int BM = 128, BN = 128; // 受线程块最大线程数限制
    // 理论上其大小不影响计算速度。为了每个线程刚好加载一个float4
    const int BK = 8;

    assert(M % BM == 0 && N % BN == 0 && K % BK == 0); // 核函数不处理边界情况
    const dim3 block_size(BN / TN, BM / TM);
    const dim3 grid_size((N + BN - 1) / BN, (M + BM - 1) / BM);

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    sgemm_gpu_kernel_v3_16_8<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

__global__ void sgemm_gpu_kernel_v3_8_16(float *__restrict__ A,
                                         float *__restrict__ B,
                                         float *__restrict__ C,
                                         const int M, const int N, const int K)
{
    const int TM = 8, TN = 16;
    const int BM = 128, BN = 128;
    const int BK = 8;

    __shared__ float s_a[BM][BK]; // 128 x 8
    __shared__ float s_b[BK][BN]; // 8 x 128

    float r_a[TM];
    float r_b[TN];
    float r_c[TM][TN] = {0.0f};

    const int thread_row = threadIdx.y * TM;
    const int thread_col = threadIdx.x * TN;

    const int global_row = blockIdx.y * BM + thread_row;
    const int global_col = blockIdx.x * BN + thread_col;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int step = 0; step < K / BK; ++step)
    {
        // 每线程加载2个 float4，共 128*2 = 256 个 float4 = 1024 float
        for (int load = 0; load < 2; ++load)
        {
            int t = tid * 2 + load;

            // ---- 加载 A ----
            if (t < (BM * BK / 4)) // 128×8 / 4 = 256 float4
            {
                int row = (t * 4) / BK;
                int col = (t * 4) % BK;
                int global_a_row = blockIdx.y * BM + row;
                int global_a_col = step * BK + col;
                int index_a = OFFSET(global_a_row, global_a_col, K);
                FETCH_FLOAT4(s_a[row][col]) = FETCH_FLOAT4(A[index_a]);
            }

            // ---- 加载 B ----
            if (t < (BK * BN / 4)) // 8×128 / 4 = 256 float4
            {
                int row = (t * 4) / BN;
                int col = (t * 4) % BN;
                int global_b_row = step * BK + row;
                int global_b_col = blockIdx.x * BN + col;
                int index_b = OFFSET(global_b_row, global_b_col, N);
                FETCH_FLOAT4(s_b[row][col]) = FETCH_FLOAT4(B[index_b]);
            }
        }

        __syncthreads();

        for (int k = 0; k < BK; ++k)
        {
            for (int i = 0; i < TM; ++i)
                r_a[i] = s_a[thread_row + i][k];

            for (int j = 0; j < TN; ++j)
                r_b[j] = s_b[k][thread_col + j];

            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    r_c[i][j] += r_a[i] * r_b[j];
        }

        __syncthreads();
    }

    // 写回 C
    for (int i = 0; i < TM; ++i)
    {
        for (int j = 0; j < TN; j += 4)
        {
            int row = global_row + i;
            int col = global_col + j;
            int index_c = OFFSET(row, col, N);
            FETCH_FLOAT4(C[index_c]) = FETCH_FLOAT4(r_c[i][j]);
        }
    }
}

CostTime sgemm_gpu_v3_8_16(float *A, float *B, float *C, const int M, const int N,
                           const int K)
{
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    const int TM = 8, TN = 16;    // 受线程最大寄存器数限制
    const int BM = 128, BN = 128; // 受线程块最大线程数限制
    // 理论上其大小不影响计算速度。为了每个线程刚好加载一个float4
    const int BK = 8;

    assert(M % BM == 0 && N % BN == 0 && K % BK == 0); // 核函数不处理边界情况
    const dim3 block_size(BN / TN, BM / TM);
    const dim3 grid_size((N + BN - 1) / BN, (M + BM - 1) / BM);

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    sgemm_gpu_kernel_v3_8_16<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

__global__ void sgemm_wmma_kernel(const __half *A, const __half *B, float *C, int M, int N, int K)
{
    // 确定 tile 坐标（以 WMMA_M/N 为粒度）
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    // 每个 warp 处理一个 tile
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // 支持多个 warp 每个 block（这里只分配一个 warp per block 处理一个 tile）
    int global_row = tile_row * WMMA_M;
    int global_col = tile_col * WMMA_N;

    // 累加 fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // 处理所有 K tile
    for (int tile_k = 0; tile_k < K; tile_k += WMMA_K)
    {
        // 定义 A/B 的 fragment
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;

        // 计算 global memory 中 A 和 B tile 的起始地址
        const __half *tile_ptr_A = A + global_row * K + tile_k;
        const __half *tile_ptr_B = B + tile_k * N + global_col;

        // 从 global memory 加载 A/B tile 到 fragment
        wmma::load_matrix_sync(a_frag, tile_ptr_A, K);
        wmma::load_matrix_sync(b_frag, tile_ptr_B, N);

        // 执行 FMA 操作
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 写回 C
    float *out_ptr = C + global_row * N + global_col;
    wmma::store_matrix_sync(out_ptr, c_frag, N, wmma::mem_row_major);
}

CostTime sgemm_gpu_wmma(float *A, float *B, float *C, const int M, const int N, const int K)
{
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    // ========== 1. 检查维度合法性（必须为16的倍数） ==========
    assert(M % 16 == 0 && N % 16 == 0 && K % 16 == 0);

    // ========== 2. 分配 GPU 内存 ==========
    __half *d_A_half, *d_B_half;
    float *d_C;
    size_t size_A_half = M * K * sizeof(__half);
    size_t size_B_half = K * N * sizeof(__half);
    size_t size_C = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A_half, size_A_half));
    CUDA_CHECK(cudaMalloc(&d_B_half, size_B_half));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // ========== 3. 主机侧 float 转 half ==========
    std::vector<__half> h_A_half(M * K), h_B_half(K * N);
    for (int i = 0; i < M * K; ++i)
        h_A_half[i] = __float2half(A[i]);
    for (int i = 0; i < K * N; ++i)
        h_B_half[i] = __float2half(B[i]);

    // ========== 4. 拷贝到 GPU ==========
    CUDA_CHECK(cudaMemcpy(d_A_half, h_A_half.data(), size_A_half, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_half, h_B_half.data(), size_B_half, cudaMemcpyHostToDevice));

    // ========== 5. 启动 kernel ==========
    dim3 gridDim(N / 16, M / 16);
    dim3 blockDim(32); // 一个 warp per block

    KernelTimer kernel_timer;
    kernel_timer.start();

    sgemm_wmma_kernel<<<gridDim, blockDim>>>(d_A_half, d_B_half, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    // ========== 6. 拷回 C ==========
    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    // ========== 7. 清理 ==========
    CUDA_CHECK(cudaFree(d_A_half));
    CUDA_CHECK(cudaFree(d_B_half));
    CUDA_CHECK(cudaFree(d_C));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

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

CostTime sgemm_gpu_mma_ptx(float *A, float *B, float *C, const int M, const int N, const int K)
{
    assert(M % 8 == 0 && N % 8 == 0 && K % 4 == 0);

    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    // ========== 1. 分配 Device 内存 ========== //
    __half *d_A_half, *d_B_half;
    float *d_C;
    size_t size_A_half = M * K * sizeof(__half);
    size_t size_B_half = K * N * sizeof(__half);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc(&d_A_half, size_A_half);
    cudaMalloc(&d_B_half, size_B_half);
    cudaMalloc(&d_C, size_C);

    // ========== 2. Host 侧 float -> half ========== //
    std::vector<__half> h_A_half(M * K);
    std::vector<__half> h_B_half(K * N);
    for (int i = 0; i < M * K; ++i)
        h_A_half[i] = __float2half(A[i]);
    for (int i = 0; i < K * N; ++i)
        h_B_half[i] = __float2half(B[i]);

    // ========== 3. 拷贝到 Device ========== //
    cudaMemcpy(d_A_half, h_A_half.data(), size_A_half, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_half, h_B_half.data(), size_B_half, cudaMemcpyHostToDevice);

    // ========== 4. 启动 Kernel ========== //
    int num_warps = (M / TILE_M) * (N / TILE_N);
    int threads_per_block = 32;
    int blocks = (num_warps + threads_per_block - 1) / threads_per_block;
    dim3 gridDim(blocks, 1);
    // dim3 gridDim(N / 8, M / 8);
    // dim3 blockDim(32); // 一个 warp per block

    KernelTimer kernel_timer;
    kernel_timer.start();
    mma_gemm_kernel<<<gridDim, threads_per_block>>>(d_A_half, d_B_half, d_C, M, N, K);
    cudaDeviceSynchronize();
    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    // ========== 5. 结果拷回 Host ========== //
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // ========== 6. 清理 ========== //
    cudaFree(d_A_half);
    cudaFree(d_B_half);
    cudaFree(d_C);
    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

__global__ void sgemm_gpu_kernel_v4(float *__restrict__ A,
                                    float *__restrict__ B,
                                    float *__restrict__ C, const int M,
                                    const int N, const int K)
{
    const int TM = 8, TN = 8;
    const int BM = 128, BN = 128;
    const int BK = 8;

    __shared__ float s_a[BK][BM]; // 相比v3，s_a改为列优先
    __shared__ float s_b[BK][BN];
    float r_a[TM];
    float r_b[TN];
    float r_c[TM][TN] = {0.0f};

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // 每次从全局内存加载到共享内存，每个线程都负责一个float4。以下是当前线程负责的这个float4的索引
    const int row_s_a = tid / 2;
    const int col_s_a = (tid % 2) * 4;
    const int row_s_b = tid / 32;
    const int col_s_b = (tid % 32) * 4;
    // 每个线程从读取的全局内存的位置，在A上的行是固定不变的，在B上列是固定不变的
    const int row_A = blockIdx.y * BM + row_s_a;
    const int col_B = blockIdx.x * BN + col_s_b;

    for (int step = 0; step < K / BK; step++)
    {
        // 从A加载到s_a
        const int col_A = step * BK + col_s_a;
        const int index_A = OFFSET(row_A, col_A, K);
        FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(A[index_A]); // 借用r_a[0]中转
        s_a[col_s_a + 0][row_s_a] = r_a[0];
        s_a[col_s_a + 1][row_s_a] = r_a[1];
        s_a[col_s_a + 2][row_s_a] = r_a[2];
        s_a[col_s_a + 3][row_s_a] = r_a[3];
        // 从B加载到s_b
        const int row_B = step * BK + row_s_b;
        const int index_B = OFFSET(row_B, col_B, N);
        FETCH_FLOAT4(s_b[row_s_b][col_s_b]) = FETCH_FLOAT4(B[index_B]);
        __syncthreads();
        for (int k = 0; k < BK; k++)
        {
            // 从s_a加载到r_a
            const int row_start = threadIdx.y * TM;
            FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(s_a[k][row_start]);
            FETCH_FLOAT4(r_a[4]) = FETCH_FLOAT4(s_a[k][row_start + 4]);
            // 从s_b加载到r_b
            const int col_start = threadIdx.x * TN;
            FETCH_FLOAT4(r_b[0]) = FETCH_FLOAT4(s_b[k][col_start]);
            FETCH_FLOAT4(r_b[4]) = FETCH_FLOAT4(s_b[k][col_start + 4]);
            // 计算
            for (int m = 0; m < TM; m++)
            {
                for (int n = 0; n < TN; n++)
                {
                    r_c[m][n] += r_a[m] * r_b[n];
                }
            }
        }
        __syncthreads();
    }

    // 写入C
    for (int m = 0; m < TM; m++)
    {
        for (int n = 0; n < TN; n += 4)
        {
            const int row = blockIdx.y * BM + threadIdx.y * TM + m;
            const int col = blockIdx.x * BN + threadIdx.x * TN + n;
            const int index_C = OFFSET(row, col, N);
            FETCH_FLOAT4(C[index_C]) = FETCH_FLOAT4(r_c[m][n]);
        }
    }
}

CostTime sgemm_gpu_v4(float *A, float *B, float *C, const int M, const int N,
                      const int K)
{
    // 除核函数，其他相比v3没有改动
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    const int TM = 8, TN = 8;     // 受线程最大寄存器数限制
    const int BM = 128, BN = 128; // 受线程块最大线程数限制
    // 理论上其大小不影响计算速度。为了每个线程刚好加载一个float4
    const int BK = 8;

    assert(M % BM == 0 && N % BN == 0 && K % BK == 0); // 核函数不处理边界情况
    const dim3 block_size(BN / TN, BM / TM);
    const dim3 grid_size((N + BN - 1) / BN, (M + BM - 1) / BM);

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    sgemm_gpu_kernel_v4<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

__global__ void sgemm_gpu_kernel_v5(float *__restrict__ A,
                                    float *__restrict__ B,
                                    float *__restrict__ C, const int M,
                                    const int N, const int K)
{
    const int TM = 8, TN = 8;
    const int BM = 128, BN = 128;
    const int BK = 8;

    __shared__ float s_a[BK][BM]; // 相比v3，s_a改为列优先
    __shared__ float s_b[BK][BN];
    float r_a[TM];
    float r_b[TN];
    float r_c[TM][TN] = {0.0f};

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // 每次从全局内存加载到共享内存，每个线程都负责一个float4。以下是负责的这个float4的索引
    const int row_s_a = tid / 2;
    const int col_s_a = (tid % 2) * 4;
    const int row_s_b = tid / 32;
    const int col_s_b = (tid % 32) * 4;
    // 每个线程从读取的全局内存的位置，在A上的行是固定不变的，在B上列是固定不变的
    const int row_A = blockIdx.y * BM + row_s_a;
    const int col_B = blockIdx.x * BN + col_s_b;

    for (int step = 0; step < K / BK; step++)
    {
        // 从A加载到s_a
        const int col_A = step * BK + col_s_a;
        const int index_A = OFFSET(row_A, col_A, K);
        FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(A[index_A]); // 借用r_a[0]中转
        s_a[col_s_a + 0][row_s_a] = r_a[0];
        s_a[col_s_a + 1][row_s_a] = r_a[1];
        s_a[col_s_a + 2][row_s_a] = r_a[2];
        s_a[col_s_a + 3][row_s_a] = r_a[3];
        // 从B加载到s_b
        const int row_B = step * BK + row_s_b;
        const int index_B = OFFSET(row_B, col_B, N);
        FETCH_FLOAT4(s_b[row_s_b][col_s_b]) = FETCH_FLOAT4(B[index_B]);
        __syncthreads();
        for (int k = 0; k < BK; k++)
        {
            // 从s_a加载到r_a
            const int row_start = threadIdx.y * TM;
            FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(s_a[k][row_start]);
            FETCH_FLOAT4(r_a[4]) = FETCH_FLOAT4(s_a[k][row_start + 4]);
            // 从s_b加载到r_b，相比v4，这里读取的位置变了
            const int col_start = threadIdx.x * (TN / 2);
            FETCH_FLOAT4(r_b[0]) = FETCH_FLOAT4(s_b[k][col_start]);
            FETCH_FLOAT4(r_b[4]) = FETCH_FLOAT4(s_b[k][col_start + BN / 2]);
            // 计算
            for (int m = 0; m < TM; m++)
            {
                for (int n = 0; n < TN; n++)
                {
                    r_c[m][n] += r_a[m] * r_b[n];
                }
            }
        }
        __syncthreads();
    }

    // 写入C，相比v4，写入位置也变了，因为操作的数据位置变了
    for (int m = 0; m < TM; m++)
    {
        const int row = blockIdx.y * BM + threadIdx.y * TM + m;
        const int col1 = blockIdx.x * BN + threadIdx.x * (TN / 2);
        const int col2 = blockIdx.x * BN + threadIdx.x * (TN / 2) + BN / 2;
        const int index1_C = OFFSET(row, col1, N);
        const int index2_C = OFFSET(row, col2, N);
        FETCH_FLOAT4(C[index1_C]) = FETCH_FLOAT4(r_c[m][0]);
        FETCH_FLOAT4(C[index2_C]) = FETCH_FLOAT4(r_c[m][4]);
    }
}

CostTime sgemm_gpu_v5(float *A, float *B, float *C, const int M, const int N,
                      const int K)
{
    // 除核函数，其他相比v3没有改动
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    const int TM = 8, TN = 8;     // 受线程最大寄存器数限制
    const int BM = 128, BN = 128; // 受线程块最大线程数限制
    // 理论上其大小不影响计算速度。为了每个线程刚好加载一个float4
    const int BK = 8;

    assert(M % BM == 0 && N % BN == 0 && K % BK == 0); // 核函数不处理边界情况
    const dim3 block_size(BN / TN, BM / TM);
    const dim3 grid_size((N + BN - 1) / BN, (M + BM - 1) / BM);

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    sgemm_gpu_kernel_v5<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

__global__ void sgemm_gpu_kernel_v6(float *__restrict__ A,
                                    float *__restrict__ B,
                                    float *__restrict__ C, const int M,
                                    const int N, const int K)
{
    const int TM = 8, TN = 8;
    const int BM = 128, BN = 128;
    const int BK = 8;

    // 相比v5，s_a, s_b变为double buffer
    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];
    float r_a[TM];
    float r_b[TN];
    float r_c[TM][TN] = {0.0f};

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // 每次从全局内存加载到共享内存，每个线程都负责一个float4。以下是负责的这个float4的索引
    const int row_s_a = tid / 2;
    const int col_s_a = (tid % 2) * 4;
    const int row_s_b = tid / 32;
    const int col_s_b = (tid % 32) * 4;
    // 每个线程从读取的全局内存的位置，在A上的行是固定不变的，在B上列是固定不变的
    const int row_A = blockIdx.y * BM + row_s_a;
    const int col_B = blockIdx.x * BN + col_s_b;

    // 第一次加载  全局 --> 共享
    // 从A加载到s_a
    const int step = 0;
    const int col_A = step * BK + col_s_a;
    const int index_A = OFFSET(row_A, col_A, K);
    FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(A[index_A]); // 借用r_a[0]中转
    s_a[0][col_s_a + 0][row_s_a] = r_a[0];
    s_a[0][col_s_a + 1][row_s_a] = r_a[1];
    s_a[0][col_s_a + 2][row_s_a] = r_a[2];
    s_a[0][col_s_a + 3][row_s_a] = r_a[3];
    // 从B加载到s_b
    const int row_B = step * BK + row_s_b;
    const int index_B = OFFSET(row_B, col_B, N);
    FETCH_FLOAT4(s_b[0][row_s_b][col_s_b]) = FETCH_FLOAT4(B[index_B]);
    __syncthreads();

    for (int step = 1; step < K / BK; step++)
    {
        const int lbi = step % 2; // load_buffer_index
        // 加载下一次迭代需要的  全局 --> 共享
        // 从A加载到s_a
        const int col_A = step * BK + col_s_a;
        const int index_A = OFFSET(row_A, col_A, K);
        FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(A[index_A]); // 借用r_a[0]中转
        s_a[lbi][col_s_a + 0][row_s_a] = r_a[0];
        s_a[lbi][col_s_a + 1][row_s_a] = r_a[1];
        s_a[lbi][col_s_a + 2][row_s_a] = r_a[2];
        s_a[lbi][col_s_a + 3][row_s_a] = r_a[3];
        // 从B加载到s_b
        const int row_B = step * BK + row_s_b;
        const int index_B = OFFSET(row_B, col_B, N);
        FETCH_FLOAT4(s_b[lbi][row_s_b][col_s_b]) = FETCH_FLOAT4(B[index_B]);
        // 相比v5，此处不再需要同步。因为加载的数据本轮迭代用不到
        // __syncthreads();

        // 使用上一次加载的做运算
        const int cbi = (step - 1) % 2; // compute_buffer_index
        for (int k = 0; k < BK; k++)
        {
            // 从s_a加载到r_a
            const int row_start = threadIdx.y * TM;
            FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(s_a[cbi][k][row_start]);
            FETCH_FLOAT4(r_a[4]) = FETCH_FLOAT4(s_a[cbi][k][row_start + 4]);
            // 从s_b加载到r_b，相比v4，这里读取的位置变了
            const int col_start = threadIdx.x * (TN / 2);
            FETCH_FLOAT4(r_b[0]) = FETCH_FLOAT4(s_b[cbi][k][col_start]);
            FETCH_FLOAT4(r_b[4]) =
                FETCH_FLOAT4(s_b[cbi][k][col_start + BN / 2]);
            // 计算
            for (int m = 0; m < TM; m++)
            {
                for (int n = 0; n < TN; n++)
                {
                    r_c[m][n] += r_a[m] * r_b[n];
                }
            }
        }
        __syncthreads();
    }

    // 补充最后一次加载到共享内存的数据对应的计算
    const int cbi = (K / BK - 1) % 2; // compute_buffer_index
    for (int k = 0; k < BK; k++)
    {
        // 从s_a加载到r_a
        const int row_start = threadIdx.y * TM;
        FETCH_FLOAT4(r_a[0]) = FETCH_FLOAT4(s_a[cbi][k][row_start]);
        FETCH_FLOAT4(r_a[4]) = FETCH_FLOAT4(s_a[cbi][k][row_start + 4]);
        // 从s_b加载到r_b，相比v4，这里读取的位置变了
        const int col_start = threadIdx.x * (TN / 2);
        FETCH_FLOAT4(r_b[0]) = FETCH_FLOAT4(s_b[cbi][k][col_start]);
        FETCH_FLOAT4(r_b[4]) = FETCH_FLOAT4(s_b[cbi][k][col_start + BN / 2]);
        // 计算
        for (int m = 0; m < TM; m++)
        {
            for (int n = 0; n < TN; n++)
            {
                r_c[m][n] += r_a[m] * r_b[n];
            }
        }
    }

    // 写入C，相比v4，写入位置也变了，因为操作的数据位置变了
    for (int m = 0; m < TM; m++)
    {
        const int row = blockIdx.y * BM + threadIdx.y * TM + m;
        const int col1 = blockIdx.x * BN + threadIdx.x * (TN / 2);
        const int col2 = blockIdx.x * BN + threadIdx.x * (TN / 2) + BN / 2;
        const int index1_C = OFFSET(row, col1, N);
        const int index2_C = OFFSET(row, col2, N);
        FETCH_FLOAT4(C[index1_C]) = FETCH_FLOAT4(r_c[m][0]);
        FETCH_FLOAT4(C[index2_C]) = FETCH_FLOAT4(r_c[m][4]);
    }
}

CostTime sgemm_gpu_v6(float *A, float *B, float *C, const int M, const int N,
                      const int K)
{
    // 除核函数，其他相比v3没有改动
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    const int TM = 8, TN = 8;     // 受线程最大寄存器数限制
    const int BM = 128, BN = 128; // 受线程块最大线程数限制
    // 理论上其大小不影响计算速度。为了每个线程刚好加载一个float4
    const int BK = 8;

    assert(M % BM == 0 && N % BN == 0 && K % BK == 0); // 核函数不处理边界情况
    const dim3 block_size(BN / TN, BM / TM);
    const dim3 grid_size((N + BN - 1) / BN, (M + BM - 1) / BM);

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    KernelTimer kernel_timer;
    kernel_timer.start();

    sgemm_gpu_kernel_v6<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

CostTime sgemm_cublas(float *A, float *B, float *C, const int M, const int N,
                      const int K)
{
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float *d_A, *d_B, *d_C;
    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    float cublas_alpha = 1.0;
    float cublas_beta = 0;

    KernelTimer kernel_timer;
    kernel_timer.start();

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_B,
                N, d_A, K, &cublas_beta, d_C, N);

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(handle));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}

CostTime sgemm_cublas_tensorcore(float *A, float *B, float *C, const int M, const int N, const int K)
{
    CostTime cost_time;
    TotalTimer total_timer;
    total_timer.start();

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // === Host to device, with half precision input ===
    __half *d_A_half, *d_B_half;
    float *d_C;
    size_t size_A_half = M * K * sizeof(__half);
    size_t size_B_half = K * N * sizeof(__half);
    size_t size_C = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A_half, size_A_half));
    CUDA_CHECK(cudaMalloc(&d_B_half, size_B_half));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Convert input float arrays to half on host
    std::vector<__half> h_A_half(M * K), h_B_half(K * N);
    for (int i = 0; i < M * K; ++i)
        h_A_half[i] = __float2half(A[i]);
    for (int i = 0; i < K * N; ++i)
        h_B_half[i] = __float2half(B[i]);

    CUDA_CHECK(cudaMemcpy(d_A_half, h_A_half.data(), size_A_half, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_half, h_B_half.data(), size_B_half, cudaMemcpyHostToDevice));

    float alpha = 1.0f;
    float beta = 0.0f;

    KernelTimer kernel_timer;
    kernel_timer.start();

    // === Tensor Core GEMM ===
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              d_B_half, CUDA_R_16F, N,
                              d_A_half, CUDA_R_16F, K,
                              &beta,
                              d_C, CUDA_R_32F, N,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    kernel_timer.end();
    cost_time.kernel = kernel_timer.cost();

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A_half));
    CUDA_CHECK(cudaFree(d_B_half));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));

    total_timer.end();
    cost_time.total = total_timer.cost();

    return cost_time;
}
