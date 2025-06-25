#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include <cub/cub.cuh>

bool check(float *out, float *res)
{
    if (abs(*out - *res) > 0.05)
        return false;
    return true;
}

int main()
{
    const int N = 32 * 1024 * 1024;
    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    float *out = (float *)malloc(sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, sizeof(float));
    float *res = (float *)malloc(sizeof(float));
    *res = 0.0f;

    for (int i = 0; i < N; i++)
    {
        a[i] = 2.0 * (float)drand48() - 1.0;
        *res += a[i];
    }

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes,
        d_a, d_out, N,
        cub::Sum(), 0.0f);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes,
        d_a, d_out, N,
        cub::Sum(), 0.0f);

    cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    if (check(out, res))
        printf("The answer is right.\n");
    else
    {
        printf("The answer is wrong.\n");
        printf("GPU result = %f, CPU result = %f\n", *out, *res);
    }

    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(d_temp_storage);

    free(a);
    free(out);
    free(res);
}
