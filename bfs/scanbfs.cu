#include <cuda.h>
#include <cub/cub.cuh>
extern CUdeviceptr d_degrees;
void my_scanDegrees(int queueSize)
{
    static void *d_temp_storage = nullptr;
    static size_t temp_storage_bytes = 0;

    // workspace allocation (一次性)
    if (temp_storage_bytes == 0)
    {
        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
                                      (int *)d_degrees, (int *)d_degrees, queueSize);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }

    // 执行 device 上的 scan：in-place，结果覆盖在 d_degrees 上
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  (int *)d_degrees, (int *)d_degrees, queueSize);
}