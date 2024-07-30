#include "test_gpu_access.h"

// KERNEL: Add two numbers
__global__ void Add_A_and_B(float *d_a, float *d_b, float *d_c)
{
    d_c[0] = d_a[0] + d_b[0];
}

void test_gpu_access_CUDA(double *cc)
{
    cc[0] = 0;         // Initialize output to 0
    cudaError_t error; // Initialie CUDA Error

    error = cudaSetDevice(0); // choose which GPU to run on
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return;
    }

    float *h_a = new float[0];
    h_a[0] = 3;

    float *h_b = new float[0];
    h_b[0] = 7;

    float *d_a, *d_b, *d_c;
    size_t sz = sizeof(float);

    error = cudaMalloc(&d_a, sz);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return;
    }

    error = cudaMalloc(&d_b, sz);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return;
    }

    error = cudaMalloc(&d_c, sz);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return;
    }

    error = cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return;
    }

    error = cudaMemcpy(d_b, h_b, sz, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return;
    }

    Add_A_and_B<<<1, 1>>>(d_a, d_b, d_c);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return;
    }
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return;
    }

    float *h_c = new float[0];
    error = cudaMemcpy(h_c, d_c, sz, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return;
    }

    if (h_c[0] != 10)
    {
        cudaDeviceReset();
        return;
    }

    int *val = new int[0];
    error = cudaDeviceGetAttribute(val, cudaDevAttrComputeCapabilityMinor, 0);
    if (error != cudaSuccess)
    {
        cudaDeviceReset();
        return;
    }

    cc[0] = (double)val[0];
    error = cudaDeviceGetAttribute(val, cudaDevAttrComputeCapabilityMajor, 0);
    cc[0] = (cc[0]) / 10 + (double)val[0];

    cudaDeviceReset();
    return;
}
