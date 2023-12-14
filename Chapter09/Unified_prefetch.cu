// //todo: unified memory prefetch
//! unified memory la async ?

#include <stdio.h>
#include <cassert>
#include <iostream>

using std::cout;


__global__ void vectorAdd(int *a, int *b, int *c, int N)
{
    
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main()
{
    const int N = 1 << 16;
    size_t bytes = N * sizeof(int);

    int *a, *b, *c;

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Get the device ID for prefetching calls
    int id = cudaGetDevice(&id);

    // Set some hints about the data and do some prefetching
    cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, id);

    // Initialize vectors
    for (int i = 0; i < N; i++)
    {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    // Pre-fetch 'a' and 'b' arrays to the specified device (GPU)
    cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);


    int BLOCK_SIZE = 1 << 10;

    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);

  
    cudaDeviceSynchronize();
    

    // Prefetch to the host (CPU)
    cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    // Verify the result on the CPU
    for (int i = 0; i < N; i++)
    {
        assert(c[i] == a[i] + b[i]);
    }

    // Free unified memory (same as memory allocated with cudaMalloc)
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    cout << "COMPLETED SUCCESSFULLY!\n";

    return 0;
}
