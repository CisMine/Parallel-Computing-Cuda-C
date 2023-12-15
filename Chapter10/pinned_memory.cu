#include <cassert>
#include <iostream>


using std::cout;
using std::end;


__global__ void vectorAdd(int *a, int *b, int *c, int N)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

void verify_result(int *a, int *b, int *c, int N)
{
    for (int i = 0; i < N; i++)
    {
        assert(c[i] == a[i] + b[i]);
    }
}

int main()
{
    constexpr int N = 100;
    size_t bytes = sizeof(int) * N;

    // Vectors for holding the host-side (CPU-side) data
    int *h_a, *h_b, *h_c;

    // Allocate pinned memory
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);

    for (int i = 0; i < N; i++)
    {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    // Allocate memory on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

   

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

 


    int NUM_THREADS = 1 << 10;
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;



    // Execute the kernel
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

  

   

    // Copy data from device to host (GPU -> CPU)
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);




    // Check result for errors
    verify_result(h_a, h_b, h_c, N);

    // Free pinned memory
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cout << "COMPLETED SUCCESSFULLY\n";

    return 0;
}
