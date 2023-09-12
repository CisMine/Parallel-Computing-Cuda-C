#include <stdio.h>
#include <stdlib.h>

// Size of the vector
#define N 100

// CUDA kernel to add two vectors
__global__ void vectorAdd(int *a, int *b, int *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int *h_a, *h_b, *h_c; // Host vectors
    int *d_a, *d_b, *d_c; // Device vectors

    // Initialize host vectors
    h_a = (int *)malloc(N * sizeof(int));
    h_b = (int *)malloc(N * sizeof(int));
    h_c = (int *)malloc(N * sizeof(int));

    // Initialize host vectors with random values
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() % 10;
        h_b[i] = rand() % 10;
    }

    // Allocate device memory for vectors
    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_b, N * sizeof(int));
    cudaMalloc((void **)&d_c, N * sizeof(int));

    // Copy data from CPU to GPU
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Call the CUDA kernel to perform vector addition
    vectorAdd<<<2, 50>>>(d_a, d_b, d_c);

    // Copy the result from GPU to CPU
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < N; i++) {
        printf("h_a[%d] %d + h_b[%d] %d = %d\n", i, h_a[i], i, h_b[i], h_c[i]);
    }

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
