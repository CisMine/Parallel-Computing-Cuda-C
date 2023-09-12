#include <stdio.h>

__constant__ int constantData[2]; // Declaration of Constant memory array

__global__ void kernel(int *d_x, int *d_y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        int x = d_x[idx];
        int a = constantData[0]; // Retrieve the value 3 from Constant memory
        int b = constantData[1]; // Retrieve the value 5 from Constant memory
        d_y[idx] = a * x + b;
    }
}

int main() {
    const int N = 10; // Number of array elements
    int h_x[N];    // Input array on the host
    int h_y[N];    // Result array on the host
    int *d_x, *d_y; // Arrays on the device

    // Initialize data on the host
    for (int i = 0; i < N; i++) {
        h_x[i] = i;
    }

    // Allocate memory for arrays on the GPU
    cudaMalloc((void**)&d_x, N * sizeof(int));
    cudaMalloc((void**)&d_y, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_x, h_x, N * sizeof(int), cudaMemcpyHostToDevice);

    // Copy the values 3 and 5 into Constant memory
    int constantValues[2] = {3, 5};
    cudaMemcpyToSymbol(constantData, constantValues, 2 * sizeof(int));

    // Launch the kernel with 1 block and N threads
    kernel<<<1, N>>>(d_x, d_y, N);
    cudaDeviceSynchronize();

    // Copy the results from the device to the host
    cudaMemcpy(h_y, d_y, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < N; i++) {
        printf("3(x= %d) + 5 => y = %d\n", h_x[i], h_y[i]);
    }

    // Free memory on the device
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
