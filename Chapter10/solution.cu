#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE (16 * 1024 * 1024)



void pageableMemoryTest() {
    float *h_data, *d_data;
    h_data = (float *)malloc(SIZE * sizeof(float));
    cudaMalloc((void **)&d_data, SIZE * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Host to Device
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Pageable - Host to Device: %f ms\n", milliseconds);

    // Device to Host
    cudaEventRecord(start);
    cudaMemcpy(h_data, d_data, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Pageable - Device to Host: %f ms\n", milliseconds);

    free(h_data);
    cudaFree(d_data);
}

void pinnedMemoryTest() {
    float *h_data, *d_data;
    cudaMallocHost((void **)&h_data, SIZE * sizeof(float));
    cudaMalloc((void **)&d_data, SIZE * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Host to Device
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Pinned - Host to Device: %f ms\n", milliseconds);

    // Device to Host
    cudaEventRecord(start);
    cudaMemcpy(h_data, d_data, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Pinned - Device to Host: %f ms\n", milliseconds);

    cudaFreeHost(h_data);
    cudaFree(d_data);
}

int main() {
    printf("Running pageable memory test...\n");
    pageableMemoryTest();

    printf("\nRunning pinned memory test...\n");
    pinnedMemoryTest();

    return 0;
}

