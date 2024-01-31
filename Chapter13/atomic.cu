#include <stdio.h>

#define NUM_THREADS 10
#define NUM_BLOCKS 2
#define ARRAY_SIZE 20 

__global__ void AtomicAdd(int *result, int *array_add)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    atomicAdd(result, array_add[tid]);

    // if (threadIdx.x == 0)
    // {
    //     atomicAdd(result, array_add[tid]);
    // }
}

__global__ void AtomicSub(int *result, int *array_sub)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    atomicSub(result, array_sub[tid]);

    // if (threadIdx.x == 0)
    // {
    //     atomicSub(result, array_sub[tid]);
    // }
}

__global__ void AtomicMax(int *result, int *array_max)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    atomicMax(result, array_max[tid]);

    // if (threadIdx.x == 0)
    // {
    //     atomicMax(result, array_max[tid]);
    // }
}

__global__ void AtomicMin(int *result, int *array_min)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    atomicMin(result, array_min[tid]);

    // if (threadIdx.x == 0)
    // {
    //     atomicMin(result, array_min[tid]);
    // }
}

int main()
{
    int *h_data = (int *)malloc(ARRAY_SIZE * sizeof(int));
    int *d_data;
    cudaMalloc((void **)&d_data, ARRAY_SIZE * sizeof(int));

    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        h_data[i] = i;
    }

    //------------ atomicAdd-------------
    int *d_result_add;
    cudaMalloc((void **)&d_result_add, sizeof(int));
    int h_result_add = 0;

    //------------ atomicSub-------------
    int *d_result_sub;
    cudaMalloc((void **)&d_result_sub, sizeof(int));
    int h_result_sub = 0;

    //------------ atomicMax-------------
    int *d_result_max;
    cudaMalloc((void **)&d_result_max, sizeof(int));
    int h_result_max = 0;

    //------------ atomicMin-------------
    int *d_result_min;
    cudaMalloc((void **)&d_result_min, sizeof(int));
    int h_result_min = 0;

    cudaMemcpy(d_data, h_data, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    AtomicAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_result_add, d_data);
    AtomicSub<<<NUM_BLOCKS, NUM_THREADS>>>(d_result_sub, d_data);
    AtomicMax<<<NUM_BLOCKS, NUM_THREADS>>>(d_result_max, d_data);
    AtomicMin<<<NUM_BLOCKS, NUM_THREADS>>>(d_result_min, d_data);

    cudaMemcpy(&h_result_add, d_result_add, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_result_sub, d_result_sub, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_result_max, d_result_max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_result_min, d_result_min, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Atomic Add Result: %d\n", h_result_add);
    printf("Atomic Sub Result: %d\n", h_result_sub);
    printf("Atomic Max Result: %d\n", h_result_max);
    printf("Atomic Min Result: %d\n", h_result_min);

    free(h_data);

    cudaFree(d_result_add);
    cudaFree(d_result_sub);
    cudaFree(d_result_max);
    cudaFree(d_result_min);
    cudaFree(d_data);

    return 0;
}
