#include <stdio.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE 4


__global__ void sum(int *d_array)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int stride = 1; stride < 4; stride *= 2)
    {
        __syncthreads();

        if (threadIdx.x % (2 * stride) == 0)
        {
            d_array[id] += d_array[id + stride];
        }
    }
    printf("blockIdx.x=%d --> %d\n", blockIdx.x, d_array[id]);
}

int main()
{
    int h_array[4] = {1, 2, 3, 4};
    int *d_array;

   cudaMalloc((void **)&d_array, sizeof(int) * ARRAY_SIZE);
   cudaMemcpy(d_array, h_array, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);

    sum<<<1, 4>>>(d_array);

    cudaFree(d_array);

    return 0;
}

