#include <stdio.h>
#include <stdlib.h>

__global__ void kernel() {
    int temp = 0;
    temp = threadIdx.x;

    printf("blockId %d ThreadIdx %d = %d\n",blockIdx.x,threadIdx.x,temp);
    
}

int main() {
    kernel<<<5,5>>>();
    cudaDeviceSynchronize();

    return 0;
}


