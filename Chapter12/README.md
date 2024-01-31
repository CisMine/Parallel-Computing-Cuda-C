



When we talk about parallelism, we often encounter the phenomenon of data hazard, a bug that can be quite headache-inducing to fix because it is a logical error. However, now we have tools like NVIDIA Compute Sanitizer which make fixing this bug somewhat easier. In this article, I will explain what a data hazard is and illustrate it.

It would be better if you read the article on [Synchronization - Asynchronization](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter08) before reading this one.

<p align="center">
 <h1 align="center"> Data Hazard </h1>
</p>


<p align="center">
  <img src="https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/2ba6daed-3b36-4324-af8b-1ee61b44e07c" />
</p>

The phenomenon where multiple threads read and write a certain value leads to conflicts, and this phenomenon is called a data hazard.

When discussing data hazards, we encounter two issues:

- Data Race: This usually relates to "write after read" or "read after write", but it mainly focuses on simultaneous access (reading and/or writing) to a stored variable without synchronization. This can lead to a situation where one thread overwrites data that another thread is reading or preparing to write, leading to a conflict in data value.
- Race Condition: This concept is broader and not limited to data access. A race condition occurs when the final result of a system shows an undefined behavior or event.
  
**In summary, just remember: when coding in CUDA, be mindful of the phenomenon where multiple threads access the same value for processing.**


<p align="center">
 <h1 align="center"> Illustration </h1>
</p>


```
#include <stdio.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE 4


__global__ void sum(int d_array)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int stride = 1; stride < 4; stride *= 2)
    {
      //  __syncthreads();   -----> barrier

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
```

This is an illustration of the principle of how the code operates.

<p align="center">
  <img src="https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/05fdcf84-baa8-4cc3-9ef0-bb4e0008209d" />
</p>

Here, we do not synchronize the threads, leading to a data race (in step 1: before 3+4 is completed, it moves to step 2, so it's 3 + 3 = 6 instead of 3 + 7 = 10).

<p align="center">
  <img src="https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/cf854c5a-d8a4-411d-ad78-efbe308d1590" />
</p>

To solve this problem, we just need to place a barrier to make the threads wait for each other until the slowest threads have finished, using the command **__syncthreads().**

<p align="center">
  <img src="https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/0eb0550a-179f-46cc-b9bd-15f9e22f238d" />
</p>

And the output after adding syncthreads.

<p align="center">
  <img src="https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/17c97dd4-51bd-4647-ac11-227baa5facf3" />
</p>



