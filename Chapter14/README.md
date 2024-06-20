

In this article, I will introduce you to how to use shared memory on the GPU using CUDA. Before reading this article, please take a look at the [Memory Types in GPU](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter06)

<p align="center">
 <h1 align="center">  Shared memory </h1>
</p>


<p align="center">
  <img src="https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/85378864-fe03-4a32-b9d8-b857e2e24762" />
</p>


Shared memory is the fastest memory (only after the register file) in the GPU, and the scope of access to shared memory is the threads within the same block.

Whenever copying data from global to shared memory, we must use __syncthreads() to synchronize the threads within the same block to **avoid race conditions**. This is because while **threads in a block run logically in parallel, not all threads can execute physically at the same time.**

You can refer to these two articles to understand better:
- [Data Hazard](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter12)
  
- [Synchronization - Asynchronization](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter08)


<p align="center">
 <h1 align="center"> Here is the process of data going from global to shared memory </h1>
</p>


<p align="center">
  <img src="https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/524d9815-17b0-42cb-8bf6-2b490a62f19f" />
</p>

`
In this article, I will focus on the concept and usage of shared memory. In future articles, I will discuss techniques to improve and optimize the use of shared memory
`

<p align="center">
 <h1 align="center"> Code </h1>
</p>

We are already familiar with the concepts of static and dynamic memory, and in shared memory, we also have these concepts.

```
__global__ void staticReverse(int *data, int n)
{
  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = data[t];
  __syncthreads();
  data[t] = s[tr];
}
```

```
__global__ void dynamicReverse(int *data, int n)
{
  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = data[t];
  __syncthreads();
  data[t] = s[tr];
}
```

In this problem, we have two steps:

- Copy data from global to shared memory in ascending order.
- Copy data from shared memory back to global memory in descending order.

```
  staticReverse<<<1,n>>>(d_data, n);
  dynamicReverse<<<1,n,n*sizeof(int)>>>(d_data, n);
```

In <<<a,b,c,d>>>

- a: the number of blocks
- b: the number of threads per block
- c: the size of shared memory
- d: the number of streams




































