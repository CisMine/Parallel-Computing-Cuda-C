

In this article, I will introduce to you a very useful built-in function in CUDA. It's important to read previous articles about [Data Hazard](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter12) and [Synchronization - Asynchronization](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter08) before diving into this one.


<p align="center">
 <h1 align="center"> Atomic Function </h1>
</p>

This library is quite simple to use. The purpose behind NVIDIA developing this library is to avoid data hazards or, in other words, to synchronize threads during the processing of operations.

For example, consider a simple code snippet:

```
for(int i=0;i<n;i++){
    y+=x[i];
    }
```

```
int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        *d_result += d_data[tid]; 
    }
```

that leads to a data hazard due to the lack of synchronization between threads, resulting in incorrect output. Therefore, the atomic function was created to solve this issue.

Atomic functions can be understood as somewhat similar to a for-loop, iterating over each thread one by one (the thread that starts first is processed first) **WHEN AND ONLY WHEN** the threads read-write the same value. Otherwise, the operations remain parallel as usual.

Here is the code using atomic operations:

```
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    atomicAdd(result, array_add[tid]);
```

And here are some atomic functions:

- atomicAdd(result, array_add[tid]); – adds array_add[tid] to result.
- atomicSub(result, array_sub[tid]); – subtracts array_sub[tid] from result.
- atomicMax(result, array_max[tid]); – computes the maximum of result and array_max[tid].
- atomicMin(result, array_min[tid]); – computes the minimum of result and array_min[tid].
  
Additionally, there are other interesting atomic functions like atomicCAS (Compare And Swap) and atomicExch (Exchange).





