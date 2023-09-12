In [Chapter06](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter06), I introduced the different types of memory within the GPU (their functions, speeds, and the thread access scope). In this lesson, I will guide you on how to use them using the CUDA-C language.

<p align="center">
 <h1 align="center">Using GPU memory</h1>
</p>

Before diving into the code, I will answer the two questions that I mentioned in Chapter06, which are:
- In the diagram, why are shared memory and L1 cache combined into a single memory instead of being separate memories?
  
- Why does the access scope of L1 involve Threads within the same block, while L2 involves all Threads across the device?

  ![image](https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/b49d6e97-34e6-45c4-a5d5-087652fb278d)

  ![image](https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/b3c0e4f2-05f4-4c78-b938-41b37d23d308)

If our perspective is the **Physical view**, then it is certain that **shared memory and L1 are two separate memory sticks**. My question is from the **Logical view**, so they will be **one**. The reason is that, as analyzed by researchers, if separated into distinct memory regions, it will be **challenging to manage and will be very resource-intensive**.

```
If you are thinking right here:

- Difficult to manage?? If they are merged, it may lead to confusion between memories, and separating them will be easier to manage. That's incorrect.

- Resource-intensive?? Whether they are merged or separated, we only have that much memory. Where is the waste of resources??
```

If you all have thoughts like that, then I would like to mention the concept of **prefetching** (extremely effective in optimizing data access performance and applied to Caches).

**Prefetching** is the process of loading data from main memory into cache or intermediate memory **before it is needed** to optimize data access performance

Example:

```
int a[100], b[100], c[100];
for(int i = 0; i < 100; i++) {
    c[i] = a[i] + b[i];
}

```

In the code snippet above, we are performing addition between two arrays, 'a' and 'b,' to store the result in array 'c.' However, **in the typical access pattern**, the program would iterate through each element one by one, and with each access, it would adjust the pointer to main memory to fetch data from 'a' and 'b.' This can lead to waiting times when accessing data from main memory.

**The prefetching mechanism** suggests an intelligent way to reduce this waiting time. Instead of waiting until each element is individually accessed, it anticipates the **upcoming data accesses** that the program may perform **based on previous access patterns**. It then prefetches these data into the cache memory, helping to reduce waiting times and optimize the overall program performance.

`This is why the cache is also referred to as temporary memory. When we push data into the cache, it is only used temporarily until that data is actually accessed. At that point, the data in the cache is fetched for processing, and the cache memory is replaced with the next set of data.`

Returning to our original issue of **'Difficult to manage - Resource-intensive'**:

- Difficult to manage: If shared memory and L1 were separate, we would need an additional mapping step to determine which data will be accessed next. When they are combined within a single memory mechanism, data can be efficiently shared between shared memory and L1 cache.

- Resource-intensive: According to researchers, if we separate them into two separate memory sticks, when implementing code, most of the time, shared memory or cache would not be fully utilized (there would be some leftover space). However, if we combine them, we can allocate exactly how much we need for shared memory, and the rest can be allocated to the cache, making it more efficient.

`For these reasons, we only need to use shared memory without needing to touch the cache (and in reality, we cannot directly manipulate the cache as NVIDIA does not provide libraries for direct cache operations).`

As for the reason why the access scope of L1 is within the threads of the same block while that of L2 encompasses all threads, it is because **global memory also requires a prefetch mechanism**, which forces us to separate the cache into two parts: **one for shared (L1) and one for global (L2).**



<p align="center">
 <h1 align="center">Now, let's move on to the code</h1>
</p>

<p align="center">
 <h1 align="center">Global memory</h1>
</p>

We will code the addition of two vectors (each with 100 elements) using global memory - the largest and slowest memory in the GPU.

```
h_: represents values on the host.

d_: represents values on the device.

The symbols h_ and d_ are commonly used in CUDA guides and documents, so I will use them here to make them familiar to everyone.
```

```sh
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
```
<p align="center">
 <h1 align="center">Analyze the kernel</h1>
</p>

```sh
__global__ void vectorAdd(int *a, int *b, int *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}
```
This kernel is quite similar to regular C code, with a few differences:
- int tid = blockIdx.x * blockDim.x + threadIdx.x;
- if (tid < N)

In regular C code, to add two vectors, we would iterate through each element and add them individually. However, in CUDA-C, we iterate through all elements in one go, but to do this, we need to determine the index (the position of the elements). So, **int tid is used to determine the index.**

![image](https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/5df425c2-4ffe-48f0-8724-df9d3e0ce96d)

![image](https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/4e167120-1066-44f5-977c-c93f90fd581e)

In the two illustrated images, the concept is quite clear. In this context, **'M' represents the number of threads in one block, also known as 'blockDim.'**

Next is the **'if (tid < N)'** statement, which serves as a barrier to determine which threads participate in the process of adding two vectors. In the given code, we only need to use 100 threads for the 100 elements. However, GPUs typically have up to 1024 threads per block, so if we don't specify the 'if' condition, it might not work properly.

`In fact, specifying 'if' is not always necessary because in 'vectorAdd<<<2, 50>>>' we've already determined that only 100 threads will be used. However, as the code becomes more complex, including the 'if' condition as a habit is recommended for better code management.`

<p align="center">
 <h1 align="center">Initialization of values on the GPU</h1>
</p>

```
cudaMalloc((void **)&d_a, N * sizeof(int));
cudaMalloc((void **)&d_b, N * sizeof(int));
cudaMalloc((void **)&d_c, N * sizeof(int));
```
The cudaMalloc function is similar to malloc in C, but it is used to dynamically allocate memory on the GPU.

A small note is that when you read various guides or documentation, you might see the following syntax:

```
 cudaMalloc(&d_a, N * sizeof(int));
 cudaMalloc(&d_b, N * sizeof(int));
 cudaMalloc(&d_c, N * sizeof(int));
```

Both code snippets are equivalent. The version with void is an older style, while the latter is more modern and commonly used.


<p align="center">
 <h1 align="center">Data Transfer</h1>
</p>

Here I use the word Transfer data to indicate copying data from Host to Device and vice versa (and later I will abbreviate it as H2D - D2H).

```
cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

vectorAdd<<<2, 50>>>(d_a, d_b, d_c);

cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
```

The cudaMemcpy function is used to transfer data between the host (CPU) and the device (GPU). It takes three parameters:
- first parameter: Destination (where to paste data - it likes Ctrl+v)
- second parameter: Source (what to copy - it likes Ctrl+c)
- third parameter: Direction of data transfer (H2D for Host to Device or D2H for Device to Host)
  
**An important note is that when you use cudaMemcpy, data is automatically copied to/from the global memory.**

<p align="center">
 <h1 align="center">Local memory and registers</h1>
</p>

Local memory and register files are two distinct memory types for each thread. Register files are the fastest memory available for each thread. When kernel variables cannot fit in register files, they will use local memory.

In other words, each thread has its own set of register files and local memory. When register files are exhausted, data is spilled into local memory. This concept is known as **register spilling.**

```sh
#include <stdio.h>
#include <stdlib.h>

__global__ void kernel() {
    int temp = 0;
    temp = threadIdx.x;

    printf("blockId %d ThreadIdx %d = %d\n", blockIdx.x, threadIdx.x, temp);
}

int main() {
    kernel<<<5, 5>>>();
    cudaDeviceSynchronize();

    return 0;
}
```
In the provided code, each thread initializes its temp variable with threadIdx.x. **While conventional thinking might suggest that temp would hold the last value of threadIdx.x**, CUDA works differently. **Each thread executes independently, so temp is a thread-local variable. Therefore, temp will indeed contain the value of threadIdx.x for each thread.**

If you revisit the global memory example:
- int tid = blockIdx.x * blockDim.x + threadIdx.x;

The **int tid** variable is local memory.

<p align="center">
 <h1 align="center">Constant Memory</h1>
</p>

Constant memory is read-only and is used to store constant values. The example code demonstrates how to use constant memory for a simple equation, y = 3x + 5, with x as a vector and 3 and 5 as constants stored in constant memory.

```sh
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
```
<p align="center">
 <h1 align="center">Initialization of values on the GPU</h1>
</p>

Unlike global memory, constant memory does not require using cudaMalloc. Instead, you need to declare that you are using constant memory by using _ _ _constant_ _ _
- _ _ _constant_ _ _ int constantData[2]

<p align="center">
 <h1 align="center">Transfer data</h1>
</p>

`cudaMemcpyToSymbol(constantData, constantValues, 2 * sizeof(int))`
  
Data transfer to constant memory is done using cudaMemcpyToSymbol.

A small note: In the code, the direction of data transfer is not explicitly specified because it defaults to H2D (Host to Device).

<p align="center">
 <h1 align="center">Summary</h1>
</p>

In summary, this tutorial covers the use of global memory, local memory (register files), and constant memory in CUDA programming. Shared memory will be discussed separately in another tutorial, and texture memory is no longer a significant concern in modern NVIDIA GPUs.

<p align="center">
 <h1 align="center">Exercises</h1>
</p>

When you run the code in the example of local memory and registers:

```sh
#include <stdio.h>
#include <stdlib.h>

__global__ void kernel() {
    int temp = 0;
    temp = threadIdx.x;

    printf("blockId %d ThreadIdx %d = %d\n", blockIdx.x, threadIdx.x, temp);
    
}

int main() {
    kernel<<<5, 5>>>();
    cudaDeviceSynchronize();

    return 0;
}
```
Why does the output not follow the order of blockId but appear mixed up as shown in the figure?

![image](https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/66e6d525-5942-4203-94ce-5947e8ebc5e0)




