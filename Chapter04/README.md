<p align="center">
 <h1 align="center">Hello world cuda-C</h1>
</p>

Parallel programming on the GPU means that we transfer data from the CPU to the GPU for processing/computation by using the Cuda C/C++ language.

Most of you may have two questions at this point:

- What is Cuda?
- How can we transfer data from the CPU to the GPU and utilize GPU cores?

In this chapter, you will learn about all these concepts and how to implement a simple "Hello World" code in CUDA C.

#### One small note is that if you are not familiar with GPUs (their functioning or components), don't worry because this chapter will not require that knowledge. Rest assured that I will create a separate chapter to explain GPUs so that readers can acquire the necessary knowledge.

<p align="center">
 <h1 align="center">What is Cuda?</h1>
</p>

CUDA (Compute Unified Device Architecture) is a parallel computing platform developed by NVIDIA. It allows programmers to utilize the GPU (Graphics Processing Unit - GPU cores) for performing computational tasks using programming languages such as C and C++.

### How Cuda works?

<p align="center">
  <img src="https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/755aaca4-cb1f-4fc3-a29d-4510e880d613" />
</p>

When we finish coding and save the file, we often add a file extension at the end. For example:

- Python uses .py
- C uses .c
- C++ uses .cpp
- Similarly, for CUDA C/C++, the file extension is .cu.

As the name suggests, CUDA C/C++ code is a combination of C (or C++) and CUDA, so we need a compiler that can compile both C/C++ binaries and CUDA binaries. To address this, NVIDIA has developed NVCC (NVIDIA CUDA Compiler), which can handle both types of code and compile them appropriately.


### NVCC (NVIDIA CUDA Compiler) is a compiler specifically designed to compile CUDA C/C++ code. It plays a crucial role in the compilation process, as it performs several important tasks to generate executable code for NVIDIA GPUs. Here is an overview of how NVCC works:

- Code Analysis: NVCC analyzes the source code to determine the portions written in CUDA C/C++ and identifies the device (GPU) and host (CPU) code sections.

- Separation of Host and Device Code: NVCC separates the host code, which runs on the CPU, from the device code, which will be executed on the GPU. It ensures that the host and device sections are handled appropriately during the compilation process.

- Compilation and Optimization: NVCC compiles the host code using a standard CPU compiler, such as GCC or MSVC, while it compiles the device code using the CUDA compiler provided by NVIDIA. The device code is optimized specifically for NVIDIA GPUs, taking advantage of their architecture and capabilities.

- GPU-specific Code Generation: NVCC generates GPU-specific machine code (PTX - Parallel Thread Execution) that represents the device code. This code is not directly executable on the GPU but serves as an intermediate representation.

- PTX Translation and Optimization: NVCC translates the PTX code into GPU-specific machine code (SASS - Scalable Assembly) using the NVIDIA GPU driver. It performs additional optimizations tailored to the target GPU architecture.

- Linking and Final Binary Generation: NVCC combines the compiled host code and the translated GPU machine code, performs linking, and generates a final executable binary file that can be executed on the target GPU.

### By providing a unified compilation process for both host and device code, NVCC simplifies the development of CUDA applications and enables efficient utilization of GPU resources.


<p align="center">
 <h1 align="center">How can we transfer data from the CPU to the GPU and utilize GPU cores?
</h1>
</p>

**In summary, you can envision the process as follows:** First, we write code in C or C++ to fetch data and store it in CPU memory. Then, from the CPU, we call a kernel (a function that runs on the GPU, written in CUDA) to **copy** the data from CPU memory to GPU memory for computation. After the computation is completed, we **copy** the results back from the GPU to the CPU to print the output.

#### One small note is that from now on, I will refer to it as CUDA C instead of CUDA C/C++. As mentioned earlier, we initially write code in C or C++ to fetch data and store it in CPU memory. Here, I will choose to code in C because it shares similar syntax with CUDA, making it easier to read the code. 

### Why copy:
The reason for using the term "copy" is that the CPU and GPU have separate memory spaces (**I will dedicate a separate chapter to discuss this in more detail**). They cannot directly access each other's memory, so data transfer between the CPU and GPU needs to occur through the PCI (bus).

### Let's run the initial lines of code together and analyze them.

```sh
#include <stdio.h>


__global__ void kernel()
{

    printf("hello world");
}

int main()
{
    kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}
```

As I have explained how CUDA works, you can save the file in the format **<filename<filename>>cu**, and then compile it using two command lines (when compiling, open the terminal and navigate to the correct directory where you have saved the code):
- nvcc <filename<filename>>cu
- ./a.out
  
![image](https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/3e1776c0-2805-4b10-ae37-c617e7c864d6)

# Code analysis

<p align="center">
  <img src="https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/1a397373-85e3-451f-a9f2-8a848e4fd206" />
</p>

Here, we have two new concepts: **Host**, which refers to the **CPU**, and **Device**, which refers to the **GPU**.
- **__ _host_ __**: It represents a normal function that is **called and executed on the CPU**. In other words, when you create a function without any additional specifications, it will be executed on the CPU.

```sh
int add(int x, int y)            __host__ int add(int x, int y)    
{                                   {
return x + y ;                         return x + y ;
}                                   }
```
For example, in the two code snippets mentioned above, they are the same. If the execution target of a function is not specified, it defaults to the CPU (i.e., the host). This is especially evident when we create the main function: int main().

- **__ _global_ __ void:** It represents a function that is **called by the host (CPU)** and **executed by the device (GPU)**. This type of function is often referred to as a **kernel function.**

**Kernel function: It executes instructions on the GPU. The CPU launches the kernel using a special syntax (as explained earlier with NVCC) to inform the GPU about the number of threads to be used.**

`I will provide a clear explanation of the meaning of this statement below, and please note that global void always goes together, meaning it does not have a return value. The reason is that the CPU and GPU are separate components that cannot communicate directly with each other. Therefore, the results cannot be returned to the CPU like regular functions. Instead, data needs to be copied back and forth between them through the PCI bus.`


```sh
int add(int x, int y)            __global__ void kernelAdd(int a, int x, int y)    
{                                   {
                                      
return x + y ;                         a = x + y ;
}                                   }
```

Here, we have two functions: **add** and **kernelAdd.**

1) add: It is called and executed on the CPU, meaning the calculation x + y will be performed by a CPU core.

2) kernelAdd: It is called by the CPU but executed on the GPU, meaning the calculation x + y will be performed by a GPU core.


- **__ _device_ __ <datatype<datatype>>:** It represents a function that is **called by the device (GPU) and executed on the device.** In simple terms, **__ _global_ __ void** can be thought of as the main function on the GPU, while **__ _device_ __ <datatype<datatype>>** is a subsidiary function. These subsidiary functions are often created and called by the main function, which is why device functions are called and executed by the GPU.
  

 ```sh
  
__device__ void PrintHello()
{
    printf("hello"); 
}

__global__ void kernel()
{
    PrintHello();
}

```

# Back to our code


```sh
#include <stdio.h>


__global__ void kernel()
{

    printf("hello world");
}

int main()
{
    kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}
```


So here, we create a kernel function to print "hello world" (which is executed by GPU cores), and we call this kernel function in the main function (CPU). There are two things we need to explain:
-  <<<1,1>>>: The **first "1"** represents the **number of blocks**, and the **second "1"** represents the **number of threads within a block.** I have explained blocks and threads earlier, but here it is a bit different from the theoretical explanation. As I mentioned before, threads represent the number of tasks, but in this case, threads actually refer to the number of GPU cores. So, in this context, threads are equivalent to GPU cores. **In this case, we specify how many "students" (GPU cores or SP) will perform the task of printing "hello world"** ==> **by using <<<1,1>>>.** It means that there is **one "class" (block)** with **one "student" (thread)** performing the task of printing "hello world". In general, **<<<N,N>>>** means that there will be **N "classes" with N "students"** performing the task of printing "hello world" ==> hello world being printed N * N times.

**Therefore, we can specify how many GPU cores (or threads) are used for execution. This leads to the statement:**

**Kernel function: It executes instructions on the GPU. The CPU launches the kernel with special syntax (as explained earlier with NVCC) to inform the GPU about the number of threads to be used.**

- Since the CPU and GPU are separate components with different processing speeds, we need synchronization between the two components. Hence, NVIDIA introduced **cudaDeviceSynchronize()**, which is a synchronization function. It ensures that all preceding computational tasks on the GPU are completed before the program proceeds to execute subsequent tasks on the CPU.

# Exercises
1) You can try creating **__ _device_ __ functions** and calling them from **__ _global_ __ void**, and then call **__ _global_ __ void** from a **__ _host_ __** function and call that **__ _host_ __** function in the main function. Additionally, you can experiment with changing the order of the function calls to observe their impact. For example, try calling **__ _global_ __ void** inside **__ _device_ __**.

```sh
#include <stdio.h>

__device__ void Device1()
{
    //
}

__device__ void Device2()
{
    //
}

__global__ void kernel()
{
    Device1();
    Device2();
}

void sub_Function_in_Host()
{
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

int main()
{
    sub_Function_in_Host();
    return 0;
}
```

2) You can try running any line of codes within the main function **before** and **after** using cudaDeviceSynchronize() to observe the output. See if it matches the theoretical expectations we discussed earlier.

3) In theory, as mentioned earlier, <<<1,1>>> indicates that there is one "class" with one "student" performing the task of printing "hello world" If we change it to:

<<<1,10>>>: Now, there will be 10 "students" in one "class" performing the task of printing "hello world" concurrently.

<<<2,5>>>: In this case, there will be 2 "classes" with 5 "students" in each class, performing the task of printing "hello world" concurrently.

Both approaches will output "hello world" ten times, but what is the difference?

Here are two hints:
- It is related to SIMT so What is SIMT?
- I mentioned in [chapter03](https://github.com/CisMine/Parallel-Computing-Cuda-C/edit/main/Chapter03/README.md)
