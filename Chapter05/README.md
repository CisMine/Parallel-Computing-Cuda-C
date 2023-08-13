In this article, we will delve a bit deeper into the operational mechanisms of the CPU and GPU. Through this exploration, we will be able to address the final question that I mentioned in [chapter04](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter04)


<p align="center">
 <h1 align="center">The operational mechanism of CPU-GPU</h1>
</p>

Back in the past, I once had a seemingly silly question: which is more important between the CPU and GPU, or when buying a computer, should we prioritize the CPU or the GPU? The answer would depend on our intended usage to determine which one should take precedence, as **CPUs and GPUs are designed for different purposes**

<p align="center">
 <h1 align="center">Approaches to processor Design</h1>
</p>

![image](https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/8f1d8d04-85c0-490e-a0b7-4b01881c2e0b)

CPUs and GPUs are designed with two different objectives, so they cannot be directly compared. So, what are these two objectives?

 

<p align="center">
 <h1 align="center">CPU: Latency-Oriented Design</h1>
</p>

This is a design approach aimed at reducing latency or response time in **processing complex tasks, but with a small number of tasks**. Why focus on reducing latency or response time? Because the CPU is designed with a **small number of high-quality, powerful, and efficient cores**. These cores can be considered multitasking (for example: executing application tasks, managing resources, controlling the system, processing information, etc.)

==> The CPU is used for processing complex tasks, so the design goal is to reduce latency or response time when handling those tasks.

As shown in the diagram, we can see that the Control Unit and Cache occupy a significant portion of the area:
- Large Control Unit: Optimizes the control of complex logic operations (AND, OR, XOR, etc.).
- Large Cache: Helps reduce data access time.
  
**Therefore, the CPU tends to prioritize using Cache and control.**

<p align="center">
 <h1 align="center">CPU: Hide short latency</h1>
</p>

One of the methods to reduce the latency or response time of the CPU is **modest multithreading**. The operation of modest multithreading is as follows:

- When the CPU executes a task, that task is divided into K smaller tasks.
- The CPU generates a few shadow threads.
- These shadow threads handle a small portion of the K tasks, while the remaining part is handled by the main threads.
- When the main threads encounter issues leading to latency (e.g., waiting for data read, waiting for data transfer), the main threads switch to processing a small portion of the task that the shadow threads are handling. This continues until the latency is resolved (i.e., when the data is read or data transfer is completed), at which point the main threads switch back to their original task.

  **This mechanism is referred to as hide short latency.**


  <p align="center">
 <h1 align="center">GPU: Throughput-Oriented Design</h1>
</p>

This is a design approach aimed at increasing the capability to **process a large number of simple tasks quickly and efficiently**. So, why focus on processing multiple tasks simultaneously in a short period? Because GPUs (Graphics Processing Units) are designed with a **high number of cores**, even though these cores might be of **lower quality** compared to CPU cores. Therefore, the goal of GPUs is to handle a large volume of simple tasks

**Therefore, GPUs excel at processing numerous simple tasks concurrently, necessitating the need to enhance their capability to handle a substantial workload in a short time.**

As shown in the diagram, a significant portion of the area is occupied by compute units:

- Multiple cores: Resulting in faster computations, which demands a steady supply of data. To meet this data computation requirement, GPUs are designed with an architecture that enhances bandwidth, making data transfer faster, and also equipped with larger memory (GPU's bandwidth-to-memory ratio or memory is significantly higher than that of CPUs).

**Hence, GPUs are tailored for processing extensive data and parallel computing.**

 <p align="center">
 <h1 align="center">GPU: Hide very high latency</h1>
</p>

One of the methods to enhance the capability of processing a large number of tasks quickly within a short period on a GPU is by employing a **massive number of threads.**

==> In simpler terms, increasing the number of tasks being executed simultaneously at a given point effectively reduces the overall time required to complete all tasks.


 <p align="center">
 <h1 align="center">Overall</h1>
</p>

CPU processes complex tasks but in smaller quantities. 

GPU processes numerous tasks but are much simpler

CPU is like a high-performance sports car with a tremendously powerful engine, while the GPU is like an extra-long bus designed to transport passengers. If only a small number of passengers need transportation, the CPU will be faster, but when the number of passengers is high, the GPU becomes an excellent choice.

Multithreading: Dividing a large task into smaller subtasks and assigning multiple threads to handle them ==> results in short latency.

Massive threads: Creating numerous threads to execute multiple tasks ==> leads to high latency.

![image](https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/916d390a-5181-4440-a2e3-7b84e7804f4a)

 <p align="center">
 <h1 align="center">A Bit of interesting About GPUs</h1>
</p>

Every time we mention **GPUs**, we refer to **graphics cards**. So, why is that name used? In the past, as the gaming industry, in particular, and fields related to graphics processing, in general, gained more recognition, the demand increased significantly. However, the quality wasn't progressing at the same rate. The reason was quite simple: for instance, a sharp and beautiful image might be around 1600x900 in resolution (this is just an example, but it helps illustrate the point). Imagine the CPU processing 1,440,000 (1600x900) pixels at once, and the task is to handle videos, meaning frames, which require processing many images with such pixel counts.

As analyzed earlier, using the **CPU to process each pixel individually would be overly resource-intensive** (to put it simply, using a for loop for each pixel, where each iteration is a simple task). This wouldn't fully leverage the CPU's capabilities; it's akin to using a sports car to transport passengers. **That's when the concept of a Graphics card was invented**, designed solely for the purpose of pixel processing. Over time, this development led to increasingly superior GPU computing power, prompting the application of GPUs in various fields related to computation. This broader use is why the term **GPGPU (General-Purpose Computing on Graphics Processing Units) came into existence.**



 <p align="center">
 <h1 align="center">Analyzing the Last Question in chapter04</h1>
</p>

Before addressing the question, let's go over two concepts: **SIMD and SIMT**.

**CPU: SIMD (Single Instruction, Multiple Data)** is a computer architecture often used in CPUs with the goal of **maximizing the processing of multiple data per instruction.**

**GPU: SIMT (Single Instruction, Multiple Threads)** is a computer architecture **developed by NVIDIA** and utilized in GPUs with the aim of utilizing as **many threads as possible for each instruction.**

Both SIMD and SIMT are parallel processing architectures used in computing devices like CPUs and GPUs.

`At first glance, SIMD and SIMT might seem similar, but they are two distinct architectures with differences in certain aspects. The reason why CPUs are SIMD-based and GPUs are SIMT-based can be attributed to these differences.`

As I mentioned:

- CPU: Due to processing complex tasks, the mechanism of SIMD divides the complex task into sub-tasks and then processes these sub-tasks in parallel.

- GPU: As it handles numerous simple tasks, the SIMT mechanism processes tasks in parallel.

For example, let's consider the problem of adding two vectors (each containing N elements).

SIMD: In this case, the task is to **perform vector addition**, and SIMD divides the vector addition problem into **N sub-problems**, which involve adding individual elements. Consequently, these **N sub-problems are executed in parallel.**

SIMT: From the SIMT perspective, each vector containing N elements becomes **N-independent tasks**. The objective is to perform N-independent addition tasks (not N sub-problems), and then **coders divide these tasks into threads for parallel execution.**

- SIMD: The computer automatically divides a large problem (e.g., adding two vectors) into N sub-problems.

- SIMT: We will be the one to divide the threads to handle these N problems.


## Analyzing the Question

Our task is to **print "hello world" 10 times**, and since it's **SIMT, We will manually allocate threads** for this printing operation. Here, I distribute threads into two types: **<<<1,10>>>** and **<<<2,5>>>**. Since there are **only 10 threads, these two methods are equivalent.** However, if the task were to **print "hello world" 64 times** and is represented as **<<<1,64>>>** and **<<<2,32>>>**, there would be **a difference** (because at each moment within a block, only 32 warps are executed). Therefore, for **<<<1,64>>>**, it would take 2-time units to complete the printing of "hello world" 64 times, whereas for **<<<2,32>>>**, it would only take 1-time unit to complete.

I provided a more detailed explanation in the [chapter03](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter03) section under the RULES part

**In summary, through this article, you now have a clearer understanding of both CPUs and GPUs. Due to the SIMT mechanism, it's crucial to intelligently distribute threads, making it essential to determine the appropriate number of threads per block.**
