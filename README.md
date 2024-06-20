
<p align="center">
 <h1 align="center">Parallel Computing Using Cuda-C </h1>
</p>

This repository contains code examples and resources for parallel computing using CUDA-C. CUDA-C is a parallel computing platform and programming model developed by NVIDIA, specifically designed for creating GPU-accelerated applications.

The goal of this repository is to provide beginners with a starting point to understand parallel computing concepts and how to utilize CUDA-C to leverage the power of GPUs for accelerating computationally intensive tasks. Whether you are a student, researcher, or developer interested in parallel computing, this repository aims to provide a practical guide and code examples to get you started.


<p align="center">
 <h1 align="center"> Introduction to CUDA-C </h1>
</p>

CUDA-C is an extension of the C programming language that allows developers to write code that can be executed on NVIDIA GPUs. It provides a set of language extensions, libraries, and tools that enable developers to harness the power of parallel processing on GPUs.

CUDA-C allows you to write parallel code using the CUDA programming model, which includes defining kernels (functions that execute on the GPU) and managing data transfers between the CPU and GPU. By writing CUDA-C code, you can achieve significant speedups for computationally intensive tasks compared to running the same code on the CPU alone.

<p align="center">
 <h1 align="center"> Why we need Cuda-C </h1>
</p>

![image](https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/281ef415-60ad-4a82-ba74-b996fd1822cc)



With the exponential growth of data and increasing demands from users, CPUs alone are no longer sufficient for efficient processing. GPUs offer parallel processing capabilities, making them well-suited for handling large-scale computations. CUDA-C, developed by NVIDIA, enables developers to leverage GPUs for accelerated processing, resulting in faster and more efficient data processing.



<p align="center">
 <h1 align="center"> Getting Started </h1>
</p>

### If your computer has GPU
Following these steps in NIVIDA to install [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)

 - If you are using Linux,  I advise you to watch [this video](https://www.youtube.com/watch?v=wxNQQP9U1Bc)

 - If you are using Windows, this is [your video](https://www.youtube.com/watch?v=cuCWbztXk4Y&t=49s)

### If your computer doesn't have GPU
 - Don't worry; I'll demonstrate how to set up and use Google Colab to code [in here](https://medium.com/@giahuy04/the-easiest-way-to-run-cuda-c-in-google-colab-831efbc33d7a)


<p align="center">
 <h1 align="center">Table of Contents </h1>
</p>

- [Chapter01: Demystifying CPUs and GPUs: What You Need to Know](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter01)

- [Chapter02: How the way a computer works](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter02)

- [Chapter03: Terminology in parallel programming](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter03)
- [Chapter04: Hello world Cuda-C](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter04)
- [Chapter05: The operational mechanism of CPU-GPU](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter05)
- [Chapter06: Memory Types in GPU](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter06)
- [Chapter07: Using GPU memory](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter07)
- [Chapter08: Synchronization and Asynchronization](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter08)
- [Chapter09: Unified memory](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter09)
- [Chapter10: Pinned memory](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter10)
- [Chapter11: Streaming](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter11)
- [Chapter12: Data Hazard](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter12)
- [Chapter13: Atomic Function](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter13)
- [Chapter14: Shared memory](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter14)



<p align="center">
 <h1 align="center">Resources </h1>
</p>

In addition to the code examples, this repository provides a curated list of resources, including books, tutorials, online courses, and research papers, to further enhance your understanding of parallel computing and CUDA-C programming. These resources will help you delve deeper into the subject and explore advanced topics and techniques.

 - [NVIDIA Practices_Guide 2023](https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf)
 - [NVIDIA Programming_Guide 2023](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)
 - [GPU Programming 2021 in youtube](https://www.youtube.com/watch?v=wFlejBXX9Gk&list=PL3xCBlatwrsXCGW4SfEoLzKiMSUCE7S_X)
 - [Cuda Programming 2023 in youtube](https://www.youtube.com/watch?v=cvo3gnInQ7M&list=PL1ysOEBe5977vlocXuRt6KBCYu_sdu1Ru)
 - [Programming Massively 2022 in youtube](https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4)
 - [Cuda training series 2022-2023 in youtube](https://www.youtube.com/playlist?list=PL6RdenZrxrw-zNX7uuGppWETdxt_JxdMj)
 - [Programming Heterogeneous Computing Systems with GPUs 2023 in youtube](https://www.youtube.com/playlist?list=PL5Q2soXY2Zi-qSKahS4ofaEwYl7_qp9mw)
 - [Cuda Thread Indexing cheatsheet](https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf)


