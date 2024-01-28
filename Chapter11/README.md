
In this article, I will guide you through a technique to optimize a program in CUDA C. This technique is relatively simple, but it will be even more beneficial if you have read the articles on [Pinned memory](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter10) and [Async-Sync](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter08).


<p align="center">
 <h1 align="center">Streaming </h1>
</p>


As I mentioned, the CPU and GPU are two separate components, and as a result, the execution of code on the CPU and GPU occurs independently of each other, without any mutual interference. We will leverage this characteristic to further optimize our program in a parallel manner.

![image](https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/62295239-9dcb-47d2-b940-4c8a31a3a68f)

When it comes to CUDA-C code, we need to focus on two concepts: compute bound and memory bound (which can be understood simply as two issues: spending too much time on computation or on memory load/store operations). The Streaming technique will help us address the memory-bound aspect.


I will delve deeper into the concepts of compute bound and memory bound in the [NVIDIA-Tools series](https://github.com/CisMine/Guide-NVIDIA-Tools) , so if you're interested, you can read more about them there. Here, I'll concentrate on the code.


As I mentioned, in order to run code on the GPU, we have to copy data from the CPU to the GPU, which can be quite time-consuming (because if we use cudaMemcpy, we have to wait for the entire copy to complete before proceeding to the next step). Instead of waiting for the entire copy, we can break it down into smaller parts (batches) to optimize this process (similar to what's shown in the diagram).

**There are two main components that always appear when we talk about Streaming: Pinned memory and Stream branches.**

- Pinned memory: The reason pinned memory is used in the Streaming technique is that it is small and fast. As mentioned earlier, we divide the data into smaller portions for copying, so we only need a small amount of memory, and pinned memory can fulfill this requirement quickly.

- Stream branches: This is how threads are organized on the GPU so that they can work independently and in parallel on the same data. Think of each stream branch as a manager responsible for dividing tasks among threads. If you don't specify branching, the default behavior will apply.

<p align="center">
  <img src="https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/bfb08af0-83fd-41eb-a001-862d34ea626c" />
</p>

<p align="center">
 <h1 align="center">Code </h1>
</p>


As I mentioned earlier, a large chunk of data will be divided and processed. The splitting and processing will be done in parallel rather than sequentially, thanks to the stream branch mechanism.

The first step when using streaming is to create stream branches by:

```
cudaStream_t stream[nStreams]
```

The rest is quite similar, with just a slight difference in:

```
cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice,stream[i])
kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
```

As I mentioned, because we're copying only a portion, we need to determine an index, also known as an offset, to maintain the correct copying across different branches.

In this context, the third parameter in the kernel, '0,' refers to shared memory, which you don't need to be concerned about at this point.

**Here, there are two methods for using streaming, and you can choose the method that suits your computer.**

### <p align="center">
 <h1 align="center">Asynchronous version 1: loop over {copy, kernel, copy} </h1>
</p>

![image](https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/82056eea-0d8d-4e28-b8e4-c381d76018b1)

```
for (int i = 0; i < nStreams; ++i) {
  int offset = i * streamSize;
  cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
  kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
  cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
}
```

### <p align="center">
 <h1 align="center">Asynchronous version 2: loop over copy, loop over kernel, loop over copy </h1>
</p>

![image](https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/3c9acae1-4510-4cb3-b139-88c0400040de)

```
for (int i = 0; i < nStreams; ++i) {
  int offset = i * streamSize;
  cudaMemcpyAsync(&d_a[offset], &a[offset], 
                  streamBytes, cudaMemcpyHostToDevice, cudaMemcpyHostToDevice, stream[i]);
}

for (int i = 0; i < nStreams; ++i) {
  int offset = i * streamSize;
  kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
}

for (int i = 0; i < nStreams; ++i) {
  int offset = i * streamSize;
  cudaMemcpyAsync(&a[offset], &d_a[offset], 
                  streamBytes, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToHost, stream[i]);
}
```

<p align="center">
 <h1 align="center">Exercise </h1>
</p>

- Code to compare the execution time of the two methods
- How to determine how many branches to divide into
  
Hint: It's not always better to divide into many branches, as I mentioned, stream branches are like managers, so assuming there are few tasks but we hire too many managers would be wasteful, while one manager would be enough.
