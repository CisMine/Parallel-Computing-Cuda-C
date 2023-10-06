To answer the question in [Chapter07](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter07), we need to go through two concepts: synchronization and asynchronization.



<p align="center">
 <h1 align="center">Synchronization and Asynchronization </h1>
</p>

Before explaining these two concepts, let's go through an example to help you visualize better.

Example:
In a school: we have N assignments, K classes, and each class has J students (N > K * J or N = K * J), and the task is to distribute these N assignments to J students. Here, I will use specific numbers to make it easier for you to understand.

So, the problem becomes: 2048 assignments, 32 classes, 32 students per class. Our task is to distribute these 2048 assignments for students to solve. In this case, we will assign 64 assignments to each class (32 * 64 = 2048), meaning each student will handle 2 assignments.

Because **each student will have a different assignment-solving speed**, the **completion speed of each class** will be different. But an interesting point here is the **synchronization among students within the same class**, meaning when they complete one assignment, they will wait until the last person (the slowest) finishes that assignment before starting the next one.


<p align="center">
  <img src="https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/4c84107b-9c9b-4e77-ba78-e3066e60d7c8" />
</p>



Here, **the students are like threads, the classes are like blocks, and N assignments are the data**. Similar to the example mentioned, the **blocks will complete their work at different speeds** (not sequentially), and the **threads within the same block will also have different completion speeds**, but they will be synchronized, meaning they will only move on to the next task when the slowest thread finishes.

The reason for synchronization is the warp mechanism, which takes 32 tasks for 32 threads at a time (I explained this in the [Warp section in Chapter03](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main/Chapter03)).

To avoid a situation where waiting for one person affects everyone, we have the concept of **Latency Hiding**.

<p align="center">
  <img src="https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/3eb5f3e3-214f-4f5d-9cc3-165e1bb9dd00" />
</p>



As shown in the image, when one warp is being executed but takes too long (possibly due to waiting for some slow threads), there is an automatic mechanism where another warp will be automatically replaced to execute (meaning there will be 32 new assignments for 32 students to work on).

Thanks to this, the threads are always busy, or in other words, we should **always keep the threads busy**.

Because of the synchronization mechanism, we encounter a phenomenon called **thread divergence** (this phenomenon significantly affects our performance).

**Thread divergence** occurs when threads perform different calculations or encounter branching conditions. Due to the synchronization mechanism, they have to wait for each other.


<p align="center">
  <img src="https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/ff464d0d-2fda-405a-b062-4cb3fc7eb239" />
</p>


In other words, instead of processing Path A and Path B in parallel, here it has to be done sequentially. Threads that satisfy Path A work first, while those satisfying Path B have to wait until Path A is finished.

### Summary:

<p align="center">
  <img src="https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/5d30ef57-33cc-40bd-9f2b-729c80631218" />
</p>


Threads within the same block synchronize but only when moving to the next task within the same job; they are still asynchronous (similar to the assignment-solving speed of the students in my example). Blocks (or threads from different blocks) are asynchronous.

Through this article, you probably have a better understanding of why we get such output.
![image](https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/a4083d4e-3479-4336-ba5a-4fc1bfa03e65)

