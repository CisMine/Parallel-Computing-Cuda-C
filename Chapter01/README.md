
<p align="center">
 <h1 align="center">Demystifying CPUs and GPUs: What You Need to Know </h1>
</p>

It’s possible that the terms CPUs and GPUs are familiar to some people, while others may have only heard of them without fully understanding their purpose. In this post, I will do my best to provide `a clear and simple explanation` for those who are unfamiliar with these terms.



<p align="center">
 <h1 align="center">CPUs </h1>
</p>

<p align="center">
  <img src="https://st4.depositphotos.com/2978065/26363/v/600/depositphotos_263634270-stock-video-render-animation-artificial-intelligence-cpu.jpg" />
</p>

<p align="center">
 <h1 align="center">What is CPUs? </h1>
</p>


CPUs or Central Processing Units are the primary components that perform most of the processing tasks in a computer system. They are also known as the “brain” of a computer.

The CPU is responsible for executing instructions that are stored in the computer’s memory, performing arithmetic and logical operations, and managing data flows between different parts of the computer. The performance of a CPU is usually measured by clock speed, which is the number of instructions it can execute per second, and the number of cores, which refers to the number of independent processing units within the CPU.

<p align="center">
 <h1 align="center">Development and the problems </h1>
</p>




<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*DvatFeAdCcXAOTvbB39tnw.png" width="450"/>  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*SWSJcjLVGgDhHXriSWT_eg.png" width="450"/> 





The general trend of these graphs shows a significant increment, but there appears to be a pause or drop in the graphs from 2004 to 2005, particularly with frequency. `So what happened at that time?`

### Before I go any further, let me just say that this article will provide “a clear and simple” explanation for those who are unfamiliar with these terms. As a result, I won’t go into great detail about what actually occurred, but if you’re interested, check out this [link](http://www.edwardbosworth.com/My5155_Slides/Chapter01/ThePowerWall.htm)

The two major issues that CPUs faced at the time were the `Power wall` and `Memory wall`.

To put it simply, more power means quicker processing from more CPUs.However, more powerful CPUs may require a higher Voltage to maintain stability at such speeds so we couldn’t clock processors faster.

Another problem was the latency when accessing the memory; if a computer has a powerful CPU but has poor memory access, it will take a long time, which is why the efficiency of many workloads is determined by memory access times. **And that is how the parallel era occurred**



<p align="center">
 <h1 align="center">GPUs </h1>
</p>

![](https://www.shutterstock.com/shutterstock/videos/1074222746/thumb/7.jpg?ip=x480")


<p align="center">
 <h1 align="center">what is GPUs? </h1>
</p>

GPUs (Graphics Processing Units) are specialized hardware components designed to accelerate the processing of graphics and parallel computing tasks. They were originally developed for use in computer gaming and graphical applications, but are now widely used in scientific and engineering applications as well.


![](http://www.nextplatform.com/wp-content/uploads/2019/07/decade-gpu-natoli.jpg")

These graphs demonstrate how much more quickly GPUs calculate than CPUs, so which features can GPUs perform at that level?

<p align="center">
 <h1 align="center">Parallel processing </h1>
</p>

lets me give an example what is parallel processing:
Assume that a teacher gives a class 10 questions to answer. The simplest solution is to have the best student in the class complete all 10 questions. However, if we can find and train the remaining 9 students to be as good as the best student, we can speed up the process of answering questions by 10 times and that’s how GPUs work.

In conclusion, saving time by breaking up large tasks into smaller ones that can be handled simultaneously is the best benefit of parallel processing. However, there are some limitations to task division, such as when a teacher assigns a class N questions but we are unable to find N students to prepare for and complete those tasks.

Parallel calculation techniques
Parallel calculation techniques is the big evolution of sequential computing. Parallel calculation techniques refer to the methods and algorithms used to divide a large computational task into smaller subtasks that can be executed simultaneously on multiple processors or computers.

Parallel computing allows for the use of multiple processing units to work together to solve complex problems faster and more efficiently than traditional sequential computing. Parallel computation can be achieved through various approaches such as shared-memory, distributed-memory, and hybrid models.


<p align="center">
 <h1 align="center">Conclusion </h1>
</p>

You now understand how to use a parallel program to get around the CPUs’ limitations.

My next post will discuss parallel programming, the languages we use, and a deeper explanation of how graphics processing units (GPUs) operate.

Last but not least, thank you for reading up until now. Please don’t be hesitant to star if you enjoyed it.


<p align="center">
 <h1 align="center">References </h1>
</p>

https://www.cs.princeton.edu/~dpw/courses/cos326-12/lec/15-parallel-intro.pdf
http://www.edwardbosworth.com/My5155_Slides/Chapter01/ThePowerWall.htm

