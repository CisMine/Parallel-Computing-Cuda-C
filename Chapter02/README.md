<p align="center">
 <h1 align="center">How the way a computer works </h1>
</p>

In this article, I will briefly discuss how computers work in terms of retrieving and processing data using an extremely intuitive and easy-to-understand example. 
And please note that this example will be referred to quite often in parallel programming lessons, so I hope you read it carefully.

<p align="center">
 <h1 align="center">Read-Write data </h1>
</p>

Example: Assuming you have 1024 cookies and there are 32 children waiting in line to receive cookies, there are two ways you can distribute the cookies.

<p align="center">
  <img src="https://github.com/CisMine/Parallel-Computing-Cuda-C/assets/122800932/510ef1d3-ec65-4637-bab4-12d4ffa5bc75" />
</p>

## Method 1: In this method, you distribute the cookies in batches of 32. Here’s how it works:
- The first child in line receives a batch of 32 cookies.
- The second child in line also receives a batch of 32 cookies.
- This process continues, and each child in line receives a batch of 32 cookies until you reach the 32nd child.
- The 32nd child receives the last batch of 32 cookies.

## Method 2: In this method, you distribute the cookies one at a time and have the child return to the end of the line after receiving a cookie. Here’s how it works:

- The first child in line receives one cookie and then goes to the back of the line.
- The second child in line receives one cookie and also goes to the back of the line.
- This process continues, with each child receiving one cookie and then joining the end of the line, until you have gone through all 32 children in the line.
- After going through the entire line once, you repeat the process for a total of 32 iterations.
- During the 32nd iteration, the last cookie (the 1,024th cookie) is given to the 32nd child in line.

  
**In the given scenario, it may seem obvious that Method 1 is faster than Method 2. However, when it comes to computers, Method 2 is actually faster. Now, let’s analyze the reasons why computers would choose Method 2.**

## Let’s analyze the example

In this analogy, the cookies represent the data that needs to be processed, and the children represent the individuals processing that data

there are two important points about how computers handle data that are often overlooked:

- Sequential Processing: After a computer finishes processing one set of data, it moves on to the next set of data. It cannot simultaneously process multiple sets of data at the same time. This sequential processing is due to the single-threaded nature of most traditional computing systems.
- Data Transfer and Processing: When data is read by the computer for processing, it is typically copied to a designated area in memory before actual processing occurs. The data cannot be processed directly at the location where it is read. This copying of data to memory allows for efficient processing and manipulation of the data without altering the original source.
  
**Both these aspects contribute to the overall functioning of a computer in handling and processing data.**

## Analyze the 1st method
In the given example, when the first child comes up to receive a cookie, he has to take it to a different place before he can sit down and eats it. After finishing the first cookie, he returns to receive the second cookie and repeats the process. This means that each child must consume their cookies one at a time, rather than eating all 32 cookies simultaneously.

This process repeats 32 times, and during this time, the remaining 31 children have to wait for their turn. You can imagine that the 32nd child has to wait for a certain duration until it’s his turn to receive and eat his cookies.

Another drawback of Method 1 is that the computer needs to perform an additional calculation to determine how many cookies to distribute to each child. In this case, the calculation is not very complex, but it can pose a problem when the number of cookies is not evenly divisible by the number of children.

For example, let’s say we have 1055 cookies (1024 + 31). In this case, it would be ideal if each child processed 33 cookies (32 + 1), while the remaining child processes 32 cookies. However, due to the maximum limit of 32 cookies that each child can handle (it is relevant to warp), we encounter a situation where we have an extra cookie that needs to be processed.

As a result, we would have a loop where the first child would have to process the remaining 31 odd cookies alone

## Analyze the 2nd method
In Method 2, when the first child comes up to receive one cookie and then returns to the waiting line, he has time to consume the first cookie while waiting for his turn to receive the second cookie. This means that the first child is processing the first cookie while the other children can proceed with reading their respective data (cookies) and start processing them while waiting in line.

By doing so, we have addressed the issue of the remaining 31 children not having to wait for the first child to finish processing his cookie. The second child, for example, can start “reading” his data (the cookie) and begin processing it while waiting in line. This allows for a more parallel processing approach, where the children can process their data concurrently rather than having to wait for each other.

Another advantage of Method 2 is that we don’t need to perform additional calculations. In the case of having 1055 cookies (1024 + 31), Method 2 allows us to distribute the 31 odd cookies evenly among the 31 remaining children. Each child can handle one extra cookie along with their designated batch. This eliminates the need for the first child to process all 31 odd cookies alone, as they are distributed evenly among the other children, enabling a more efficient distribution of workload.

**Through the analysis, we can see that Method 2 has cleverly handled the data in a nearly parallel manner, while Method 1 is sequential.**
