#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess)
  {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void kernel(float *a, int offset)
{
  int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
  float x = (float)i;
  float s = sinf(x);
  float c = cosf(x);
  a[i] = a[i] + sqrtf(s * s + c * c);
}

float maxError(float *a, int n)
{
  float maxE = 0;
  for (int i = 0; i < n; i++)
  {
    float error = fabs(a[i] - 1.0f);
    if (error > maxE)
      maxE = error;
  }
  return maxE;
}

int main(int argc, char **argv)
{
  const int blockSize = 256, nStreams = 4;
  const int n = 4 * 1024 * blockSize * nStreams;
  const int streamSize = n / nStreams;
  const int streamBytes = streamSize * sizeof(float);
  const int bytes = n * sizeof(float);

  // Allocate pinned host memory and device memory
  float *a, *d_a;
  checkCuda(cudaMallocHost((void **)&a, bytes)); // Host pinned
  checkCuda(cudaMalloc((void **)&d_a, bytes));   // Device

  float ms; // Elapsed time in milliseconds

  // Create events and streams
  cudaStream_t stream[nStreams];

  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));

  for (int i = 0; i < nStreams; ++i)
    checkCuda(cudaStreamCreate(&stream[i]));

  // Baseline case - sequential transfer and execute
  memset(a, 0, bytes);
  checkCuda(cudaEventRecord(startEvent, 0));
  checkCuda(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
  kernel<<<n / blockSize, blockSize>>>(d_a, 0);
  checkCuda(cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost));
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for sequential transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // Asynchronous version 1: loop over {copy, kernel, copy}
  memset(a, 0, bytes);
  checkCuda(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    checkCuda(cudaMemcpyAsync(&d_a[offset], &a[offset],
                              streamBytes, cudaMemcpyHostToDevice,
                              stream[i]));

    kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);

    checkCuda(cudaMemcpyAsync(&a[offset], &d_a[offset],
                              streamBytes, cudaMemcpyDeviceToHost,
                              stream[i]));
  }

  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // Asynchronous version 2:
  // Loop over copy, loop over kernel, loop over copy memset(a, 0, bytes);
  memset(a, 0, bytes);
  checkCuda(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    checkCuda(cudaMemcpyAsync(&d_a[offset], &a[offset],
                              streamBytes, cudaMemcpyHostToDevice,
                              stream[i]));
  }

  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
  }

  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    checkCuda(cudaMemcpyAsync(&a[offset], &d_a[offset],
                              streamBytes, cudaMemcpyDeviceToHost,
                              stream[i]));
  }

  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // Cleanup
  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));
  for (int i = 0; i < nStreams; ++i)
    checkCuda(cudaStreamDestroy(stream[i]));
  cudaFree(d_a);
  cudaFreeHost(a);

  return 0;
}

