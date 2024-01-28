#include <stdio.h>

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

int main()
{
    const int blockSize = 768, nStreams = 4;
    const int n = 256 * 256;
    const int streamSize = n / nStreams;
    const int streamBytes = streamSize * sizeof(float);
    const int bytes = n * sizeof(float);
    float ms;

    // Host array
    float *a;
    cudaMallocHost((void **)&a, bytes);
    memset(a, 0, bytes);

    // Device pointer array
    float **d_a = (float **)malloc(nStreams * sizeof(float *));
    cudaStream_t stream[nStreams];

    cudaEvent_t startEvent, stopEvent;
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    for (int i = 0; i < nStreams; ++i)
    {
        cudaStreamCreate(&stream[i]);
        int offset = i * streamSize;

        checkCuda(cudaMallocAsync((void **)&d_a[i], streamBytes, stream[i])); // Allocate memory on the device asynchronously

        checkCuda(cudaMemcpyAsync(d_a[i], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]));

        kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(*d_a, offset);

        checkCuda(cudaMemcpyAsync(&a[offset], d_a[i], streamBytes, cudaMemcpyDeviceToHost, stream[i]));
    }

    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Time for sequential transfer and execute (ms): %f\n", ms);
    printf("  max error: %e\n", maxError(a, n));

    // Synchronize and clean up
    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));
    for (int i = 0; i < nStreams; ++i)
        checkCuda(cudaStreamDestroy(stream[i]));

    cudaFree(d_a);
    cudaFreeHost(a);

    return 0;
}