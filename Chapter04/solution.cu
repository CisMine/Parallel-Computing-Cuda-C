// host call global
// global call device

#include <stdio.h>

__device__ void Device1()
{
    printf("Device1\n");
}

__device__ void Device2()
{
    printf("Device2");
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

// Device1
// Device2

//-----------------------------------------------------------------------
// host call device
#include <stdio.h>

__device__ void Device1()
{
    printf("Device1\n");
}

__device__ void Device2()
{
    printf("Device2");
}

void sub_Function_in_Host()
{
    Device1();
}

int main()
{
    sub_Function_in_Host();
    Device2();
    cudaDeviceSynchronize();
    return 0;
}

// error: calling a __device__ function("Device1") from a __host__ function("sub_Function_in_Host") is not allowed
// error: calling a __device__ function("Device2") from a __host__ function("main") is not allowed

//---------------------------------------------------------------------
// device call host
#include <stdio.h>

void sub_Function_in_Host()
{
    printf("host function");
}

__device__ void Device1()
{
    sub_Function_in_Host();
}

int main()
{
    Device1();
    cudaDeviceSynchronize();
    return 0;
}

// error: calling a __host__ function("sub_Function_in_Host") from a __device__ function("Device1") is not allowed
// error: identifier "sub_Function_in_Host" is undefined in device code

//------------------------------------------------------------
// global call host

#include <stdio.h>

void sub_Function_in_Host()
{
    printf("host function");
}

__global__ void kernel()
{
    sub_Function_in_Host();
}

int main()
{
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}

// error: calling a __host__ function("sub_Function_in_Host") from a __global__ function("kernel") is not allowed
// error: identifier "sub_Function_in_Host" is undefined in device code

//-----------------------------------------------------------------
// device call global

#include <stdio.h>

__global__ void kernel()
{
    printf("kernel function");
}

__device__ void Device1()
{
    kernel<<<1, 1>>>();
}

int main()
{
    Device1();
    cudaDeviceSynchronize();
    return 0;
}

// error: calling a __global__ function("kernel") from a __device__ function("Device1") is only allowed on the compute_35 architecture or above

// -----------------------------------------------------------
#include <stdio.h>

__global__ void kernel1()
{
    printf("kernel1\n");
}

__global__ void kernel2()
{
    printf("kernel2\n");
}

int main()
{
    kernel1<<<1, 1>>>();
    printf("CPU here\n");
    kernel2<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("CPU also here\n");
    return 0;
}

// CPU here
// kernel1
// kernel2
// CPU also here
