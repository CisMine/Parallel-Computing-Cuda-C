#include <stdio.h>

__constant__ int constantData[2]; // Khai báo mảng Constant memory

__global__ void kernel(int *d_x, int *d_y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        int x = d_x[idx];
        int a = constantData[0]; // Lấy giá trị 3 từ Constant memory
        int b = constantData[1]; // Lấy giá trị 5 từ Constant memory
        d_y[idx] = a * x + b;
    }
}

int main() {
    const int N = 10; // Số phần tử mảng
    int h_x[N];    // Mảng đầu vào trên host
    int h_y[N];    // Mảng kết quả trên host
    int *d_x, *d_y; // Mảng trên device

    // Khởi tạo dữ liệu trên host
    for (int i = 0; i < N; i++) {
        h_x[i] = i;
    }

    // Khởi tạo vector trên GPU
    cudaMalloc((void**)&d_x, N * sizeof(int));
    cudaMalloc((void**)&d_y, N * sizeof(int));

    // Sao chép dữ liệu từ host vào device
    cudaMemcpy(d_x, h_x, N * sizeof(int), cudaMemcpyHostToDevice);

    // Sao chép giá trị 3 và 5 vào Constant memory
    int constantValues[2] = {3, 5};
    cudaMemcpyToSymbol(constantData, constantValues, 2 * sizeof(int));

    // Gọi kernel với 1 block và N threads
    kernel<<<1, N>>>(d_x, d_y, N);
    cudaDeviceSynchronize();

    // Sao chép kết quả từ device về host
    cudaMemcpy(h_y, d_y, N * sizeof(int), cudaMemcpyDeviceToHost);

    // In kết quả
    for (int i = 0; i < N; i++) {
        printf("3(x= %d) + 5 => y = %d\n", h_x[i], h_y[i]);
    }

    // Giải phóng bộ nhớ trên device
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}


