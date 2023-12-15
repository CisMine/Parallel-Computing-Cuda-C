#include <iostream>

__managed__ int  y=2; 
__global__ void kernel() {
printf("%d\n", y);
}
int main() {
kernel<<< 1, 1 >>>();

// Error on some GPUs, all CC < 6.0
cudaDeviceSynchronize();
y =20;
printf("%d\n", y);
return 0;
}

