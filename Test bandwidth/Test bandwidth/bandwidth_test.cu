#include<cstdio>
#include<cuda_runtime.h>

#define CUDA_CHECK(call)                                 \
do {                                                     \
	cudaError_t err = call;                              \
	if (err != cudaSuccess) {                            \
		fprintf(stderr, "CUDA error %s:%d: %s\n",        \
		__FILE__, __LINE__, cudaGetErrorString(err));    \
		exit(EXIT_FAILURE);                              \
	}                                                    \
	}while(0)

__global__ void copyKernel(const float* in, float* out, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		out[i] = in[i];
	}
}

int main()
{
	const int dev = 0;
	cudaDeviceProp p;
	CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

	// 理论峰值带宽
	int memClockKHz = 0;
	CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev));
	double theoryBW = 2.0 * memClockKHz * (p.memoryBusWidth / 8) / 1.0e6;
	printf("Device: %s\n", p.name);
	printf("Bus width: %d bit, Mem clock: %.0f MHz\n", p.memoryBusWidth, memClockKHz / 1000.0);
	printf("Theoretical bandwidth: %.1f GB/s\n\n", theoryBW);

	// 数据准备
	const int N = 1 << 26;
	const size_t bytes = N * sizeof(float);

	float* d_in, * d_out;
	CUDA_CHECK(cudaMalloc(&d_in, bytes));
	CUDA_CHECK(cudaMalloc(&d_out, bytes));
	CUDA_CHECK(cudaMemset(d_in, 1, bytes));

	dim3 block(256);
	dim3 grid((N + block.x - 1) / block.x);
	
	// warm up
	for (int i = 0;i < 10;i++) copyKernel << <grid, block >> > (d_in, d_out, N);
	
	// 计时

	// 实测带宽

	// 清理
	return 0;
}
