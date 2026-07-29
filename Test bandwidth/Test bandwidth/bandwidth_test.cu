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


int main()
{
	const int dev = 0;
	cudaDeviceProp p;
	CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

	// 理论峰值带宽
	int memClockKHz = 0;
	cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev);
	double theoryBW = 2.0 * memClockKHz * (p.memoryBusWidth / 8) / 1.0e6;
	printf("Device: %s\n", p.name);
	printf("Bus width: %d bit, Mem clock: %.0f MHz\n", p.memoryBusWidth, memClockKHz / 1000.0);
	printf("Theoretical bandwidth: %.1f GB/s\n\n", theoryBW);
	// 数据准备

	// warm up

	// 计时

	// 实测带宽

	// 清理
	return 0;
}
