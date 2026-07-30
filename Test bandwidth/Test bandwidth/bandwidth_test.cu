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
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// 计时
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	int iters = 100;
	CUDA_CHECK(cudaEventRecord(start));
	for (int i = 0;i < iters;i++) copyKernel << <grid, block >> > (d_in, d_out, N);
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
	
	// 实测带宽
	double movedBytes = 2 * bytes;
	double avgMs = ms / iters;                        
	double actualBW = movedBytes / ((avgMs / 1000) * 1e9);

	printf("Kernel time (average)   : %.4f ms\n", avgMs);
	printf("Measured bandwidth      : %.1f GB/s\n", actualBW);
	printf("Efficiency              : %.1f%% of theoretical peak\n", actualBW / theoryBW * 100);
	// 清理
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}
