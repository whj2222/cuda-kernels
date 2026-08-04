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

#define BLOCK_SIZE 256

//GPU sum = 16777216, CPU sum = 16777216 -- > PASS
//Time(avg) : 0.4612 ms
//Effective BW : 145.5 GB / S

template<unsigned int blockSize>
__global__ void reduce6(int* g_idata, int* g_odata, int n)
{
	extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int i = (blockDim.x * 2) * blockIdx.x + threadIdx.x;
	int gridSize = blockDim.x * 2 * gridDim.x;
	sdata[tid] = 0;
	while (i < n)
	{
		sdata[tid] += g_idata[i] + g_idata[i + blockSize];
		i += gridSize;
	}
	__syncthreads();

	if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; } __syncthreads();
	if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; } __syncthreads();
	if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[tid + 64]; } __syncthreads();
	if (tid < 32)
	{
		volatile int* v = sdata;
		if (blockSize >= 64) v[tid] += v[tid + 32]; __syncwarp();
		if (blockSize >= 32) v[tid] += v[tid + 16]; __syncwarp();
		if (blockSize >= 16) v[tid] += v[tid + 8]; __syncwarp();
		if (blockSize >= 8) v[tid] += v[tid + 4]; __syncwarp();
		if (blockSize >= 4) v[tid] += v[tid + 2]; __syncwarp();
		if (blockSize >= 2) v[tid] += v[tid + 1]; __syncwarp();
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


int main()
{
	// 准备数据
	const int N = 1 << 24;
	const size_t bytes = N * sizeof(int);

	int* h_in = (int*)malloc(bytes);
	int cpuSum = 0;
	for (int i = 0;i < N;i++)
	{
		h_in[i] = 1;
		cpuSum += h_in[i];
	}

	// 启动配置
	dim3 block(BLOCK_SIZE);
	dim3 grid(1024);
	size_t smem = BLOCK_SIZE * sizeof(int);

	//设备内存
	int* d_in, * d_out;
	CUDA_CHECK(cudaMalloc(&d_in, bytes));
	CUDA_CHECK(cudaMalloc(&d_out, grid.x * sizeof(int)));
	CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

	// warm up
	for (int t = 0;t < 10;t++)
	{
		reduce6<BLOCK_SIZE> << <grid, block, smem >> > (d_in, d_out, N);
		CUDA_CHECK(cudaGetLastError());
	}
	CUDA_CHECK(cudaDeviceSynchronize());

	// 计时
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	const int iter = 100;
	CUDA_CHECK(cudaEventRecord(start));
	for (int i = 0;i < iter;i++)
	{
		reduce6<BLOCK_SIZE> << <grid, block, smem >> > (d_in, d_out, N);
		CUDA_CHECK(cudaGetLastError());
		reduce6<BLOCK_SIZE> << <1, block, smem >> > (d_out, d_out, grid.x);
		CUDA_CHECK(cudaGetLastError());
	}
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
	ms /= iter;
	// 结果
	int gpuSum = 0;
	CUDA_CHECK(cudaMemcpy(&gpuSum, d_out, sizeof(int), cudaMemcpyDeviceToHost));
	double actualBW = (double)bytes / (ms / 1e3) / 1e9;

	printf("GPU sum = %d, CPU sum = %d --> %s\n", gpuSum, cpuSum, (gpuSum == cpuSum) ? "PASS" : "FAIL");
	printf("Time (avg)     : %.4f ms\n", ms);
	printf("Effective BW   : %.1f GB/S\n", actualBW);

	// 清理
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_in);
	cudaFree(d_out);
	free(h_in);
	return 0;
}