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
//Time(avg) : 2.4913 ms
//Effective BW : 26.9 GB / S

__global__ void reduce3(int* g_idata, int* g_odata, int n)
{
	extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	sdata[tid] = (i < n) ? g_idata[i] : 0;
	__syncthreads();

	for (int s = blockDim.x / 2;s > 0;s /= 2)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int* runReduce(int* d_in, int* d_buf1, int* d_buf2, int N, size_t smem)
{
	int curN = N;
	int* src = d_in;
	int* dst = d_buf1;
	int* spare = d_buf2;
	bool first = true;
	while (curN > 1)
	{
		int blocks = (curN + BLOCK_SIZE - 1) / BLOCK_SIZE;
		reduce3 << <blocks, BLOCK_SIZE, smem >> > (src, dst, curN);
		CUDA_CHECK(cudaGetLastError());
		curN = blocks;

		if (first)
		{
			src = dst; dst = spare; first = false;
		}
		else
		{
			int* tmp = src; src = dst; dst = tmp;
		}
	}
	return src;
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
	dim3 grid((N + block.x - 1) / block.x);
	size_t smem = BLOCK_SIZE * sizeof(int);

	//设备内存
	int* d_in, * d_buf1, * d_buf2;
	CUDA_CHECK(cudaMalloc(&d_in, bytes));
	CUDA_CHECK(cudaMalloc(&d_buf1, grid.x * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_buf2, grid.x * sizeof(int)));
	CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

	// warm up
	for (int t = 0;t < 10;t++)
	{
		runReduce(d_in, d_buf1, d_buf2, N, smem);
	}
	CUDA_CHECK(cudaDeviceSynchronize());

	// 计时
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	const int iter = 100;
	int* d_result = nullptr;
	CUDA_CHECK(cudaEventRecord(start));
	for (int i = 0;i < iter;i++)
	{
		d_result = runReduce(d_in, d_buf1, d_buf2, N, smem);
	}
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
	ms /= iter;
	// 结果
	int gpuSum = 0;
	CUDA_CHECK(cudaMemcpy(&gpuSum, d_result, sizeof(int), cudaMemcpyDeviceToHost));
	double actualBW = (double)bytes / (ms / 1e3) / 1e9;

	printf("GPU sum = %d, CPU sum = %d --> %s\n", gpuSum, cpuSum, (gpuSum == cpuSum) ? "PASS" : "FAIL");
	printf("Time (avg)     : %.4f ms\n", ms);
	printf("Effective BW   : %.1f GB/S\n", actualBW);

	// 清理
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_in);
	cudaFree(d_buf1);
	cudaFree(d_buf2);
	free(h_in);
	return 0;
}