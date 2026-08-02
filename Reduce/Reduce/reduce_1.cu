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

__global__ void reduce0(int* g_idata, int* g_odata, int n)
{
	extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	sdata[tid] = (i < n) ? g_idata[i] : 0;
	__syncthreads();

	for (int s = 1;s < blockDim.x;s *= 2)
	{
		if (tid % (2 * s) == 0)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int* runReduce(int* src, int* dst, int N, size_t smem)
{
	int curN = N;
	while (curN > 1)
	{
		int blocks = (curN + BLOCK_SIZE - 1) / BLOCK_SIZE;
		reduce0 << <blocks, BLOCK_SIZE, smem >> > (src, dst, curN);
		CUDA_CHECK(cudaGetLastError());
		curN = blocks;

		int* tmp = src; src = dst; dst = tmp;
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
	int* d_in, * d_partial;
	CUDA_CHECK(cudaMalloc(&d_in, bytes));
	CUDA_CHECK(cudaMalloc(&d_partial, grid.x * sizeof(int)));
	CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

	// warm up
	for (int t = 0;t < 10;t++)
	{
		runReduce(d_in, d_partial, N, smem);
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
		d_result = runReduce(d_in, d_partial, N, smem);
	}
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	
	float ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
	// 结果

	// 清理
	cudaFree(d_in);
	cudaFree(d_partial);
	free(h_in);
	return 0;
}