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
	sdata[i] = (i < n) ? g_idata[i] : 0;
	__syncthreads();

	for (int s = 1;s < blockDim.x;s * 2)
	{
		if (tid % (2 * s) == 0)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
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
	dim3 grid((N + block.x - 1) / block.x);

	//设备内存
	int* d_in, * d_partial;
	CUDA_CHECK(cudaMalloc(&d_in, bytes));
	CUDA_CHECK(cudaMalloc(&d_partial, grid.x * sizeof(int)));
	CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

	// 多趟归约

	// warm up

	// 计时

	// 结果

	// 清理
	cudaFree(d_in);
	cudaFree(d_partial);
	free(h_in);
	return 0;
}