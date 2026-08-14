
#include<cstdio>
#include<cuda_runtime.h>

#define CUDA_CHECK(call)\
do {\
	cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error));\
		exit(1);\
	}\
} while (0)

//========== Softmax Performance ==========
//Matrix size : 4096 x 4096
//Memory size : 64.00 MB
//======================================== =
//PASS : Result match!
//Performance States :
//Matrix Size : 4096 x 4096
//Avg Time per run : 6.46 ms
//Effective Bandwidth : 20.77 GB / s
//Throughput(approx) : 2.60 GFLOPS(based on element count)

//========== Softmax Performance ==========
//Matrix size : 5120 x 5120
//Memory size : 100.00 MB
//======================================== =
//PASS : Result match!
//Performance States :
//Matrix Size : 5120 x 5120
//Avg Time per run : 8.30 ms
//Effective Bandwidth : 25.28 GB / s
//Throughput(approx) : 3.16 GFLOPS(based on element count)

// 一个block处理一行
__global__ void softmax_v1(float* input, float* output, int M, int N)
{
	extern __shared__ float smem[];

	int row = blockIdx.x;
	int tid = threadIdx.x;

	float* x = input + row * N;
	float* y = output + row * N;

	// 并行计算max
	float max_val = -INFINITY;
	for (int i = tid; i < N;i += blockDim.x)
	{
		max_val = fmaxf(max_val, x[i]);
	}
	__syncthreads();
	smem[tid] = max_val;

	// 归约计算全局最大值
	for (int i = blockDim.x / 2; i > 0;i >>= 1)
	{
		if (tid < i) smem[tid] = fmaxf(smem[tid], smem[tid + i]);
		__syncthreads();
	}
	max_val = smem[0];
	__syncthreads();

	// 并行计算指数和
	float sum = 0.0f;
	for (int i = tid;i < N;i += blockDim.x)
	{
		sum += expf(x[i] - max_val);
	}
	smem[tid] = sum;
	__syncthreads();

	// 归约计算总指数和
	for (int i = blockDim.x / 2;i > 0;i >>= 1)
	{
		if (tid < i) smem[tid] += smem[tid + i];
		__syncthreads();
	}

	sum = smem[0];
	__syncthreads();

	// 计算softmax
	for (int i = tid; i < N;i += blockDim.x)
	{
		y[i] = expf(x[i] - max_val) / sum;
	}

}

// CPU 端的 Softmax 实现，用于结果验证
void softmax_cpu(float* input, float* output, int M, int N) {
	for (int row = 0; row < M; row++) {
		float* x = input + row * N;
		float* y = output + row * N;

		float max_val = -INFINITY;
		for (int i = 0; i < N; i++) {
			if (x[i] > max_val) max_val = x[i];
		}

		float sum = 0.0f;
		for (int i = 0; i < N; i++) {
			sum += expf(x[i] - max_val);
		}

		for (int i = 0; i < N; i++) {
			y[i] = expf(x[i] - max_val) / sum;
		}
	}
}

bool check_result(float* gpu_res, float* cpu_res, int M, int N)
{
	float eps = 1e-5;
	for (int i = 0;i < M * N;i++)
	{
		if (fabs(gpu_res[i] - cpu_res[i]) > eps)
		{
			return false;
		}
	}
	return true;
}

int main()
{
	int M = 6144;
	int N = 6144;
	size_t byte = M * N * sizeof(float);
	int num_elements = M * N;

	printf("========== Softmax Performance ==========\n");
	printf("Matrix size: %d x %d\n", M, N);
	printf("Memory size: %.2f MB\n", byte / (1024.0 * 1024.0));
	printf("=========================================\n");

	float* h_input = (float*)malloc(byte);
	float* h_output_gpu = (float*)malloc(byte);
	float* h_output_cpu = (float*)malloc(byte);


	for (int i = 0; i < M * N;i++)
	{
		h_input[i] = (float)rand() / RAND_MAX * 10.0F;
	}

	float* d_input;
	float* d_output;
	CUDA_CHECK(cudaMalloc(&d_input, byte));
	CUDA_CHECK(cudaMalloc(&d_output, byte));
	CUDA_CHECK(cudaMemcpy(d_input, h_input, byte, cudaMemcpyHostToDevice));


	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	dim3 threadPerBlock(256);
	int smem_size = threadPerBlock.x * sizeof(float);
	float milliseconds = 0;

	// warm up
	for (int i = 0;i < 20;i++)
	{
		softmax_v1 << <M, threadPerBlock, smem_size >> > (d_input, d_output, M, N);
	}
	CUDA_CHECK(cudaDeviceSynchronize());

	int repeat = 20;

	CUDA_CHECK(cudaEventRecord(start));
	for (int i = 0;i < repeat;i++)
	{
		softmax_v1 << <M, threadPerBlock, smem_size >> > (d_input, d_output, M, N);
	}
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventSynchronize(stop));
	CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, byte, cudaMemcpyDeviceToHost));



	float avg_time_ms = milliseconds / repeat;

	softmax_cpu(h_input, h_output_cpu, M, N);
	if (check_result(h_output_gpu, h_output_cpu, M, N))
	{
		printf("PASS: Result match!\n");
	}
	else {
		printf("FAIL: Result mismatch.\n");
	}

	double total_ops = (double)M * (double)N;
	double time_sec = avg_time_ms / 1000.0;
	double effective_bandwidth = (2 * M * N * sizeof(float)) / (time_sec * 1e9);
	double gflops = total_ops / time_sec * 1e-9;

	printf("Performance States:\n");
	printf("  Matrix Size: %d x %d\n", M, N);
	printf("  Avg Time per run: %.2f ms\n", avg_time_ms);
	printf("  Effective Bandwidth: %.2f GB/s\n", effective_bandwidth);
	printf("  Throughput (approx): %.2f GFLOPS (based on element count)\n", gflops);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	free(h_input);
	free(h_output_gpu);
	free(h_output_cpu);
	cudaFree(d_input);
	cudaFree(d_output);

	system("pause");
	return 0;
}