
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
//Matrix size : 1024 x 1024
//Memory size : 4.00 MB
//======================================== =
//PASS : Result match!
//Performance States :
//Matrix Size : 1024 x 1024
//Avg Time per run : 0.08 ms
//Effective Bandwidth : 102.91 GB / s
//Throughput(approx) : 12.86 GFLOPS(based on element count)

// Warp内归约: max
__device__ float warpReduceMax(float val)
{
	for (int offset = 16; offset > 0;offset >>= 1)
	{
		val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
	}
	return val;
}

// Warp内归约: sum
__device__ float warpReduceSum(float val)
{
	for (int offset = 16;offset > 0;offset >>= 1)
	{
		val += __shfl_down_sync(0xffffffff, val, offset);
	}
	return val;
}

// block内归约: max
__device__ float blockReduceMaxShuffle(float val)
{
	__shared__ float  warp_max[32];

	int lane = threadIdx.x % 32;
	int wid = threadIdx.x / 32;

	val = warpReduceMax(val);

	if (lane == 0) warp_max[wid] = val;
	__syncthreads();

	int num_warps = blockDim.x / 32;
	val = (lane < num_warps) ? warp_max[lane] : -INFINITY;
	if (wid == 0) val = warpReduceMax(val);

	__shared__ float block_result;
	if (threadIdx.x == 0) block_result = val;
	__syncthreads();
	return block_result;
}

// block内归约: sum
__device__ float blockReduceSumShuffle(float val)
{
	__shared__ float warp_sum[32];

	int lane = threadIdx.x % 32;
	int wid = threadIdx.x / 32;

	val = warpReduceSum(val);

	if (lane == 0) warp_sum[wid] = val;
	__syncthreads();

	int num_warps = blockDim.x / 32;
	val = (lane < num_warps) ? warp_sum[lane] : 0.0f;
	if (wid == 0) val = warpReduceSum(val);

	__shared__ float block_result;
	if (threadIdx.x == 0) block_result = val;
	__syncthreads();
	return block_result;
}


__global__ void softmax_v4(float* input, float* output, int M, int N)
{
	extern __shared__ float smem[];

	int row = blockIdx.x;
	int tid = threadIdx.x;

	float* x = input + row * N;
	float* y = output + row * N;

	// 向量化指针
	float4* x4 = reinterpret_cast<float4*>(x);
	float4* y4 = reinterpret_cast<float4*>(y);
	float4* smem4 = reinterpret_cast<float4*>(smem);
	int N4 = N / 4;

	// 求最大值 
	float local_max = -INFINITY;
	for (int i = tid;i < N4;i += blockDim.x)
	{
		float4 data = x4[i];
		local_max = fmaxf(local_max, fmaxf(fmaxf(data.w, data.x), fmaxf(data.y, data.z)));
	}
	for (int i = N4 * 4 + tid;i < N;i += blockDim.x)
	{
		local_max = fmaxf(local_max, x[i]);
	}
	float max_val = blockReduceMaxShuffle(local_max);

	// 求指数和
	float local_sum = 0.0f;
	for (int i = tid; i < N4; i += blockDim.x) {
		float4 data = x4[i];
		float4 e;
		e.x = expf(data.x - max_val);
		e.y = expf(data.y - max_val);
		e.z = expf(data.z - max_val);
		e.w = expf(data.w - max_val);
		smem4[i] = e;                                   // 一次存 4 个
		local_sum += e.x + e.y + e.z + e.w;
	}
	for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
		float e = expf(x[i] - max_val);
		smem[i] = e;
		local_sum += e;
	}
	float sum_val = blockReduceSumShuffle(local_sum);
	// 归一化
	float inv_sum = 1.0f / sum_val;                     // 乘法比除法快
	for (int i = tid; i < N4; i += blockDim.x) {
		float4 e = smem4[i];                            // 一次读 4 个
		float4 result;
		result.x = e.x * inv_sum;
		result.y = e.y * inv_sum;
		result.z = e.z * inv_sum;
		result.w = e.w * inv_sum;
		y4[i] = result;                                 // 一次写 4 个
	}
	for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
		y[i] = smem[i] * inv_sum;
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
			printf("i = %d, GPU result: %f, CPU result: %f\n", i, gpu_res[i], cpu_res[i]);
			return false;
		}
	}
	return true;
}

int main()
{
	int M = 1024;
	int N = 1024;
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
	int smem_size = N * sizeof(float);
	float milliseconds = 0;

	// warm up
	for (int i = 0;i < 20;i++)
	{
		softmax_v4 << <M, threadPerBlock, smem_size >> > (d_input, d_output, M, N);
	}
	CUDA_CHECK(cudaDeviceSynchronize());

	int repeat = 20;

	CUDA_CHECK(cudaEventRecord(start));
	for (int i = 0;i < repeat;i++)
	{
		softmax_v4 << <M, threadPerBlock, smem_size >> > (d_input, d_output, M, N);
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