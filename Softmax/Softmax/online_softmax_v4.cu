
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

__device__ void warpReduceOnline(float& m, float& d)
{
	for (int offset = 16;offset > 0;offset >>= 1)
	{
		float m2 = __shfl_down_sync(0xffffffff, m, offset);
		float d2 = __shfl_down_sync(0xffffffff, d, offset);

		float m_new = fmaxf(m, m2);
		d = d * expf(m - m_new) + d2 * expf(m2 - m_new);
		m = m_new;
	}
}

#define MAX_ELEMS_PER_THREAD 32

__global__ void online_softmax_v4(float* input, float* output, int M, int N)
{
	int row = blockIdx.x;
	int tid = threadIdx.x;
	int lane = tid % 32;
	int wid = tid / 32;

	for (int row = blockIdx.x;row < M;row += gridDim.x)
	{
		float* x = input + row * N;
		float* y = output + row * N;

		float4* x4 = reinterpret_cast<float4*>(x);
		float4* y4 = reinterpret_cast<float4*>(y);
		int N4 = N / 4;

		float4 reg_cache[MAX_ELEMS_PER_THREAD];

		// 每个线程处理一段
		float local_m = -INFINITY;
		float local_d = 0.0f;

		int count = 0;
		for (int i = tid;i < N4;i += blockDim.x)
		{
			float4 data = x4[i];
			reg_cache[count] = data;
			count++;
			float vals[4] = { data.x, data.y, data.z, data.w };
			for (int k = 0;k < 4;k++)
			{
				float m_new = fmaxf(local_m, vals[k]);
				local_d = local_d * expf(local_m - m_new) + expf(vals[k] - m_new);
				local_m = m_new;
			}
		}
		for (int i = N4 * 4;i < N;i += blockDim.x)
		{
			float m_new = fmaxf(local_m, x[i]);
			local_d = local_d * expf(local_m - m_new) + expf(x[i] - m_new);
			local_m = m_new;
		}

		// warp合并
		warpReduceOnline(local_m, local_d);

		__shared__ float warp_m[32];
		__shared__ float warp_d[32];

		if (lane == 0)
		{
			warp_m[wid] = local_m;
			warp_d[wid] = local_d;
		}
		__syncthreads();

		// 最后一个warp归约
		int num_warp = blockDim.x / 32;
		if (wid == 0)
		{
			local_m = (lane < num_warp) ? warp_m[lane] : -INFINITY;
			local_d = (lane < num_warp) ? warp_d[lane] : 0.0f;
			warpReduceOnline(local_m, local_d);
		}
		// 广播最终结果

		__shared__ float final_m;
		__shared__ float final_d;
		if (tid == 0)
		{
			final_m = local_m;
			final_d = local_d;
		}
		__syncthreads();
		// 归一化
		int idx = 0;
		for (int i = tid;i < N4;i += blockDim.x)
		{
			float4 data = reg_cache[idx];
			float4 out;
			out.x = exp(data.x - final_m) / final_d;
			out.y = exp(data.y - final_m) / final_d;
			out.z = exp(data.z - final_m) / final_d;
			out.w = exp(data.w - final_m) / final_d;
			y4[i] = out;
		}
		for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
			y[i] = expf(x[i] - final_m) / final_d;
		}
		__syncthreads();
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
	int M = 4096;
	int N = 4096;
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
		online_softmax_v4 << <M, threadPerBlock >> > (d_input, d_output, M, N);
	}
	CUDA_CHECK(cudaDeviceSynchronize());

	int repeat = 20;

	CUDA_CHECK(cudaEventRecord(start));
	for (int i = 0;i < repeat;i++)
	{
		online_softmax_v4 << <M, threadPerBlock >> > (d_input, d_output, M, N);
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