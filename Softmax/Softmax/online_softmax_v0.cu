
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


__global__ void online_softmax_v0(float* input, float* output, int M, int N)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= M) return;

	float* x = input + row * N;
	float* y = output + row * N;

	// online softmax同时计算max和sum
	float m = -INFINITY;
	float d = 0.0f;

	for (int i = 0;i < N;i++)
	{
		float xi = x[i];
		float m_new = fmax(m, xi);
		d = d * expf(m - m_new) + expf(xi - m_new);
		m = m_new;
	}

	// 归一化
	for (int i = 0;i < N;i++)
	{
		y[i] = expf(x[i] - m) / d;
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
		online_softmax_v0 << <M, threadPerBlock >> > (d_input, d_output, M, N);
	}
	CUDA_CHECK(cudaDeviceSynchronize());

	int repeat = 20;

	CUDA_CHECK(cudaEventRecord(start));
	for (int i = 0;i < repeat;i++)
	{
		online_softmax_v0 << <M, threadPerBlock >> > (d_input, d_output, M, N);
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