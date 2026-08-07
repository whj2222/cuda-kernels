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

__global__ void softmax_v0(float* input, float* output, int M, int N)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= M) return;
	float* x = input + row * N;
	float* y = output + row * N;

	// 求最大值
	float maxNum = -INFINITY;
	for (int i = 0;i < N;i++)
	{
		maxNum = fmaxf(maxNum, x[i]);
	}
	
	// 求和
	float sum = 0.0f;
	for (int i = 0; i < N;i++)
	{
		sum += expf(x[i] - maxNum);
	}

	// 最终结果
	for (int i = 0;i < N;i++)
	{
		y[i] = expf((x[i] - maxNum)) / sum;
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
	int M = 1024;
	int N = 1024;
	size_t byte = M * N * sizeof(float);

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

	dim3 threadPerBlock(256);
	dim3 blockPerGrid((M * N + threadPerBlock.x - 1) / threadPerBlock.x);

	softmax_v0 << <blockPerGrid, threadPerBlock >> > (d_input, d_output, M, N);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, byte, cudaMemcpyDeviceToHost));

	softmax_cpu(h_input, h_output_cpu, M, N);
	if (check_result(h_output_gpu, h_output_cpu, M, N))
	{
		printf("PASS: Result match!\n");
	}else{
		printf("FAIL: Result mismatch.\n");
	}

	free(h_input);
	free(h_output_gpu);
	free(h_output_cpu);
	cudaFree(d_input);
	cudaFree(d_output);

	return 0;
}