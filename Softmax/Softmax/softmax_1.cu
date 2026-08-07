#include<cstdio>
#include<cuda_runtime.h>


__global__ void softmax_v0(float* input, float* output, int N, int M)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	float* x = input + row * N;
	float* y = input + row * N;

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
		y[i] = expf((x[i] - maxNum) / sum);
	}
}


int main()
{
	return 0;
}