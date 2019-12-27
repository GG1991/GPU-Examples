#include <iostream>

using namespace std;

__global__
void dot_prod_kernel(const double *arr1_d, const double *arr2_d, double *g_res, const int n)
{    
	// There is a different shared memory for each block
	extern __shared__ double sdata[]; 
	int it = threadIdx.x + blockDim.x * blockIdx.x;

	sdata[threadIdx.x] = (it < n) ? arr1_d[it] * arr2_d[it] : 0; // mv data to shared memory
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (threadIdx.x % (2 * s) == 0) {
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}
		__syncthreads();
	}
	//save result for this block on global memory
	if (threadIdx.x == 0) g_res[blockIdx.x] = sdata[0];
}

int main()
{
	int grid = 1000;
	int blocks = 128;
	int shared = 128 * sizeof(double);

	int n = 1 << 22;
	double *arr1_h, *arr2_h, *arr1_d, *arr2_d;
	double *g_res_h, *g_res_d;
	
	arr1_h = (double *)malloc(n * sizeof(double));
	arr2_h = (double *)malloc(n * sizeof(double));
	g_res_h = (double *)malloc(blocks * sizeof(double));

	cudaMalloc((void **)&g_res_d, blocks * sizeof(double));
	cudaMalloc((void **)&arr1_d, n * sizeof(double));
	cudaMalloc((void **)&arr2_d, n * sizeof(double));

	for (int i = 0; i < n; ++i) {
		arr1_h[i] = 1.0;
		arr2_h[i] = 1.0;
	}

	cudaMemcpy(arr1_d, arr1_h, n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(arr2_d, arr2_h, n * sizeof(double), cudaMemcpyHostToDevice);

	dot_prod_kernel<<<grid, blocks, shared>>>(arr1_d, arr2_d, g_res_d, n);

	cudaMemcpy(g_res_h, g_res_d, blocks * sizeof(double), cudaMemcpyDeviceToHost);

	double result = 0.0;
	for (int i = 0; i < blocks; ++i) {
		result += g_res_h[i];
	}
	cout << "result = " << result << endl;

	free(arr1_h);
	free(arr2_h);
	free(g_res_h);
	cudaFree(arr1_d);
	cudaFree(arr2_d);
	cudaFree(g_res_d);

	return 0;
}
