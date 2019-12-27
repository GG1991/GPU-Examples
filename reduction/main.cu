#include <iostream>

using namespace std;

__global__
void reduce(const double *arr_d, double *g_res, const int n)
{    
	// There is a different shared memory for each block
	extern __shared__ double sdata[]; 
	int it = threadIdx.x + blockDim.x * blockIdx.x;

	sdata[threadIdx.x] = (it < n) ? arr_d[it] : 0; // mv data to shared memory
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
	double *arr_h, *arr_d;
	double *g_res_h, *g_res_d;
	
	arr_h = (double *)malloc(n * sizeof(double));
	g_res_h = (double *)malloc(blocks * sizeof(double));

	cudaMalloc((void **)&g_res_d, blocks * sizeof(double));
	cudaMalloc((void **)&arr_d, n * sizeof(double));

	for (int i = 0; i < n; ++i) {
		arr_h[i] = 1.1;
	}

	cudaMemcpy(arr_d, arr_h, n * sizeof(double), cudaMemcpyHostToDevice);

	cout << "launching kernel : ";
	reduce<<<grid, blocks, shared>>>(arr_d, g_res_d, n);
	cout << "OK " << endl;

	cudaMemcpy(g_res_h, g_res_d, blocks * sizeof(double), cudaMemcpyDeviceToHost);

	double result = 0.0;
	for (int i = 0; i < blocks; ++i) {
		result += g_res_h[i];
	}
	cout << "result = " << result << endl;

	free(arr_h);
	free(g_res_h);
	cudaFree(arr_d);
	cudaFree(g_res_d);

	return 0;
}
