

#define NPE 8
#define NVOI 6
#define DIM 3

struct CUDA_vars {

	int nex, ney, nez;
	int nx, ny, nz;
	double bmat_cache[NPE][NVOI][NPE * DIM];

};

struct CUDA_vars CUDA_vars_h;
__device__ struct CUDA_vars CUDA_vars_d;

int main() {

	CUDA_vars_h.nex = 45;
	CUDA_vars_h.ney = 66;
	CUDA_vars_h.nez = 11;

	CUDA_vars_h.nx = 45;
	CUDA_vars_h.ny = 66;
	CUDA_vars_h.nz = 11;

	for (int i = 0; i < NPE; ++i) {
		for (int j = 0; j < NVOI; ++j) {
			for (int k = 0; k < NPE * DIM; ++k) {
				CUDA_vars_h.bmat_cache[i][j][k] = 1.0;
			}
		}
	}

	cudaMemcpy(&CUDA_vars_d, &CUDA_vars_h,
		       	sizeof(CUDA_vars), cudaMemcpyHostToDevice);

}
